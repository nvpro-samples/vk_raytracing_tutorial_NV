/* Copyright (c) 2014-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


#include "raytrace.hpp"
#include "nvh/fileoperations.hpp"
#include "nvvk/descriptorsets_vk.hpp"

#include "nvvk/shaders_vk.hpp"
#include "obj_loader.h"

extern std::vector<std::string> defaultSearchPaths;


void Raytracer::setup(const vk::Device&         device,
                      const vk::PhysicalDevice& physicalDevice,
                      nvvk::Allocator*          allocator,
                      uint32_t                  queueFamily)
{
  m_device         = device;
  m_physicalDevice = physicalDevice;
  m_alloc          = allocator;

  m_graphicsQueueIndex = queueFamily;

  // Requesting ray tracing properties
  auto properties = m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                                    vk::PhysicalDeviceRayTracingPropertiesNV>();
  m_rtProperties  = properties.get<vk::PhysicalDeviceRayTracingPropertiesNV>();
  m_rtBuilder.setup(m_device, allocator, m_graphicsQueueIndex);

  m_debug.setup(device);
}


void Raytracer::destroy()
{
  m_rtBuilder.destroy();
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_alloc->destroy(m_rtSBTBuffer);
}

//--------------------------------------------------------------------------------------------------
// Converting a OBJ primitive to the ray tracing geometry used for the BLAS
//
vk::GeometryNV Raytracer::objectToVkGeometryNV(const ObjModel& model)
{
  vk::GeometryTrianglesNV triangles;
  triangles.setVertexData(model.vertexBuffer.buffer);
  triangles.setVertexOffset(0);  // Start at the beginning of the buffer
  triangles.setVertexCount(model.nbVertices);
  triangles.setVertexStride(sizeof(VertexObj));
  triangles.setVertexFormat(vk::Format::eR32G32B32Sfloat);  // 3xfloat32 for vertices
  triangles.setIndexData(model.indexBuffer.buffer);
  triangles.setIndexOffset(0 * sizeof(uint32_t));
  triangles.setIndexCount(model.nbIndices);
  triangles.setIndexType(vk::IndexType::eUint32);  // 32-bit indices
  vk::GeometryDataNV geoData;
  geoData.setTriangles(triangles);
  vk::GeometryNV geometry;
  geometry.setGeometry(geoData);
  // Consider the geometry opaque for optimization
  geometry.setFlags(vk::GeometryFlagBitsNV::eNoDuplicateAnyHitInvocation);
  return geometry;
}

//--------------------------------------------------------------------------------------------------
// Returning the ray tracing geometry used for the BLAS, containing all spheres
//
vk::GeometryNV Raytracer::implicitToVkGeometryNV(const ImplInst& implicitObj)
{
  vk::GeometryAABBNV aabb;
  aabb.setAabbData(implicitObj.implBuf.buffer);
  aabb.setNumAABBs(static_cast<uint32_t>(implicitObj.objImpl.size()));
  aabb.setStride(sizeof(ObjImplicit));
  aabb.setOffset(0);
  vk::GeometryDataNV geoData;
  geoData.setAabbs(aabb);
  vk::GeometryNV geometry;
  geometry.setGeometryType(vk::GeometryTypeNV::eAabbs);
  geometry.setGeometry(geoData);
  // Consider the geometry opaque for optimization
  geometry.setFlags(vk::GeometryFlagBitsNV::eNoDuplicateAnyHitInvocation);
  return geometry;
}


void Raytracer::createBottomLevelAS(std::vector<ObjModel>& models, ImplInst& implicitObj)
{
  // BLAS - Storing each primitive in a geometry
  std::vector<std::vector<VkGeometryNV>> blas;
  blas.reserve(models.size());
  for(const auto& obj : models)
  {
    auto geo = objectToVkGeometryNV(obj);
    // We could add more geometry in each BLAS, but we add only one for now
    blas.push_back({geo});
  }

  // Adding implicit
  if(!implicitObj.objImpl.empty())
  {
    blas.push_back({implicitToVkGeometryNV(implicitObj)});
    implicitObj.blasId = static_cast<int>(blas.size() - 1);  // remember blas ID for tlas
  }

  m_rtBuilder.buildBlas(blas, vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace);
}

void Raytracer::createTopLevelAS(std::vector<ObjInstance>& instances, ImplInst& implicitObj)
{
  std::vector<nvvk::RaytracingBuilderNV::Instance> tlas;
  tlas.reserve(instances.size());
  for(int i = 0; i < static_cast<int>(instances.size()); i++)
  {
    nvvk::RaytracingBuilderNV::Instance rayInst;
    rayInst.transform  = instances[i].transform;  // Position of the instance
    rayInst.instanceId = i;                       // gl_InstanceID
    rayInst.blasId     = instances[i].objIndex;
    rayInst.hitGroupId = 0;  // We will use the same hit group for all objects
    rayInst.flags      = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
    tlas.emplace_back(rayInst);
  }

  // Add the blas containing all implicit
  if(!implicitObj.objImpl.empty())
  {
    nvvk::RaytracingBuilderNV::Instance rayInst;
    rayInst.transform  = implicitObj.transform;                      // Position of the instance
    rayInst.instanceId = static_cast<uint32_t>(implicitObj.blasId);  // Same for material index
    rayInst.blasId     = static_cast<uint32_t>(implicitObj.blasId);
    rayInst.hitGroupId = 1;  // We will use the same hit group for all objects (the second one)
    rayInst.flags      = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;
    tlas.emplace_back(rayInst);
  }

  m_rtBuilder.buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void Raytracer::createRtDescriptorSet(const vk::ImageView& outputImage)
{
  using vkDT   = vk::DescriptorType;
  using vkSS   = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(0, vkDT::eAccelerationStructureNV, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));  // TLAS
  m_rtDescSetLayoutBind.addBinding(
      vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenNV));  // Output image

  m_rtDescPool      = m_rtDescSetLayoutBind.createPool(m_device);
  m_rtDescSetLayout = m_rtDescSetLayoutBind.createLayout(m_device);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::AccelerationStructureNV                   tlas = m_rtBuilder.getAccelerationStructure();
  vk::WriteDescriptorSetAccelerationStructureNV descASInfo{1, &tlas};
  vk::DescriptorImageInfo imageInfo{{}, outputImage, vk::ImageLayout::eGeneral};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 0, &descASInfo));
  writes.emplace_back(m_rtDescSetLayoutBind.makeWrite(m_rtDescSet, 1, &imageInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void Raytracer::updateRtDescriptorSet(const vk::ImageView& outputImage)
{
  using vkDT = vk::DescriptorType;

  // (1) Output buffer
  vk::DescriptorImageInfo imageInfo{{}, outputImage, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet  wds{m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
  m_device.updateDescriptorSets(wds, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void Raytracer::createRtPipeline(vk::DescriptorSetLayout& sceneDescLayout)
{
  std::vector<std::string> paths = defaultSearchPaths;

  vk::ShaderModule raygenSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace.rgen.spv", true, paths));
  vk::ShaderModule missSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace.rmiss.spv", true, paths));

  // The second miss shader is invoked when a shadow ray misses the geometry. It
  // simply indicates that no occlusion has been found
  vk::ShaderModule shadowmissSM =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/raytraceShadow.rmiss.spv", true, paths));


  std::vector<vk::PipelineShaderStageCreateInfo> stages;

  // Raygen
  vk::RayTracingShaderGroupCreateInfoNV rg{vk::RayTracingShaderGroupTypeNV::eGeneral,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eRaygenNV, raygenSM, "main"});
  rg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(rg);

  // Miss
  vk::RayTracingShaderGroupCreateInfoNV mg{vk::RayTracingShaderGroupTypeNV::eGeneral,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, missSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);
  // Shadow Miss
  stages.push_back({{}, vk::ShaderStageFlagBits::eMissNV, shadowmissSM, "main"});
  mg.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(mg);

  // Hit Group0 - Closest Hit + AnyHit
  vk::ShaderModule chitSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace.rchit.spv", true, paths));
  vk::ShaderModule ahitSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace.rahit.spv", true, paths));

  vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                           VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  stages.push_back({{}, vk::ShaderStageFlagBits::eAnyHitNV, ahitSM, "main"});
  hg.setAnyHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(hg);


  // Hit Group1 - Closest Hit + Intersection (procedural)
  vk::ShaderModule chit2SM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace2.rchit.spv", true, paths));
  vk::ShaderModule ahit2SM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace2.rahit.spv", true, paths));
  vk::ShaderModule rintSM =
      nvvk::createShaderModule(m_device,  //
                               nvh::loadFile("shaders/raytrace.rint.spv", true, paths));
  {
    vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eProceduralHitGroup,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
    stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chit2SM, "main"});
    hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
    stages.push_back({{}, vk::ShaderStageFlagBits::eAnyHitNV, ahit2SM, "main"});
    hg.setAnyHitShader(static_cast<uint32_t>(stages.size() - 1));
    stages.push_back({{}, vk::ShaderStageFlagBits::eIntersectionNV, rintSM, "main"});
    hg.setIntersectionShader(static_cast<uint32_t>(stages.size() - 1));
    m_rtShaderGroups.push_back(hg);
  }

  // Callable shaders
  vk::RayTracingShaderGroupCreateInfoNV callGroup{vk::RayTracingShaderGroupTypeNV::eGeneral,
                                                  VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                                  VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};

  vk::ShaderModule call0 =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/light_point.rcall.spv", true, paths));
  vk::ShaderModule call1 =
      nvvk::createShaderModule(m_device,
                               nvh::loadFile("shaders/light_spot.rcall.spv", true, paths));
  vk::ShaderModule call2 =
      nvvk::createShaderModule(m_device, nvh::loadFile("shaders/light_inf.rcall.spv", true, paths));

  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableNV, call0, "main"});
  callGroup.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(callGroup);
  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableNV, call1, "main"});
  callGroup.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(callGroup);
  stages.push_back({{}, vk::ShaderStageFlagBits::eCallableNV, call2, "main"});
  callGroup.setGeneralShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(callGroup);


  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

  // Push constant: we want to be able to update constants used by the shaders
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenNV
                                         | vk::ShaderStageFlagBits::eClosestHitNV
                                         | vk::ShaderStageFlagBits::eMissNV
                                         | vk::ShaderStageFlagBits::eCallableNV,
                                     0, sizeof(RtPushConstants)};
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<vk::DescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, sceneDescLayout};
  pipelineLayoutCreateInfo.setSetLayoutCount(static_cast<uint32_t>(rtDescSetLayouts.size()));
  pipelineLayoutCreateInfo.setPSetLayouts(rtDescSetLayouts.data());

  m_rtPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Assemble the shader stages and recursion depth info into the ray tracing pipeline
  vk::RayTracingPipelineCreateInfoNV rayPipelineInfo;
  rayPipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));  // Stages are shaders
  rayPipelineInfo.setPStages(stages.data());

  rayPipelineInfo.setGroupCount(static_cast<uint32_t>(
      m_rtShaderGroups.size()));  // 1-raygen, n-miss, n-(hit[+anyhit+intersect])
  rayPipelineInfo.setPGroups(m_rtShaderGroups.data());

  rayPipelineInfo.setMaxRecursionDepth(2);  // Ray depth
  rayPipelineInfo.setLayout(m_rtPipelineLayout);
  m_rtPipeline = static_cast<const vk::Pipeline&>(m_device.createRayTracingPipelineNV({}, rayPipelineInfo));

  m_device.destroy(raygenSM);
  m_device.destroy(missSM);
  m_device.destroy(shadowmissSM);
  m_device.destroy(chitSM);
  m_device.destroy(ahitSM);
  m_device.destroy(chit2SM);
  m_device.destroy(ahit2SM);
  m_device.destroy(rintSM);
  m_device.destroy(call0);
  m_device.destroy(call1);
  m_device.destroy(call2);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and writing them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void Raytracer::createRtShaderBindingTable()
{
  auto groupCount =
      static_cast<uint32_t>(m_rtShaderGroups.size());               // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier
  uint32_t baseAlignment   = m_rtProperties.shaderGroupBaseAlignment;  // Size of shader alignment

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t sbtSize = groupCount * baseAlignment;

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  m_device.getRayTracingShaderGroupHandlesNV(m_rtPipeline, 0, groupCount, sbtSize,
                                             shaderHandleStorage.data());
  // Write the handles in the SBT
  m_rtSBTBuffer = m_alloc->createBuffer(sbtSize, vk::BufferUsageFlagBits::eTransferSrc,
                                        vk::MemoryPropertyFlagBits::eHostVisible
                                            | vk::MemoryPropertyFlagBits::eHostCoherent);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, std::string("SBT").c_str());

  // Write the handles in the SBT
  void* mapped = m_alloc->map(m_rtSBTBuffer);
  auto* pData  = reinterpret_cast<uint8_t*>(mapped);
  for(uint32_t g = 0; g < groupCount; g++)
  {
    memcpy(pData, shaderHandleStorage.data() + g * groupHandleSize, groupHandleSize);  // raygen
    pData += baseAlignment;
  }
  m_alloc->unmap(m_rtSBTBuffer);

  m_alloc->finalizeAndReleaseStaging();
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void Raytracer::raytrace(const vk::CommandBuffer& cmdBuf,
                         const nvmath::vec4f&     clearColor,
                         vk::DescriptorSet&       sceneDescSet,
                         vk::Extent2D&            size,
                         ObjPushConstants&        sceneConstants)
{
  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_rtPushConstants.clearColor           = clearColor;
  m_rtPushConstants.lightPosition        = sceneConstants.lightPosition;
  m_rtPushConstants.lightIntensity       = sceneConstants.lightIntensity;
  m_rtPushConstants.lightDirection       = sceneConstants.lightDirection;
  m_rtPushConstants.lightSpotCutoff      = sceneConstants.lightSpotCutoff;
  m_rtPushConstants.lightSpotOuterCutoff = sceneConstants.lightSpotOuterCutoff;
  m_rtPushConstants.lightType            = sceneConstants.lightType;
  m_rtPushConstants.frame                = sceneConstants.frame;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, m_rtPipelineLayout, 0,
                            {m_rtDescSet, sceneDescSet}, {});
  cmdBuf.pushConstants<RtPushConstants>(m_rtPipelineLayout,
                                        vk::ShaderStageFlagBits::eRaygenNV
                                            | vk::ShaderStageFlagBits::eClosestHitNV
                                            | vk::ShaderStageFlagBits::eMissNV
                                            | vk::ShaderStageFlagBits::eCallableNV,
                                        0, m_rtPushConstants);

  vk::DeviceSize progSize =
      m_rtProperties.shaderGroupBaseAlignment;         // Size of a program identifier
  vk::DeviceSize rayGenOffset        = 0u * progSize;  // Start at the beginning of m_sbtBuffer
  vk::DeviceSize missOffset          = 1u * progSize;  // Jump over raygen
  vk::DeviceSize missStride          = progSize;
  vk::DeviceSize hitGroupOffset      = 3u * progSize;  // Jump over the previous shaders
  vk::DeviceSize hitGroupStride      = progSize;
  vk::DeviceSize callableGroupOffset = 5u * progSize;  // Jump over the previous shaders
  vk::DeviceSize callableGroupStride = progSize;

  // m_sbtBuffer holds all the shader handles: raygen, n-miss, hit...
  cmdBuf.traceRaysNV(m_rtSBTBuffer.buffer, rayGenOffset,                              //
                     m_rtSBTBuffer.buffer, missOffset, missStride,                    //
                     m_rtSBTBuffer.buffer, hitGroupOffset, hitGroupStride,            //
                     m_rtSBTBuffer.buffer, callableGroupOffset, callableGroupStride,  //
                     size.width, size.height,                                         //
                     1);                                                              // depth

  m_debug.endLabel(cmdBuf);
}
