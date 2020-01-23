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

#include <sstream>
#include <vulkan/vulkan.hpp>

extern std::vector<std::string> defaultSearchPaths;

#define STB_IMAGE_IMPLEMENTATION
#include "fileformats/stb_image.h"
#include "obj_loader.h"

#include "hello_vulkan.h"
#include "nvh//cameramanipulator.hpp"
#include "nvvkpp/descriptorsets_vkpp.hpp"
#include "nvvkpp/pipeline_vkpp.hpp"

#include "nvh/fileoperations.hpp"
#include "nvvkpp/commands_vkpp.hpp"
#include "nvvkpp/renderpass_vkpp.hpp"
#include "nvvkpp/utilities_vkpp.hpp"
#include <random>

// Holding the camera matrices
struct CameraMatrices
{
  nvmath::mat4f view;
  nvmath::mat4f proj;
  nvmath::mat4f viewInverse;
  // #VKRay
  nvmath::mat4f projInverse;
};

//--------------------------------------------------------------------------------------------------
// Keep the handle on the device
// Initialize the tool to do all our allocations: buffers, images
//
void HelloVulkan::setup(const vk::Device&         device,
                        const vk::PhysicalDevice& physicalDevice,
                        uint32_t                  queueFamily)
{
  AppBase::setup(device, physicalDevice, queueFamily);
  m_alloc.init(device, physicalDevice);
  m_debug.setup(m_device);
}

//--------------------------------------------------------------------------------------------------
// Called at each frame to update the camera matrix
//
void HelloVulkan::updateUniformBuffer()
{
  const float aspectRatio = m_size.width / static_cast<float>(m_size.height);

  CameraMatrices ubo = {};
  ubo.view           = CameraManip.getMatrix();
  ubo.proj           = nvmath::perspectiveVK(CameraManip.getFov(), aspectRatio, 0.1f, 1000.0f);
  // ubo.proj[1][1] *= -1;  // Inverting Y for Vulkan
  ubo.viewInverse = nvmath::invert(ubo.view);
  // #VKRay
  ubo.projInverse = nvmath::invert(ubo.proj);

  void* data = m_device.mapMemory(m_cameraMat.allocation, 0, sizeof(ubo));
  memcpy(data, &ubo, sizeof(ubo));
  m_device.unmapMemory(m_cameraMat.allocation);
}

//--------------------------------------------------------------------------------------------------
// Describing the layout pushed when rendering
//
void HelloVulkan::createDescriptorSetLayout()
{
  using vkDS     = vk::DescriptorSetLayoutBinding;
  using vkDT     = vk::DescriptorType;
  using vkSS     = vk::ShaderStageFlagBits;
  uint32_t nbTxt = static_cast<uint32_t>(m_textures.size());
  uint32_t nbObj = static_cast<uint32_t>(m_objModel.size());

  // Camera matrices (binding = 0)
  m_descSetLayoutBind.emplace_back(
      vkDS(0, vkDT::eUniformBuffer, 1, vkSS::eVertex | vkSS::eRaygenNV));
  // Materials (binding = 1)
  m_descSetLayoutBind.emplace_back(vkDS(1, vkDT::eStorageBuffer, nbObj + 1,
                                        vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitNV));
  // Scene description (binding = 2)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(2, vkDT::eStorageBuffer, 1, vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitNV));
  // Textures (binding = 3)
  m_descSetLayoutBind.emplace_back(
      vkDS(3, vkDT::eCombinedImageSampler, nbTxt, vkSS::eFragment | vkSS::eClosestHitNV));
  // Materials Index (binding = 4)
  m_descSetLayoutBind.emplace_back(
      vkDS(4, vkDT::eStorageBuffer, nbObj + 1, vkSS::eFragment | vkSS::eClosestHitNV));
  // Storing vertices (binding = 5)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(5, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitNV));
  // Storing indices (binding = 6)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(6, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitNV));
  // Storing spheres (binding = 7)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(7, vkDT::eStorageBuffer, 1, vkSS::eClosestHitNV | vkSS::eIntersectionNV));


  m_descSetLayout = nvvkpp::util::createDescriptorSetLayout(m_device, m_descSetLayoutBind);
  m_descPool      = nvvkpp::util::createDescriptorPool(m_device, m_descSetLayoutBind, 1);
  m_descSet       = nvvkpp::util::createDescriptorSet(m_device, m_descPool, m_descSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Setting up the buffers in the descriptor set
//
void HelloVulkan::updateDescriptorSet()
{
  std::vector<vk::WriteDescriptorSet> writes;

  // Camera matrices and scene description
  vk::DescriptorBufferInfo dbiUnif{m_cameraMat.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[0], &dbiUnif));
  vk::DescriptorBufferInfo dbiSceneDesc{m_sceneDesc.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[2], &dbiSceneDesc));

  // All material buffers, 1 buffer per OBJ
  std::vector<vk::DescriptorBufferInfo> dbiMat;
  std::vector<vk::DescriptorBufferInfo> dbiMatIdx;
  std::vector<vk::DescriptorBufferInfo> dbiVert;
  std::vector<vk::DescriptorBufferInfo> dbiIdx;
  for(auto& model : m_objModel)
  {
    dbiMat.emplace_back(model.matColorBuffer.buffer, 0, VK_WHOLE_SIZE);
    dbiMatIdx.emplace_back(model.matIndexBuffer.buffer, 0, VK_WHOLE_SIZE);
    dbiVert.emplace_back(model.vertexBuffer.buffer, 0, VK_WHOLE_SIZE);
    dbiIdx.emplace_back(model.indexBuffer.buffer, 0, VK_WHOLE_SIZE);
  }
  dbiMat.emplace_back(m_spheresMatColorBuffer.buffer, 0, VK_WHOLE_SIZE);
  dbiMatIdx.emplace_back(m_spheresMatIndexBuffer.buffer, 0, VK_WHOLE_SIZE);

  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[1], dbiMat.data()));
  writes.emplace_back(
      nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[4], dbiMatIdx.data()));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[5], dbiVert.data()));
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[6], dbiIdx.data()));

  vk::DescriptorBufferInfo dbiSpheres{m_spheresBuffer.buffer, 0, VK_WHOLE_SIZE};
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[7], &dbiSpheres));

  // All texture samplers
  std::vector<vk::DescriptorImageInfo> diit;
  for(auto& texture : m_textures)
  {
    diit.push_back(texture.descriptor);
  }
  writes.emplace_back(nvvkpp::util::createWrite(m_descSet, m_descSetLayoutBind[3], diit.data()));

  // Writing the information
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Creating the pipeline layout
//
void HelloVulkan::createGraphicsPipeline()
{
  using vkSS = vk::ShaderStageFlagBits;

  vk::PushConstantRange pushConstantRanges = {vkSS::eVertex | vkSS::eFragment, 0,
                                              sizeof(ObjPushConstant)};

  // Creating the Pipeline Layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  vk::DescriptorSetLayout      descSetLayout(m_descSetLayout);
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&descSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_pipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Creating the Pipeline
  std::vector<std::string>          paths = defaultSearchPaths;
  nvvkpp::GraphicsPipelineGenerator gpb(m_device, m_pipelineLayout, m_offscreenRenderPass);
  gpb.depthStencilState = {true};
  gpb.addShader(nvh::loadFile("shaders/vert_shader.vert.spv", true, paths), vkSS::eVertex);
  gpb.addShader(nvh::loadFile("shaders/frag_shader.frag.spv", true, paths), vkSS::eFragment);
  gpb.vertexInputState.bindingDescriptions   = {{0, sizeof(VertexObj)}};
  gpb.vertexInputState.attributeDescriptions = {
      {0, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, pos)},
      {1, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, nrm)},
      {2, 0, vk::Format::eR32G32B32Sfloat, offsetof(VertexObj, color)},
      {3, 0, vk::Format::eR32G32Sfloat, offsetof(VertexObj, texCoord)}};

  m_graphicsPipeline = gpb.create();
  m_debug.setObjectName(m_graphicsPipeline, "Graphics");
}

//--------------------------------------------------------------------------------------------------
// Loading the OBJ file and setting up all buffers
//
void HelloVulkan::loadModel(const std::string& filename, nvmath::mat4f transform)
{
  using vkBU = vk::BufferUsageFlagBits;

  ObjLoader loader;
  loader.loadModel(filename);

  // Converting from Srgb to linear
  for(auto& m : loader.m_materials)
  {
    m.ambient  = nvmath::pow(m.ambient, 2.2f);
    m.diffuse  = nvmath::pow(m.diffuse, 2.2f);
    m.specular = nvmath::pow(m.specular, 2.2f);
  }

  ObjInstance instance;
  instance.objIndex    = static_cast<uint32_t>(m_objModel.size());
  instance.transform   = transform;
  instance.transformIT = nvmath::transpose(nvmath::invert(transform));
  instance.txtOffset   = static_cast<uint32_t>(m_textures.size());

  ObjModel model;
  model.nbIndices  = static_cast<uint32_t>(loader.m_indices.size());
  model.nbVertices = static_cast<uint32_t>(loader.m_vertices.size());

  // Create the buffers on Device and copy vertices, indices and materials
  nvvkpp::SingleCommandBuffer cmdBufGet(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer           cmdBuf = cmdBufGet.createCommandBuffer();
  model.vertexBuffer =
      m_alloc.createBuffer(cmdBuf, loader.m_vertices, vkBU::eVertexBuffer | vkBU::eStorageBuffer);
  model.indexBuffer =
      m_alloc.createBuffer(cmdBuf, loader.m_indices, vkBU::eIndexBuffer | vkBU::eStorageBuffer);
  model.matColorBuffer = m_alloc.createBuffer(cmdBuf, loader.m_materials, vkBU::eStorageBuffer);
  model.matIndexBuffer = m_alloc.createBuffer(cmdBuf, loader.m_matIndx, vkBU::eStorageBuffer);
  // Creates all textures found
  createTextureImages(cmdBuf, loader.m_textures);
  cmdBufGet.flushCommandBuffer(cmdBuf);
  m_alloc.flushStaging();

  std::string objNb = std::to_string(instance.objIndex);
  m_debug.setObjectName(model.vertexBuffer.buffer, (std::string("vertex_" + objNb).c_str()));
  m_debug.setObjectName(model.indexBuffer.buffer, (std::string("index_" + objNb).c_str()));
  m_debug.setObjectName(model.matColorBuffer.buffer, (std::string("mat_" + objNb).c_str()));
  m_debug.setObjectName(model.matIndexBuffer.buffer, (std::string("matIdx_" + objNb).c_str()));

  m_objModel.emplace_back(model);
  m_objInstance.emplace_back(instance);
}

//--------------------------------------------------------------------------------------------------
// Creating the uniform buffer holding the camera matrices
// - Buffer is host visible
//
void HelloVulkan::createUniformBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  using vkMP = vk::MemoryPropertyFlagBits;

  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices), vkBU::eUniformBuffer,
                                     vkMP::eHostVisible | vkMP::eHostCoherent);
  m_debug.setObjectName(m_cameraMat.buffer, "cameraMat");
}

//--------------------------------------------------------------------------------------------------
// Create a storage buffer containing the description of the scene elements
// - Which geometry is used by which instance
// - Transformation
// - Offset for texture
//
void HelloVulkan::createSceneDescriptionBuffer()
{
  using vkBU = vk::BufferUsageFlagBits;
  nvvkpp::SingleCommandBuffer cmdGen(m_device, m_graphicsQueueIndex);

  auto cmdBuf = cmdGen.createCommandBuffer();
  m_sceneDesc = m_alloc.createBuffer(cmdBuf, m_objInstance, vkBU::eStorageBuffer);
  cmdGen.flushCommandBuffer(cmdBuf);
  m_alloc.flushStaging();
  m_debug.setObjectName(m_sceneDesc.buffer, "sceneDesc");
}

//--------------------------------------------------------------------------------------------------
// Creating all textures and samplers
//
void HelloVulkan::createTextureImages(const vk::CommandBuffer&        cmdBuf,
                                      const std::vector<std::string>& textures)
{
  using vkIU = vk::ImageUsageFlagBits;

  vk::SamplerCreateInfo samplerCreateInfo{
      {}, vk::Filter::eLinear, vk::Filter::eLinear, vk::SamplerMipmapMode::eLinear};
  samplerCreateInfo.setMaxLod(FLT_MAX);
  vk::Format format = vk::Format::eR8G8B8A8Srgb;

  // If no textures are present, create a dummy one to accommodate the pipeline layout
  if(textures.empty() && m_textures.empty())
  {
    nvvkTexture texture;

    std::array<uint8_t, 4> color{255u, 255u, 255u, 255u};
    vk::DeviceSize         bufferSize      = sizeof(color);
    auto                   imgSize         = vk::Extent2D(1, 1);
    auto                   imageCreateInfo = nvvkpp::image::create2DInfo(imgSize, format);

    // Creating the VKImage
    texture = m_alloc.createImage(cmdBuf, bufferSize, color.data(), imageCreateInfo);
    // Setting up the descriptor used by the shader
    texture.descriptor =
        nvvkpp::image::create2DDescriptor(m_device, texture.image, samplerCreateInfo, format);
    // The image format must be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
    nvvkpp::image::setImageLayout(cmdBuf, texture.image, vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eShaderReadOnlyOptimal);
    m_textures.push_back(texture);
  }
  else
  {
    // Uploading all images
    for(const auto& texture : textures)
    {
      std::stringstream o;
      int               texWidth, texHeight, texChannels;
      o << "media/textures/" << texture;
      std::string txtFile = nvh::findFile(o.str(), defaultSearchPaths);

      stbi_uc* pixels =
          stbi_load(txtFile.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

      // Handle failure
      if(!pixels)
      {
        texWidth = texHeight = 1;
        texChannels          = 4;
        std::array<uint8_t, 4> color{255u, 0u, 255u, 255u};
        pixels = reinterpret_cast<stbi_uc*>(color.data());
      }

      vk::DeviceSize bufferSize = static_cast<uint64_t>(texWidth) * texHeight * sizeof(uint8_t) * 4;
      auto           imgSize    = vk::Extent2D(texWidth, texHeight);
      auto imageCreateInfo = nvvkpp::image::create2DInfo(imgSize, format, vkIU::eSampled, true);

      nvvkTexture texture;
      texture = m_alloc.createImage(cmdBuf, bufferSize, pixels, imageCreateInfo);

      nvvkpp::image::generateMipmaps(cmdBuf, texture.image, format, imgSize,
                                     imageCreateInfo.mipLevels);
      texture.descriptor =
          nvvkpp::image::create2DDescriptor(m_device, texture.image, samplerCreateInfo, format);
      m_textures.push_back(texture);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// Destroying all allocations
//
void HelloVulkan::destroyResources()
{
  m_device.destroy(m_graphicsPipeline);
  m_device.destroy(m_pipelineLayout);
  m_device.destroy(m_descPool);
  m_device.destroy(m_descSetLayout);
  m_alloc.destroy(m_cameraMat);
  m_alloc.destroy(m_sceneDesc);

  for(auto& m : m_objModel)
  {
    m_alloc.destroy(m.vertexBuffer);
    m_alloc.destroy(m.indexBuffer);
    m_alloc.destroy(m.matColorBuffer);
    m_alloc.destroy(m.matIndexBuffer);
  }

  for(auto& t : m_textures)
  {
    m_alloc.destroy(t);
  }

  //#Post
  m_device.destroy(m_postPipeline);
  m_device.destroy(m_postPipelineLayout);
  m_device.destroy(m_postDescPool);
  m_device.destroy(m_postDescSetLayout);
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);
  m_device.destroy(m_offscreenRenderPass);
  m_device.destroy(m_offscreenFramebuffer);

  // #VKRay
  m_rtBuilder.destroy();
  m_device.destroy(m_rtDescPool);
  m_device.destroy(m_rtDescSetLayout);
  m_device.destroy(m_rtPipeline);
  m_device.destroy(m_rtPipelineLayout);
  m_alloc.destroy(m_rtSBTBuffer);

  m_alloc.destroy(m_spheresBuffer);
  m_alloc.destroy(m_spheresAabbBuffer);
  m_alloc.destroy(m_spheresMatColorBuffer);
  m_alloc.destroy(m_spheresMatIndexBuffer);
}

//--------------------------------------------------------------------------------------------------
// Drawing the scene in raster mode
//
void HelloVulkan::rasterize(const vk::CommandBuffer& cmdBuf)
{
  using vkPBP = vk::PipelineBindPoint;
  using vkSS  = vk::ShaderStageFlagBits;
  vk::DeviceSize offset{0};

  m_debug.beginLabel(cmdBuf, "Rasterize");

  // Dynamic Viewport
  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  // Drawing all triangles
  cmdBuf.bindPipeline(vkPBP::eGraphics, m_graphicsPipeline);
  cmdBuf.bindDescriptorSets(vkPBP::eGraphics, m_pipelineLayout, 0, {m_descSet}, {});
  for(int i = 0; i < m_objInstance.size(); ++i)
  {
    auto& inst                = m_objInstance[i];
    auto& model               = m_objModel[inst.objIndex];
    m_pushConstant.instanceId = i;  // Telling which instance is drawn
    cmdBuf.pushConstants<ObjPushConstant>(m_pipelineLayout, vkSS::eVertex | vkSS::eFragment, 0,
                                          m_pushConstant);

    cmdBuf.bindVertexBuffers(0, 1, &model.vertexBuffer.buffer, &offset);
    cmdBuf.bindIndexBuffer(model.indexBuffer.buffer, 0, vk::IndexType::eUint32);
    cmdBuf.drawIndexed(model.nbIndices, 1, 0, 0, 0);
  }
  m_debug.endLabel(cmdBuf);
}

//--------------------------------------------------------------------------------------------------
// Handling resize of the window
//
void HelloVulkan::onResize(int /*w*/, int /*h*/)
{
  createOffscreenRender();
  updatePostDescriptorSet();
  updateRtDescriptorSet();
}

//////////////////////////////////////////////////////////////////////////
// Post-processing
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Creating an offscreen frame buffer and the associated render pass
//
void HelloVulkan::createOffscreenRender()
{
  m_alloc.destroy(m_offscreenColor);
  m_alloc.destroy(m_offscreenDepth);

  // Creating the color image
  auto colorCreateInfo = nvvkpp::image::create2DInfo(m_size, m_offscreenColorFormat,
                                                     vk::ImageUsageFlagBits::eColorAttachment
                                                         | vk::ImageUsageFlagBits::eSampled
                                                         | vk::ImageUsageFlagBits::eStorage);
  m_offscreenColor     = m_alloc.createImage(colorCreateInfo);

  m_offscreenColor.descriptor =
      nvvkpp::image::create2DDescriptor(m_device, m_offscreenColor.image, vk::SamplerCreateInfo{},
                                        m_offscreenColorFormat, vk::ImageLayout::eGeneral);

  // Creating the depth buffer
  auto depthCreateInfo =
      nvvkpp::image::create2DInfo(m_size, m_offscreenDepthFormat,
                                  vk::ImageUsageFlagBits::eDepthStencilAttachment);
  m_offscreenDepth = m_alloc.createImage(depthCreateInfo);

  vk::ImageViewCreateInfo depthStencilView;
  depthStencilView.setViewType(vk::ImageViewType::e2D);
  depthStencilView.setFormat(m_offscreenDepthFormat);
  depthStencilView.setSubresourceRange({vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1});
  depthStencilView.setImage(m_offscreenDepth.image);
  m_offscreenDepth.descriptor.imageView = m_device.createImageView(depthStencilView);

  // Setting the image layout for both color and depth
  {
    nvvkpp::SingleCommandBuffer genCmdBuf(m_device, m_graphicsQueueIndex);
    auto                        cmdBuf = genCmdBuf.createCommandBuffer();
    nvvkpp::image::setImageLayout(cmdBuf, m_offscreenColor.image, vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eGeneral);
    nvvkpp::image::setImageLayout(cmdBuf, m_offscreenDepth.image, vk::ImageAspectFlagBits::eDepth,
                                  vk::ImageLayout::eUndefined,
                                  vk::ImageLayout::eDepthStencilAttachmentOptimal);

    genCmdBuf.flushCommandBuffer(cmdBuf);
  }

  // Creating a renderpass for the offscreen
  if(!m_offscreenRenderPass)
  {
    m_offscreenRenderPass =
        nvvkpp::util::createRenderPass(m_device, {m_offscreenColorFormat}, m_offscreenDepthFormat,
                                       1, true, true, vk::ImageLayout::eGeneral,
                                       vk::ImageLayout::eGeneral);
  }

  // Creating the frame buffer for offscreen
  std::vector<vk::ImageView> attachments = {m_offscreenColor.descriptor.imageView,
                                            m_offscreenDepth.descriptor.imageView};

  m_device.destroy(m_offscreenFramebuffer);
  vk::FramebufferCreateInfo info;
  info.setRenderPass(m_offscreenRenderPass);
  info.setAttachmentCount(2);
  info.setPAttachments(attachments.data());
  info.setWidth(m_size.width);
  info.setHeight(m_size.height);
  info.setLayers(1);
  m_offscreenFramebuffer = m_device.createFramebuffer(info);
}

//--------------------------------------------------------------------------------------------------
// The pipeline is how things are rendered, which shaders, type of primitives, depth test and more
//
void HelloVulkan::createPostPipeline()
{
  // Push constants in the fragment shader
  vk::PushConstantRange pushConstantRanges = {vk::ShaderStageFlagBits::eFragment, 0, sizeof(float)};

  // Creating the pipeline layout
  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;
  pipelineLayoutCreateInfo.setSetLayoutCount(1);
  pipelineLayoutCreateInfo.setPSetLayouts(&m_postDescSetLayout);
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstantRanges);
  m_postPipelineLayout = m_device.createPipelineLayout(pipelineLayoutCreateInfo);

  // Pipeline: completely generic, no vertices
  std::vector<std::string> paths = defaultSearchPaths;

  nvvkpp::GraphicsPipelineGenerator pipelineGenerator(m_device, m_postPipelineLayout, m_renderPass);
  pipelineGenerator.addShader(nvh::loadFile("shaders/passthrough.vert.spv", true, paths),
                              vk::ShaderStageFlagBits::eVertex);
  pipelineGenerator.addShader(nvh::loadFile("shaders/post.frag.spv", true, paths),
                              vk::ShaderStageFlagBits::eFragment);
  pipelineGenerator.rasterizationState.setCullMode(vk::CullModeFlagBits::eNone);
  m_postPipeline = pipelineGenerator.create();
  m_debug.setObjectName(m_postPipeline, "post");
}

//--------------------------------------------------------------------------------------------------
// The descriptor layout is the description of the data that is passed to the vertex or the
// fragment program.
//
void HelloVulkan::createPostDescriptor()
{
  using vkDS = vk::DescriptorSetLayoutBinding;
  using vkDT = vk::DescriptorType;
  using vkSS = vk::ShaderStageFlagBits;

  m_postDescSetLayoutBind.emplace_back(vkDS(0, vkDT::eCombinedImageSampler, 1, vkSS::eFragment));
  m_postDescSetLayout = nvvkpp::util::createDescriptorSetLayout(m_device, m_postDescSetLayoutBind);
  m_postDescPool      = nvvkpp::util::createDescriptorPool(m_device, m_postDescSetLayoutBind);
  m_postDescSet = nvvkpp::util::createDescriptorSet(m_device, m_postDescPool, m_postDescSetLayout);
}

//--------------------------------------------------------------------------------------------------
// Update the output
//
void HelloVulkan::updatePostDescriptorSet()
{
  vk::WriteDescriptorSet writeDescriptorSets =
      nvvkpp::util::createWrite(m_postDescSet, m_postDescSetLayoutBind[0],
                                &m_offscreenColor.descriptor);
  m_device.updateDescriptorSets(writeDescriptorSets, nullptr);
}

//--------------------------------------------------------------------------------------------------
// Draw a full screen quad with the attached image
//
void HelloVulkan::drawPost(vk::CommandBuffer cmdBuf)
{
  m_debug.beginLabel(cmdBuf, "Post");

  cmdBuf.setViewport(0, {vk::Viewport(0, 0, (float)m_size.width, (float)m_size.height, 0, 1)});
  cmdBuf.setScissor(0, {{{0, 0}, {m_size.width, m_size.height}}});

  auto aspectRatio = static_cast<float>(m_size.width) / static_cast<float>(m_size.height);
  cmdBuf.pushConstants<float>(m_postPipelineLayout, vk::ShaderStageFlagBits::eFragment, 0,
                              aspectRatio);
  cmdBuf.bindPipeline(vk::PipelineBindPoint::eGraphics, m_postPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, m_postPipelineLayout, 0,
                            m_postDescSet, {});
  cmdBuf.draw(3, 1, 0, 0);

  m_debug.endLabel(cmdBuf);
}

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////

//--------------------------------------------------------------------------------------------------
// Initialize Vulkan ray tracing
// #VKRay
void HelloVulkan::initRayTracing()
{
  // Requesting ray tracing properties
  auto properties = m_physicalDevice.getProperties2<vk::PhysicalDeviceProperties2,
                                                    vk::PhysicalDeviceRayTracingPropertiesNV>();
  m_rtProperties  = properties.get<vk::PhysicalDeviceRayTracingPropertiesNV>();
  m_rtBuilder.setup(m_device, m_physicalDevice, m_graphicsQueueIndex);
}

//--------------------------------------------------------------------------------------------------
// Converting a OBJ primitive to the ray tracing geometry used for the BLAS
//
vk::GeometryNV HelloVulkan::objectToVkGeometryNV(const ObjModel& model)
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
  geometry.setFlags(vk::GeometryFlagBitsNV::eOpaque);
  return geometry;
}

//--------------------------------------------------------------------------------------------------
// Returning the ray tracing geometry used for the BLAS, containing all spheres
//
vk::GeometryNV HelloVulkan::sphereToVkGeometryNV()
{
  vk::GeometryAABBNV aabb;
  aabb.setAabbData(m_spheresAabbBuffer.buffer);
  aabb.setNumAABBs(static_cast<uint32_t>(m_spheres.size()));
  aabb.setStride(sizeof(Aabb));
  aabb.setOffset(0);
  vk::GeometryDataNV geoData;
  geoData.setAabbs(aabb);
  vk::GeometryNV geometry;
  geometry.setGeometryType(vk::GeometryTypeNV::eAabbs);
  geometry.setGeometry(geoData);
  // Consider the geometry opaque for optimization
  geometry.setFlags(vk::GeometryFlagBitsNV::eOpaque);
  return geometry;
}

//--------------------------------------------------------------------------------------------------
// Creating all spheres
//
void HelloVulkan::createSpheres()
{
  std::random_device                    rd{};
  std::mt19937                          gen{rd()};
  std::normal_distribution<float>       xzd{0.f, 5.f};
  std::normal_distribution<float>       yd{3.f, 1.f};
  std::uniform_real_distribution<float> radd{.05f, .2f};

  // All spheres
  Sphere s;
  for(uint32_t i = 0; i < 2000000; i++)
  {
    s.center = nvmath::vec3f(xzd(gen), yd(gen), xzd(gen));
    s.radius = radd(gen);
    m_spheres.emplace_back(s);
  }

  // Axis aligned bounding box of each sphere
  std::vector<Aabb> aabbs;
  for(const auto& s : m_spheres)
  {
    Aabb aabb;
    aabb.minimum = s.center - nvmath::vec3f(s.radius);
    aabb.maximum = s.center + nvmath::vec3f(s.radius);
    aabbs.emplace_back(aabb);
  }

  // Creating two materials
  MatrialObj mat;
  mat.diffuse = vec3f(0, 1, 1);
  std::vector<MatrialObj> materials;
  std::vector<int>        matIdx;
  materials.emplace_back(mat);
  mat.diffuse = vec3f(1, 1, 0);
  materials.emplace_back(mat);

  // Assign a material to each sphere
  for(size_t i = 0; i < m_spheres.size(); i++)
  {
    matIdx.push_back(i % 2);
  }

  // Creating all buffers
  using vkBU = vk::BufferUsageFlagBits;
  nvvkpp::SingleCommandBuffer genCmdBuf(m_device, m_graphicsQueueIndex);
  auto                        cmdBuf = genCmdBuf.createCommandBuffer();
  m_spheresBuffer         = m_alloc.createBuffer(cmdBuf, m_spheres, vkBU::eStorageBuffer);
  m_spheresAabbBuffer     = m_alloc.createBuffer(cmdBuf, aabbs);
  m_spheresMatIndexBuffer = m_alloc.createBuffer(cmdBuf, matIdx, vkBU::eStorageBuffer);
  m_spheresMatColorBuffer = m_alloc.createBuffer(cmdBuf, materials, vkBU::eStorageBuffer);
  genCmdBuf.flushCommandBuffer(cmdBuf);

  // Debug information
  m_debug.setObjectName(m_spheresBuffer.buffer, "spheres");
  m_debug.setObjectName(m_spheresAabbBuffer.buffer, "spheresAabb");
  m_debug.setObjectName(m_spheresMatColorBuffer.buffer, "spheresMat");
  m_debug.setObjectName(m_spheresMatIndexBuffer.buffer, "spheresMatIdx");
}

void HelloVulkan::createBottomLevelAS()
{
  // BLAS - Storing each primitive in a geometry
  std::vector<std::vector<vk::GeometryNV>> blas;
  blas.reserve(m_objModel.size());
  for(const auto& obj : m_objModel)
  {
    auto geo = objectToVkGeometryNV(obj);
    // We could add more geometry in each BLAS, but we add only one for now
    blas.push_back({geo});
  }

  // Spheres
  auto geo = sphereToVkGeometryNV();
  blas.push_back({geo});

  m_rtBuilder.buildBlas(blas, vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace);
}

void HelloVulkan::createTopLevelAS()
{
  std::vector<nvvkpp::RaytracingBuilder::Instance> tlas;
  tlas.reserve(m_objInstance.size());
  for(int i = 0; i < static_cast<int>(m_objInstance.size()); i++)
  {
    nvvkpp::RaytracingBuilder::Instance rayInst;
    rayInst.transform  = m_objInstance[i].transform;  // Position of the instance
    rayInst.instanceId = i;                           // gl_InstanceID
    rayInst.blasId     = m_objInstance[i].objIndex;
    rayInst.hitGroupId = 0;  // We will use the same hit group for all objects
    rayInst.flags      = vk::GeometryInstanceFlagBitsNV::eTriangleCullDisable;
    tlas.emplace_back(rayInst);
  }

  // Add the blas containing all spheres
  {
    nvvkpp::RaytracingBuilder::Instance rayInst;
    rayInst.transform  = m_objInstance[0].transform;          // Position of the instance
    rayInst.instanceId = static_cast<uint32_t>(tlas.size());  // gl_InstanceID
    rayInst.blasId     = static_cast<uint32_t>(m_objModel.size());
    rayInst.hitGroupId = 1;  // We will use the same hit group for all objects
    rayInst.flags      = vk::GeometryInstanceFlagBitsNV::eTriangleCullDisable;
    tlas.emplace_back(rayInst);
  }

  m_rtBuilder.buildTlas(tlas, vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace);
}

//--------------------------------------------------------------------------------------------------
// This descriptor set holds the Acceleration structure and the output image
//
void HelloVulkan::createRtDescriptorSet()
{
  using vkDT   = vk::DescriptorType;
  using vkSS   = vk::ShaderStageFlagBits;
  using vkDSLB = vk::DescriptorSetLayoutBinding;

  m_rtDescSetLayoutBind.emplace_back(
      vkDSLB(0, vkDT::eAccelerationStructureNV, 1, vkSS::eRaygenNV | vkSS::eClosestHitNV));  // TLAS
  m_rtDescSetLayoutBind.emplace_back(
      vkDSLB(1, vkDT::eStorageImage, 1, vkSS::eRaygenNV));  // Output image

  m_rtDescPool      = nvvkpp::util::createDescriptorPool(m_device, m_rtDescSetLayoutBind);
  m_rtDescSetLayout = nvvkpp::util::createDescriptorSetLayout(m_device, m_rtDescSetLayoutBind);
  m_rtDescSet       = m_device.allocateDescriptorSets({m_rtDescPool, 1, &m_rtDescSetLayout})[0];

  vk::WriteDescriptorSetAccelerationStructureNV descASInfo;
  descASInfo.setAccelerationStructureCount(1);
  descASInfo.setPAccelerationStructures(&m_rtBuilder.getAccelerationStructure());
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};

  std::vector<vk::WriteDescriptorSet> writes;
  writes.emplace_back(
      nvvkpp::util::createWrite(m_rtDescSet, m_rtDescSetLayoutBind[0], &descASInfo));
  writes.emplace_back(nvvkpp::util::createWrite(m_rtDescSet, m_rtDescSetLayoutBind[1], &imageInfo));
  m_device.updateDescriptorSets(static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Writes the output image to the descriptor set
// - Required when changing resolution
//
void HelloVulkan::updateRtDescriptorSet()
{
  using vkDT = vk::DescriptorType;

  // (1) Output buffer
  vk::DescriptorImageInfo imageInfo{
      {}, m_offscreenColor.descriptor.imageView, vk::ImageLayout::eGeneral};
  vk::WriteDescriptorSet wds{m_rtDescSet, 1, 0, 1, vkDT::eStorageImage, &imageInfo};
  m_device.updateDescriptorSets(wds, nullptr);
}


//--------------------------------------------------------------------------------------------------
// Pipeline for the ray tracer: all shaders, raygen, chit, miss
//
void HelloVulkan::createRtPipeline()
{
  std::vector<std::string> paths = defaultSearchPaths;

  vk::ShaderModule raygenSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvh::loadFile("shaders/raytrace.rgen.spv", true, paths));
  vk::ShaderModule missSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvh::loadFile("shaders/raytrace.rmiss.spv", true, paths));

  // The second miss shader is invoked when a shadow ray misses the geometry. It
  // simply indicates that no occlusion has been found
  vk::ShaderModule shadowmissSM = nvvkpp::util::createShaderModule(
      m_device, nvh::loadFile("shaders/raytraceShadow.rmiss.spv", true, paths));


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

  // Hit Group0 - Closest Hit
  vk::ShaderModule chitSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvh::loadFile("shaders/raytrace.rchit.spv", true, paths));

  {
    vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
    stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chitSM, "main"});
    hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
    m_rtShaderGroups.push_back(hg);
  }

  // Hit Group1 - Closest Hit + Intersection (procedural)
  vk::ShaderModule chit2SM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvh::loadFile("shaders/raytrace2.rchit.spv", true, paths));
  vk::ShaderModule rintSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvh::loadFile("shaders/raytrace.rint.spv", true, paths));
  {
    vk::RayTracingShaderGroupCreateInfoNV hg{vk::RayTracingShaderGroupTypeNV::eProceduralHitGroup,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV,
                                             VK_SHADER_UNUSED_NV, VK_SHADER_UNUSED_NV};
    stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chit2SM, "main"});
    hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
    stages.push_back({{}, vk::ShaderStageFlagBits::eIntersectionNV, rintSM, "main"});
    hg.setIntersectionShader(static_cast<uint32_t>(stages.size() - 1));
    m_rtShaderGroups.push_back(hg);
  }

  vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo;

  // Push constant: we want to be able to update constants used by the shaders
  vk::PushConstantRange pushConstant{vk::ShaderStageFlagBits::eRaygenNV
                                         | vk::ShaderStageFlagBits::eClosestHitNV
                                         | vk::ShaderStageFlagBits::eMissNV,
                                     0, sizeof(RtPushConstant)};
  pipelineLayoutCreateInfo.setPushConstantRangeCount(1);
  pipelineLayoutCreateInfo.setPPushConstantRanges(&pushConstant);

  // Descriptor sets: one specific to ray tracing, and one shared with the rasterization pipeline
  std::vector<vk::DescriptorSetLayout> rtDescSetLayouts = {m_rtDescSetLayout, m_descSetLayout};
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
  m_rtPipeline = m_device.createRayTracingPipelineNV({}, rayPipelineInfo);

  m_device.destroy(raygenSM);
  m_device.destroy(missSM);
  m_device.destroy(shadowmissSM);
  m_device.destroy(chitSM);
  m_device.destroy(chit2SM);
  m_device.destroy(rintSM);
}

//--------------------------------------------------------------------------------------------------
// The Shader Binding Table (SBT)
// - getting all shader handles and writing them in a SBT buffer
// - Besides exception, this could be always done like this
//   See how the SBT buffer is used in run()
//
void HelloVulkan::createRtShaderBindingTable()
{
  auto groupCount =
      static_cast<uint32_t>(m_rtShaderGroups.size());               // 3 shaders: raygen, miss, chit
  uint32_t groupHandleSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier

  // Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
  uint32_t sbtSize = groupCount * groupHandleSize;

  std::vector<uint8_t> shaderHandleStorage(sbtSize);
  m_device.getRayTracingShaderGroupHandlesNV(m_rtPipeline, 0, groupCount, sbtSize,
                                             shaderHandleStorage.data());
  // Write the handles in the SBT
  nvvkpp::SingleCommandBuffer genCmdBuf(m_device, m_graphicsQueueIndex);
  vk::CommandBuffer           cmdBuf = genCmdBuf.createCommandBuffer();

  m_rtSBTBuffer =
      m_alloc.createBuffer(cmdBuf, shaderHandleStorage, vk::BufferUsageFlagBits::eRayTracingNV);
  m_debug.setObjectName(m_rtSBTBuffer.buffer, "SBT");


  genCmdBuf.flushCommandBuffer(cmdBuf);

  m_alloc.flushStaging();
}

//--------------------------------------------------------------------------------------------------
// Ray Tracing the scene
//
void HelloVulkan::raytrace(const vk::CommandBuffer& cmdBuf, const nvmath::vec4f& clearColor)
{
  m_debug.beginLabel(cmdBuf, "Ray trace");
  // Initializing push constant values
  m_rtPushConstants.clearColor     = clearColor;
  m_rtPushConstants.lightPosition  = m_pushConstant.lightPosition;
  m_rtPushConstants.lightIntensity = m_pushConstant.lightIntensity;
  m_rtPushConstants.lightType      = m_pushConstant.lightType;

  cmdBuf.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, m_rtPipeline);
  cmdBuf.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, m_rtPipelineLayout, 0,
                            {m_rtDescSet, m_descSet}, {});
  cmdBuf.pushConstants<RtPushConstant>(m_rtPipelineLayout,
                                       vk::ShaderStageFlagBits::eRaygenNV
                                           | vk::ShaderStageFlagBits::eClosestHitNV
                                           | vk::ShaderStageFlagBits::eMissNV,
                                       0, m_rtPushConstants);

  vk::DeviceSize progSize = m_rtProperties.shaderGroupHandleSize;  // Size of a program identifier
  vk::DeviceSize rayGenOffset   = 0u * progSize;  // Start at the beginning of m_sbtBuffer
  vk::DeviceSize missOffset     = 1u * progSize;  // Jump over raygen
  vk::DeviceSize missStride     = progSize;
  vk::DeviceSize hitGroupOffset = 3u * progSize;  // Jump over the previous shaders
  vk::DeviceSize hitGroupStride = progSize;

  // m_sbtBuffer holds all the shader handles: raygen, n-miss, hit...
  cmdBuf.traceRaysNV(m_rtSBTBuffer.buffer, rayGenOffset,                    //
                     m_rtSBTBuffer.buffer, missOffset, missStride,          //
                     m_rtSBTBuffer.buffer, hitGroupOffset, hitGroupStride,  //
                     m_rtSBTBuffer.buffer, 0, 0,                            //
                     m_size.width, m_size.height,                           //
                     1);                                                    // depth

  m_debug.endLabel(cmdBuf);
}
