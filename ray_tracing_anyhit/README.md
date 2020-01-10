![logo](http://nvidianews.nvidia.com/_ir/219/20157/NV_Designworks_logo_horizontal_greenblack.png)

By [Martin-Karl Lefrançois](https://devblogs.nvidia.com/author/mlefrancois/), Neil Bickford

Updated **December 2019**

# NVIDIA Vulkan Ray Tracing Tutorial - Any Hit Shader

![anyhit](images/anyhit.png)

This is an extension of the Vulkan ray tracing [tutorial](../ray_tracing__simple).

Like closest hit shaders, any hit shaders operate on intersections between rays and geometry. However, the any hit shader will be executed for all hits along the ray. The closest hit shader will then be invoked on the closest accepted intersection.

The any hit shader can be useful for discarding intersections, such as for alpha cutouts for example, but can also be used for simple transparency. In this example we will show what is needed to do to add this shader type and to create a transparency effect.

**Note**:
> This example is based on many elements from the [Antialiasing Tutorial](vkrt_tuto_jitter_cam.md).

# Any Hit

## `raytrace.rahit`

Create a new shader file `raytrace.rahit` and rerun CMake to have it added to the solution.

This shader starts like `raytrace.chit`, but uses less information.

~~~~ C++
#version 460
#extension GL_NV_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_GOOGLE_include_directive : enable

#include "random.glsl"
#include "raycommon.glsl"
#include "wavefront.glsl"

// clang-format off
layout(location = 0) rayPayloadInNV hitPayload prd;

layout(binding = 2, set = 1, scalar) buffer ScnDesc { sceneDesc i[]; } scnDesc;
layout(binding = 4, set = 1)  buffer MatIndexColorBuffer { int i[]; } matIndex[];
layout(binding = 5, set = 1, scalar) buffer Vertices { Vertex v[]; } vertices[];
layout(binding = 6, set = 1) buffer Indices { uint i[]; } indices[];
layout(binding = 1, set = 1, scalar) buffer MatColorBufferObject { WaveFrontMaterial m[]; } materials[];
// clang-format on
~~~~

**Note**:
> You can find the source of `random.glsl` in the Antialiasing Tutorial [here](../ray_tracing_jitter_cam/README.md#toc1.1).

For the any hit shader, we need to know which material we hit, and whether that material supports transparency. If it is opaque, we simply return, which means that the hit will be accepted.

~~~~ C++
void main()
{
  // Object of this instance
  uint objId = scnDesc.i[gl_InstanceID].objId;
  // Indices of the triangle
  uint ind = indices[objId].i[3 * gl_PrimitiveID + 0];
  // Vertex of the triangle
  Vertex v0 = vertices[objId].v[ind.x];

  // Material of the object
  int               matIdx = matIndex[objId].i[gl_PrimitiveID];
  WaveFrontMaterial mat    = materials[objId].m[matIdx];

  if (mat.illum != 4)
    return;
~~~~

Now we will apply transparency:

~~~~ C++
  if (mat.dissolve == 0.0)
      ignoreIntersectionNV();
  else if(rnd(prd.seed) > mat.dissolve)
     ignoreIntersectionNV();
}
~~~~

As you can see, we are using a random number generator to determine if the ray hits or ignores the object. If we accumulate enough rays, the final result will converge to what we want.

## `raycommon.glsl`

The random `seed` also needs to be passed in the ray payload.

In `raycommon.glsl`, add the seed:

~~~~ C++
struct hitPayload
{
  vec3 hitValue;
  uint seed;
};
~~~~

## Adding Any Hit to `createRtPipeline`

The any hit shader will be part of the hit shader group. Currently, the hit shader group only contains the closest hit shader.

In `createRtPipeline()`, after loading `raytrace.rchit.spv`, load `raytrace.rahit.spv`

~~~~ C++
  vk::ShaderModule ahitSM =
      nvvkpp::util::createShaderModule(m_device,  //
                                       nvh::loadFile("shaders/raytrace.rahit.spv", true, paths));
~~~~

add the any hit shader to the hit group

~~~~ C++
  stages.push_back({{}, vk::ShaderStageFlagBits::eClosestHitNV, chitSM, "main"});
  hg.setClosestHitShader(static_cast<uint32_t>(stages.size() - 1));
  stages.push_back({{}, vk::ShaderStageFlagBits::eAnyHitNV, ahitSM, "main"});
  hg.setAnyHitShader(static_cast<uint32_t>(stages.size() - 1));
  m_rtShaderGroups.push_back(hg);
~~~~

and at the end, delete it:

~~~~ C++
m_device.destroy(ahitSM);
~~~~

## Give access of the buffers to the Any Hit shader

In `createDescriptorSetLayout()`, we need to allow the Any Hit shader to access some buffers.

This is the case for the material and scene description buffers

~~~~ C++
  // Materials (binding = 1)
  m_descSetLayoutBind.emplace_back(
      vkDS(1, vkDT::eStorageBuffer, nbObj,
           vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitNV | vkSS::eAnyHitNV));
  // Scene description (binding = 2)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(2, vkDT::eStorageBuffer, 1,
           vkSS::eVertex | vkSS::eFragment | vkSS::eClosestHitNV | vkSS::eAnyHitNV));
~~~~

and also for the vertex, index and material index buffers:

~~~~ C++
  // Materials (binding = 4)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(4, vkDT::eStorageBuffer, nbObj,
           vkSS::eFragment | vkSS::eClosestHitNV | vkSS::eAnyHitNV));
  // Storing vertices (binding = 5)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(5, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitNV | vkSS::eAnyHitNV));
  // Storing indices (binding = 6)
  m_descSetLayoutBind.emplace_back(  //
      vkDS(6, vkDT::eStorageBuffer, nbObj, vkSS::eClosestHitNV | vkSS::eAnyHitNV));
~~~~

## Opaque Flag

In the example, when creating `VkGeometryNV` objects, we set their flags to `vk::GeometryFlagBitsNV::eOpaque`. However, this avoided invoking the any hit shader.

We could remove all of the flags, but another issue could happen: the any hit shader could be called multiple times for the same triangle. To have the any hit shader process only one hit per triangle, set the `eNoDuplicateAnyHitInvocation` flag:

~~~~ C++
geometry.setFlags(vk::GeometryFlagBitsNV::eNoDuplicateAnyHitInvocation);
~~~~

## `raytrace.rgen`

If you have done the previous [Jitter Camera/Antialiasing](../ray_tracing_jitter_cam) tutorial,
you will need just a few changes.

First, `seed` will need to be available in the any hit shader, which is the reason we have added it to the hitPayload structure.

Change the local `seed` to `prd.seed` everywhere.

~~~~ C++
prd.seed = tea(gl_LaunchIDNV.y * gl_LaunchSizeNV.x + gl_LaunchIDNV.x, pushC.frame);
~~~~

For optimization, the `TraceNV` call was using the `gl_RayFlagsOpaqueNV` flag. But
this will skip the any hit shader, so change it to

~~~~ C++
uint  rayFlags = gl_RayFlagsNoneNV;
~~~~

## `raytrace.rchit`

Similarly, in the closest hit shader, change the flag to `gl_RayFlagsSkipClosestHitShaderNV`, as we want to enable the any hit and miss shaders, but we still don't care
about the closest hit shader for shadow rays. This will enable transparent shadows.

~~~~ C++
uint  flags = gl_RayFlagsSkipClosestHitShaderNV;
~~~~

# Scene and Model

For a more interesting scene, you can replace the `helloVk.loadModel` calls in `main()` with the following scene:

~~~~ C++
  helloVk.loadModel(nvh::findFile("media/scenes/wuson.obj", defaultSearchPaths));
  helloVk.loadModel(nvh::findFile("media/scenes/sphere.obj", defaultSearchPaths),
                    nvmath::scale_mat4(nvmath::vec3f(1.5f))
                        * nvmath::translation_mat4(nvmath::vec3f(0.0f, 1.0f, 0.0f)));
  helloVk.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths));
~~~~

## OBJ Materials

By default, all objects are opaque, you will need to change the material description.

Edit the first few lines of `media/scenes/wuson.mtl` and `media/scenes/sphere.mtl` to use a new illumination model (4) with a dissolve value of 0.5:

~~~~ C++
newmtl  default
illum 4
d 0.5
...
~~~~

# Accumulation

As mentioned earlier, for the effect to work, we need to accumulate frames over time. Please implement the following from [Jitter Camera/Antialiasing](vkrt_tuto_jitter_cam.md):

* [Frame Number](vkrt_tuto_jitter_cam.md#toc1.2)
* [Storing or Updating](vkrt_tuto_jitter_cam.md#toc1.4)
* [Application Frame Update](vkrt_tuto_jitter_cam.md#toc2)
