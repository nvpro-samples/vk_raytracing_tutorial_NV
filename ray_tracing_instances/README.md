﻿![logo](http://nvidianews.nvidia.com/_ir/219/20157/NV_Designworks_logo_horizontal_greenblack.png)

By [Martin-Karl Lefrançois](https://devblogs.nvidia.com/author/mlefrancois/), Neil Bickford

Updated **December 2019**

# NVIDIA Vulkan Ray Tracing Tutorial - Instances

![instances](images/VkInstances.png)

This is an extension of the Vulkan ray tracing [tutorial](../ray_tracing__simple/README.md).

Ray tracing can easily handle having many object instances at once. For instance, a top level acceleration structure can have many different instances of a bottom level acceleration structure. However, when we have many different objects, we can run into problems with memory allocation. Many Vulkan implementations support no more than 4096 allocations, while our current application creates 4 allocations per object (Vertex, Index, and Material), then one for the BLAS. That means we are hitting the limit with just above 1000 objects.

We could modify the code and do clever memory allocation, but we could also use one of the memory allocators available online. In this tutorial we will use the [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) (VMA).

# Helpers

Download [vk_mem_alloc.h](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/blob/master/src/vk_mem_alloc.h) from GitHub and add this to the `current` folder.

There is already a variation of the allocator for VMA, which is located under [nvpro-samples](https://github.com/nvpro-samples/shared_sources/tree/master/nvvkpp). This allocator has the same simple interface as the `AllocatorDedicated` class in `allocator_dedicated_vkpp.hpp`, but will use VMA for memory management.

# Many Instances

First, let's look how the scene would look like when we have just a few objects, with many instances.

In `main.cpp`, add the following includes:

~~~~ C++
#include <random>
~~~~

VMA might use dedicated memory, which we do, so you need to add the following extension to the 
creation of the context.

~~~~ C++
  contextInfo.addDeviceExtension(VK_KHR_BIND_MEMORY_2_EXTENSION_NAME);
~~~~  

Then replace the calls to `helloVk.loadModel` in `main()` by

~~~~ C++
  // Creation of the example
  helloVk.loadModel(nvh::findFile("media/scenes/cube.obj", defaultSearchPaths));
  helloVk.loadModel(nvh::findFile("media/scenes/cube_multi.obj", defaultSearchPaths));
  helloVk.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths));

  std::random_device              rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937                    gen(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<float> dis(1.0f, 1.0f);
  std::normal_distribution<float> disn(0.05f, 0.05f);

  for(int n = 0; n < 2000; ++n)
  {
    HelloVulkan::ObjInstance inst;
    inst.objIndex       = n % 2;
    inst.txtOffset      = 0;
    float         scale = fabsf(disn(gen));
    nvmath::mat4f mat =
        nvmath::translation_mat4(nvmath::vec3f{dis(gen), 2.0f + dis(gen), dis(gen)});
    mat              = mat * nvmath::rotation_mat4_x(dis(gen));
    mat              = mat * nvmath::scale_mat4(nvmath::vec3f(scale));
    inst.transform   = mat;
    inst.transformIT = nvmath::transpose(nvmath::invert((inst.transform)));
    helloVk.m_objInstance.push_back(inst);
  }
~~~~

**Note**:
> This will create 3 models (OBJ) and their instances, and then add 2000 instances distributed between green cubes and cubes with one color per face.

# Many Objects

Instead of creating many instances, create many objects.

Remove the previous code and replace it with the following

~~~~ C++
  // Creation of the example
  std::random_device              rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937                    gen(rd());  //Standard mersenne_twister_engine seeded with rd()
  std::normal_distribution<float> dis(1.0f, 1.0f);
  std::normal_distribution<float> disn(0.05f, 0.05f);
  for(int n = 0; n < 2000; ++n)
  {
    helloVk.loadModel(nvh::findFile("media/scenes/cube_multi.obj", defaultSearchPaths));
    HelloVulkan::ObjInstance& inst = helloVk.m_objInstance.back();

    float         scale = fabsf(disn(gen));
    nvmath::mat4f mat =
        nvmath::translation_mat4(nvmath::vec3f{dis(gen), 2.0f + dis(gen), dis(gen)});
    mat              = mat * nvmath::rotation_mat4_x(dis(gen));
    mat              = mat * nvmath::scale_mat4(nvmath::vec3f(scale));
    inst.transform   = mat;
    inst.transformIT = nvmath::transpose(nvmath::invert((inst.transform)));
  }

  helloVk.loadModel(nvh::findFile("media/scenes/plane.obj", defaultSearchPaths));
~~~~

The example might still work, but the console will print the following error after loading 1363 objects. All other objects allocated after the 1363rd will fail to be displayed.

**Error**:
> Error: VUID_Undefined
    Number of currently valid memory objects is not less than the maximum allowed (4096).
      - Object[0] -  Type Device

**Note**:
> This is the best case; the application can run out of memory and crash if substantially more objects are created (e.g. 20,000)

# Vulkan Memory Allocator (VMA)

It is possible to use a memory allocator to fix this issue.

## `hello_vulkan.h`

In `hello_vulkan.h`, add the downloaded helper:

~~~~ C++
#include "nvvkpp/allocator_vma_vkpp.hpp"
~~~~

Indicate to the `RaytracingBuilder` that we will use VMA:

~~~~ C++
// #VKRay
#define ALLOC_VMA
//#define ALLOC_DEDICATED
~~~~

Replace the definition of buffers and textures

~~~~ C++
#if defined(ALLOC_DEDICATED)
using nvvkBuffer  = nvvkpp::BufferDedicated;
using nvvkTexture = nvvkpp::TextureDedicated;
#elif defined(ALLOC_VMA)
using nvvkBuffer  = nvvkpp::BufferVma;
using nvvkTexture = nvvkpp::TextureVma;
#endif
~~~~

And do the same for the allocator

~~~~ C++
#if defined(ALLOC_DEDICATED)
  nvvkpp::AllocatorDedicated m_alloc;  // Allocator for buffer, images, acceleration structures
#elif defined(ALLOC_VMA)
  nvvkpp::AllocatorVma m_alloc;  // Allocator for buffer, images, acceleration structures
  VmaAllocator         m_vmaAllocator;
#endif
~~~~

## `hello_vulkan.cpp`

In the source file there are also a few changes to make.

First, the following should only be defined once in the entire program, and it should be defined before `#include "hello_vulkan.h"`:

~~~~ C++
#define VMA_IMPLEMENTATION
~~~~

VMA needs to be created, which will be done in the `setup()` function:

~~~~ C++
#if defined(ALLOC_DEDICATED)
  m_alloc.init(device, physicalDevice);
#elif defined(ALLOC_VMA)
  VmaAllocatorCreateInfo allocatorInfo = {};
  allocatorInfo.physicalDevice         = physicalDevice;
  allocatorInfo.device                 = device;
  allocatorInfo.flags |=
      VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT | VMA_ALLOCATOR_CREATE_KHR_BIND_MEMORY2_BIT;
  vmaCreateAllocator(&allocatorInfo, &m_vmaAllocator);
  m_alloc.init(device, m_vmaAllocator);
#endif
~~~~

When using VMA, memory buffer mapping is done through the VMA interface (instead of the VKDevice). Therefore, change the lines at the end of `updateUniformBuffer()` to

~~~~ C++
#if defined(ALLOC_DEDICATED)
  void* data = m_device.mapMemory(m_cameraMat.allocation, 0, sizeof(CameraMatrices));
  memcpy(data, &ubo, sizeof(ubo));
  m_device.unmapMemory(m_cameraMat.allocation);
#elif defined(ALLOC_VMA)
  void* data;
  vmaMapMemory(m_vmaAllocator, m_cameraMat.allocation, &data);
  memcpy(data, &ubo, sizeof(ubo));
  vmaUnmapMemory(m_vmaAllocator, m_cameraMat.allocation);
#endif
~~~~

Additionally, VMA has its own usage flags, so since `VMA_MEMORY_USAGE_CPU_TO_GPU` maps to `vkMP::eHostVisible` and `vkMP::eHostCoherent`, change the call to `m_alloc.createBuffer` in `HelloVulkan::createUniformBuffer()` to

~~~~ C++
  m_cameraMat = m_alloc.createBuffer(sizeof(CameraMatrices), vkBU::eUniformBuffer,
#if defined(ALLOC_DEDICATED)
                                     vkMP::eHostVisible | vkMP::eHostCoherent
#elif defined(ALLOC_VMA)
                                     VMA_MEMORY_USAGE_CPU_TO_GPU
#endif
  );
~~~~

The RaytracerBuilder was made to allow various allocators, but we still need to pass the right one in its setup function. Change the last line of `initRayTracing()` to

~~~~ C++
#if defined(ALLOC_DEDICATED)
  m_rtBuilder.setup(m_device, m_physicalDevice, m_graphicsQueueIndex);
#elif defined(ALLOC_VMA)
  m_rtBuilder.setup(m_device, m_vmaAllocator, m_graphicsQueueIndex);
#endif
~~~~

## Destruction

The VMA allocator need to be released in `HelloVulkan::destroyResources()` after the last `m_alloc.destroy`.

~~~~ C++
#if defined(ALLOC_VMA)
  vmaDestroyAllocator(m_vmaAllocator);
#endif
~~~~

# Result

Instead of thousands of allocations, our example will have only 14 allocations. Some of these allocations are even allocated by Dear ImGui, and not by VMA. These are the 14 objects with blue borders below:

![Memory](images/VkInstanceNsight1.png)

Finally, here is the Vulkan Device Memory view from Nsight Graphics:
![VkMemory](images/VkInstanceNsight2.png)