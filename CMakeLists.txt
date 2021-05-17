cmake_minimum_required(VERSION 3.5)
project(vk_raytracing_tutorial)


#####################################################################################
# look for nvpro_core 1) as a sub-folder 2) at some other locations
# this cannot be put anywhere else since we still didn't find setup.cmake yet
#
if(NOT BASE_DIRECTORY)

  find_path(BASE_DIRECTORY
    NAMES nvpro_core/cmake/setup.cmake
    PATHS ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/../.. 
    REQUIRED
    DOC "Directory containing nvpro_core"
    )
endif()
if(EXISTS ${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
  include(${BASE_DIRECTORY}/nvpro_core/cmake/setup.cmake)
else()
  message(FATAL_ERROR "could not find base directory, please set BASE_DIRECTORY to folder containing nvpro_core")
endif()

set(CMAKE_CXX_STANDARD 17)

_add_package_VulkanSDK()
_add_package_ImGUI()

_add_nvpro_core_lib()

add_subdirectory(ray_tracing__advance) 
add_subdirectory(ray_tracing__before) 
add_subdirectory(ray_tracing__simple)
add_subdirectory(ray_tracing_animation)
add_subdirectory(ray_tracing_anyhit)
add_subdirectory(ray_tracing_callable)
add_subdirectory(ray_tracing_jitter_cam)
add_subdirectory(ray_tracing_manyhits)
add_subdirectory(ray_tracing_reflections)
add_subdirectory(ray_tracing_instances)
add_subdirectory(ray_tracing_intersection)

install(DIRECTORY "media" CONFIGURATIONS Release DESTINATION "bin_${ARCH}")
install(DIRECTORY "media" CONFIGURATIONS Debug DESTINATION "bin_${ARCH}_debug")