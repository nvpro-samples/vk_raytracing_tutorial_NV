#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout(location = 1) rayPayloadInNV shadowPayload prd;

void main()
{
  prd.isHit = false;
}
