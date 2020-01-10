#version 460
#extension GL_NV_ray_tracing : require
#extension GL_GOOGLE_include_directive : enable

#include "raycommon.glsl"

layout(location = 0) rayPayloadInNV hitPayload prd;
layout(shaderRecordNV) buffer sr_
{
  vec4 c;
}
shaderRec;

void main()
{
  prd.hitValue = shaderRec.c.rgb;
}
