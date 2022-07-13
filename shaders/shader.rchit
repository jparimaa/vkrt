#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#define M_PI 3.1415926535897932384626433832795

hitAttributeEXT vec2 hitCoordinate;

layout(location = 0) rayPayloadInEXT Payload {
  vec3 rayOrigin;
  vec3 rayDirection;
  vec3 previousNormal;
  vec3 directColor;
  vec3 indirectColor;
  int rayDepth;
  int rayActive;
}
payload;

layout(location = 1) rayPayloadEXT bool isShadow;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0) uniform Camera {
  vec4 position;
  vec4 right;
  vec4 up;
  vec4 forward;
  uint frameCount;
}
camera;

layout(set = 0, binding = 2) buffer IndexBuffer { uint data[]; }
indexBuffer;
layout(set = 0, binding = 3) buffer VertexBuffer { float data[]; }
vertexBuffer;

layout(set = 1, binding = 0) buffer MaterialIndexBuffer { uint data[]; }
materialIndexBuffer;

layout(set = 2, binding = 0) uniform texture2D textures[];


void main() {
  payload.rayActive = 0;

  uint materialIndex = materialIndexBuffer.data[gl_PrimitiveID];
  const vec3 barycentricCoords = vec3(1.0f - hitCoordinate.x - hitCoordinate.y, hitCoordinate.x, hitCoordinate.y);
  payload.directColor = barycentricCoords * (10.0 / materialIndex);


  return;
}
