#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#define M_PI 3.1415926535897932384626433832795

hitAttributeEXT vec2 attribs;

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

struct Vertex
{
	vec3 position;
	vec3 normal;
	vec2 uv;
	vec4 tangent;
};

// todo: better alignment
layout(set = 0, binding = 2) buffer IndexBuffer { uvec4 data[]; }
indexBuffer;
layout(set = 0, binding = 3) buffer VertexBuffer { Vertex data[]; }
vertexBuffer;

layout(set = 1, binding = 0) buffer MaterialIndexBuffer { uint data[]; }
materialIndexBuffer;

layout(set = 2, binding = 0) uniform sampler2D textures[];



void main() {
	payload.rayActive = 0;

	const uvec4 index = indexBuffer.data[gl_PrimitiveID];
	const Vertex v0 = vertexBuffer.data[index.x];
	const Vertex v1 = vertexBuffer.data[index.y];
	const Vertex v2 = vertexBuffer.data[index.z];

	const vec3 barycentrics = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
	const vec2 texCoord = v0.uv.xy * barycentrics.x + v1.uv.xy * barycentrics.y + v2.uv.xy * barycentrics.z;

	uint materialIndex = materialIndexBuffer.data[gl_PrimitiveID];

	payload.directColor = texture(textures[materialIndex], texCoord).xyz;

	return;
}
