#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;
layout(location = 3) in vec4 inTangent;

layout(set = 0, binding = 0) uniform UBO
{
    mat4 wvpMatrix;
}
ubo;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUv;

void main()
{
    gl_Position = ubo.wvpMatrix * vec4(inPosition, 1.0);
    outNormal = inNormal;
    outUv = inUv;
}
