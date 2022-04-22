#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(set = 1, binding = 0) uniform sampler2D baseColor;
layout(set = 1, binding = 1) uniform sampler2D metallicRoughness;
layout(set = 1, binding = 2) uniform sampler2D normal;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inUv;

layout(location = 0) out vec4 outColor;

void main()
{
    vec4 color = texture(baseColor, inUv);
    if (color.a < 0.1)
    {
        // A bit nasty but works for this case.
        discard;
    }
    outColor = color;
}
