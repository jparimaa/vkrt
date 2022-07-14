#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT Payload
{
    vec3 directColor;
}
payload;

void main()
{
    payload.directColor = vec3(0.0, 0.0, 0.2);
}