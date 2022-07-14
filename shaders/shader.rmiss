#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT Payload
{
    vec3 hitValue;
    int done;
    int depth;
    vec3 rayOrigin;
    vec3 rayDir;
}
payload;

void main()
{
    payload.hitValue = vec3(0.0, 0.0, 0.2);
}