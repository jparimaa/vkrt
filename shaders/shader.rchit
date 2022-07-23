#version 460

#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

hitAttributeEXT vec2 attribs;

layout(location = 0) rayPayloadInEXT Payload
{
    vec3 hitValue;
    int done;
    int depth;
    float attenuation;
    vec3 rayOrigin;
    vec3 rayDir;
}
payload;

layout(location = 1) rayPayloadEXT bool isShadowed;

layout(binding = 0, set = 0) uniform accelerationStructureEXT topLevelAS;
layout(binding = 1, set = 0) uniform CommonUniformBuffer
{
    mat4 viewInverse;
    mat4 projInverse;
    vec4 position;
    vec4 right;
    vec4 up;
    vec4 forward;
    vec4 lightPositions[4];
}
commonBuffer;

// Todo: could fix vertex alignment so there's no need to have everything as vec4
struct Vertex
{
    vec4 position;
    vec4 normal;
    vec4 uv;
    vec4 tangent;
};

struct MaterialInfo
{
    int baseColorTextureIndex;
    int metallicRoughnessTextureIndex;
    int normalTextureIndex;
    int indexBufferOffset;
};

struct IndexInfo
{
    uint x;
    uint y;
    uint z;
};

layout(std430, set = 0, binding = 2) buffer IndexBuffer
{
    IndexInfo data[];
}
indexBuffer;

layout(set = 0, binding = 3) buffer VertexBuffer
{
    Vertex data[];
}
vertexBuffer;

layout(set = 1, binding = 0) buffer MaterialIndexBuffer
{
    MaterialInfo data[];
}
materialIndexBuffer;

layout(set = 2, binding = 0) uniform sampler2D textures[];

mat3 getTBN(vec3 normal, vec3 tangent, mat3 M)
{
    const vec3 N = normal;
    const vec3 T = normalize(mat3(M) * tangent);
    const vec3 B = cross(T, N);
    return mat3(T, B, N);
}

void main()
{
    const int indexBufferOffset = materialIndexBuffer.data[gl_GeometryIndexEXT].indexBufferOffset;
    const IndexInfo index = indexBuffer.data[indexBufferOffset + gl_PrimitiveID];
    const Vertex v0 = vertexBuffer.data[index.x];
    const Vertex v1 = vertexBuffer.data[index.y];
    const Vertex v2 = vertexBuffer.data[index.z];

    const vec3 barycentrics = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec2 uv = v0.uv.xy * barycentrics.x + v1.uv.xy * barycentrics.y + v2.uv.xy * barycentrics.z;

    const vec3 position = v0.position.xyz * barycentrics.x + v1.position.xyz * barycentrics.y + v2.position.xyz * barycentrics.z;
    const vec3 worldPos = vec3(gl_ObjectToWorldEXT * vec4(position, 1.0));

    const vec3 normal = v0.normal.xyz * barycentrics.x + v1.normal.xyz * barycentrics.y + v2.normal.xyz * barycentrics.z;
    const vec3 worldNormal = normalize(vec3(normal * gl_WorldToObjectEXT)); // Transforming the normal to world space

    const vec3 tangent = v0.tangent.xyz * barycentrics.x + v1.tangent.xyz * barycentrics.y + v2.tangent.xyz * barycentrics.z;

    const mat3 TBN = getTBN(worldNormal, tangent, mat3(1.0));
    uint normalTextureIndex = materialIndexBuffer.data[gl_GeometryIndexEXT].normalTextureIndex;
    const vec3 mapNormal = texture(textures[normalTextureIndex], uv).xyz;
    const vec3 perturbedNormal = normalize(TBN * normalize(mapNormal * 2.0 - vec3(1.0)));

    float totalLightAmount = 0.0;
    const float lightIntensity = 10.0;

    const uint flags = //
        gl_RayFlagsTerminateOnFirstHitEXT | // Terminate on first hit, no need to go further
        gl_RayFlagsOpaqueEXT | // Will not call the any hit shader, so all objects will be opaque
        gl_RayFlagsSkipClosestHitShaderEXT; // Will not invoke the hit shader, only the miss shader

    // Light + shadow
    for (int i = 0; i < 4; ++i)
    {
        const vec3 lightVec = commonBuffer.lightPositions[i].xyz - worldPos;
        const float lightDistance = length(lightVec);
        const vec3 lightDir = normalize(lightVec);

        const float diffuse = clamp(dot(perturbedNormal, lightDir), 0, 1);
        const float lightPower = lightIntensity / (lightDistance * lightDistance);

        float shadowMultiplier = 1.0;
        if (dot(perturbedNormal, lightDir) > 0)
        {
            isShadowed = true;
            traceRayEXT(topLevelAS, // acceleration structure
                        flags, // rayFlags
                        0xFF, // cullMask
                        0, // sbtRecordOffset
                        0, // sbtRecordStride
                        1, // missIndex to use shadow miss shader
                        worldPos, // ray origin
                        0.001, // ray min range
                        lightDir, // ray direction
                        lightDistance, // ray max range
                        1 // payload location to check if shadowed
            );

            if (isShadowed)
            {
                shadowMultiplier = 0.3;
            }
        }

        totalLightAmount += diffuse * lightPower * shadowMultiplier;
    }

    const float ambient = 0.1;

    uint baseColorTextureIndex = materialIndexBuffer.data[gl_GeometryIndexEXT].baseColorTextureIndex;
    const vec3 baseColor = texture(textures[baseColorTextureIndex], uv).xyz;
    payload.hitValue = baseColor * totalLightAmount * payload.attenuation + baseColor * ambient;

    // Reflection
    const uint metallicRoughnessTextureIndex = materialIndexBuffer.data[gl_GeometryIndexEXT].metallicRoughnessTextureIndex;
    const float metallic = texture(textures[metallicRoughnessTextureIndex], uv).b;
    if (metallic > 0.1) // Not very realistic but works in this case
    {
        const float reflectAmount = 0.5f * metallic;
        payload.attenuation *= reflectAmount;
        payload.hitValue *= (1.0 - payload.attenuation);
        payload.done = 0;
        payload.rayOrigin = worldPos;
        payload.rayDir = reflect(gl_WorldRayDirectionEXT, perturbedNormal);
    }
}
