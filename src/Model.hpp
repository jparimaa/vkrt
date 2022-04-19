#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <unordered_map>

class Model final
{
public:
    struct Vertex
    {
        glm::vec3 position{};
        glm::vec3 normal{};
        glm::vec2 uv{};
        glm::vec4 tangent{};
    };

    struct Material
    {
        int baseColor = -1;
        int metallicRoughnessImage = -1;
        int normalImage = -1;
    };

    struct Image
    {
        unsigned int width;
        unsigned int height;
        unsigned int components;
        unsigned int bitsPerChannel;
        std::vector<unsigned char> data;
    };

    struct Primitive
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
        int material = -1;
    };

    using Index = uint32_t;

    Model(const std::string& filename);
    ~Model() {}

    std::vector<Primitive> primitives;
    std::vector<Material> materials;
    std::vector<Image> images;

    uint64_t vertexBufferSizeInBytes = 0;
    uint64_t indexBufferSizeInBytes = 0;
};
