#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
#include <unordered_map>

class Model final
{
public:
    // todo: better alignment
    struct Vertex
    {
        glm::vec4 position{};
        glm::vec4 normal{};
        glm::vec4 uv{};
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

    using Index = uint32_t;

    struct Primitive
    {
        std::vector<Vertex> vertices;
        std::vector<Index> indices;
        int material = -1;
    };

    Model(const std::string& filename);
    ~Model() {}

    std::vector<Primitive> primitives;
    std::vector<Material> materials;
    std::vector<Image> images;

    uint64_t vertexBufferSizeInBytes = 0;
    uint64_t indexBufferSizeInBytes = 0;
};
