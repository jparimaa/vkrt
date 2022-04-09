#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>

class Model final
{
public:
    struct Vertex
    {
        glm::vec3 position;
        glm::vec2 uv;
        glm::vec3 normal;
    };

    struct Material
    {
        int baseColor = -1;
        int metallicRoughnessImage = -1;
        int normalImage = -1;
        int emissiveImage = -1;
        int occlusionImage = -1;
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

    Model(const std::string& filename);
    ~Model() {}

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<Material> materials;
    std::vector<Image> images;
};
