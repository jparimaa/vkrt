#include "Model.hpp"
#include "Utils.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NOEXCEPTION
#include <tiny_gltf.h>

#include <string>
#include <cstring>
#include <unordered_map>

namespace
{
const std::unordered_map<int, size_t> c_componentTypeSizes{
    {TINYGLTF_COMPONENT_TYPE_BYTE, 1},
    {TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE, 1},
    {TINYGLTF_COMPONENT_TYPE_SHORT, 2},
    {TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT, 2},
    {TINYGLTF_COMPONENT_TYPE_INT, 4},
    {TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT, 4},
    {TINYGLTF_COMPONENT_TYPE_FLOAT, 4},
    {TINYGLTF_COMPONENT_TYPE_DOUBLE, 8},
};

const std::unordered_map<int, size_t> c_typeCounts{
    {TINYGLTF_TYPE_SCALAR, 1},
    {TINYGLTF_TYPE_VEC2, 2},
    {TINYGLTF_TYPE_VEC3, 3},
    {TINYGLTF_TYPE_VEC4, 4},
};

size_t getAccessorElementSizeInBytes(const tinygltf::Accessor& accessor)
{
    const size_t componentTypeSize = c_componentTypeSizes.at(accessor.componentType);
    const size_t typeCount = c_typeCounts.at(accessor.type);
    return componentTypeSize * typeCount;
}

std::vector<Model::Vertex> loadVertices(const tinygltf::Model& model)
{
    std::vector<Model::Vertex> vertices;
    for (const auto& [attributeName, attributeIndex] : model.meshes[0].primitives[0].attributes)
    {
        const tinygltf::Accessor& accessor = model.accessors[attributeIndex];
        const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
        const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

        if (vertices.empty())
        {
            vertices.resize(accessor.count);
        }
        CHECK(vertices.size() == accessor.count);

        const size_t elementSizeInBytes = getAccessorElementSizeInBytes(accessor);
        const size_t offset = bufferView.byteOffset + accessor.byteOffset;
        const unsigned char* bufferPtr = &buffer.data[offset];
        for (size_t i = 0; i < accessor.count; ++i)
        {
            CHECK(bufferPtr < &buffer.data[offset + bufferView.byteLength]);
            if (attributeName == "POSITION")
            {
                std::memcpy(&vertices[i].position, bufferPtr, elementSizeInBytes);
            }
            else if (attributeName == "NORMAL")
            {
                std::memcpy(&vertices[i].normal, bufferPtr, elementSizeInBytes);
            }
            else if (attributeName == "TEXCOORD_0")
            {
                std::memcpy(&vertices[i].uv, bufferPtr, elementSizeInBytes);
            }
            bufferPtr += elementSizeInBytes + bufferView.byteStride;
        }
    }
    return vertices;
}

std::vector<uint32_t> loadIndices(const tinygltf::Model& model)
{
    std::vector<uint32_t> indices;

    const tinygltf::Accessor& indicesAccessor = model.accessors[model.meshes[0].primitives[0].indices];
    const tinygltf::BufferView& indexBufferView = model.bufferViews[indicesAccessor.bufferView];
    const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
    indices.resize(indicesAccessor.count);

    const size_t indexOffset = indexBufferView.byteOffset + indicesAccessor.byteOffset;
    const unsigned char* indexBufferPtr = &indexBuffer.data[indexOffset];
    unsigned short indexValue = 0;
    for (size_t i = 0; i < indicesAccessor.count; ++i)
    {
        CHECK(indexBufferPtr < &indexBuffer.data[indexOffset + indexBufferView.byteLength]);
        std::memcpy(&indexValue, indexBufferPtr, sizeof(unsigned short));
        indices[i] = indexValue;
        indexBufferPtr += indexBufferView.byteStride + sizeof(unsigned short);
    }

    return indices;
}

std::vector<Model::Material> loadMaterials(const tinygltf::Model& model)
{
    std::vector<Model::Material> materials(model.materials.size());

    for (size_t i = 0; i < model.materials.size(); ++i)
    {
        const tinygltf::Material& m = model.materials[i];
        materials[i].baseColor = model.textures[m.pbrMetallicRoughness.baseColorTexture.index].source;
        materials[i].metallicRoughnessImage = model.textures[m.pbrMetallicRoughness.metallicRoughnessTexture.index].source;
        materials[i].normalImage = model.textures[m.normalTexture.index].source;
        materials[i].emissiveImage = model.textures[m.emissiveTexture.index].source;
        materials[i].occlusionImage = model.textures[m.occlusionTexture.index].source;
    }

    return materials;
}

std::vector<Model::Image> loadImages(tinygltf::Model& model)
{
    std::vector<Model::Image> images(model.images.size());

    for (size_t i = 0; i < model.images.size(); ++i)
    {
        images[i].width = model.images[i].width;
        images[i].height = model.images[i].height;
        images[i].components = model.images[i].component;
        images[i].bitsPerChannel = model.images[i].bits;
        images[i].data = std::move(model.images[i].image);
    }
    return images;
}
} // namespace

Model::Model(const std::string& filename)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string errorMessage;
    std::string warningMessage;

    const std::string filepath = c_modelsFolder + filename;
    printf("Loading model %s... ", filepath.c_str());
    const bool modelLoaded = loader.LoadBinaryFromFile(&model, &errorMessage, &warningMessage, filepath);

    if (!warningMessage.empty())
    {
        LOGW(warningMessage.c_str());
        abort();
    }

    if (!errorMessage.empty())
    {
        LOGE(errorMessage.c_str());
        abort();
    }

    CHECK(modelLoaded);
    CHECK(!model.meshes.empty());

    vertices = loadVertices(model);
    indices = loadIndices(model);
    materials = loadMaterials(model);
    images = loadImages(model);

    printf("Completed\n");
}
