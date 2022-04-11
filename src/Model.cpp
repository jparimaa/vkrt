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

int getSourceOrMinusOne(const std::vector<tinygltf::Texture>& textures, int index)
{
    if (index < 0)
    {
        return -1;
    }
    return textures[index].source;
}

std::vector<Model::Primitive> loadPrimitives(const tinygltf::Model& model)
{
    std::vector<Model::Primitive> primitives(model.meshes[0].primitives.size());
    for (size_t i = 0; i < model.meshes[0].primitives.size(); ++i)
    {
        const tinygltf::Primitive& gltfPrimitive = model.meshes[0].primitives[i];

        { // Indices
            const tinygltf::Accessor& accessor = model.accessors[gltfPrimitive.indices];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            std::vector<uint32_t>& indices = primitives[i].indices;
            indices.resize(accessor.count);

            const size_t elementSizeInBytes = getAccessorElementSizeInBytes(accessor);
            const size_t indexOffset = bufferView.byteOffset + accessor.byteOffset;
            const unsigned char* bufferPtr = &buffer.data[indexOffset];
            unsigned short indexValue = 0;
            const size_t lastIndex = indexOffset + bufferView.byteLength - 1;

            for (size_t i = 0; i < accessor.count; ++i)
            {
                CHECK(bufferPtr < &buffer.data[lastIndex]);
                std::memcpy(&indexValue, bufferPtr, sizeof(unsigned short));
                indices[i] = indexValue;
                bufferPtr += bufferView.byteStride + elementSizeInBytes;
            }
        }

        // Vertices
        for (const auto& [attributeName, attributeIndex] : gltfPrimitive.attributes)
        {
            const tinygltf::Accessor& accessor = model.accessors[attributeIndex];
            const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
            const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

            std::vector<Model::Vertex>& vertices = primitives[i].vertices;
            vertices.resize(accessor.count);

            const size_t elementSizeInBytes = getAccessorElementSizeInBytes(accessor);
            const size_t offset = bufferView.byteOffset + accessor.byteOffset;
            const unsigned char* bufferPtr = &buffer.data[offset];
            const size_t lastIndex = offset + bufferView.byteLength - 1;

            for (size_t accessorIndex = 0; accessorIndex < accessor.count; ++accessorIndex)
            {
                CHECK(bufferPtr < &buffer.data[lastIndex]);

                if (attributeName == "POSITION")
                {
                    std::memcpy(&vertices[accessorIndex].position, bufferPtr, elementSizeInBytes);
                }
                else if (attributeName == "NORMAL")
                {
                    std::memcpy(&vertices[accessorIndex].normal, bufferPtr, elementSizeInBytes);
                }
                else if (attributeName == "TEXCOORD_0")
                {
                    std::memcpy(&vertices[accessorIndex].uv, bufferPtr, elementSizeInBytes);
                }
                else if (attributeName == "TANGENT")
                {
                    std::memcpy(&vertices[accessorIndex].tangent, bufferPtr, elementSizeInBytes);
                }
                bufferPtr += bufferView.byteStride;
            }
        }
    }
    return primitives;
}

std::vector<Model::Material> loadMaterials(const tinygltf::Model& gltfModel)
{
    std::vector<Model::Material> materials(gltfModel.materials.size());
    const std::vector<tinygltf::Texture>& t = gltfModel.textures;

    for (size_t i = 0; i < gltfModel.materials.size(); ++i)
    {
        const tinygltf::Material& m = gltfModel.materials[i];
        materials[i].baseColor = getSourceOrMinusOne(t, m.pbrMetallicRoughness.baseColorTexture.index);
        materials[i].metallicRoughnessImage = getSourceOrMinusOne(t, m.pbrMetallicRoughness.metallicRoughnessTexture.index);
        materials[i].normalImage = getSourceOrMinusOne(t, m.normalTexture.index);
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
    tinygltf::Model gltfModel;
    tinygltf::TinyGLTF loader;
    std::string errorMessage;
    std::string warningMessage;

    const std::string filepath = c_modelsFolder + filename;
    printf("Loading model %s... ", filepath.c_str());
    const bool modelLoaded = loader.LoadASCIIFromFile(&gltfModel, &errorMessage, &warningMessage, filepath);

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
    CHECK(!gltfModel.meshes.empty());

    primitives = loadPrimitives(gltfModel);
    materials = loadMaterials(gltfModel);
    images = loadImages(gltfModel);

    printf("Completed\n");
}
