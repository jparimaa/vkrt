#pragma once

#include "Context.hpp"
#include "Camera.hpp"
#include "Model.hpp"
#include <vector>
#include <chrono>
#include <unordered_map>
#include <memory>

class Raytracer final
{
public:
    Raytracer(Context& context);
    ~Raytracer();

    bool render();

private:
    bool update(uint32_t imageIndex);

    void getFunctionPointers();
    void loadModel();
    void setupCamera();
    void updateCamera(double deltaTime);
    void createColorImage();
    void createSwapchainImageViews();
    void createSampler();
    void createTextures();
    void createMipmaps(VkImage image, uint32_t mipLevels, glm::uvec2 imageSize);
    void createVertexAndIndexBuffer();
    void createDescriptorPool();
    void createCommonDescriptorSetLayoutAndAllocate();
    void createMaterialIndexDescriptorSetLayoutAndAllocate();
    void createTexturesDescriptorSetLayoutAndAllocate();
    void createPipeline();
    void createCommonBuffer();
    void createMaterialIndexBuffer();
    void allocateCommandBuffers();
    void createBLAS();
    void createTLAS();
    void updateCommonDescriptorSets();
    void updateMaterialIndexDescriptorSet();
    void updateTexturesDescriptorSets();
    void createShaderBindingTable();

    Context& m_context;
    VkDevice m_device;

    PFN_vkCreateRayTracingPipelinesKHR m_pvkCreateRayTracingPipelinesKHR;
    PFN_vkGetBufferDeviceAddressKHR m_pvkGetBufferDeviceAddressKHR;
    PFN_vkGetAccelerationStructureBuildSizesKHR m_pvkGetAccelerationStructureBuildSizesKHR;
    PFN_vkCreateAccelerationStructureKHR m_pvkCreateAccelerationStructureKHR;
    PFN_vkGetAccelerationStructureDeviceAddressKHR m_pvkGetAccelerationStructureDeviceAddressKHR;
    PFN_vkCmdBuildAccelerationStructuresKHR m_pvkCmdBuildAccelerationStructuresKHR;
    PFN_vkGetRayTracingShaderGroupHandlesKHR m_pvkGetRayTracingShaderGroupHandlesKHR;
    PFN_vkCmdTraceRaysKHR m_pvkCmdTraceRaysKHR;
    PFN_vkDestroyAccelerationStructureKHR m_pvkDestroyAccelerationStructureKHR;

    std::unique_ptr<Model> m_model{nullptr};
    Camera m_camera;
    std::chrono::steady_clock::time_point m_lastRenderTime;
    std::unordered_map<int, bool> m_keysDown;
    VkImage m_colorImage;
    VkDeviceMemory m_colorImageMemory;
    VkImageView m_colorImageView;
    std::vector<VkImageView> m_swapchainImageViews;
    VkSampler m_sampler;
    std::vector<VkImage> m_images;
    VkDeviceMemory m_imageMemory;
    std::vector<VkImageView> m_imageViews;
    VkDescriptorSetLayout m_commonDescriptorSetLayout;
    VkDescriptorSetLayout m_materialIndexDescriptorSetLayout;
    VkDescriptorSetLayout m_texturesDescriptorSetLayout;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;
    VkDescriptorPool m_descriptorPool;
    VkDescriptorSet m_commonDescriptorSet;
    VkDescriptorSet m_materialIndexDescriptorSet;
    VkDescriptorSet m_texturesDescriptorSet;
    VkBuffer m_vertexBuffer;
    VkDeviceMemory m_vertexBufferMemory;
    VkBuffer m_indexBuffer;
    VkDeviceMemory m_indexBufferMemory;
    VkBuffer m_primitiveIndexBuffer;
    VkDeviceMemory m_primitiveIndexBufferMemory;
    size_t m_triangleCount;
    size_t m_vertexDataSize;
    size_t m_indexDataSize;
    VkBuffer m_commonBuffer;
    VkDeviceMemory m_commonBufferMemory;
    VkBuffer m_materialIndexBuffer;
    VkDeviceMemory m_materialIndexBufferMemory;

    VkBuffer m_blasBuffer;
    VkDeviceMemory m_blasMemory;
    VkAccelerationStructureKHR m_blas;
    VkDeviceAddress m_blasDeviceAddress;

    VkBuffer m_blasGeometryInstanceBuffer;
    VkDeviceMemory m_blasGeometryInstanceMemory;
    VkBuffer m_tlasBuffer;
    VkDeviceMemory m_tlasMemory;
    VkAccelerationStructureKHR m_tlas;

    VkBuffer m_shaderBindingTableBuffer;
    VkDeviceMemory m_shaderBindingTableMemory;
    VkStridedDeviceAddressRegionKHR m_rchitShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR m_rgenShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR m_rmissShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR m_callableShaderBindingTable{};

    std::vector<VkCommandBuffer> m_commandBuffers;
    float m_fps;
};
