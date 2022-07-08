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
    struct PrimitiveInfo
    {
        int32_t vertexCountOffset{0};
        VkDeviceSize indexOffset{0};
        uint32_t indexCount;
        uint32_t firstIndex;
        int material;
    };

    bool update(uint32_t imageIndex);

    void getFunctionPointers();
    void loadModel();
    void releaseModel();
    void setupCamera();
    void updateCamera(double deltaTime);
    void createMsaaColorImage();
    void createDepthImage();
    void createSwapchainImageViews();
    void createSampler();
    void createTextures();
    void createMipmaps(VkImage image, uint32_t mipLevels, glm::uvec2 imageSize);
    void createCommonDescriptorSetLayout();
    void createMaterialDescriptorSetLayout();
    void createTexturesDescriptorSetLayout();
    void createPipeline();
    void createDescriptorPool();
    void allocateCommonDescriptorSets();
    void allocateMaterialIndexDescriptorSets();
    void allocateTextureDescriptorSets();
    void createCommonUniformBuffer();
    void updateCommonDescriptorSets();
    void updateMaterialDescriptorSet();
    void updateTexturesDescriptorSets();
    void createVertexAndIndexBuffer();
    void allocateCommandBuffers();
    void createBLAS();
    void createTLAS();
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

    std::unique_ptr<Model> m_model{nullptr};
    Camera m_camera;
    std::chrono::steady_clock::time_point m_lastRenderTime;
    std::unordered_map<int, bool> m_keysDown;
    VkImage m_msaaColorImage;
    VkDeviceMemory m_msaaColorImageMemory;
    VkImageView m_msaaColorImageView;
    VkImage m_depthImage;
    VkDeviceMemory m_depthImageMemory;
    VkImageView m_depthImageView;
    std::vector<VkImageView> m_swapchainImageViews;
    VkSampler m_sampler;
    std::vector<VkImage> m_images;
    VkDeviceMemory m_imageMemory;
    std::vector<VkImageView> m_imageViews;
    VkDescriptorSetLayout m_commonDescriptorSetLayout;
    VkDescriptorSetLayout m_materialDescriptorSetLayout;
    VkDescriptorSetLayout m_texturesDescriptorSetLayout;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_pipeline;
    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_commonDescriptorSets;
    std::vector<VkDescriptorSet> m_texturesDescriptorSets;
    VkBuffer m_commonUniformBuffer;
    VkDeviceMemory m_commonUniformBufferMemory;
    VkBuffer m_attributeBuffer;
    VkDeviceMemory m_attributeBufferMemory;
    std::vector<PrimitiveInfo> m_primitiveInfos;
    size_t m_vertexDataSize;
    size_t m_indexDataSize;

    VkBuffer m_blasBuffer;
    VkDeviceMemory m_blasMemory;
    VkAccelerationStructureKHR m_blas;
    VkDeviceAddress m_blasDeviceAddress;
    VkBuffer m_blasScratchBuffer;
    VkDeviceMemory m_blasScratchMemory;

    VkBuffer m_blasGeometryInstanceBuffer;
    VkDeviceMemory m_blasGeometryInstanceMemory;
    VkBuffer m_tlasBuffer;
    VkDeviceMemory m_tlasMemory;
    VkAccelerationStructureKHR m_tlas;
    VkBuffer m_tlasScratchBuffer;
    VkDeviceMemory m_tlasScratchMemory;
    VkBuffer m_shaderBindingTableBuffer;
    VkDeviceMemory m_shaderBindingTableMemory;
    VkStridedDeviceAddressRegionKHR m_rchitShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR m_rgenShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR m_rmissShaderBindingTable{};
    VkStridedDeviceAddressRegionKHR m_callableShaderBindingTable{};

    std::vector<VkCommandBuffer> m_commandBuffers;
    float m_fps;
};
