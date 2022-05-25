#pragma once

#include "Context.hpp"
#include "Camera.hpp"
#include "Model.hpp"
#include "GUI.hpp"
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
    void createRenderPass();
    void createMsaaColorImage();
    void createDepthImage();
    void createSwapchainImageViews();
    void createFramebuffers();
    void createSampler();
    void createTextures();
    void createMipmaps(VkImage image, uint32_t mipLevels, glm::uvec2 imageSize);
    void createCommonDescriptorSetLayout();
    void createMaterialDescriptorSetLayout();
    void createTexturesDescriptorSetLayout();
    void createPipeline();
    void createDescriptorPool();
    void createCommonDescriptorSets();
    void createMaterialDescriptorSet();
    void createTextureDescriptorSet();
    void createUniformBuffer();
    void updateCommonDescriptorSets();
    void updateMaterialDescriptorSet();
    void updateTexturesDescriptorSets();
    void createVertexAndIndexBuffer();
    void createBLAS();
    void allocateCommandBuffers();
    void initializeGUI();

    Context& m_context;
    VkDevice m_device;

    PFN_vkCreateRayTracingPipelinesKHR m_pvkCreateRayTracingPipelinesKHR;
    PFN_vkGetBufferDeviceAddressKHR m_pvkGetBufferDeviceAddressKHR;
    std::unique_ptr<Model> m_model{nullptr};
    Camera m_camera;
    std::chrono::steady_clock::time_point m_lastRenderTime;
    std::unordered_map<int, bool> m_keysDown;
    VkRenderPass m_renderPass;
    VkImage m_msaaColorImage;
    VkDeviceMemory m_msaaColorImageMemory;
    VkImageView m_msaaColorImageView;
    VkImage m_depthImage;
    VkDeviceMemory m_depthImageMemory;
    VkImageView m_depthImageView;
    std::vector<VkImageView> m_swapchainImageViews;
    std::vector<VkFramebuffer> m_framebuffers;
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
    VkDescriptorSet m_materialDescriptorSet;
    std::vector<VkDescriptorSet> m_texturesDescriptorSets;
    VkBuffer m_commonUniformBuffer;
    VkDeviceMemory m_commonUniformBufferMemory;
    VkBuffer m_attributeBuffer;
    VkDeviceMemory m_attributeBufferMemory;
    std::vector<PrimitiveInfo> m_primitiveInfos;
    std::vector<VkCommandBuffer> m_commandBuffers;
    std::unique_ptr<GUI> m_gui;
    float m_fps;
};
