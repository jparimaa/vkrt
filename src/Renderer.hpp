#pragma once

#include "Context.hpp"
#include "Camera.hpp"
#include "Model.hpp"
#include "GUI.hpp"
#include <vector>
#include <chrono>
#include <unordered_map>
#include <memory>

class Renderer final
{
public:
    Renderer(Context& context);
    ~Renderer();

    bool render();

private:
    struct PrimitiveInfo
    {
        int32_t vertexCountOffset{0};
        VkDeviceSize indexOffset{0};
        uint32_t indexCount;
    };

    bool update(uint32_t imageIndex);

    void loadModel();
    void releaseModel();
    void setupCamera();
    void updateCamera(double deltaTime);
    void createRenderPass();
    void createDepthImage();
    void createSwapchainImageViews();
    void createFramebuffers();
    void createSampler();
    void createTextures();
    void createUboDescriptorSetLayouts();
    void createTexturesDescriptorSetLayouts();
    void createGraphicsPipeline();
    void createDescriptorPool();
    void createUboDescriptorSets();
    void createTextureDescriptorSet();
    void createUniformBuffer();
    void updateUboDescriptorSets();
    void updateTexturesDescriptorSets();
    void createVertexAndIndexBuffer();
    void allocateCommandBuffers();
    void initializeGUI();

    Context& m_context;
    VkDevice m_device;

    std::unique_ptr<Model> m_model{nullptr};
    Camera m_camera;
    std::chrono::steady_clock::time_point m_lastRenderTime;
    std::unordered_map<int, bool> m_keysDown;
    VkRenderPass m_renderPass;
    VkImage m_depthImage;
    VkDeviceMemory m_depthImageMemory;
    std::vector<VkImageView> m_swapchainImageViews;
    VkImageView m_depthImageView;
    std::vector<VkFramebuffer> m_framebuffers;
    VkSampler m_sampler;
    std::vector<VkImage> m_images;
    std::vector<VkDeviceMemory> m_imageMemories;
    std::vector<VkImageView> m_imageViews;
    VkDescriptorSetLayout m_uboDescriptorSetLayout;
    VkDescriptorSetLayout m_texturesDescriptorSetLayout;
    VkPipelineLayout m_pipelineLayout;
    VkPipeline m_graphicsPipeline;
    VkDescriptorPool m_descriptorPool;
    std::vector<VkDescriptorSet> m_uboDescriptorSets;
    std::vector<VkDescriptorSet> m_texturesDescriptorSets;
    VkBuffer m_uniformBuffer;
    VkDeviceMemory m_uniformBufferMemory;
    VkBuffer m_attributeBuffer;
    VkDeviceMemory m_attributeBufferMemory;
    std::vector<PrimitiveInfo> m_primitiveInfos;
    std::vector<VkCommandBuffer> m_commandBuffers;
    std::unique_ptr<GUI> m_gui;
};
