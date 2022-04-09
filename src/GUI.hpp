#pragma once

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

class GUI final
{
public:
    struct InitData
    {
        VkCommandPool graphicsCommandPool;
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkInstance instance;
        int graphicsFamily;
        VkQueue graphicsQueue;
        VkFormat colorFormat;
        VkFormat depthFormat;
        GLFWwindow* glfwWindow;
        uint32_t imageCount;
        VkDescriptorPool descriptorPool;
    };

    GUI(const InitData& initData);
    ~GUI();

    void beginFrame();
    void endFrame(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer);

private:
    void createRenderPass(VkFormat colorFormat, VkFormat depthFormat);

    VkDevice m_device;
    VkRenderPass m_renderPass;
    VkExtent2D m_extent;
};
