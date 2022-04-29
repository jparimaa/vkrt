#pragma once

#include "VulkanUtils.hpp"
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <vector>

class Context final
{
public:
    struct KeyEvent
    {
        int key;
        int action;
    };

    Context();
    ~Context();

    GLFWwindow* getGlfwWindow() const;
    VkPhysicalDevice getPhysicalDevice() const;
    VkDevice getDevice() const;
    VkInstance getInstance() const;
    const std::vector<VkImage>& getSwapchainImages() const;
    VkQueue getGraphicsQueue() const;
    VkCommandPool getGraphicsCommandPool() const;
    VkSurfaceKHR getSurface() const;

    bool update();
    std::vector<KeyEvent> getKeyEvents();
    glm::dvec2 getCursorPosition();
    uint32_t acquireNextSwapchainImage();
    void submitCommandBuffers(const std::vector<VkCommandBuffer>& commandBuffers);

private:
    void initGLFW();
    void createInstance();
    void createWindow();
    void handleKey(GLFWwindow* /*window*/, int key, int /*scancode*/, int action, int /*mods*/);
    void enumeratePhysicalDevice();
    void createDevice();
    void createSwapchain();
    void createCommandPools();
    void createSemaphores();
    void createFences();

    VkInstance m_instance;
    VkDebugUtilsMessengerEXT m_debugMessenger;
    GLFWwindow* m_window;
    bool m_shouldQuit = false;
    std::vector<KeyEvent> m_keyEvents;
    glm::dvec2 m_cursorPosition;
    VkSurfaceKHR m_surface;
    VkPhysicalDevice m_physicalDevice;
    VkPhysicalDeviceProperties m_physicalDeviceProperties;
    VkDevice m_device;
    VkQueue m_graphicsQueue;
    VkQueue m_computeQueue;
    VkQueue m_presentQueue;
    VkSwapchainKHR m_swapchain;
    std::vector<VkImage> m_swapchainImages;
    VkCommandPool m_graphicsCommandPool;
    VkCommandPool m_computeCommandPool;
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;
    std::vector<VkFence> m_inFlightFences;
    uint32_t m_frameIndex{0};
    uint32_t m_imageIndex;
};
