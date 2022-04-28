#include "Context.hpp"
#include "Utils.hpp"
#include "DebugMarker.hpp"

#include <set>
#include <algorithm>

namespace
{
const uint64_t c_timeout = 10'000'000'000;
const VkPresentModeKHR c_presentMode = VK_PRESENT_MODE_MAILBOX_KHR;

VKAPI_ATTR VkBool32 VKAPI_CALL debugUtilsCallback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
                                                  VkDebugUtilsMessageTypeFlagsEXT message_type,
                                                  const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
                                                  void* user_data)
{
    if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        printf("Vulkan warning ");
    }
    else if (message_severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        printf("Vulkan error ");
    }
    else
    {
        return VK_FALSE;
    }

    printf("(%d)\n%s\n%s\n\n", callback_data->messageIdNumber, callback_data->pMessageIdName, callback_data->pMessage);
    return VK_FALSE;
}

void glfwErrorCallback(int error, const char* description)
{
    printf("GLFW error %d: %s\n", error, description);
}
} // namespace

Context::Context()
{
    initGLFW();
    createInstance();
    createWindow();
    enumeratePhysicalDevice();
    createDevice();
    createSwapchain();
    createCommandPools();
    createSemaphores();
    createFences();
}

Context::~Context()
{
    vkDeviceWaitIdle(m_device);

    for (VkFence fence : m_inFlightFences)
    {
        vkDestroyFence(m_device, fence, nullptr);
    }

    vkDestroySemaphore(m_device, m_renderFinished, nullptr);
    vkDestroySemaphore(m_device, m_imageAvailable, nullptr);
    vkDestroyCommandPool(m_device, m_computeCommandPool, nullptr);
    vkDestroyCommandPool(m_device, m_graphicsCommandPool, nullptr);

    vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);

    vkDestroyDevice(m_device, nullptr);

    vkDestroySurfaceKHR(m_instance, m_surface, nullptr);
    glfwDestroyWindow(m_window);
    glfwTerminate();

    auto vkDestroyDebugUtilsMessengerEXT = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
    CHECK(vkDestroyDebugUtilsMessengerEXT);
    vkDestroyDebugUtilsMessengerEXT(m_instance, m_debugMessenger, nullptr);
    vkDestroyInstance(m_instance, nullptr);
}

GLFWwindow* Context::getGlfwWindow() const
{
    return m_window;
}

VkPhysicalDevice Context::getPhysicalDevice() const
{
    return m_physicalDevice;
}

VkDevice Context::getDevice() const
{
    return m_device;
}

VkInstance Context::getInstance() const
{
    return m_instance;
}

const std::vector<VkImage>& Context::getSwapchainImages() const
{
    return m_swapchainImages;
}

VkQueue Context::getGraphicsQueue() const
{
    return m_graphicsQueue;
}

VkCommandPool Context::getGraphicsCommandPool() const
{
    return m_graphicsCommandPool;
}

VkSurfaceKHR Context::getSurface() const
{
    return m_surface;
}

bool Context::update()
{
    glfwPollEvents();
    glfwGetCursorPos(m_window, &m_cursorPosition.x, &m_cursorPosition.y);
    return !(glfwWindowShouldClose(m_window) || m_shouldQuit);
}

std::vector<Context::KeyEvent> Context::getKeyEvents()
{
    std::vector<KeyEvent> events = m_keyEvents;
    m_keyEvents.clear();
    return events;
}

glm::dvec2 Context::getCursorPosition()
{
    return m_cursorPosition;
}

uint32_t Context::acquireNextSwapchainImage()
{
    VK_CHECK(vkAcquireNextImageKHR(m_device, m_swapchain, c_timeout, m_imageAvailable, VK_NULL_HANDLE, &m_imageIndex));
    VK_CHECK(vkWaitForFences(m_device, 1, &m_inFlightFences[m_imageIndex], true, c_timeout));
    VK_CHECK(vkResetFences(m_device, 1, &m_inFlightFences[m_imageIndex]));
    return m_imageIndex;
}

void Context::submitCommandBuffers(const std::vector<VkCommandBuffer>& commandBuffers)
{
    VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &m_imageAvailable;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = ui32Size(commandBuffers);
    submitInfo.pCommandBuffers = commandBuffers.data();
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &m_renderFinished;

    VK_CHECK(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_inFlightFences[m_imageIndex]));

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &m_renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &m_imageIndex;
    presentInfo.pResults = nullptr;

    VK_CHECK(vkQueuePresentKHR(m_presentQueue, &presentInfo));
}

void Context::initGLFW()
{
    glfwSetErrorCallback(glfwErrorCallback);
    const int glfwInitialized = glfwInit();
    CHECK(glfwInitialized == GLFW_TRUE);

    const int vulkanSupported = glfwVulkanSupported();
    CHECK(vulkanSupported == GLFW_TRUE);
}

void Context::createInstance()
{
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "MyApp";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkDebugUtilsMessengerCreateInfoEXT debugUtilsCreateInfo{};
    debugUtilsCreateInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugUtilsCreateInfo.messageSeverity = //
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | //
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT;
    debugUtilsCreateInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
    debugUtilsCreateInfo.pfnUserCallback = debugUtilsCallback;

    const std::vector<const char*> extensions = getRequiredInstanceExtensions();

    VkInstanceCreateInfo instanceCreateInfo{};
    instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceCreateInfo.pApplicationInfo = &appInfo;
    instanceCreateInfo.enabledExtensionCount = ui32Size(extensions);
    instanceCreateInfo.ppEnabledExtensionNames = extensions.data();
    instanceCreateInfo.enabledLayerCount = ui32Size(c_validationLayers);
    instanceCreateInfo.ppEnabledLayerNames = c_validationLayers.data();
    instanceCreateInfo.pNext = &debugUtilsCreateInfo;

    VK_CHECK(vkCreateInstance(&instanceCreateInfo, nullptr, &m_instance));

    auto vkCreateDebugUtilsMessengerEXT = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    CHECK(vkCreateDebugUtilsMessengerEXT);
    VK_CHECK(vkCreateDebugUtilsMessengerEXT(m_instance, &debugUtilsCreateInfo, nullptr, &m_debugMessenger));
}

void Context::createWindow()
{
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    m_window = glfwCreateWindow(c_windowWidth, c_windowHeight, "Vulkan", nullptr, nullptr);
    CHECK(m_window);
    glfwSetWindowPos(m_window, 1200, 200);

    auto keyCallback = [](GLFWwindow* window, int key, int scancode, int action, int mods) {
        static_cast<Context*>(glfwGetWindowUserPointer(window))->handleKey(window, key, scancode, action, mods);
    };

    //glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetKeyCallback(m_window, keyCallback);

    VK_CHECK(glfwCreateWindowSurface(m_instance, m_window, nullptr, &m_surface));
}

void Context::handleKey(GLFWwindow* /*window*/, int key, int /*scancode*/, int action, int /*mods*/)
{
    if (action == GLFW_RELEASE && key == GLFW_KEY_ESCAPE)
    {
        m_shouldQuit = true;
    }
    m_keyEvents.push_back({key, action});
}

void Context::enumeratePhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, nullptr);
    CHECK(deviceCount);

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(m_instance, &deviceCount, devices.data());

    for (VkPhysicalDevice device : devices)
    {
        if (isDeviceSuitable(device, m_surface))
        {
            m_physicalDevice = device;
            break;
        }
    }
    CHECK(m_physicalDevice != VK_NULL_HANDLE);

    //printDeviceExtensions(m_physicalDevice);
    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_physicalDeviceProperties);
    printPhysicalDeviceName(m_physicalDeviceProperties);
}

void Context::createDevice()
{
    const QueueFamilyIndices indices = getQueueFamilies(m_physicalDevice, m_surface);

    const std::set<int> uniqueQueueFamilies = //
        {
            indices.graphicsFamily,
            indices.computeFamily,
            indices.presentFamily //
        };

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    const float queuePriority = 1.0f;
    for (int queueFamily : uniqueQueueFamilies)
    {
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = ui32Size(queueCreateInfos);
    createInfo.pQueueCreateInfos = queueCreateInfos.data();
    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = ui32Size(c_deviceExtensions);
    createInfo.ppEnabledExtensionNames = c_deviceExtensions.data();
    createInfo.enabledLayerCount = ui32Size(c_validationLayers);
    createInfo.ppEnabledLayerNames = c_validationLayers.data();

    VK_CHECK(vkCreateDevice(m_physicalDevice, &createInfo, nullptr, &m_device));

    vkGetDeviceQueue(m_device, indices.graphicsFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, indices.computeFamily, 0, &m_computeQueue);
    vkGetDeviceQueue(m_device, indices.presentFamily, 0, &m_presentQueue);

    DebugMarker::initialize(m_instance, m_device);
}

void Context::createSwapchain()
{
    const SwapchainCapabilities capabilities = getSwapchainCapabilities(m_physicalDevice, m_surface);

    bool formatAvailable = true;
    for (const VkSurfaceFormatKHR& format : capabilities.formats)
    {
        formatAvailable = formatAvailable || (c_surfaceFormat.format == format.format && format.colorSpace == c_surfaceFormat.colorSpace);
    }
    CHECK(formatAvailable);

    const auto foundPresentMode = std::find(std::begin(capabilities.presentModes), std::end(capabilities.presentModes), c_presentMode);
    CHECK(foundPresentMode != std::end(capabilities.presentModes));

    const VkExtent2D extent{c_windowWidth, c_windowHeight};
    CHECK(extent.width <= capabilities.surfaceCapabilities.maxImageExtent.width);
    CHECK(extent.width >= capabilities.surfaceCapabilities.minImageExtent.width);
    CHECK(extent.height <= capabilities.surfaceCapabilities.maxImageExtent.height);
    CHECK(extent.height >= capabilities.surfaceCapabilities.minImageExtent.height);

    CHECK(c_swapchainImageCount > capabilities.surfaceCapabilities.minImageCount);
    CHECK(c_swapchainImageCount < capabilities.surfaceCapabilities.maxImageCount);

    const QueueFamilyIndices indices = getQueueFamilies(m_physicalDevice, m_surface);
    uint32_t queueFamilyIndices[] = {(uint32_t)indices.graphicsFamily, (uint32_t)indices.presentFamily};
    CHECK(indices.graphicsFamily == indices.presentFamily);

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = m_surface;
    createInfo.minImageCount = c_swapchainImageCount;
    createInfo.imageFormat = c_surfaceFormat.format;
    createInfo.imageColorSpace = c_surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.queueFamilyIndexCount = 0;
    createInfo.pQueueFamilyIndices = nullptr;
    createInfo.preTransform = capabilities.surfaceCapabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = c_presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain));

    uint32_t queriedImageCount;
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &queriedImageCount, nullptr);
    CHECK(queriedImageCount == c_swapchainImageCount);
    m_swapchainImages.resize(c_swapchainImageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &queriedImageCount, m_swapchainImages.data());
}

void Context::createCommandPools()
{
    const QueueFamilyIndices indices = getQueueFamilies(m_physicalDevice, m_surface);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = indices.graphicsFamily;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_graphicsCommandPool));

    poolInfo.queueFamilyIndex = indices.computeFamily;
    VK_CHECK(vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_computeCommandPool));
}

void Context::createSemaphores()
{
    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VK_CHECK(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_imageAvailable));
    VK_CHECK(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_renderFinished));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_SEMAPHORE, m_imageAvailable, "Semaphore - Image available");
    DebugMarker::setObjectName(VK_OBJECT_TYPE_SEMAPHORE, m_renderFinished, "Semaphore - Render finished");
}

void Context::createFences()
{
    m_inFlightFences.resize(m_swapchainImages.size());

    VkFenceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    createInfo.pNext = nullptr;
    createInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (VkFence& fence : m_inFlightFences)
    {
        vkCreateFence(m_device, &createInfo, nullptr, &fence);
    }
}
