#include "GUI.hpp"
#include "Utils.hpp"
#include "VulkanUtils.hpp"
#include <cstdio>

namespace
{
void imguiCallback(VkResult r)
{
    if (r == 0)
    {
        return;
    }

    printf("ImGUI error: VkResult = %d\n", r);
}
} // namespace

GUI::GUI(const InitData& initData)
{
    m_device = initData.device;

    int width;
    int height;
    glfwGetFramebufferSize(initData.glfwWindow, &width, &height);
    m_extent = {static_cast<unsigned int>(width), static_cast<unsigned int>(height)};

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    CHECK(ImGui_ImplGlfw_InitForVulkan(initData.glfwWindow, true));

    ImGui_ImplVulkan_InitInfo imguiInitInfo{};
    imguiInitInfo.Instance = initData.instance;
    imguiInitInfo.PhysicalDevice = initData.physicalDevice;
    imguiInitInfo.Device = initData.device;
    imguiInitInfo.QueueFamily = initData.graphicsFamily;
    imguiInitInfo.Queue = initData.graphicsQueue;
    imguiInitInfo.PipelineCache = VK_NULL_HANDLE;
    imguiInitInfo.DescriptorPool = initData.descriptorPool;
    imguiInitInfo.Subpass = 0;
    imguiInitInfo.MinImageCount = initData.imageCount;
    imguiInitInfo.ImageCount = initData.imageCount;
    imguiInitInfo.MSAASamples = initData.sampleCount;
    imguiInitInfo.Allocator = nullptr;
    imguiInitInfo.CheckVkResultFn = imguiCallback;

    CHECK(ImGui_ImplVulkan_Init(&imguiInitInfo, initData.renderPass));

    const SingleTimeCommand command = beginSingleTimeCommands(initData.graphicsCommandPool, m_device);
    ImGui_ImplVulkan_CreateFontsTexture(command.commandBuffer);
    endSingleTimeCommands(initData.graphicsQueue, command);
    ImGui_ImplVulkan_DestroyFontUploadObjects();
}

GUI::~GUI()
{
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void GUI::beginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void GUI::endFrame(VkCommandBuffer commandBuffer, VkFramebuffer framebuffer)
{
    ImGui::Render();

    ImDrawData* drawData = ImGui::GetDrawData();
    ImGui_ImplVulkan_RenderDrawData(drawData, commandBuffer);
}
