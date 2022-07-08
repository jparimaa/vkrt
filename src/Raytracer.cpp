#include "Raytracer.hpp"
#include "VulkanUtils.hpp"
#include "Utils.hpp"
#include "DebugMarker.hpp"
#include <imgui.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/transform.hpp>
#include <array>
#include <cmath>

namespace
{
struct UniformBufferInfo
{
    glm::mat4 wvp;
    glm::mat4 camera;
};

const size_t c_uniformBufferSize = sizeof(UniformBufferInfo);
const VkImageSubresourceRange c_defaultSubresourceRance{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
const VkSampleCountFlagBits c_msaaSampleCount = VK_SAMPLE_COUNT_8_BIT;

VkMemoryAllocateFlagsInfo c_memoryAllocateFlagsInfo{
    VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, //
    NULL, //
    VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, //
    0 //
};
} // namespace

Raytracer::Raytracer(Context& context) :
    m_context(context),
    m_device(context.getDevice()),
    m_lastRenderTime(std::chrono::high_resolution_clock::now())
{
    getFunctionPointers();
    loadModel();
    setupCamera();
    createRenderPass();
    createMsaaColorImage();
    createDepthImage();
    createSwapchainImageViews();
    createFramebuffers();
    createSampler();
    createTextures();
    createCommonDescriptorSetLayout();
    createMaterialDescriptorSetLayout();
    createTexturesDescriptorSetLayout();
    createPipeline();
    allocateCommonDescriptorSets();
    allocateTextureDescriptorSets();
    createCommonUniformBuffer();
    updateCommonDescriptorSets();
    updateMaterialDescriptorSet();
    updateTexturesDescriptorSets();
    createVertexAndIndexBuffer();
    allocateCommandBuffers();
    createBLAS();
    releaseModel();
    initializeGUI();
}

Raytracer::~Raytracer()
{
    vkDeviceWaitIdle(m_device);

    m_gui.reset();

    vkDestroyBuffer(m_device, m_attributeBuffer, nullptr);
    vkFreeMemory(m_device, m_attributeBufferMemory, nullptr);
    vkDestroyBuffer(m_device, m_commonUniformBuffer, nullptr);
    vkFreeMemory(m_device, m_commonUniformBufferMemory, nullptr);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_texturesDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_commonDescriptorSetLayout, nullptr);

    for (const VkImageView& imageView : m_imageViews)
    {
        vkDestroyImageView(m_device, imageView, nullptr);
    }

    for (const VkImage& image : m_images)
    {
        vkDestroyImage(m_device, image, nullptr);
    }

    vkFreeMemory(m_device, m_imageMemory, nullptr);

    vkDestroySampler(m_device, m_sampler, nullptr);

    for (const VkFramebuffer& framebuffer : m_framebuffers)
    {
        vkDestroyFramebuffer(m_device, framebuffer, nullptr);
    }

    for (const VkImageView& imageView : m_swapchainImageViews)
    {
        vkDestroyImageView(m_device, imageView, nullptr);
    }

    vkDestroyImageView(m_device, m_depthImageView, nullptr);
    vkFreeMemory(m_device, m_depthImageMemory, nullptr);
    vkDestroyImage(m_device, m_depthImage, nullptr);

    vkDestroyImageView(m_device, m_msaaColorImageView, nullptr);
    vkFreeMemory(m_device, m_msaaColorImageMemory, nullptr);
    vkDestroyImage(m_device, m_msaaColorImage, nullptr);

    vkDestroyRenderPass(m_device, m_renderPass, nullptr);
}

bool Raytracer::render()
{
    const uint32_t imageIndex = m_context.acquireNextSwapchainImage();

    if (!update(imageIndex))
    {
        return false;
    }

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    beginInfo.pInheritanceInfo = nullptr;

    VkCommandBuffer cb = m_commandBuffers[imageIndex];
    vkResetCommandBuffer(cb, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT);
    vkBeginCommandBuffer(cb, &beginInfo);

    std::array<VkClearValue, 2> clearValues{};
    clearValues[0].color = {0.0f, 0.0f, 0.2f, 1.0f};
    clearValues[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_renderPass;
    renderPassInfo.framebuffer = m_framebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = c_windowExtent;
    renderPassInfo.clearValueCount = ui32Size(clearValues);
    renderPassInfo.pClearValues = clearValues.data();

    vkCmdBeginRenderPass(cb, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    {
        DebugMarker::beginLabel(cb, "Render", DebugMarker::blue);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cb, 0, 1, &m_attributeBuffer, offsets);
        vkCmdBindIndexBuffer(cb, m_attributeBuffer, m_primitiveInfos[0].indexOffset, VK_INDEX_TYPE_UINT32);
        for (size_t i = 0; i < m_primitiveInfos.size(); ++i)
        {
            const PrimitiveInfo& primitiveInfo = m_primitiveInfos[i];
            const std::vector<VkDescriptorSet> descriptorSets{m_commonDescriptorSets[imageIndex], m_texturesDescriptorSets[primitiveInfo.material]};
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, ui32Size(descriptorSets), descriptorSets.data(), 0, nullptr);
            vkCmdDrawIndexed(cb, primitiveInfo.indexCount, 1, primitiveInfo.firstIndex, primitiveInfo.vertexCountOffset, 0);
        }

        DebugMarker::endLabel(cb);
    }

    {
        DebugMarker::beginLabel(cb, "GUI");

        m_gui->beginFrame();
        ImGui::Begin("GUI");
        ImGui::Text("FPS %f", m_fps);
        ImGui::End();
        m_gui->endFrame(cb, m_framebuffers[imageIndex]);

        DebugMarker::endLabel(cb);
    }

    vkCmdEndRenderPass(cb);

    VK_CHECK(vkEndCommandBuffer(cb));

    m_context.submitCommandBuffers({cb});

    return true;
}

bool Raytracer::update(uint32_t imageIndex)
{
    bool running = m_context.update();
    if (!running)
    {
        return false;
    }

    using namespace std::chrono;
    const double deltaTime = static_cast<double>(duration_cast<nanoseconds>(high_resolution_clock::now() - m_lastRenderTime).count()) / 1'000'000'000.0;
    m_fps = 1.0f / deltaTime;
    m_lastRenderTime = high_resolution_clock::now();

    updateCamera(deltaTime);

    void* dst;
    VK_CHECK(vkMapMemory(m_device, m_commonUniformBufferMemory, imageIndex * c_uniformBufferSize, c_uniformBufferSize, 0, &dst));
    UniformBufferInfo uniformBufferInfo{};
    const glm::mat4 scaleMatrix = glm::scale(glm::vec3(0.01f, 0.01f, 0.01f));
    uniformBufferInfo.wvp = m_camera.getProjectionMatrix() * m_camera.getViewMatrix() * scaleMatrix;
    std::memcpy(dst, &uniformBufferInfo, static_cast<size_t>(c_uniformBufferSize));
    vkUnmapMemory(m_device, m_commonUniformBufferMemory);

    return true;
}

void Raytracer::getFunctionPointers()
{
    m_pvkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(m_device, "vkCreateRayTracingPipelinesKHR");
    CHECK(m_pvkCreateRayTracingPipelinesKHR);
    m_pvkGetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(m_device, "vkGetBufferDeviceAddressKHR");
    CHECK(m_pvkGetBufferDeviceAddressKHR);
    m_pvkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureBuildSizesKHR");
    CHECK(m_pvkGetAccelerationStructureBuildSizesKHR);
    m_pvkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(m_device, "vkCreateAccelerationStructureKHR");
    CHECK(m_pvkCreateAccelerationStructureKHR);
    m_pvkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureDeviceAddressKHR");
    CHECK(m_pvkGetAccelerationStructureDeviceAddressKHR);
    m_pvkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(m_device, "vkCmdBuildAccelerationStructuresKHR");
    CHECK(m_pvkCmdBuildAccelerationStructuresKHR);
}

void Raytracer::loadModel()
{
    m_model.reset(new Model("sponza/Sponza.gltf"));
}

void Raytracer::releaseModel()
{
    m_model.reset();
}

void Raytracer::setupCamera()
{
    m_camera.setPosition(glm::vec3{-4.0f, 2.0f, -0.2f});
    m_camera.setRotation(glm::vec3{0.0f, 1.51f, 0.0f});
}

void Raytracer::updateCamera(double deltaTime)
{
    std::vector<Context::KeyEvent> keyEvents = m_context.getKeyEvents();
    for (const Context::KeyEvent& keyEvent : keyEvents)
    {
        if (keyEvent.action == GLFW_PRESS || keyEvent.action == GLFW_REPEAT)
        {
            m_keysDown[keyEvent.key] = true;
        }
        if (keyEvent.action == GLFW_RELEASE)
        {
            m_keysDown[keyEvent.key] = false;
        }
    }

    const float translationSpeed = 5.0f;
    const float rotationSpeed = 1.5f;
    const float translationAmout = translationSpeed * deltaTime;
    const float rotationAmout = rotationSpeed * deltaTime;
    if (m_keysDown[GLFW_KEY_W])
    {
        m_camera.translate(m_camera.getForward() * translationAmout);
    }
    if (m_keysDown[GLFW_KEY_S])
    {
        m_camera.translate(-m_camera.getForward() * translationAmout);
    }
    if (m_keysDown[GLFW_KEY_A])
    {
        m_camera.translate(m_camera.getLeft() * translationAmout);
    }
    if (m_keysDown[GLFW_KEY_D])
    {
        m_camera.translate(-m_camera.getLeft() * translationAmout);
    }
    if (m_keysDown[GLFW_KEY_E])
    {
        m_camera.translate(m_camera.getUp() * translationAmout);
    }
    if (m_keysDown[GLFW_KEY_Q])
    {
        m_camera.translate(-m_camera.getUp() * translationAmout);
    }
    if (m_keysDown[GLFW_KEY_Z])
    {
        m_camera.rotate(c_up, rotationAmout);
    }
    if (m_keysDown[GLFW_KEY_C])
    {
        m_camera.rotate(-c_up, rotationAmout);
    }
}

void Raytracer::createRenderPass()
{
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorAttachmentResolveRef{};
    colorAttachmentResolveRef.attachment = 2;
    colorAttachmentResolveRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = c_surfaceFormat.format;
    colorAttachment.samples = c_msaaSampleCount;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = c_depthFormat;
    depthAttachment.samples = c_msaaSampleCount;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.format = c_surfaceFormat.format;
    colorAttachmentResolve.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachmentResolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachmentResolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachmentResolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachmentResolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachmentResolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    const std::array<VkAttachmentDescription, 3> attachments = {colorAttachment, depthAttachment, colorAttachmentResolve};

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = ui32Size(attachments);
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_RENDER_PASS, m_renderPass, "Render pass - Main");
}

void Raytracer::createMsaaColorImage()
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = c_windowWidth;
    imageInfo.extent.height = c_windowHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = c_surfaceFormat.format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    imageInfo.samples = c_msaaSampleCount;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags = 0;

    VK_CHECK(vkCreateImage(m_device, &imageInfo, nullptr, &m_msaaColorImage));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_IMAGE, m_msaaColorImage, "Image - MSAA color");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, m_msaaColorImage, &memRequirements);

    const MemoryTypeResult memoryTypeResult = findMemoryType(m_context.getPhysicalDevice(), memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_msaaColorImageMemory));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_msaaColorImageMemory, "Memory - MSAA color image");
    VK_CHECK(vkBindImageMemory(m_device, m_msaaColorImage, m_msaaColorImageMemory, 0));

    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = m_msaaColorImage;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = c_surfaceFormat.format;
    createInfo.subresourceRange = c_defaultSubresourceRance;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    VK_CHECK(vkCreateImageView(m_device, &createInfo, nullptr, &m_msaaColorImageView));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, m_msaaColorImageView, "Image view - MSAA color");
}

void Raytracer::createDepthImage()
{
    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = c_windowWidth;
    imageInfo.extent.height = c_windowHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = c_depthFormat;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imageInfo.samples = c_msaaSampleCount;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags = 0;

    VK_CHECK(vkCreateImage(m_device, &imageInfo, nullptr, &m_depthImage));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_IMAGE, m_msaaColorImage, "Image - MSAA depth");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, m_depthImage, &memRequirements);

    const MemoryTypeResult memoryTypeResult = findMemoryType(m_context.getPhysicalDevice(), memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_depthImageMemory));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_msaaColorImageMemory, "Memory - MSAA depth image");
    VK_CHECK(vkBindImageMemory(m_device, m_depthImage, m_depthImageMemory, 0));

    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = m_depthImage;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = c_depthFormat;
    createInfo.subresourceRange = c_defaultSubresourceRance;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VK_CHECK(vkCreateImageView(m_device, &createInfo, nullptr, &m_depthImageView));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, m_depthImageView, "Image view - MSAA depth");
}

void Raytracer::createSwapchainImageViews()
{
    const std::vector<VkImage>& swapchainImages = m_context.getSwapchainImages();

    m_swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); ++i)
    {
        VkImageViewCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createInfo.image = swapchainImages[i];
        createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createInfo.format = c_surfaceFormat.format;
        createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createInfo.subresourceRange = c_defaultSubresourceRance;

        VK_CHECK(vkCreateImageView(m_device, &createInfo, nullptr, &m_swapchainImageViews[i]));
    }
}

void Raytracer::createFramebuffers()
{
    m_framebuffers.resize(m_swapchainImageViews.size());

    VkFramebufferCreateInfo framebufferInfo{};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = m_renderPass;
    framebufferInfo.width = c_windowWidth;
    framebufferInfo.height = c_windowHeight;
    framebufferInfo.layers = 1;

    for (size_t i = 0; i < m_swapchainImageViews.size(); ++i)
    {
        const std::array<VkImageView, 3> attachments = {m_msaaColorImageView, m_depthImageView, m_swapchainImageViews[i]};
        framebufferInfo.attachmentCount = ui32Size(attachments);
        framebufferInfo.pAttachments = attachments.data();

        VK_CHECK(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[i]));
        DebugMarker::setObjectName(VK_OBJECT_TYPE_FRAMEBUFFER, m_framebuffers[i], "Framebuffer " + std::to_string(i));
    }
}

void Raytracer::createSampler()
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;

    VK_CHECK(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_sampler));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_SAMPLER, m_sampler, "Sampler - Main");
}

void Raytracer::createTextures()
{
    const std::vector<Model::Image>& images = m_model->images;
    const size_t imageCount = images.size();
    m_images.resize(imageCount);
    m_imageViews.resize(imageCount);
    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

    for (size_t i = 0; i < imageCount; ++i)
    {
        const Model::Image& image = images[i];
        const glm::uvec2 imageResolution{images[i].width, images[i].height};
        const unsigned int mipLevelCount = static_cast<uint32_t>(std::floor(std::log2(std::max(imageResolution.x, imageResolution.y))) + 1);

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = image.width;
        imageInfo.extent.height = image.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = mipLevelCount;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.flags = 0;

        VK_CHECK(vkCreateImage(m_device, &imageInfo, nullptr, &m_images[i]));
        DebugMarker::setObjectName(VK_OBJECT_TYPE_IMAGE, m_images[i], "Image - Sponza " + std::to_string(i));
    }

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, m_images[0], &memRequirements);

    const VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const MemoryTypeResult memoryTypeResult = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, memoryProperties);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size * imageCount;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;
    const VkDeviceSize singleImageSize = memRequirements.size;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_imageMemory));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_msaaColorImageMemory, "Memory - Texture images");

    for (size_t i = 0; i < imageCount; ++i)
    {
        vkBindImageMemory(m_device, m_images[i], m_imageMemory, i * singleImageSize);

        const Model::Image& image = images[i];
        const glm::uvec2 imageResolution{images[i].width, images[i].height};
        const unsigned int mipLevelCount = static_cast<uint32_t>(std::floor(std::log2(std::max(imageResolution.x, imageResolution.y))) + 1);

        const StagingBuffer stagingBuffer = createStagingBuffer(m_device, physicalDevice, image.data.data(), image.data.size());

        VkImageMemoryBarrier transferDstBarrier{};
        transferDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        transferDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        transferDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        transferDstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        transferDstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        transferDstBarrier.image = m_images[i];
        transferDstBarrier.subresourceRange = c_defaultSubresourceRance;
        transferDstBarrier.subresourceRange.levelCount = mipLevelCount;
        transferDstBarrier.srcAccessMask = 0;
        transferDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        const VkPipelineStageFlags transferSrcFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        const VkPipelineStageFlags transferDstFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {image.width, image.height, 1};

        const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
        const VkCommandBuffer& cb = command.commandBuffer;

        vkCmdPipelineBarrier(cb, transferSrcFlags, transferDstFlags, 0, 0, nullptr, 0, nullptr, 1, &transferDstBarrier);
        vkCmdCopyBufferToImage(cb, stagingBuffer.buffer, m_images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        endSingleTimeCommands(m_context.getGraphicsQueue(), command);

        releaseStagingBuffer(m_device, stagingBuffer);

        createMipmaps(m_images[i], mipLevelCount, imageResolution);

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = c_defaultSubresourceRance;
        viewInfo.subresourceRange.levelCount = mipLevelCount;

        VK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_imageViews[i]));
        DebugMarker::setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, m_imageViews[i], "Image view - Sponza " + std::to_string(i));
    }
}

void Raytracer::createMipmaps(VkImage image, uint32_t mipLevels, glm::uvec2 imageSize)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    const VkPipelineStageFlagBits transferStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
    const VkImageLayout transferDstLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    const VkImageLayout transferSrcLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    int32_t mipWidth = imageSize.x;
    int32_t mipHeight = imageSize.y;

    const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
    const VkCommandBuffer& cb = command.commandBuffer;

    for (uint32_t i = 1; i < mipLevels; ++i)
    {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = transferDstLayout;
        barrier.newLayout = transferSrcLayout;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(cb, transferStageMask, transferStageMask, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mipWidth > 1 ? mipWidth / 2 : 1, mipHeight > 1 ? mipHeight / 2 : 1, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(cb, image, transferSrcLayout, image, transferDstLayout, 1, &blit, VK_FILTER_LINEAR);

        barrier.oldLayout = transferSrcLayout;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(cb, transferStageMask, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

        mipWidth = mipWidth > 1 ? mipWidth / 2 : mipWidth;
        mipHeight = mipHeight > 1 ? mipHeight / 2 : mipHeight;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = transferDstLayout;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(cb, transferStageMask, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(m_context.getGraphicsQueue(), command);
}

void Raytracer::createCommonDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(5);
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[0].pImmutableSamplers = nullptr;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[1].pImmutableSamplers = nullptr;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[2].pImmutableSamplers = nullptr;
    bindings[3].binding = 3;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[3].descriptorCount = 1;
    bindings[3].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[3].pImmutableSamplers = nullptr;
    bindings[4].binding = 4;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[4].descriptorCount = 1;
    bindings[4].stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    bindings[4].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = ui32Size(bindings);
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_commonDescriptorSetLayout));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, m_commonDescriptorSetLayout, "Desc set layout - Common");
}

void Raytracer::createMaterialDescriptorSetLayout()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(2);
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[0].pImmutableSamplers = nullptr;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[1].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = ui32Size(bindings);
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_materialDescriptorSetLayout));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, m_materialDescriptorSetLayout, "Desc set layout - Material");
}

void Raytracer::createTexturesDescriptorSetLayout()
{
    const uint32_t imageCount = 3;
    std::vector<VkDescriptorSetLayoutBinding> bindings(imageCount);

    for (uint32_t i = 0; i < imageCount; ++i)
    {
        bindings[i].binding = i;
        bindings[i].descriptorCount = 1;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[i].pImmutableSamplers = nullptr;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = ui32Size(bindings);
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_texturesDescriptorSetLayout));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, m_texturesDescriptorSetLayout, "Desc set layout - Texture");
}

void Raytracer::createPipeline()
{
    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts{m_commonDescriptorSetLayout, m_materialDescriptorSetLayout, m_texturesDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = ui32Size(descriptorSetLayouts);
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    VK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, m_pipelineLayout, "Pipeline layout - Raytracer");

    const std::filesystem::path currentPath = getCurrentExecutableDirectory();
    VkShaderModule closesHitShaderModule = createShaderModule(m_device, currentPath / "shader.rchit.spv");
    VkShaderModule rayGenShaderModule = createShaderModule(m_device, currentPath / "shader.rgen.spv");
    VkShaderModule missShaderModule = createShaderModule(m_device, currentPath / "shader.rmiss.spv");
    VkShaderModule shadowMissShaderModule = createShaderModule(m_device, currentPath / "shader_shadow.rmiss.spv");

    std::vector<VkPipelineShaderStageCreateInfo> pipelineShaderStageCreateInfoList(4);

    pipelineShaderStageCreateInfoList[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfoList[0].pNext = NULL;
    pipelineShaderStageCreateInfoList[0].flags = 0;
    pipelineShaderStageCreateInfoList[0].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    pipelineShaderStageCreateInfoList[0].module = closesHitShaderModule;
    pipelineShaderStageCreateInfoList[0].pName = "main";
    pipelineShaderStageCreateInfoList[0].pSpecializationInfo = NULL;
    pipelineShaderStageCreateInfoList[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfoList[1].pNext = NULL;
    pipelineShaderStageCreateInfoList[1].flags = 0;
    pipelineShaderStageCreateInfoList[1].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    pipelineShaderStageCreateInfoList[1].module = rayGenShaderModule;
    pipelineShaderStageCreateInfoList[1].pName = "main";
    pipelineShaderStageCreateInfoList[1].pSpecializationInfo = NULL;
    pipelineShaderStageCreateInfoList[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfoList[2].pNext = NULL;
    pipelineShaderStageCreateInfoList[2].flags = 0;
    pipelineShaderStageCreateInfoList[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    pipelineShaderStageCreateInfoList[2].module = missShaderModule;
    pipelineShaderStageCreateInfoList[2].pName = "main";
    pipelineShaderStageCreateInfoList[2].pSpecializationInfo = NULL;
    pipelineShaderStageCreateInfoList[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineShaderStageCreateInfoList[3].pNext = NULL;
    pipelineShaderStageCreateInfoList[3].flags = 0;
    pipelineShaderStageCreateInfoList[3].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    pipelineShaderStageCreateInfoList[3].module = shadowMissShaderModule;
    pipelineShaderStageCreateInfoList[3].pName = "main";
    pipelineShaderStageCreateInfoList[3].pSpecializationInfo = NULL;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> rayTracingShaderGroupCreateInfoList(4);

    rayTracingShaderGroupCreateInfoList[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    rayTracingShaderGroupCreateInfoList[0].pNext = NULL;
    rayTracingShaderGroupCreateInfoList[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    rayTracingShaderGroupCreateInfoList[0].generalShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[0].closestHitShader = 0;
    rayTracingShaderGroupCreateInfoList[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[0].intersectionShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[0].pShaderGroupCaptureReplayHandle = NULL;
    rayTracingShaderGroupCreateInfoList[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    rayTracingShaderGroupCreateInfoList[1].pNext = NULL;
    rayTracingShaderGroupCreateInfoList[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    rayTracingShaderGroupCreateInfoList[1].generalShader = 1;
    rayTracingShaderGroupCreateInfoList[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[1].intersectionShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[1].pShaderGroupCaptureReplayHandle = NULL;
    rayTracingShaderGroupCreateInfoList[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    rayTracingShaderGroupCreateInfoList[2].pNext = NULL;
    rayTracingShaderGroupCreateInfoList[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    rayTracingShaderGroupCreateInfoList[2].generalShader = 2;
    rayTracingShaderGroupCreateInfoList[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[2].intersectionShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[2].pShaderGroupCaptureReplayHandle = NULL;
    rayTracingShaderGroupCreateInfoList[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    rayTracingShaderGroupCreateInfoList[3].pNext = NULL;
    rayTracingShaderGroupCreateInfoList[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    rayTracingShaderGroupCreateInfoList[3].generalShader = 3;
    rayTracingShaderGroupCreateInfoList[3].closestHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[3].anyHitShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[3].intersectionShader = VK_SHADER_UNUSED_KHR;
    rayTracingShaderGroupCreateInfoList[3].pShaderGroupCaptureReplayHandle = NULL;

    VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfo{};
    rayTracingPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rayTracingPipelineCreateInfo.pNext = NULL;
    rayTracingPipelineCreateInfo.flags = 0;
    rayTracingPipelineCreateInfo.stageCount = 4;
    rayTracingPipelineCreateInfo.pStages = pipelineShaderStageCreateInfoList.data();
    rayTracingPipelineCreateInfo.groupCount = 4;
    rayTracingPipelineCreateInfo.pGroups = rayTracingShaderGroupCreateInfoList.data();
    rayTracingPipelineCreateInfo.maxPipelineRayRecursionDepth = 1;
    rayTracingPipelineCreateInfo.pLibraryInfo = NULL;
    rayTracingPipelineCreateInfo.pLibraryInterface = NULL;
    rayTracingPipelineCreateInfo.pDynamicState = NULL;
    rayTracingPipelineCreateInfo.layout = m_pipelineLayout;
    rayTracingPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
    rayTracingPipelineCreateInfo.basePipelineIndex = 0;

    VK_CHECK(m_pvkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &rayTracingPipelineCreateInfo, NULL, &m_pipeline));

    vkDestroyShaderModule(m_device, closesHitShaderModule, nullptr);
    vkDestroyShaderModule(m_device, rayGenShaderModule, nullptr);
    vkDestroyShaderModule(m_device, missShaderModule, nullptr);
    vkDestroyShaderModule(m_device, shadowMissShaderModule, nullptr);
}

void Raytracer::createDescriptorPool()
{
    const uint32_t swapchainLength = static_cast<uint32_t>(m_context.getSwapchainImages().size());
    const uint32_t numSetsForGUI = 1;
    const uint32_t numSetsForModel = ui32Size(m_model->materials);

    const uint32_t imageDescriptorCount = numSetsForModel + numSetsForGUI;

    std::array<VkDescriptorPoolSize, 4> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = swapchainLength;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = imageDescriptorCount;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[2].descriptorCount = 1;
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[3].descriptorCount = 1;

    const uint32_t maxSets = swapchainLength + numSetsForModel + numSetsForGUI;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = ui32Size(poolSizes);
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSets;

    VK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL, m_descriptorPool, "Descriptor pool - Raytracer");
}

void Raytracer::allocateCommonDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> descriptorSetLayouts{
        m_commonDescriptorSetLayout, //
        m_materialDescriptorSetLayout //
    };

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = NULL;
    descriptorSetAllocateInfo.descriptorPool = m_descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = ui32Size(descriptorSetLayouts);
    descriptorSetAllocateInfo.pSetLayouts = descriptorSetLayouts.data();

    m_commonDescriptorSets.resize(ui32Size(descriptorSetLayouts));

    VK_CHECK(vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, m_commonDescriptorSets.data()));
    for (size_t i = 0; i < m_commonDescriptorSets.size(); ++i)
    {
        DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, m_commonDescriptorSets[i], "Desc set - Common " + std::to_string(i));
    }
}

void Raytracer::allocateTextureDescriptorSets()
{
    const size_t materialCount = m_model->materials.size();
    m_texturesDescriptorSets.resize(materialCount);
    std::vector<VkDescriptorSetLayout> layouts(materialCount, m_texturesDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = ui32Size(layouts);
    allocInfo.pSetLayouts = layouts.data();
    VK_ERROR_FRAGMENTED_POOL;
    VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, m_texturesDescriptorSets.data()));
    for (size_t i = 0; i < m_texturesDescriptorSets.size(); ++i)
    {
        DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, m_texturesDescriptorSets[i], "Desc set - Texture " + std::to_string(i));
    }
}

void Raytracer::createCommonUniformBuffer()
{
    const VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const uint64_t bufferSize = c_uniformBufferSize * m_context.getSwapchainImages().size();

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_commonUniformBuffer));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_commonUniformBuffer, "Buffer - Raytracer common uniform buffer");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, m_commonUniformBuffer, &memRequirements);

    const MemoryTypeResult memoryTypeResult = findMemoryType(m_context.getPhysicalDevice(), memRequirements.memoryTypeBits, memoryProperties);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_commonUniformBufferMemory));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_commonUniformBufferMemory, "Memory - Raytracer common uniform buffer");

    VK_CHECK(vkBindBufferMemory(m_device, m_commonUniformBuffer, m_commonUniformBufferMemory, 0));
}

void Raytracer::updateCommonDescriptorSets()
{
    // Infos
    VkWriteDescriptorSetAccelerationStructureKHR accelerationStructureDescriptorInfo{};
    accelerationStructureDescriptorInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    accelerationStructureDescriptorInfo.pNext = NULL;
    accelerationStructureDescriptorInfo.accelerationStructureCount = 1;
    accelerationStructureDescriptorInfo.pAccelerationStructures = &m_tlas;

    VkDescriptorBufferInfo uniformDescriptorInfo{};
    uniformDescriptorInfo.buffer = m_commonUniformBuffer;
    uniformDescriptorInfo.offset = 0;
    uniformDescriptorInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo indexDescriptorInfo{};
    indexDescriptorInfo.buffer = m_attributeBuffer;
    indexDescriptorInfo.offset = m_primitiveInfos[0].indexOffset;
    indexDescriptorInfo.range = m_indexDataSize;

    VkDescriptorBufferInfo vertexDescriptorInfo{};
    vertexDescriptorInfo.buffer = m_attributeBuffer;
    vertexDescriptorInfo.offset = 0;
    vertexDescriptorInfo.range = m_vertexDataSize;

    VkDescriptorImageInfo imageDescriptorInfo{};
    imageDescriptorInfo.sampler = VK_NULL_HANDLE;
    imageDescriptorInfo.imageView = m_msaaColorImageView;
    imageDescriptorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Write sets
    VkWriteDescriptorSet writeAccelerationStructure{};
    writeAccelerationStructure.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeAccelerationStructure.pNext = &accelerationStructureDescriptorInfo;
    writeAccelerationStructure.dstSet = m_commonDescriptorSets[0];
    writeAccelerationStructure.dstBinding = 0;
    writeAccelerationStructure.dstArrayElement = 0;
    writeAccelerationStructure.descriptorCount = 1;
    writeAccelerationStructure.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    writeAccelerationStructure.pImageInfo = NULL;
    writeAccelerationStructure.pBufferInfo = NULL;
    writeAccelerationStructure.pTexelBufferView = NULL;

    VkWriteDescriptorSet writeUniformBuffer{};
    writeUniformBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeUniformBuffer.pNext = NULL;
    writeUniformBuffer.dstSet = m_commonDescriptorSets[0];
    writeUniformBuffer.dstBinding = 1;
    writeUniformBuffer.dstArrayElement = 0;
    writeUniformBuffer.descriptorCount = 1;
    writeUniformBuffer.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    writeUniformBuffer.pImageInfo = NULL;
    writeUniformBuffer.pBufferInfo = &uniformDescriptorInfo;
    writeUniformBuffer.pTexelBufferView = NULL;

    VkWriteDescriptorSet writeIndexBuffer{};
    writeIndexBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeIndexBuffer.pNext = NULL;
    writeIndexBuffer.dstSet = m_commonDescriptorSets[0];
    writeIndexBuffer.dstBinding = 2;
    writeIndexBuffer.dstArrayElement = 0;
    writeIndexBuffer.descriptorCount = 1;
    writeIndexBuffer.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeIndexBuffer.pImageInfo = NULL;
    writeIndexBuffer.pBufferInfo = &indexDescriptorInfo;
    writeIndexBuffer.pTexelBufferView = NULL;

    VkWriteDescriptorSet writeVertexBuffer{};
    writeVertexBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeVertexBuffer.pNext = NULL;
    writeVertexBuffer.dstSet = m_commonDescriptorSets[0];
    writeVertexBuffer.dstBinding = 3;
    writeVertexBuffer.dstArrayElement = 0;
    writeVertexBuffer.descriptorCount = 1;
    writeVertexBuffer.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeVertexBuffer.pImageInfo = NULL;
    writeVertexBuffer.pBufferInfo = &vertexDescriptorInfo;
    writeVertexBuffer.pTexelBufferView = NULL;

    VkWriteDescriptorSet writeImage{};
    writeImage.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeImage.pNext = NULL;
    writeImage.dstSet = m_commonDescriptorSets[0];
    writeImage.dstBinding = 4;
    writeImage.dstArrayElement = 0;
    writeImage.descriptorCount = 1;
    writeImage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writeImage.pImageInfo = &imageDescriptorInfo;
    writeImage.pBufferInfo = NULL;
    writeImage.pTexelBufferView = NULL;

    std::vector<VkWriteDescriptorSet> writeDescriptorSets{
        writeAccelerationStructure, //
        writeUniformBuffer, //
        writeIndexBuffer, //
        writeVertexBuffer, //
        writeImage //
    };

    vkUpdateDescriptorSets(m_device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
}

void Raytracer::updateMaterialDescriptorSet()
{
}

void Raytracer::updateTexturesDescriptorSets()
{
    const std::vector<Model::Material>& materials = m_model->materials;
    const size_t materialCount = m_model->materials.size();
    std::vector<VkWriteDescriptorSet> descriptorWrites(materialCount * 3);

    const size_t imageCount = m_model->images.size();
    std::vector<VkDescriptorImageInfo> imageInfos(imageCount);
    for (size_t i = 0; i < imageCount; ++i)
    {
        VkDescriptorImageInfo& imageInfo = imageInfos[i];
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = m_imageViews[i];
        imageInfo.sampler = m_sampler;
    }

    for (size_t i = 0; i < materialCount; ++i)
    {
        const size_t offset = i * 3;
        const std::vector<int> materialIndices{materials[i].baseColor, materials[i].metallicRoughnessImage, materials[i].normalImage};
        for (size_t j = 0; j < materialIndices.size(); ++j)
        {
            descriptorWrites[offset + j].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[offset + j].dstSet = m_texturesDescriptorSets[i];
            descriptorWrites[offset + j].dstBinding = j;
            descriptorWrites[offset + j].dstArrayElement = 0;
            descriptorWrites[offset + j].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[offset + j].descriptorCount = 1;
            const int materialIndex = materialIndices[j] != -1 ? materialIndices[j] : 0;
            descriptorWrites[offset + j].pImageInfo = &imageInfos[materialIndex];
        }
    }

    vkUpdateDescriptorSets(m_device, ui32Size(descriptorWrites), descriptorWrites.data(), 0, nullptr);
}

void Raytracer::createVertexAndIndexBuffer()
{
    m_primitiveInfos.resize(m_model->primitives.size());
    const uint64_t bufferSize = m_model->vertexBufferSizeInBytes + m_model->indexBufferSizeInBytes;
    std::vector<uint8_t> data(bufferSize, 0);
    size_t vertexOffset = 0;
    size_t indexOffset = m_model->vertexBufferSizeInBytes;
    int32_t vertexCountOffset = 0;
    uint32_t firstIndex = 0;
    for (size_t i = 0; i < m_model->primitives.size(); ++i)
    {
        const Model::Primitive& primitive = m_model->primitives[i];

        m_primitiveInfos[i].indexCount = ui32Size(primitive.indices);
        m_primitiveInfos[i].vertexCountOffset = vertexCountOffset;
        m_primitiveInfos[i].indexOffset = indexOffset;
        m_primitiveInfos[i].firstIndex = firstIndex;
        m_primitiveInfos[i].material = primitive.material;

        vertexCountOffset += static_cast<int32_t>(primitive.vertices.size());
        firstIndex += ui32Size(primitive.indices);

        const size_t vertexDataSize = sizeof(Model::Vertex) * primitive.vertices.size();
        const size_t indexDataSize = sizeof(Model::Index) * primitive.indices.size();
        std::memcpy(&data[vertexOffset], primitive.vertices.data(), vertexDataSize);
        std::memcpy(&data[indexOffset], primitive.indices.data(), indexDataSize);
        vertexOffset += vertexDataSize;
        indexOffset += indexDataSize;
    }

    m_vertexDataSize = vertexOffset;
    m_indexDataSize = indexOffset;

    VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();
    StagingBuffer stagingBuffer = createStagingBuffer(m_device, physicalDevice, data.data(), bufferSize);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = //
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | //
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | //
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT | //
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_attributeBuffer));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_attributeBuffer, "Buffer - Attribute");

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(m_device, m_attributeBuffer, &memoryRequirements);
    VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    const VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const MemoryTypeResult memoryTypeResult = findMemoryType(physicalDevice, memoryRequirements.memoryTypeBits, memoryProperties);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.pNext = &c_memoryAllocateFlagsInfo;
    allocInfo.allocationSize = memoryRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_attributeBufferMemory));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_attributeBufferMemory, "Memory - Attribute buffer");

    VK_CHECK(vkBindBufferMemory(m_device, m_attributeBuffer, m_attributeBufferMemory, 0));

    const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);

    VkBufferCopy vertexCopyRegion{};
    vertexCopyRegion.size = bufferSize;
    vertexCopyRegion.srcOffset = 0;
    vertexCopyRegion.dstOffset = 0;

    vkCmdCopyBuffer(command.commandBuffer, stagingBuffer.buffer, m_attributeBuffer, 1, &vertexCopyRegion);

    endSingleTimeCommands(m_context.getGraphicsQueue(), command);

    releaseStagingBuffer(m_device, stagingBuffer);
}

void Raytracer::createBLAS()
{
    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();

    // Setup geometry and get build size
    VkBufferDeviceAddressInfo vertexBufferDeviceAddressInfo{};
    vertexBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    vertexBufferDeviceAddressInfo.pNext = NULL;
    vertexBufferDeviceAddressInfo.buffer = m_attributeBuffer;

    VkDeviceAddress vertexBufferDeviceAddress = m_pvkGetBufferDeviceAddressKHR(m_device, &vertexBufferDeviceAddressInfo);
    VkDeviceAddress indexBufferDeviceAddress = vertexBufferDeviceAddress + m_primitiveInfos[0].indexOffset;

    uint32_t vertexCount = 0;
    uint32_t triangleCount = 0;
    for (size_t i = 0; i < m_model->primitives.size(); ++i)
    {
        vertexCount += static_cast<uint32_t>(m_model->primitives[i].vertices.size());
        CHECK(m_model->primitives[i].indices.size() % 3 == 0);
        triangleCount += static_cast<uint32_t>(m_model->primitives[i].indices.size() / 3);
    }

    VkAccelerationStructureGeometryDataKHR blasGeometryData{};
    blasGeometryData.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    blasGeometryData.triangles.pNext = NULL;
    blasGeometryData.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    blasGeometryData.triangles.vertexData = VkDeviceOrHostAddressConstKHR{vertexBufferDeviceAddress};
    blasGeometryData.triangles.vertexStride = sizeof(Model::Vertex);
    blasGeometryData.triangles.maxVertex = vertexCount;
    blasGeometryData.triangles.indexType = VK_INDEX_TYPE_UINT32;
    blasGeometryData.triangles.indexData = VkDeviceOrHostAddressConstKHR{indexBufferDeviceAddress};
    blasGeometryData.triangles.transformData = VkDeviceOrHostAddressConstKHR{0};

    VkAccelerationStructureGeometryKHR blasGeometry{};
    blasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    blasGeometry.pNext = NULL;
    blasGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    blasGeometry.geometry = blasGeometryData;
    blasGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR blasBuildGeometryInfo{};
    blasBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    blasBuildGeometryInfo.pNext = NULL;
    blasBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blasBuildGeometryInfo.flags = 0;
    blasBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    blasBuildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    blasBuildGeometryInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    blasBuildGeometryInfo.geometryCount = 1;
    blasBuildGeometryInfo.pGeometries = &blasGeometry;
    blasBuildGeometryInfo.ppGeometries = NULL;
    blasBuildGeometryInfo.scratchData = VkDeviceOrHostAddressKHR{0};

    VkAccelerationStructureBuildSizesInfoKHR blasBuildSizesInfo{};
    blasBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    blasBuildSizesInfo.pNext = NULL;
    blasBuildSizesInfo.accelerationStructureSize = 0;
    blasBuildSizesInfo.updateScratchSize = 0;
    blasBuildSizesInfo.buildScratchSize = 0;

    const std::vector<uint32_t> maxPrimitiveCounts = {triangleCount};

    const VkAccelerationStructureBuildTypeKHR buildType = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;
    m_pvkGetAccelerationStructureBuildSizesKHR(m_device, buildType, &blasBuildGeometryInfo, maxPrimitiveCounts.data(), &blasBuildSizesInfo);

    // Create BLAS buffer
    m_blasBuffer = createBuffer(m_device, blasBuildSizesInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_blasMemory = allocateAndBindMemory(m_device, physicalDevice, m_blasBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create BLAS
    VkAccelerationStructureCreateInfoKHR blasCreateInfo{};
    blasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    blasCreateInfo.pNext = NULL;
    blasCreateInfo.createFlags = 0;
    blasCreateInfo.buffer = m_blasBuffer;
    blasCreateInfo.offset = 0;
    blasCreateInfo.size = blasBuildSizesInfo.accelerationStructureSize;
    blasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blasCreateInfo.deviceAddress = 0;

    VK_CHECK(m_pvkCreateAccelerationStructureKHR(m_device, &blasCreateInfo, NULL, &m_blas));

    VkAccelerationStructureDeviceAddressInfoKHR blasDeviceAddressInfo{};
    blasDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    blasDeviceAddressInfo.pNext = NULL;
    blasDeviceAddressInfo.accelerationStructure = m_blas;

    m_blasDeviceAddress = m_pvkGetAccelerationStructureDeviceAddressKHR(m_device, &blasDeviceAddressInfo);

    // Create BLAS scratch buffer
    m_blasScratchBuffer = createBuffer(m_device, blasBuildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_blasScratchMemory = allocateAndBindMemory(m_device, physicalDevice, m_blasScratchBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkBufferDeviceAddressInfo blasScratchBufferDeviceAddressInfo{};
    blasScratchBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    blasScratchBufferDeviceAddressInfo.pNext = NULL;
    blasScratchBufferDeviceAddressInfo.buffer = m_blasScratchBuffer;

    VkDeviceAddress blasScratchBufferDeviceAddress = m_pvkGetBufferDeviceAddressKHR(m_device, &blasScratchBufferDeviceAddressInfo);

    // Build BLAS
    blasBuildGeometryInfo.dstAccelerationStructure = m_blas;
    blasBuildGeometryInfo.scratchData.deviceAddress = blasScratchBufferDeviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR blasBuildRangeInfo{};
    blasBuildRangeInfo.primitiveCount = triangleCount;
    blasBuildRangeInfo.primitiveOffset = 0;
    blasBuildRangeInfo.firstVertex = 0;
    blasBuildRangeInfo.transformOffset = 0;

    const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
    const VkCommandBuffer& cb = command.commandBuffer;

    const VkAccelerationStructureBuildRangeInfoKHR* blasBuildRangeInfos = &blasBuildRangeInfo;

    m_pvkCmdBuildAccelerationStructuresKHR(cb, 1, &blasBuildGeometryInfo, &blasBuildRangeInfos);

    endSingleTimeCommands(m_context.getGraphicsQueue(), command);
}

void Raytracer::createTLAS()
{
    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();

    // Setup BLAS instance buffer
    VkAccelerationStructureInstanceKHR blasInstance{};
    blasInstance.transform.matrix = {{1.0f, 0.0f, 0.0f, 0.0f},
                                     {0.0f, 1.0f, 0.0f, 0.0f},
                                     {0.0f, 0.0f, 1.0f, 0.0f}};
    blasInstance.instanceCustomIndex = 0;
    blasInstance.mask = 0xFF;
    blasInstance.instanceShaderBindingTableRecordOffset = 0;
    blasInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    blasInstance.accelerationStructureReference = m_blasDeviceAddress;

    m_blasGeometryInstanceBuffer = createBuffer(m_device, sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_blasGeometryInstanceMemory = allocateAndBindMemory(m_device, physicalDevice, m_blasGeometryInstanceBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    void* hostBlasGeometryInstanceMemoryMapped;
    VK_CHECK(vkMapMemory(m_device, m_blasGeometryInstanceMemory, 0, sizeof(VkAccelerationStructureInstanceKHR), 0, &hostBlasGeometryInstanceMemoryMapped));
    memcpy(hostBlasGeometryInstanceMemoryMapped, &blasInstance, sizeof(VkAccelerationStructureInstanceKHR));
    vkUnmapMemory(m_device, m_blasGeometryInstanceMemory);

    VkBufferDeviceAddressInfo blasGeometryInstanceDeviceAddressInfo{};
    blasGeometryInstanceDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    blasGeometryInstanceDeviceAddressInfo.pNext = NULL;
    blasGeometryInstanceDeviceAddressInfo.buffer = m_blasGeometryInstanceBuffer;

    VkDeviceAddress blasGeometryInstanceDeviceAddress = m_pvkGetBufferDeviceAddressKHR(m_device, &blasGeometryInstanceDeviceAddressInfo);

    // Setup TLAS build size info
    VkAccelerationStructureGeometryDataKHR tlasGeometryData{};
    tlasGeometryData.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeometryData.instances.pNext = NULL;
    tlasGeometryData.instances.arrayOfPointers = VK_FALSE;
    tlasGeometryData.instances.data.deviceAddress = blasGeometryInstanceDeviceAddress;

    VkAccelerationStructureGeometryKHR tlasGeometry{};
    tlasGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeometry.pNext = NULL;
    tlasGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeometry.geometry = tlasGeometryData;
    tlasGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

    VkAccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo{};
    tlasBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    tlasBuildGeometryInfo.pNext = NULL;
    tlasBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasBuildGeometryInfo.flags = 0;
    tlasBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tlasBuildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    tlasBuildGeometryInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    tlasBuildGeometryInfo.geometryCount = 1;
    tlasBuildGeometryInfo.pGeometries = &tlasGeometry;
    tlasBuildGeometryInfo.ppGeometries = NULL;
    tlasBuildGeometryInfo.scratchData.deviceAddress = 0;

    VkAccelerationStructureBuildSizesInfoKHR tlasBuildSizesInfo{};
    tlasBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    tlasBuildSizesInfo.pNext = NULL;
    tlasBuildSizesInfo.accelerationStructureSize = 0;
    tlasBuildSizesInfo.updateScratchSize = 0;
    tlasBuildSizesInfo.buildScratchSize = 0;

    std::vector<uint32_t> topLevelMaxPrimitiveCountList = {1};

    m_pvkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasBuildGeometryInfo, topLevelMaxPrimitiveCountList.data(), &tlasBuildSizesInfo);

    // Create TLAS buffer
    m_tlasBuffer = createBuffer(m_device, tlasBuildSizesInfo.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR);
    m_tlasMemory = allocateAndBindMemory(m_device, physicalDevice, m_tlasBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // Create TLAS
    VkAccelerationStructureCreateInfoKHR tlasCreateInfo{};
    tlasCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    tlasCreateInfo.pNext = NULL;
    tlasCreateInfo.createFlags = 0;
    tlasCreateInfo.buffer = m_tlasBuffer;
    tlasCreateInfo.offset = 0;
    tlasCreateInfo.size = tlasBuildSizesInfo.accelerationStructureSize;
    tlasCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tlasCreateInfo.deviceAddress = 0;

    VK_CHECK(m_pvkCreateAccelerationStructureKHR(m_device, &tlasCreateInfo, NULL, &m_tlas));

    // TLAS scratch buffer
    VkAccelerationStructureDeviceAddressInfoKHR tlasDeviceAddressInfo{};
    tlasDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    tlasDeviceAddressInfo.pNext = NULL;
    tlasDeviceAddressInfo.accelerationStructure = m_tlas;

    VkDeviceAddress tlasDeviceAddress = m_pvkGetAccelerationStructureDeviceAddressKHR(m_device, &tlasDeviceAddressInfo);

    m_tlasScratchBuffer = createBuffer(m_device, tlasBuildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    m_tlasScratchMemory = allocateAndBindMemory(m_device, physicalDevice, m_tlasScratchBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkBufferDeviceAddressInfo tlasScratchBufferDeviceAddressInfo{};
    tlasScratchBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    tlasScratchBufferDeviceAddressInfo.pNext = NULL;
    tlasScratchBufferDeviceAddressInfo.buffer = m_tlasBuffer;

    VkDeviceAddress tlasScratchBufferDeviceAddress = m_pvkGetBufferDeviceAddressKHR(m_device, &tlasScratchBufferDeviceAddressInfo);

    // Build TLAS
    tlasBuildGeometryInfo.dstAccelerationStructure = m_tlas;
    tlasBuildGeometryInfo.scratchData.deviceAddress = tlasScratchBufferDeviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR tlasBuildRangeInfo{};
    tlasBuildRangeInfo.primitiveCount = 1;
    tlasBuildRangeInfo.primitiveOffset = 0;
    tlasBuildRangeInfo.firstVertex = 0;
    tlasBuildRangeInfo.transformOffset = 0;

    const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
    const VkCommandBuffer& cb = command.commandBuffer;

    const VkAccelerationStructureBuildRangeInfoKHR* tlasBuildRangeInfos = &tlasBuildRangeInfo;

    m_pvkCmdBuildAccelerationStructuresKHR(cb, 1, &tlasBuildGeometryInfo, &tlasBuildRangeInfos);

    endSingleTimeCommands(m_context.getGraphicsQueue(), command);
}

void Raytracer::allocateCommandBuffers()
{
    m_commandBuffers.resize(m_framebuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_context.getGraphicsCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = ui32Size(m_commandBuffers);

    VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()));
}

void Raytracer::initializeGUI()
{
    const QueueFamilyIndices indices = getQueueFamilies(m_context.getPhysicalDevice(), m_context.getSurface());

    GUI::InitData initData{};
    initData.graphicsCommandPool = m_context.getGraphicsCommandPool();
    initData.physicalDevice = m_context.getPhysicalDevice();
    initData.device = m_device;
    initData.instance = m_context.getInstance();
    initData.graphicsFamily = indices.graphicsFamily;
    initData.graphicsQueue = m_context.getGraphicsQueue();
    initData.colorFormat = c_surfaceFormat.format;
    initData.depthFormat = c_depthFormat;
    initData.glfwWindow = m_context.getGlfwWindow();
    initData.imageCount = c_swapchainImageCount;
    initData.sampleCount = c_msaaSampleCount;
    initData.descriptorPool = m_descriptorPool;
    initData.renderPass = m_renderPass;

    m_gui.reset(new GUI(initData));
}
