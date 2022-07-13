#include "Renderer.hpp"
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
const size_t c_uniformBufferSize = sizeof(glm::mat4);
const VkImageSubresourceRange c_defaultSubresourceRance{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
const VkSampleCountFlagBits c_msaaSampleCount = VK_SAMPLE_COUNT_8_BIT;
} // namespace

Renderer::Renderer(Context& context) :
    m_context(context),
    m_device(context.getDevice()),
    m_lastRenderTime(std::chrono::high_resolution_clock::now())
{
    loadModel();
    setupCamera();
    createRenderPass();
    createMsaaColorImage();
    createDepthImage();
    createSwapchainImageViews();
    createFramebuffers();
    createSampler();
    createTextures();
    createUboDescriptorSetLayouts();
    createTexturesDescriptorSetLayouts();
    createGraphicsPipeline();
    createDescriptorPool();
    createUboDescriptorSets();
    createTextureDescriptorSet();
    createUniformBuffer();
    updateUboDescriptorSets();
    updateTexturesDescriptorSets();
    createVertexAndIndexBuffer();
    allocateCommandBuffers();
    releaseModel();
    initializeGUI();
}

Renderer::~Renderer()
{
    vkDeviceWaitIdle(m_device);

    m_gui.reset();

    vkDestroyBuffer(m_device, m_attributeBuffer, nullptr);
    vkFreeMemory(m_device, m_attributeBufferMemory, nullptr);
    vkDestroyBuffer(m_device, m_uniformBuffer, nullptr);
    vkFreeMemory(m_device, m_uniformBufferMemory, nullptr);
    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_texturesDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_uboDescriptorSetLayout, nullptr);

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

bool Renderer::render()
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
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cb, 0, 1, &m_attributeBuffer, offsets);
        vkCmdBindIndexBuffer(cb, m_attributeBuffer, m_primitiveInfos[0].indexOffset, VK_INDEX_TYPE_UINT32);
        for (size_t i = 0; i < m_primitiveInfos.size(); ++i)
        {
            const PrimitiveInfo& primitiveInfo = m_primitiveInfos[i];
            const std::vector<VkDescriptorSet> descriptorSets{m_uboDescriptorSets[imageIndex], m_texturesDescriptorSets[primitiveInfo.material]};
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

bool Renderer::update(uint32_t imageIndex)
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
    VK_CHECK(vkMapMemory(m_device, m_uniformBufferMemory, imageIndex * c_uniformBufferSize, c_uniformBufferSize, 0, &dst));
    glm::mat4 scaleMatrix = glm::scale(glm::vec3(0.01f, 0.01f, 0.01f));
    const glm::mat4 wvpMatrix = m_camera.getProjectionMatrix() * m_camera.getViewMatrix() * scaleMatrix;
    std::memcpy(dst, &wvpMatrix[0], static_cast<size_t>(c_uniformBufferSize));
    vkUnmapMemory(m_device, m_uniformBufferMemory);

    return true;
}

void Renderer::loadModel()
{
    m_model.reset(new Model("sponza/Sponza.gltf"));
}

void Renderer::releaseModel()
{
    m_model.reset();
}

void Renderer::setupCamera()
{
    m_camera.setPosition(glm::vec3{-4.0f, 2.0f, -0.2f});
    m_camera.setRotation(glm::vec3{0.0f, 1.51f, 0.0f});
}

void Renderer::updateCamera(double deltaTime)
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

void Renderer::createRenderPass()
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

void Renderer::createMsaaColorImage()
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

void Renderer::createDepthImage()
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

void Renderer::createSwapchainImageViews()
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

void Renderer::createFramebuffers()
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

void Renderer::createSampler()
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

void Renderer::createTextures()
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

void Renderer::createMipmaps(VkImage image, uint32_t mipLevels, glm::uvec2 imageSize)
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

void Renderer::createUboDescriptorSetLayouts()
{
    VkDescriptorSetLayoutBinding uboLayoutBinding{};
    uboLayoutBinding.binding = 0;
    uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboLayoutBinding.descriptorCount = 1;
    uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    uboLayoutBinding.pImmutableSamplers = nullptr;

    const std::vector<VkDescriptorSetLayoutBinding> bindings{uboLayoutBinding};
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = ui32Size(bindings);
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_uboDescriptorSetLayout));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, m_uboDescriptorSetLayout, "Desc set layout - UBO");
}

void Renderer::createTexturesDescriptorSetLayouts()
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

void Renderer::createGraphicsPipeline()
{
    const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts{m_uboDescriptorSetLayout, m_texturesDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = ui32Size(descriptorSetLayouts);
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    VK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_PIPELINE_LAYOUT, m_pipelineLayout, "Pipeline layout - Renderer");

    VkVertexInputBindingDescription vertexDescription{};
    vertexDescription.binding = 0;
    vertexDescription.stride = sizeof(Model::Vertex);
    vertexDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(4);

    attributeDescriptions[0].binding = 0;
    attributeDescriptions[0].location = 0;
    attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[0].offset = offsetof(Model::Vertex, position);

    attributeDescriptions[1].binding = 0;
    attributeDescriptions[1].location = 1;
    attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
    attributeDescriptions[1].offset = offsetof(Model::Vertex, normal);

    attributeDescriptions[2].binding = 0;
    attributeDescriptions[2].location = 2;
    attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
    attributeDescriptions[2].offset = offsetof(Model::Vertex, uv);

    attributeDescriptions[3].binding = 0;
    attributeDescriptions[3].location = 3;
    attributeDescriptions[3].format = VK_FORMAT_R32G32B32A32_SFLOAT;
    attributeDescriptions[3].offset = offsetof(Model::Vertex, tangent);

    VkPipelineVertexInputStateCreateInfo vertexInputState{};
    vertexInputState.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputState.vertexBindingDescriptionCount = 1;
    vertexInputState.pVertexBindingDescriptions = &vertexDescription;
    vertexInputState.vertexAttributeDescriptionCount = ui32Size(attributeDescriptions);
    vertexInputState.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState{};
    inputAssemblyState.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssemblyState.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(c_windowExtent.width);
    viewport.height = static_cast<float>(c_windowExtent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = c_windowExtent;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.pViewports = &viewport;
    viewportState.scissorCount = 1;
    viewportState.pScissors = &scissor;

    VkPipelineRasterizationStateCreateInfo rasterizationState{};
    rasterizationState.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizationState.depthClampEnable = VK_FALSE;
    rasterizationState.rasterizerDiscardEnable = VK_FALSE;
    rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizationState.lineWidth = 1.0f;
    rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizationState.depthBiasEnable = VK_FALSE;
    rasterizationState.depthBiasConstantFactor = 0.0f;
    rasterizationState.depthBiasClamp = 0.0f;
    rasterizationState.depthBiasSlopeFactor = 0.0f;

    VkPipelineMultisampleStateCreateInfo multisampleState{};
    multisampleState.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampleState.sampleShadingEnable = VK_FALSE;
    multisampleState.rasterizationSamples = c_msaaSampleCount;
    multisampleState.minSampleShading = 1.0f;
    multisampleState.pSampleMask = nullptr;
    multisampleState.alphaToCoverageEnable = VK_FALSE;
    multisampleState.alphaToOneEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo depthStencilState{};
    depthStencilState.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencilState.depthTestEnable = VK_TRUE;
    depthStencilState.depthWriteEnable = VK_TRUE;
    depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS;
    depthStencilState.depthBoundsTestEnable = VK_FALSE;
    depthStencilState.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState colorBlendAttachmentState{};
    colorBlendAttachmentState.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachmentState.blendEnable = VK_FALSE;
    colorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachmentState.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachmentState.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachmentState.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    colorBlendAttachmentState.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlendState{};
    colorBlendState.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlendState.logicOpEnable = VK_FALSE;
    colorBlendState.logicOp = VK_LOGIC_OP_COPY;
    colorBlendState.attachmentCount = 1;
    colorBlendState.pAttachments = &colorBlendAttachmentState;
    colorBlendState.blendConstants[0] = 0.0f;
    colorBlendState.blendConstants[1] = 0.0f;
    colorBlendState.blendConstants[2] = 0.0f;
    colorBlendState.blendConstants[3] = 0.0f;

    const std::filesystem::path currentPath = getCurrentExecutableDirectory();
    VkShaderModule vertexShaderModule = createShaderModule(m_device, currentPath / "shader.vert.spv");
    VkShaderModule fragmentShaderModule = createShaderModule(m_device, currentPath / "shader.frag.spv");

    VkPipelineShaderStageCreateInfo vertexShaderStageInfo{};
    vertexShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertexShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertexShaderStageInfo.module = vertexShaderModule;
    vertexShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragmentShaderStageInfo{};
    fragmentShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragmentShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragmentShaderStageInfo.module = fragmentShaderModule;
    fragmentShaderStageInfo.pName = "main";

    std::vector<VkPipelineShaderStageCreateInfo> shaderStages{vertexShaderStageInfo, fragmentShaderStageInfo};

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = ui32Size(shaderStages);
    pipelineInfo.pStages = shaderStages.data();
    pipelineInfo.pVertexInputState = &vertexInputState;
    pipelineInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizationState;
    pipelineInfo.pMultisampleState = &multisampleState;
    pipelineInfo.pDepthStencilState = &depthStencilState;
    pipelineInfo.pColorBlendState = &colorBlendState;
    pipelineInfo.pDynamicState = nullptr;
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.renderPass = m_renderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
    pipelineInfo.basePipelineIndex = -1;

    VK_CHECK(vkCreateGraphicsPipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_graphicsPipeline));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_PIPELINE, m_graphicsPipeline, "Pipeline - Renderer");

    for (const VkPipelineShaderStageCreateInfo& stage : shaderStages)
    {
        vkDestroyShaderModule(m_device, stage.module, nullptr);
    }
}

void Renderer::createDescriptorPool()
{
    const uint32_t swapchainLength = static_cast<uint32_t>(m_context.getSwapchainImages().size());
    const uint32_t numSetsForGUI = 1;
    const uint32_t numSetsForModel = ui32Size(m_model->materials);

    const uint32_t descriptorCount = numSetsForModel + numSetsForGUI;

    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = swapchainLength;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = descriptorCount;

    const uint32_t maxSets = swapchainLength + numSetsForModel + numSetsForGUI;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = ui32Size(poolSizes);
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSets;

    VK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL, m_descriptorPool, "Descriptor pool - Renderer");
}

void Renderer::createUboDescriptorSets()
{
    const uint32_t swapchainLength = static_cast<uint32_t>(m_context.getSwapchainImages().size());
    m_uboDescriptorSets.resize(swapchainLength);

    std::vector<VkDescriptorSetLayout> layouts(swapchainLength, m_uboDescriptorSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = ui32Size(layouts);
    allocInfo.pSetLayouts = layouts.data();

    VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, m_uboDescriptorSets.data()));
    for (size_t i = 0; i < m_uboDescriptorSets.size(); ++i)
    {
        DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, m_uboDescriptorSets[i], "Desc set - UBO " + std::to_string(i));
    }
}

void Renderer::createTextureDescriptorSet()
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

void Renderer::createUniformBuffer()
{
    const VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const uint64_t bufferSize = c_uniformBufferSize * m_context.getSwapchainImages().size();

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_uniformBuffer));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_uniformBuffer, "Buffer - Renderer uniform buffer");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, m_uniformBuffer, &memRequirements);

    const MemoryTypeResult memoryTypeResult = findMemoryType(m_context.getPhysicalDevice(), memRequirements.memoryTypeBits, memoryProperties);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_uniformBufferMemory));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_uniformBufferMemory, "Memory - Renderer uniform buffer");

    VK_CHECK(vkBindBufferMemory(m_device, m_uniformBuffer, m_uniformBufferMemory, 0));
}

void Renderer::updateUboDescriptorSets()
{
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_uniformBuffer;
    bufferInfo.range = c_uniformBufferSize;

    std::vector<VkWriteDescriptorSet> descriptorWrites(m_uboDescriptorSets.size());

    for (size_t i = 0; i < m_uboDescriptorSets.size(); ++i)
    {
        bufferInfo.offset = i * c_uniformBufferSize;
        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = m_uboDescriptorSets[i];
        descriptorWrites[i].dstBinding = 0;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pBufferInfo = &bufferInfo;
    }

    vkUpdateDescriptorSets(m_device, ui32Size(descriptorWrites), descriptorWrites.data(), 0, nullptr);
}

void Renderer::updateTexturesDescriptorSets()
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

void Renderer::createVertexAndIndexBuffer()
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

    VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();
    StagingBuffer stagingBuffer = createStagingBuffer(m_device, physicalDevice, data.data(), bufferSize);

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = bufferSize;
    bufferInfo.usage = //
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | //
        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | //
        VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(m_device, &bufferInfo, nullptr, &m_attributeBuffer));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_attributeBuffer, "Buffer - Attribute");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, m_attributeBuffer, &memRequirements);

    const VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const MemoryTypeResult memoryTypeResult = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, memoryProperties);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
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

void Renderer::allocateCommandBuffers()
{
    m_commandBuffers.resize(m_framebuffers.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_context.getGraphicsCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = ui32Size(m_commandBuffers);

    VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()));
}

void Renderer::initializeGUI()
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
