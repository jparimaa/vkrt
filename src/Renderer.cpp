#include "Renderer.hpp"
#include "VulkanUtils.hpp"
#include "Utils.hpp"
#include "DebugMarker.hpp"
#include <imgui.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include <array>
#include <glm/gtc/matrix_transform.hpp>
namespace
{
const size_t c_uniformBufferSize = sizeof(glm::mat4);
const VkImageSubresourceRange c_defaultSubresourceRance{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
} // namespace

Renderer::Renderer(Context& context) :
    m_context(context),
    m_device(context.getDevice()),
    m_lastRenderTime(std::chrono::high_resolution_clock::now())
{
    DebugMarker::initialize(m_context.getInstance(), m_device);

    loadModel();
    setupCamera();
    createRenderPass();
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
    updateTexturesDescriptorSet();
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
    vkDestroyImageView(m_device, m_depthImageView, nullptr);

    for (const VkImageView& imageView : m_imageViews)
    {
        vkDestroyImageView(m_device, imageView, nullptr);
    }

    for (const VkImage& image : m_images)
    {
        vkDestroyImage(m_device, image, nullptr);
    }

    for (const VkDeviceMemory& imageMemory : m_imageMemories)
    {
        vkFreeMemory(m_device, imageMemory, nullptr);
    }

    vkDestroySampler(m_device, m_sampler, nullptr);

    for (const VkFramebuffer& framebuffer : m_framebuffers)
    {
        vkDestroyFramebuffer(m_device, framebuffer, nullptr);
    }

    for (const VkImageView& imageView : m_swapchainImageViews)
    {
        vkDestroyImageView(m_device, imageView, nullptr);
    }

    if (m_depthImage != VK_NULL_HANDLE)
    {
        vkDestroyImage(m_device, m_depthImage, nullptr);
    }

    if (m_depthImageMemory != VK_NULL_HANDLE)
    {
        vkFreeMemory(m_device, m_depthImageMemory, nullptr);
    }

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

    {
        DebugMarker::beginLabel(cb, "Render", DebugMarker::blue);

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
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_graphicsPipeline);

        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cb, 0, 1, &m_attributeBuffer, offsets);
        vkCmdBindIndexBuffer(cb, m_attributeBuffer, m_offsetToIndexData, VK_INDEX_TYPE_UINT32);
        const std::vector<VkDescriptorSet> descriptorSets{m_uboDescriptorSets[imageIndex], m_texturesDescriptorSet};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, ui32Size(descriptorSets), descriptorSets.data(), 0, nullptr);
        vkCmdDrawIndexed(cb, m_numIndices, 1, 0, 0, 0);

        vkCmdEndRenderPass(cb);

        DebugMarker::endLabel(cb);
    }

    {
        DebugMarker::beginLabel(cb, "GUI");

        m_gui->beginFrame();
        ImGui::Begin("Hello, world!");
        ImGui::Text("This is some useful text.");
        ImGui::End();
        m_gui->endFrame(cb, m_framebuffers[imageIndex]);

        DebugMarker::endLabel(cb);
    }

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
    m_lastRenderTime = high_resolution_clock::now();

    updateCamera(deltaTime);

    void* dst;
    VK_CHECK(vkMapMemory(m_device, m_uniformBufferMemory, imageIndex * c_uniformBufferSize, c_uniformBufferSize, 0, &dst));
    const glm::mat4 viewProjectionMatrix = m_camera.getProjectionMatrix() * m_camera.getViewMatrix();
    std::memcpy(dst, &viewProjectionMatrix[0], static_cast<size_t>(c_uniformBufferSize));
    vkUnmapMemory(m_device, m_uniformBufferMemory);

    return true;
}

void Renderer::loadModel()
{
    m_model.reset(new Model("sponza/Sponza.gltf"));
    m_numIndices = m_model->indices.size();
}

void Renderer::releaseModel()
{
    m_model.reset();
}

void Renderer::setupCamera()
{
    m_camera.setPosition(glm::vec3{0.0f, 0.0f, 10.0f});
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

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = c_surfaceFormat.format;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depthAttachment{};
    depthAttachment.format = c_depthFormat;
    depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    const std::array<VkAttachmentDescription, 2> attachments = {colorAttachment, depthAttachment};

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = ui32Size(attachments);
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass));
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
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags = 0;

    VK_CHECK(vkCreateImage(m_device, &imageInfo, nullptr, &m_depthImage));

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, m_depthImage, &memRequirements);

    const MemoryTypeResult memoryTypeResult = findMemoryType(m_context.getPhysicalDevice(), memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_depthImageMemory));
    VK_CHECK(vkBindImageMemory(m_device, m_depthImage, m_depthImageMemory, 0));

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_depthImage;
    barrier.subresourceRange = c_defaultSubresourceRance;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    const VkPipelineStageFlags barrierSrcFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
    const VkPipelineStageFlags barrierDstFlags = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;

    const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);

    vkCmdPipelineBarrier(command.commandBuffer, barrierSrcFlags, barrierDstFlags, 0, 0, nullptr, 0, nullptr, 1, &barrier);

    endSingleTimeCommands(m_context.getGraphicsQueue(), command);
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

    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = m_depthImage;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = c_depthFormat;
    createInfo.subresourceRange = c_defaultSubresourceRance;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VK_CHECK(vkCreateImageView(m_device, &createInfo, nullptr, &m_depthImageView));
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
        const std::array<VkImageView, 2> attachments = {m_swapchainImageViews[i], m_depthImageView};
        framebufferInfo.attachmentCount = ui32Size(attachments);
        framebufferInfo.pAttachments = attachments.data();

        VK_CHECK(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_framebuffers[i]));
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
    samplerInfo.maxAnisotropy = 16;
    samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;
    samplerInfo.compareOp = VK_COMPARE_OP_ALWAYS;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 512.0f;

    VK_CHECK(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_sampler));
}

void Renderer::createTextures()
{
    const size_t numImages = m_model->images.size();
    m_images.resize(numImages);
    m_imageMemories.resize(numImages);
    m_imageViews.resize(numImages);
    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();
    const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    const Model::Material& mat = m_model->materials[0];
    const std::vector<int> imageIndices{mat.baseColor, mat.metallicRoughnessImage, mat.normalImage};

    for (size_t i = 0; i < imageIndices.size(); ++i)
    {
        const Model::Image& image = m_model->images[imageIndices[i]];

        const StagingBuffer stagingBuffer = createStagingBuffer(m_device, physicalDevice, image.data.data(), image.data.size());

        const VkImageUsageFlags imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = image.width;
        imageInfo.extent.height = image.height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = imageUsage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        imageInfo.flags = 0;

        VK_CHECK(vkCreateImage(m_device, &imageInfo, nullptr, &m_images[i]));

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(m_device, m_images[i], &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;

        const VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        const MemoryTypeResult memoryTypeResult = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, memoryProperties);
        CHECK(memoryTypeResult.found);

        VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_imageMemories[i]));

        vkBindImageMemory(m_device, m_images[i], m_imageMemories[i], 0);

        {
            VkImageMemoryBarrier transferDstBarrier{};
            transferDstBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            transferDstBarrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            transferDstBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            transferDstBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            transferDstBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            transferDstBarrier.image = m_images[i];
            transferDstBarrier.subresourceRange = c_defaultSubresourceRance;
            transferDstBarrier.srcAccessMask = 0;
            transferDstBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            VkImageMemoryBarrier readOnlyBarrier = transferDstBarrier;
            readOnlyBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            readOnlyBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            readOnlyBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            readOnlyBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

            const VkPipelineStageFlags transferSrcFlags = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            const VkPipelineStageFlags transferDstFlags = VK_PIPELINE_STAGE_TRANSFER_BIT;
            const VkPipelineStageFlags readOnlySrcFlags = transferDstFlags;
            const VkPipelineStageFlags readOnlyDstFlags = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

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
            vkCmdPipelineBarrier(cb, readOnlySrcFlags, readOnlyDstFlags, 0, 0, nullptr, 0, nullptr, 1, &readOnlyBarrier);

            endSingleTimeCommands(m_context.getGraphicsQueue(), command);
        }

        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange = c_defaultSubresourceRance;

        VK_CHECK(vkCreateImageView(m_device, &viewInfo, nullptr, &m_imageViews[i]));

        releaseStagingBuffer(m_device, stagingBuffer);
    }
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
}

void Renderer::createTexturesDescriptorSetLayouts()
{
    const uint32_t imageCount = ui32Size(m_model->images);
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
}

void Renderer::createGraphicsPipeline()
{
    const std::array<VkDescriptorSetLayout, 2> descriptorSetLayouts{m_uboDescriptorSetLayout, m_texturesDescriptorSetLayout};
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = ui32Size(descriptorSetLayouts);
    pipelineLayoutInfo.pSetLayouts = descriptorSetLayouts.data();

    VK_CHECK(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));

    VkVertexInputBindingDescription vertexDescription{};
    vertexDescription.binding = 0;
    vertexDescription.stride = sizeof(Model::Vertex);
    vertexDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::vector<VkVertexInputAttributeDescription> attributeDescriptions(3);

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
    multisampleState.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
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
    colorBlendAttachmentState.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    colorBlendAttachmentState.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
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

    VkShaderModule vertexShaderModule = createShaderModule(m_device, "shaders/shader.vert.spv");
    VkShaderModule fragmentShaderModule = createShaderModule(m_device, "shaders/shader.frag.spv");

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

    for (const VkPipelineShaderStageCreateInfo& stage : shaderStages)
    {
        vkDestroyShaderModule(m_device, stage.module, nullptr);
    }
}

void Renderer::createDescriptorPool()
{
    const uint32_t swapchainLength = static_cast<uint32_t>(m_context.getSwapchainImages().size());
    const uint32_t numSetsForGUI = 1;
    const uint32_t numSetsForModel = 1;

    const uint32_t descriptorCount = ui32Size(m_model->images) + numSetsForGUI;

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
}

void Renderer::createTextureDescriptorSet()
{
    std::vector<VkDescriptorSetLayout> layouts{m_texturesDescriptorSetLayout};

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = ui32Size(layouts);
    allocInfo.pSetLayouts = layouts.data();
    VK_ERROR_FRAGMENTED_POOL;
    VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_texturesDescriptorSet));
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

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, m_uniformBuffer, &memRequirements);

    const MemoryTypeResult memoryTypeResult = findMemoryType(m_context.getPhysicalDevice(), memRequirements.memoryTypeBits, memoryProperties);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_uniformBufferMemory));
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

void Renderer::updateTexturesDescriptorSet()
{
    std::vector<VkWriteDescriptorSet> descriptorWrites(m_imageViews.size());
    std::vector<VkDescriptorImageInfo> imageInfos(m_imageViews.size());

    for (size_t i = 0; i < m_imageViews.size(); ++i)
    {
        VkDescriptorImageInfo& imageInfo = imageInfos[i];
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = m_imageViews[i];
        imageInfo.sampler = m_sampler;

        descriptorWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[i].dstSet = m_texturesDescriptorSet;
        descriptorWrites[i].dstBinding = i;
        descriptorWrites[i].dstArrayElement = 0;
        descriptorWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorWrites[i].descriptorCount = 1;
        descriptorWrites[i].pImageInfo = &imageInfo;
    }

    vkUpdateDescriptorSets(m_device, ui32Size(descriptorWrites), descriptorWrites.data(), 0, nullptr);
}

void Renderer::createVertexAndIndexBuffer()
{
    VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();
    const VkMemoryPropertyFlags memoryProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    const uint64_t vertexBufferSize = sizeof(Model::Vertex) * m_model->vertices.size();
    const uint64_t indexBufferSize = sizeof(Model::Index) * m_model->indices.size();
    const uint64_t bufferSize = vertexBufferSize + indexBufferSize;
    m_offsetToIndexData = vertexBufferSize;
    std::vector<uint8_t> data(bufferSize, 0);
    std::memcpy(&data[0], m_model->vertices.data(), vertexBufferSize);
    std::memcpy(&data[m_offsetToIndexData], m_model->indices.data(), indexBufferSize);
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

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(m_device, m_attributeBuffer, &memRequirements);

    const MemoryTypeResult memoryTypeResult = findMemoryType(physicalDevice, memRequirements.memoryTypeBits, memoryProperties);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_attributeBufferMemory));
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
    initData.descriptorPool = m_descriptorPool;

    m_gui.reset(new GUI(initData));
}
