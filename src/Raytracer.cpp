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
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
    glm::vec4 position;
    glm::vec4 right;
    glm::vec4 up;
    glm::vec4 forward;
    std::array<glm::vec4, 4> lightPositions;
};

const std::array<glm::vec4, 4> c_lightPositions{
    glm::vec4{6.0f, 6.0f, 0.0f, 0.0f}, //
    glm::vec4{2.0f, 5.0f, 0.0f, 0.0f}, //
    glm::vec4{-2.0f, 4.0f, 0.0f, 0.0f}, //
    glm::vec4{-6.0f, 3.0f, 0.0f, 0.0f} //
};

struct MaterialInfo
{
    int baseColorTextureIndex = -1;
    int metallicRoughnessTextureIndex = -1;
    int normalTextureIndex = -1;
    int indexBufferOffset = 0;
};

const size_t c_uniformBufferSize = sizeof(UniformBufferInfo);
const VkImageSubresourceRange c_defaultSubresourceRance{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
const uint32_t c_shaderCount = 4;
const uint32_t c_shaderGroupCount = 4;

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
    createColorImage();
    createSwapchainImageViews();
    createSampler();
    createTextures();
    createVertexAndIndexBuffer();
    createDescriptorPool();
    createCommonDescriptorSetLayoutAndAllocate();
    createMaterialIndexDescriptorSetLayoutAndAllocate();
    createTexturesDescriptorSetLayoutAndAllocate();
    createPipeline();
    createCommonBuffer();
    createMaterialIndexBuffer();
    allocateCommandBuffers();
    createBLAS();
    createTLAS();
    updateCommonDescriptorSets();
    updateMaterialIndexDescriptorSet();
    updateTexturesDescriptorSets();
    createShaderBindingTable();

    m_model.reset();
}

Raytracer::~Raytracer()
{
    vkDeviceWaitIdle(m_device);

    destroyBufferAndFreeMemory(m_device, m_vertexBuffer, m_vertexBufferMemory);
    destroyBufferAndFreeMemory(m_device, m_indexBuffer, m_indexBufferMemory);
    destroyBufferAndFreeMemory(m_device, m_primitiveIndexBuffer, m_primitiveIndexBufferMemory);
    destroyBufferAndFreeMemory(m_device, m_commonBuffer, m_commonBufferMemory);
    destroyBufferAndFreeMemory(m_device, m_materialIndexBuffer, m_materialIndexBufferMemory);
    destroyBufferAndFreeMemory(m_device, m_tlasBuffer, m_tlasMemory);
    destroyBufferAndFreeMemory(m_device, m_blasBuffer, m_blasMemory);
    destroyBufferAndFreeMemory(m_device, m_blasGeometryInstanceBuffer, m_blasGeometryInstanceMemory);
    destroyBufferAndFreeMemory(m_device, m_shaderBindingTableBuffer, m_shaderBindingTableMemory);

    m_pvkDestroyAccelerationStructureKHR(m_device, m_tlas, nullptr);
    m_pvkDestroyAccelerationStructureKHR(m_device, m_blas, nullptr);

    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_texturesDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_materialIndexDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(m_device, m_commonDescriptorSetLayout, nullptr);

    vkDestroySampler(m_device, m_sampler, nullptr);

    for (const VkImageView& imageView : m_imageViews)
    {
        vkDestroyImageView(m_device, imageView, nullptr);
    }

    for (const VkImage& image : m_images)
    {
        vkDestroyImage(m_device, image, nullptr);
    }

    vkFreeMemory(m_device, m_imageMemory, nullptr);

    for (const VkImageView& imageView : m_swapchainImageViews)
    {
        vkDestroyImageView(m_device, imageView, nullptr);
    }

    vkDestroyImageView(m_device, m_colorImageView, nullptr);
    vkFreeMemory(m_device, m_colorImageMemory, nullptr);
    vkDestroyImage(m_device, m_colorImage, nullptr);
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

    {
        DebugMarker::beginLabel(cb, "Render", DebugMarker::blue);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipeline);

        const std::vector<VkDescriptorSet> descriptorSets{m_commonDescriptorSet, m_materialIndexDescriptorSet, m_texturesDescriptorSet};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_pipelineLayout, 0, ui32Size(descriptorSets), descriptorSets.data(), 0, nullptr);

        m_pvkCmdTraceRaysKHR(cb, &m_rgenShaderBindingTable, &m_rmissShaderBindingTable, &m_rchitShaderBindingTable, &m_callableShaderBindingTable, c_windowWidth, c_windowHeight, 1);

        {
            const std::vector<VkImage>& swapchainImages = m_context.getSwapchainImages();

            VkImageMemoryBarrier swapchainLayoutBarrier{};
            swapchainLayoutBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
            swapchainLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            swapchainLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            swapchainLayoutBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainLayoutBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            swapchainLayoutBarrier.image = swapchainImages[imageIndex];
            swapchainLayoutBarrier.subresourceRange = c_defaultSubresourceRance;
            swapchainLayoutBarrier.srcAccessMask = 0;
            swapchainLayoutBarrier.dstAccessMask = 0;

            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapchainLayoutBarrier);

            VkImageCopy region{};
            region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.srcSubresource.baseArrayLayer = 0;
            region.srcSubresource.mipLevel = 0;
            region.srcSubresource.layerCount = 1;
            region.srcOffset = {0, 0, 0};
            region.dstSubresource = region.srcSubresource;
            region.dstOffset = region.srcOffset;
            region.extent = {c_windowWidth, c_windowHeight, 1};

            vkCmdCopyImage(cb, m_colorImage, VK_IMAGE_LAYOUT_GENERAL, swapchainImages[imageIndex], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            swapchainLayoutBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            swapchainLayoutBarrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &swapchainLayoutBarrier);
        }

        DebugMarker::endLabel(cb);
    }

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
    // Todo: ring buffer
    VK_CHECK(vkMapMemory(m_device, m_commonBufferMemory, 0, c_uniformBufferSize, 0, &dst));

    UniformBufferInfo uniformBufferInfo{};
    uniformBufferInfo.forward = toVec4(m_camera.getForward(), 0.0f);
    uniformBufferInfo.right = toVec4(-m_camera.getLeft(), 0.0f);
    uniformBufferInfo.up = toVec4(m_camera.getUp(), 0.0f);
    uniformBufferInfo.position = toVec4(m_camera.getPosition(), 1.0f);

    uniformBufferInfo.projInverse = glm::inverse(m_camera.getProjectionMatrix());
    uniformBufferInfo.viewInverse = glm::inverse(m_camera.getViewMatrix());
    uniformBufferInfo.lightPositions = c_lightPositions;

    std::memcpy(dst, &uniformBufferInfo, static_cast<size_t>(c_uniformBufferSize));
    vkUnmapMemory(m_device, m_commonBufferMemory);

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
    m_pvkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(m_device, "vkGetRayTracingShaderGroupHandlesKHR");
    CHECK(m_pvkGetRayTracingShaderGroupHandlesKHR);
    m_pvkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(m_device, "vkCmdTraceRaysKHR");
    CHECK(m_pvkCmdTraceRaysKHR);
    m_pvkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(m_device, "vkDestroyAccelerationStructureKHR");
    CHECK(m_pvkDestroyAccelerationStructureKHR);
}

void Raytracer::loadModel()
{
    m_model.reset(new Model("sponza/Sponza.gltf"));
}

void Raytracer::setupCamera()
{
    m_camera.setPosition({6.3f, 4.5f, -0.7f});
    m_camera.setRotation({0.0f, 1.57f, 0.0f});
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

void Raytracer::createColorImage()
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
    imageInfo.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.flags = 0;

    VK_CHECK(vkCreateImage(m_device, &imageInfo, nullptr, &m_colorImage));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_IMAGE, m_colorImage, "Image - Color");

    VkMemoryRequirements memRequirements;
    vkGetImageMemoryRequirements(m_device, m_colorImage, &memRequirements);

    const MemoryTypeResult memoryTypeResult = findMemoryType(m_context.getPhysicalDevice(), memRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(memoryTypeResult.found);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = memoryTypeResult.typeIndex;

    VK_CHECK(vkAllocateMemory(m_device, &allocInfo, nullptr, &m_colorImageMemory));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_colorImageMemory, "Memory - Color image");
    VK_CHECK(vkBindImageMemory(m_device, m_colorImage, m_colorImageMemory, 0));

    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = m_colorImage;
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = c_surfaceFormat.format;
    createInfo.subresourceRange = c_defaultSubresourceRance;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    VK_CHECK(vkCreateImageView(m_device, &createInfo, nullptr, &m_colorImageView));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_IMAGE_VIEW, m_colorImageView, "Image view - Color");

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.image = m_colorImage;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcAccessMask = VK_ACCESS_NONE;
    barrier.dstAccessMask = VK_ACCESS_NONE;

    const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
    const VkCommandBuffer& cb = command.commandBuffer;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    endSingleTimeCommands(m_context.getGraphicsQueue(), command);
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

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.image = swapchainImages[i];
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        barrier.srcAccessMask = VK_ACCESS_NONE;
        barrier.dstAccessMask = VK_ACCESS_NONE;

        const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
        const VkCommandBuffer& cb = command.commandBuffer;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
        endSingleTimeCommands(m_context.getGraphicsQueue(), command);
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
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_colorImageMemory, "Memory - Texture images");

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

void Raytracer::createVertexAndIndexBuffer()
{
    m_vertexDataSize = m_model->vertexBufferSizeInBytes;
    m_indexDataSize = m_model->indexBufferSizeInBytes;
    std::vector<uint8_t> vertexData(m_vertexDataSize, 0);
    std::vector<uint8_t> indexData(m_indexDataSize, 0);

    const int indexCount = m_indexDataSize / sizeof(Model::Index);
    std::vector<Model::Index> indices(indexCount);
    int indexCounter = 0;
    Model::Index indexCounterOffset = 0;
    size_t vertexByteOffset = 0;
    size_t indexByteOffset = 0;

    for (const Model::Primitive& primitive : m_model->primitives)
    {
        Model::Index highestIndex = 0;
        for (Model::Index index : primitive.indices)
        {
            indices[indexCounter] = indexCounterOffset + index;
            highestIndex = std::max(highestIndex, index);
            ++indexCounter;
        }

        m_primitiveInfos.push_back(
            PrimitiveInfo{
                highestIndex, //
                ui32Size(primitive.indices) / 3, //
                indexByteOffset //
            } //
        );

        indexCounterOffset += primitive.vertices.size();

        const size_t vertexSize = sizeof(Model::Vertex) * primitive.vertices.size();
        std::memcpy(&vertexData[vertexByteOffset], primitive.vertices.data(), vertexSize);
        vertexByteOffset += vertexSize;
        indexByteOffset += sizeof(Model::Index) * primitive.indices.size();
    }

    m_triangleCount = indices.size() / 3;

    std::vector<glm::uvec4> primitiveIndices(m_triangleCount);
    size_t counter = 0;
    for (size_t i = 0; i < indices.size(); i += 3)
    {
        primitiveIndices[counter].x = indices[i + 0];
        primitiveIndices[counter].y = indices[i + 1];
        primitiveIndices[counter].z = indices[i + 2];
        primitiveIndices[counter].w = 0;
        ++counter;
    }

    CHECK(m_indexDataSize == (sizeof(Model::Index) * indices.size()));
    std::memcpy(&indexData[0], indices.data(), m_indexDataSize);

    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();
    const VkBufferUsageFlags usage = //
        VK_BUFFER_USAGE_TRANSFER_DST_BIT | //
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | //
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | //
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;

    { // Vertex
        StagingBuffer stagingBuffer = createStagingBuffer(m_device, physicalDevice, vertexData.data(), m_vertexDataSize);

        m_vertexBuffer = createBuffer(m_device, m_vertexDataSize, usage);
        m_vertexBufferMemory = allocateAndBindMemory(m_device, physicalDevice, m_vertexBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_vertexBuffer, "Buffer - Vertex");
        DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_vertexBufferMemory, "Memory - Vertex buffer");

        copyRegion.size = m_vertexDataSize;

        const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
        vkCmdCopyBuffer(command.commandBuffer, stagingBuffer.buffer, m_vertexBuffer, 1, &copyRegion);
        endSingleTimeCommands(m_context.getGraphicsQueue(), command);

        releaseStagingBuffer(m_device, stagingBuffer);
    }
    { // Index
        StagingBuffer stagingBuffer = createStagingBuffer(m_device, physicalDevice, indexData.data(), m_indexDataSize);

        m_indexBuffer = createBuffer(m_device, m_indexDataSize, usage);
        m_indexBufferMemory = allocateAndBindMemory(m_device, physicalDevice, m_indexBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_indexBuffer, "Buffer - Index");
        DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_indexBufferMemory, "Memory - Index buffer");

        copyRegion.size = m_indexDataSize;

        const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
        vkCmdCopyBuffer(command.commandBuffer, stagingBuffer.buffer, m_indexBuffer, 1, &copyRegion);
        endSingleTimeCommands(m_context.getGraphicsQueue(), command);

        releaseStagingBuffer(m_device, stagingBuffer);
    }
    { // Primitive Index
        const uint64_t primitiveIndicesSize = primitiveIndices.size() * sizeof(primitiveIndices[0]);
        StagingBuffer stagingBuffer = createStagingBuffer(m_device, physicalDevice, primitiveIndices.data(), primitiveIndicesSize);

        m_primitiveIndexBuffer = createBuffer(m_device, primitiveIndicesSize, usage);
        m_primitiveIndexBufferMemory = allocateAndBindMemory(m_device, physicalDevice, m_primitiveIndexBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_primitiveIndexBuffer, "Buffer - Index");
        DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_primitiveIndexBufferMemory, "Memory - Index buffer");

        copyRegion.size = primitiveIndicesSize;

        const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
        vkCmdCopyBuffer(command.commandBuffer, stagingBuffer.buffer, m_primitiveIndexBuffer, 1, &copyRegion);
        endSingleTimeCommands(m_context.getGraphicsQueue(), command);

        releaseStagingBuffer(m_device, stagingBuffer);
    }
}

void Raytracer::createDescriptorPool()
{
    std::array<VkDescriptorPoolSize, 5> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = ui32Size(m_context.getSwapchainImages());
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = ui32Size(m_model->materials);
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    poolSizes[2].descriptorCount = 1;
    poolSizes[3].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[3].descriptorCount = 1;
    poolSizes[4].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[4].descriptorCount = 1;

    const uint32_t maxSets = ui32Size(m_model->materials) + 64;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = ui32Size(poolSizes);
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = maxSets;

    VK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_POOL, m_descriptorPool, "Descriptor pool - Raytracer");
}

void Raytracer::createCommonDescriptorSetLayoutAndAllocate()
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

    const std::vector<VkDescriptorSetLayout> layouts{m_commonDescriptorSetLayout};

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = NULL;
    descriptorSetAllocateInfo.descriptorPool = m_descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = ui32Size(layouts);
    descriptorSetAllocateInfo.pSetLayouts = layouts.data();

    VK_CHECK(vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, &m_commonDescriptorSet));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, m_commonDescriptorSet, "Desc set - Common");
}

void Raytracer::createMaterialIndexDescriptorSetLayoutAndAllocate()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings(1);
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[0].pImmutableSamplers = nullptr;

    VkDescriptorBindingFlagsEXT bindFlag = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT;

    VkDescriptorSetLayoutBindingFlagsCreateInfoEXT extendedInfo{};
    extendedInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO_EXT;
    extendedInfo.pNext = nullptr;
    extendedInfo.bindingCount = 1u;
    extendedInfo.pBindingFlags = &bindFlag;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.pNext = &extendedInfo;
    layoutInfo.bindingCount = ui32Size(bindings);
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_materialIndexDescriptorSetLayout));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, m_materialIndexDescriptorSetLayout, "Desc set layout - Material Index");

    const std::vector<VkDescriptorSetLayout> layouts{m_materialIndexDescriptorSetLayout};

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo{};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = NULL;
    descriptorSetAllocateInfo.descriptorPool = m_descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = ui32Size(layouts);
    descriptorSetAllocateInfo.pSetLayouts = layouts.data();

    VK_CHECK(vkAllocateDescriptorSets(m_device, &descriptorSetAllocateInfo, &m_materialIndexDescriptorSet));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, m_materialIndexDescriptorSet, "Desc set - Material index");
}

void Raytracer::createTexturesDescriptorSetLayoutAndAllocate()
{
    std::array<VkDescriptorSetLayoutBinding, 1> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorCount = ui32Size(m_images);
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[0].stageFlags = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    bindings[0].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = ui32Size(bindings);
    layoutInfo.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_texturesDescriptorSetLayout));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET_LAYOUT, m_texturesDescriptorSetLayout, "Desc set layout - Textures");

    const std::vector<VkDescriptorSetLayout> layouts{m_texturesDescriptorSetLayout};

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = ui32Size(layouts);
    allocInfo.pSetLayouts = layouts.data();
    VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &m_texturesDescriptorSet));
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DESCRIPTOR_SET, m_texturesDescriptorSet, "Desc set - Textures");
}

void Raytracer::createPipeline()
{
    const std::vector<VkDescriptorSetLayout> descriptorSetLayouts{m_commonDescriptorSetLayout, m_materialIndexDescriptorSetLayout, m_texturesDescriptorSetLayout};
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

    std::array<VkPipelineShaderStageCreateInfo, c_shaderCount> shaderStageCreateInfoList;

    shaderStageCreateInfoList[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfoList[0].pNext = NULL;
    shaderStageCreateInfoList[0].flags = 0;
    shaderStageCreateInfoList[0].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    shaderStageCreateInfoList[0].module = closesHitShaderModule;
    shaderStageCreateInfoList[0].pName = "main";
    shaderStageCreateInfoList[0].pSpecializationInfo = NULL;
    shaderStageCreateInfoList[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfoList[1].pNext = NULL;
    shaderStageCreateInfoList[1].flags = 0;
    shaderStageCreateInfoList[1].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    shaderStageCreateInfoList[1].module = rayGenShaderModule;
    shaderStageCreateInfoList[1].pName = "main";
    shaderStageCreateInfoList[1].pSpecializationInfo = NULL;
    shaderStageCreateInfoList[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfoList[2].pNext = NULL;
    shaderStageCreateInfoList[2].flags = 0;
    shaderStageCreateInfoList[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    shaderStageCreateInfoList[2].module = missShaderModule;
    shaderStageCreateInfoList[2].pName = "main";
    shaderStageCreateInfoList[2].pSpecializationInfo = NULL;
    shaderStageCreateInfoList[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfoList[3].pNext = NULL;
    shaderStageCreateInfoList[3].flags = 0;
    shaderStageCreateInfoList[3].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    shaderStageCreateInfoList[3].module = shadowMissShaderModule;
    shaderStageCreateInfoList[3].pName = "main";
    shaderStageCreateInfoList[3].pSpecializationInfo = NULL;

    std::array<VkRayTracingShaderGroupCreateInfoKHR, c_shaderGroupCount> shaderGroupCreateInfoList;

    shaderGroupCreateInfoList[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shaderGroupCreateInfoList[0].pNext = NULL;
    shaderGroupCreateInfoList[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    shaderGroupCreateInfoList[0].generalShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[0].closestHitShader = 0;
    shaderGroupCreateInfoList[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[0].intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[0].pShaderGroupCaptureReplayHandle = NULL;
    shaderGroupCreateInfoList[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shaderGroupCreateInfoList[1].pNext = NULL;
    shaderGroupCreateInfoList[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    shaderGroupCreateInfoList[1].generalShader = 1;
    shaderGroupCreateInfoList[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[1].intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[1].pShaderGroupCaptureReplayHandle = NULL;
    shaderGroupCreateInfoList[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shaderGroupCreateInfoList[2].pNext = NULL;
    shaderGroupCreateInfoList[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    shaderGroupCreateInfoList[2].generalShader = 2;
    shaderGroupCreateInfoList[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[2].intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[2].pShaderGroupCaptureReplayHandle = NULL;
    shaderGroupCreateInfoList[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    shaderGroupCreateInfoList[3].pNext = NULL;
    shaderGroupCreateInfoList[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    shaderGroupCreateInfoList[3].generalShader = 3;
    shaderGroupCreateInfoList[3].closestHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[3].anyHitShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[3].intersectionShader = VK_SHADER_UNUSED_KHR;
    shaderGroupCreateInfoList[3].pShaderGroupCaptureReplayHandle = NULL;

    VkRayTracingPipelineCreateInfoKHR rayTracingPipelineCreateInfo{};
    rayTracingPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    rayTracingPipelineCreateInfo.pNext = NULL;
    rayTracingPipelineCreateInfo.flags = 0;
    rayTracingPipelineCreateInfo.stageCount = ui32Size(shaderStageCreateInfoList);
    rayTracingPipelineCreateInfo.pStages = shaderStageCreateInfoList.data();
    rayTracingPipelineCreateInfo.groupCount = ui32Size(shaderGroupCreateInfoList);
    rayTracingPipelineCreateInfo.pGroups = shaderGroupCreateInfoList.data();
    rayTracingPipelineCreateInfo.maxPipelineRayRecursionDepth = 2;
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

void Raytracer::createCommonBuffer()
{
    const uint64_t bufferSize = c_uniformBufferSize;

    m_commonBuffer = createBuffer(m_device, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    m_commonBufferMemory = allocateAndBindMemory(m_device, m_context.getPhysicalDevice(), m_commonBuffer, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_commonBuffer, "Buffer - Common uniform buffer");
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_commonBufferMemory, "Memory - Common uniform memory");
}

void Raytracer::createMaterialIndexBuffer()
{
    const uint64_t bufferSize = sizeof(MaterialInfo) * m_model->primitives.size();

    m_materialIndexBuffer = createBuffer(m_device, bufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    m_materialIndexBufferMemory = allocateAndBindMemory(m_device, m_context.getPhysicalDevice(), m_materialIndexBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    DebugMarker::setObjectName(VK_OBJECT_TYPE_BUFFER, m_materialIndexBuffer, "Buffer - Material index buffer");
    DebugMarker::setObjectName(VK_OBJECT_TYPE_DEVICE_MEMORY, m_materialIndexBufferMemory, "Memory - Material index memory");
}

void Raytracer::allocateCommandBuffers()
{
    m_commandBuffers.resize(m_swapchainImageViews.size());

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_context.getGraphicsCommandPool();
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = ui32Size(m_commandBuffers);

    VK_CHECK(vkAllocateCommandBuffers(m_device, &allocInfo, m_commandBuffers.data()));
}

void Raytracer::createBLAS()
{
    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();

    // Setup geometry and get build size
    VkBufferDeviceAddressInfo vertexBufferDeviceAddressInfo{};
    vertexBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    vertexBufferDeviceAddressInfo.pNext = NULL;
    vertexBufferDeviceAddressInfo.buffer = m_vertexBuffer;

    VkBufferDeviceAddressInfo indexBufferDeviceAddressInfo{};
    indexBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    indexBufferDeviceAddressInfo.pNext = NULL;
    indexBufferDeviceAddressInfo.buffer = m_indexBuffer;

    const VkDeviceAddress vertexBufferDeviceAddress = m_pvkGetBufferDeviceAddressKHR(m_device, &vertexBufferDeviceAddressInfo);
    const VkDeviceAddress indexBufferDeviceAddress = m_pvkGetBufferDeviceAddressKHR(m_device, &indexBufferDeviceAddressInfo);

    std::vector<VkAccelerationStructureGeometryKHR> geometries;
    geometries.reserve(m_primitiveInfos.size());
    std::vector<uint32_t> triangleCounts;
    triangleCounts.reserve(m_primitiveInfos.size());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
    rangeInfos.reserve(m_primitiveInfos.size());

    for (const PrimitiveInfo& info : m_primitiveInfos)
    {
        VkAccelerationStructureGeometryDataKHR geometryData{};
        geometryData.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        geometryData.triangles.pNext = NULL;
        geometryData.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        geometryData.triangles.vertexData = VkDeviceOrHostAddressConstKHR{vertexBufferDeviceAddress};
        geometryData.triangles.vertexStride = sizeof(Model::Vertex);
        geometryData.triangles.maxVertex = info.maxVertex;
        geometryData.triangles.indexType = VK_INDEX_TYPE_UINT32;
        geometryData.triangles.indexData = VkDeviceOrHostAddressConstKHR{indexBufferDeviceAddress};
        geometryData.triangles.transformData = VkDeviceOrHostAddressConstKHR{0};

        VkAccelerationStructureGeometryKHR geometry{};
        geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
        geometry.pNext = NULL;
        geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometry.geometry = geometryData;
        geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        geometries.push_back(geometry);
        triangleCounts.push_back(info.triangleCount);

        VkAccelerationStructureBuildRangeInfoKHR blasBuildRangeInfo{};
        blasBuildRangeInfo.primitiveCount = info.triangleCount;
        blasBuildRangeInfo.primitiveOffset = info.indexByteOffset;
        blasBuildRangeInfo.firstVertex = 0;
        blasBuildRangeInfo.transformOffset = 0;
        rangeInfos.push_back(blasBuildRangeInfo);
    }

    VkAccelerationStructureBuildGeometryInfoKHR blasBuildGeometryInfo{};
    blasBuildGeometryInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    blasBuildGeometryInfo.pNext = NULL;
    blasBuildGeometryInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    blasBuildGeometryInfo.flags = 0;
    blasBuildGeometryInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    blasBuildGeometryInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    blasBuildGeometryInfo.dstAccelerationStructure = VK_NULL_HANDLE;
    blasBuildGeometryInfo.geometryCount = ui32Size(geometries);
    blasBuildGeometryInfo.pGeometries = geometries.data();
    blasBuildGeometryInfo.ppGeometries = NULL;
    blasBuildGeometryInfo.scratchData = VkDeviceOrHostAddressKHR{0};

    VkAccelerationStructureBuildSizesInfoKHR blasBuildSizesInfo{};
    blasBuildSizesInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    blasBuildSizesInfo.pNext = NULL;
    blasBuildSizesInfo.accelerationStructureSize = 0;
    blasBuildSizesInfo.updateScratchSize = 0;
    blasBuildSizesInfo.buildScratchSize = 0;

    const VkAccelerationStructureBuildTypeKHR buildType = VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR;
    m_pvkGetAccelerationStructureBuildSizesKHR(m_device, buildType, &blasBuildGeometryInfo, triangleCounts.data(), &blasBuildSizesInfo);

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
    VkBuffer blasScratchBuffer;
    VkDeviceMemory blasScratchMemory;

    blasScratchBuffer = createBuffer(m_device, blasBuildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    blasScratchMemory = allocateAndBindMemory(m_device, physicalDevice, blasScratchBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkBufferDeviceAddressInfo blasScratchBufferDeviceAddressInfo{};
    blasScratchBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    blasScratchBufferDeviceAddressInfo.pNext = NULL;
    blasScratchBufferDeviceAddressInfo.buffer = blasScratchBuffer;

    VkDeviceAddress blasScratchBufferDeviceAddress = m_pvkGetBufferDeviceAddressKHR(m_device, &blasScratchBufferDeviceAddressInfo);

    // Build BLAS
    blasBuildGeometryInfo.dstAccelerationStructure = m_blas;
    blasBuildGeometryInfo.scratchData.deviceAddress = blasScratchBufferDeviceAddress;

    const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
    const VkCommandBuffer& cb = command.commandBuffer;
    const VkAccelerationStructureBuildRangeInfoKHR* blasBuildRangeInfos = rangeInfos.data();
    m_pvkCmdBuildAccelerationStructuresKHR(cb, 1, &blasBuildGeometryInfo, &blasBuildRangeInfos);
    endSingleTimeCommands(m_context.getGraphicsQueue(), command);

    destroyBufferAndFreeMemory(m_device, blasScratchBuffer, blasScratchMemory);
}

void Raytracer::createTLAS()
{
    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();

    // Setup BLAS instance buffer
    // clang-format off
    const std::vector<float> matrixData{
        0.01f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.01f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.01f, 0.0f
    };
    // clang-format on
    VkAccelerationStructureInstanceKHR blasInstance{};
    std::memcpy(blasInstance.transform.matrix, matrixData.data(), sizeof(float) * matrixData.size());
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
    VkBuffer tlasScratchBuffer;
    VkDeviceMemory tlasScratchMemory;

    tlasScratchBuffer = createBuffer(m_device, tlasBuildSizesInfo.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    tlasScratchMemory = allocateAndBindMemory(m_device, physicalDevice, tlasScratchBuffer, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkBufferDeviceAddressInfo tlasScratchBufferDeviceAddressInfo{};
    tlasScratchBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    tlasScratchBufferDeviceAddressInfo.pNext = NULL;
    tlasScratchBufferDeviceAddressInfo.buffer = tlasScratchBuffer;

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

    destroyBufferAndFreeMemory(m_device, tlasScratchBuffer, tlasScratchMemory);
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
    uniformDescriptorInfo.buffer = m_commonBuffer;
    uniformDescriptorInfo.offset = 0;
    uniformDescriptorInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo indexDescriptorInfo{};
    indexDescriptorInfo.buffer = m_primitiveIndexBuffer;
    indexDescriptorInfo.offset = 0;
    indexDescriptorInfo.range = VK_WHOLE_SIZE;

    VkDescriptorBufferInfo vertexDescriptorInfo{};
    vertexDescriptorInfo.buffer = m_vertexBuffer;
    vertexDescriptorInfo.offset = 0;
    vertexDescriptorInfo.range = VK_WHOLE_SIZE;

    VkDescriptorImageInfo imageDescriptorInfo{};
    imageDescriptorInfo.sampler = VK_NULL_HANDLE;
    imageDescriptorInfo.imageView = m_colorImageView;
    imageDescriptorInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    // Write sets
    VkWriteDescriptorSet writeAccelerationStructure{};
    writeAccelerationStructure.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeAccelerationStructure.pNext = &accelerationStructureDescriptorInfo;
    writeAccelerationStructure.dstSet = m_commonDescriptorSet;
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
    writeUniformBuffer.dstSet = m_commonDescriptorSet;
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
    writeIndexBuffer.dstSet = m_commonDescriptorSet;
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
    writeVertexBuffer.dstSet = m_commonDescriptorSet;
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
    writeImage.dstSet = m_commonDescriptorSet;
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

void Raytracer::updateMaterialIndexDescriptorSet()
{
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = m_materialIndexBuffer;
    bufferInfo.offset = 0;
    bufferInfo.range = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writeIndexBuffer{};
    writeIndexBuffer.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeIndexBuffer.pNext = NULL;
    writeIndexBuffer.dstSet = m_materialIndexDescriptorSet;
    writeIndexBuffer.dstBinding = 0;
    writeIndexBuffer.dstArrayElement = 0;
    writeIndexBuffer.descriptorCount = 1;
    writeIndexBuffer.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writeIndexBuffer.pImageInfo = NULL;
    writeIndexBuffer.pBufferInfo = &bufferInfo;
    writeIndexBuffer.pTexelBufferView = NULL;

    const std::vector<VkWriteDescriptorSet> writeDescriptorSets{writeIndexBuffer};

    vkUpdateDescriptorSets(m_device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

    // Copy data
    std::vector<MaterialInfo> materialInfo(m_model->primitives.size());
    size_t counter = 0;
    int indexBufferOffset = 0;
    for (const Model::Primitive& primitive : m_model->primitives)
    {
        materialInfo[counter].baseColorTextureIndex = m_model->materials[primitive.material].baseColor;
        materialInfo[counter].normalTextureIndex = m_model->materials[primitive.material].normalImage;
        materialInfo[counter].metallicRoughnessTextureIndex = m_model->materials[primitive.material].metallicRoughnessImage;
        materialInfo[counter].indexBufferOffset = indexBufferOffset;

        indexBufferOffset += primitive.indices.size() / 3;

        // For some materials there's no normal or metallicRoughess, just use some image in that case to avoid crashes
        materialInfo[counter].normalTextureIndex = std::max(materialInfo[counter].normalTextureIndex, 0);
        materialInfo[counter].metallicRoughnessTextureIndex = std::max(materialInfo[counter].metallicRoughnessTextureIndex, 0);
        ++counter;
    }
    CHECK(counter == materialInfo.size());

    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = materialInfo.size() * sizeof(materialInfo[0]);

    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();
    StagingBuffer stagingBuffer = createStagingBuffer(m_device, physicalDevice, materialInfo.data(), copyRegion.size);

    const SingleTimeCommand command = beginSingleTimeCommands(m_context.getGraphicsCommandPool(), m_device);
    vkCmdCopyBuffer(command.commandBuffer, stagingBuffer.buffer, m_materialIndexBuffer, 1, &copyRegion);
    endSingleTimeCommands(m_context.getGraphicsQueue(), command);

    releaseStagingBuffer(m_device, stagingBuffer);
}

void Raytracer::updateTexturesDescriptorSets()
{
    const size_t imageCount = m_images.size();
    std::vector<VkDescriptorImageInfo> imageInfos(imageCount);
    for (size_t i = 0; i < imageCount; ++i)
    {
        VkDescriptorImageInfo& imageInfo = imageInfos[i];
        imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        imageInfo.imageView = m_imageViews[i];
        imageInfo.sampler = m_sampler;
    }

    VkWriteDescriptorSet writeSet{};
    writeSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeSet.dstBinding = 0;
    writeSet.dstArrayElement = 0;
    writeSet.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writeSet.descriptorCount = ui32Size(imageInfos);
    writeSet.pBufferInfo = 0;
    writeSet.dstSet = m_texturesDescriptorSet;
    writeSet.pImageInfo = imageInfos.data();

    vkUpdateDescriptorSets(m_device, 1, &writeSet, 0, nullptr);
}

void Raytracer::createShaderBindingTable()
{
    const VkPhysicalDevice physicalDevice = m_context.getPhysicalDevice();

    VkPhysicalDeviceProperties physicalDeviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);

    VkPhysicalDeviceRayTracingPipelinePropertiesKHR physicalDeviceRayTracingPipelineProperties{};
    physicalDeviceRayTracingPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    physicalDeviceRayTracingPipelineProperties.pNext = NULL;

    VkPhysicalDeviceProperties2 physicalDeviceProperties2{};
    physicalDeviceProperties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    physicalDeviceProperties2.pNext = &physicalDeviceRayTracingPipelineProperties;
    physicalDeviceProperties2.properties = physicalDeviceProperties;

    vkGetPhysicalDeviceProperties2(physicalDevice, &physicalDeviceProperties2);

    const VkDeviceSize shaderGroupHandleSize = static_cast<VkDeviceSize>(physicalDeviceRayTracingPipelineProperties.shaderGroupHandleSize);
    const VkDeviceSize shaderBindingTableSize = shaderGroupHandleSize * c_shaderGroupCount;
    const uint32_t shaderGroupBaseAlignment = physicalDeviceRayTracingPipelineProperties.shaderGroupBaseAlignment;

    const VkBufferUsageFlags usage = VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    m_shaderBindingTableBuffer = createBuffer(m_device, shaderBindingTableSize, usage);
    m_shaderBindingTableMemory = allocateAndBindMemory(m_device, physicalDevice, m_shaderBindingTableBuffer, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);

    std::vector<char> shaderGroupHandlesBuffer(shaderBindingTableSize, 0);
    VK_CHECK(m_pvkGetRayTracingShaderGroupHandlesKHR(m_device, m_pipeline, 0, c_shaderGroupCount, shaderBindingTableSize, shaderGroupHandlesBuffer.data()));

    void* sbtMemoryMapped;
    VK_CHECK(vkMapMemory(m_device, m_shaderBindingTableMemory, 0, shaderBindingTableSize, 0, &sbtMemoryMapped));

    for (uint32_t i = 0; i < c_shaderGroupCount; ++i)
    {
        memcpy(sbtMemoryMapped, &shaderGroupHandlesBuffer[i * shaderGroupHandleSize], shaderGroupHandleSize);
        sbtMemoryMapped = (char*)sbtMemoryMapped + shaderGroupBaseAlignment;
    }

    vkUnmapMemory(m_device, m_shaderBindingTableMemory);

    VkBufferDeviceAddressInfo shaderBindingTableBufferDeviceAddressInfo{};
    shaderBindingTableBufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    shaderBindingTableBufferDeviceAddressInfo.pNext = NULL;
    shaderBindingTableBufferDeviceAddressInfo.buffer = m_shaderBindingTableBuffer;

    const VkDeviceAddress shaderBindingTableBufferDeviceAddress = m_pvkGetBufferDeviceAddressKHR(m_device, &shaderBindingTableBufferDeviceAddressInfo);

    const VkDeviceSize groupSize = c_shaderCount * shaderGroupBaseAlignment;

    m_rchitShaderBindingTable.deviceAddress = shaderBindingTableBufferDeviceAddress + 0 * shaderGroupBaseAlignment;
    m_rchitShaderBindingTable.stride = shaderGroupBaseAlignment;
    m_rchitShaderBindingTable.size = groupSize * 1;

    m_rgenShaderBindingTable.deviceAddress = shaderBindingTableBufferDeviceAddress + 1 * shaderGroupBaseAlignment;
    m_rgenShaderBindingTable.stride = groupSize;
    m_rgenShaderBindingTable.size = groupSize * 1;

    m_rmissShaderBindingTable.deviceAddress = shaderBindingTableBufferDeviceAddress + 2 * shaderGroupBaseAlignment;
    m_rmissShaderBindingTable.stride = shaderGroupBaseAlignment;
    m_rmissShaderBindingTable.size = groupSize * 2;
}
