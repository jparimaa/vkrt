#pragma once

#include <vulkan/vulkan.h>
#include <array>
#include <string>

class DebugMarker final
{
public:
    inline static const std::array<float, 4> red{0.9f, 0.7f, 0.7f, 1.0f};
    inline static const std::array<float, 4> green{0.7f, 0.9f, 0.7f, 1.0f};
    inline static const std::array<float, 4> blue{0.7f, 0.7f, 0.9f, 1.0f};
    inline static const std::array<float, 4> white{1.0f, 1.0f, 1.0f, 1.0f};

    DebugMarker() = delete;

    static void initialize(VkInstance instance, VkDevice device);

    static void beginLabel(VkCommandBuffer cb, const std::string& name, std::array<float, 4> color = white);
    static void endLabel(VkCommandBuffer cb);
    static void setObjectName(VkObjectType type, uint64_t handle, const std::string& name);

private:
    static VkInstance s_instance;
    static VkDevice s_device;
    static bool s_initialized;

    static PFN_vkCmdBeginDebugUtilsLabelEXT vkCmdBeginDebugUtilsLabelEXT;
    static PFN_vkCmdEndDebugUtilsLabelEXT vkCmdEndDebugUtilsLabelEXT;
    static PFN_vkSetDebugUtilsObjectNameEXT vkSetDebugUtilsObjectNameEXT;
};
