#include "DebugMarker.hpp"
#include "Utils.hpp"

VkInstance DebugMarker::s_instance;
VkDevice DebugMarker::s_device;
bool DebugMarker::s_initialized = false;

PFN_vkCmdBeginDebugUtilsLabelEXT DebugMarker::vkCmdBeginDebugUtilsLabelEXT;
PFN_vkCmdEndDebugUtilsLabelEXT DebugMarker::vkCmdEndDebugUtilsLabelEXT;
PFN_vkSetDebugUtilsObjectNameEXT DebugMarker::vkSetDebugUtilsObjectNameEXT;

void DebugMarker::initialize(VkInstance instance, VkDevice device)
{
    s_instance = instance;
    s_device = device;
    vkCmdBeginDebugUtilsLabelEXT = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(s_instance, "vkCmdBeginDebugUtilsLabelEXT");
    vkCmdEndDebugUtilsLabelEXT = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(s_instance, "vkCmdEndDebugUtilsLabelEXT");
    vkSetDebugUtilsObjectNameEXT = (PFN_vkSetDebugUtilsObjectNameEXT)vkGetInstanceProcAddr(s_instance, "vkSetDebugUtilsObjectNameEXT");
    s_initialized = true;
}

void DebugMarker::beginLabel(VkCommandBuffer cb, const std::string& name, std::array<float, 4> color)
{
    CHECK(s_initialized);

    VkDebugUtilsLabelEXT beginDebug{};
    beginDebug.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    beginDebug.pLabelName = name.c_str();
    for (size_t i = 0; i < color.size(); ++i)
    {
        beginDebug.color[i] = color[i];
    }
    vkCmdBeginDebugUtilsLabelEXT(cb, &beginDebug);
}

void DebugMarker::endLabel(VkCommandBuffer cb)
{
    CHECK(s_initialized);

    vkCmdEndDebugUtilsLabelEXT(cb);
}

void DebugMarker::setObjectName(VkObjectType type, uint64_t handle, const std::string& name)
{
    CHECK(s_initialized);

    VkDebugUtilsObjectNameInfoEXT nameInfo{};
    nameInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    nameInfo.objectType = type;
    nameInfo.objectHandle = handle;
    nameInfo.pObjectName = name.c_str();
    vkSetDebugUtilsObjectNameEXT(s_device, &nameInfo);
}
