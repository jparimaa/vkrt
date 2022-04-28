#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <filesystem>
#include <string>

#define CHECK(f)                                                           \
    do                                                                     \
    {                                                                      \
        if (!(f))                                                          \
        {                                                                  \
            printf("Abort. %s failed at %s:%d\n", #f, __FILE__, __LINE__); \
            abort();                                                       \
        }                                                                  \
    } while (false)

#define LOGE(f)                                                \
    do                                                         \
    {                                                          \
        printf("ERROR: %s at %s:%d\n", f, __FILE__, __LINE__); \
        abort();                                               \
    } while (false)

#define LOGW(f)                                                  \
    do                                                           \
    {                                                            \
        printf("WARNING: %s at %s:%d\n", f, __FILE__, __LINE__); \
    } while (false)

const std::string c_modelsFolder = MODELS_FOLDER;
const int c_windowWidth = 1600;
const int c_windowHeight = 1200;

const glm::vec3 c_forward(0.0f, 0.0f, -1.0f);
const glm::vec4 c_forwardZero(c_forward.x, c_forward.y, c_forward.z, 0.0f);
const glm::vec3 c_backward(0.0f, 0.0f, 1.0f);
const glm::vec3 c_up(0.0f, 1.0f, 0.0f);
const glm::vec4 c_upZero(c_up.x, c_up.y, c_up.z, 0.0f);
const glm::vec3 c_down(0.0f, -1.0f, 0.0f);
const glm::vec3 c_left(-1.0f, 0.0f, 0.0f);
const glm::vec4 c_leftZero(c_left.x, c_left.y, c_left.z, 0.0f);
const glm::vec3 c_right(1.0f, 0.0f, 0.0f);

template<typename T>
uint32_t ui32Size(const T& container)
{
    return static_cast<uint32_t>(container.size());
}

std::filesystem::path getCurrentExecutableDirectory();
