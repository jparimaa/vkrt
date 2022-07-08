#include "Utils.hpp"
#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

glm::vec4 toVec4(glm::vec3 v, float w)
{
    return glm::vec4(v.x, v.y, v.x, w);
}

std::filesystem::path getCurrentExecutableDirectory()
{
    char path[260] = {0};
#ifdef _WIN32
    if (GetModuleFileNameA(nullptr, path, _countof(path)) == 0)
        return "";
#else
    // /proc/self/exe is mostly linux-only, but can't hurt to try it elsewhere
    if (readlink("/proc/self/exe", path, std::size(path)) <= 0)
    {
        // portable but assumes executable dir == cwd
        if (!getcwd(path, std::size(path)))
            return ""; // failure
    }
#endif

    std::filesystem::path result = path;
    result = result.parent_path();
    return result;
}