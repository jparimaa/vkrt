cmake_minimum_required(VERSION 3.18)
project(imgui)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sources, exe
set(_imgui_dir "${CMAKE_CURRENT_SOURCE_DIR}/../imgui")
set(_target "imgui")
add_library(${_target}
    ${_imgui_dir}/backends/imgui_impl_glfw.cpp
    ${_imgui_dir}/backends/imgui_impl_vulkan.cpp
    ${_imgui_dir}/imgui.cpp
    ${_imgui_dir}/imgui_draw.cpp
    ${_imgui_dir}/imgui_demo.cpp
    ${_imgui_dir}/imgui_tables.cpp
    ${_imgui_dir}/imgui_widgets.cpp
)

# Includes, libraries, compile options
find_package(Vulkan REQUIRED)
target_include_directories(${_target} PUBLIC ${_imgui_dir} ${_imgui_dir}/backends)
set(_submodule_dir "${CMAKE_CURRENT_SOURCE_DIR}/../")
target_include_directories(${_target} PRIVATE ${_submodule_dir}/glfw/include/ ${Vulkan_INCLUDE_DIRS})
