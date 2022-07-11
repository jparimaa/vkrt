cmake_minimum_required(VERSION 3.18)
project(vkrt)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sources, exe
set(_src_dir "${CMAKE_CURRENT_SOURCE_DIR}/src")
file(GLOB _source_list "${_src_dir}/*.cpp" "${_src_dir}/*.hpp")
set(_target "vkrt")
add_executable(${_target} ${_source_list})

# Includes, libraries, compile options
find_package(Vulkan REQUIRED)
add_subdirectory(submodules/glfw)
set(TINYGLTF_HEADER_ONLY OFF CACHE INTERNAL "" FORCE)
set(TINYGLTF_INSTALL OFF CACHE INTERNAL "" FORCE)
set(TINYGLTF_BUILD_LOADER_EXAMPLE OFF CACHE INTERNAL "" FORCE)
add_subdirectory(submodules/tinygltf)
add_subdirectory(submodules/glm)
add_subdirectory(submodules/imgui_cmake)
target_include_directories(${_target} PRIVATE ${_src_dir} ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${_target} PRIVATE glfw tinygltf ${Vulkan_LIBRARIES} glm::glm imgui)
target_compile_options(${_target} PRIVATE "/wd26812")
target_compile_definitions(${_target} PRIVATE MODELS_FOLDER="${CMAKE_CURRENT_SOURCE_DIR}/models/")

# Shaders
function(add_shader TARGET SHADER)
    find_program(GLSLC glslc)

    set(_shader_src_path ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER})
    set(_shader_output_path ${CMAKE_BINARY_DIR}/shaders/${SHADER}.spv)

    get_filename_component(_shader_output_dir ${_shader_output_path} DIRECTORY)
    file(MAKE_DIRECTORY ${_shader_output_dir})

    add_custom_command(
           OUTPUT ${_shader_output_path}
           COMMAND ${GLSLC} --target-env=vulkan1.2 -o ${_shader_output_path} ${_shader_src_path}
           DEPENDS ${_shader_src_path}
           IMPLICIT_DEPENDS CXX ${_shader_src_path}
           VERBATIM)

    set_source_files_properties(${_shader_output_path} PROPERTIES GENERATED TRUE)
    target_sources(${TARGET} PRIVATE ${_shader_output_path})
    add_custom_command(
        TARGET ${TARGET} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy ${_shader_output_path} $<TARGET_FILE_DIR:${TARGET}>/${SHADER}.spv
      )
endfunction(add_shader)

file(GLOB _shader_list "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*")
foreach(_shader ${_shader_list})
    get_filename_component(_shader_filename ${_shader} NAME)
    add_shader(${_target} ${_shader_filename})
endforeach()

add_custom_target(shaders SOURCES  ${_shader_list}) # Just for grouping shaders in VS