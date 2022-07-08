#pragma once

#include "Utils.hpp"
#include <glm/glm.hpp>

class Camera final
{
public:
    Camera();

    glm::vec3 getPosition() const;
    glm::vec3 getForward() const;
    glm::vec3 getLeft() const;
    glm::vec3 getUp() const;
    void setPosition(const glm::vec3& pos);
    void setRotation(const glm::vec3& rot);
    void translate(const glm::vec3& translation);
    void rotate(const glm::vec3& axis, float amount);

    const glm::mat4x4& getViewMatrix() const;
    const glm::mat4x4& getProjectionMatrix() const;

private:
    void updateViewMatrix();

    glm::vec3 m_position = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 m_rotation = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 m_forward = c_forward;
    glm::vec3 m_up = c_up;
    glm::vec3 m_left = c_left;
    glm::mat4 m_viewMatrix;
    glm::mat4 m_projectionMatrix;
};
