#pragma once

#include "three_body_physics.h"

#include <array>
#include <string>
#include <vector>

struct GLFWwindow;

namespace tableau::demos {

struct CameraState {
    float yaw_degrees{40.0F};
    float pitch_degrees{25.0F};
    float distance{7.0F};
    Vec3 target{};
};

struct IntegratorRenderData {
    bool visible{true};
    std::array<float, 3> color{0.9F, 0.9F, 0.9F};
    std::array<Vec3, 3> body_positions{};
    std::array<std::vector<Vec3>, 3> trails{};
};

class OpenGLRenderer {
public:
    OpenGLRenderer() = default;
    ~OpenGLRenderer();

    bool initialize(int width, int height, const char* title);
    void shutdown();

    bool shouldClose() const;
    void pollEvents() const;

    void beginFrame();
    void drawFrame(
        const std::array<IntegratorRenderData, 3>& integrator_data,
        const CameraState& camera_state,
        int focused_integrator);
    void endFrame();

    GLFWwindow* window() const { return window_; }
    void setWindowTitle(const std::string& title) const;
    void setShowGrid(bool enabled) { show_grid_ = enabled; }

private:
    bool createProgram();
    void destroyProgram();
    void ensureBuffers();
    void generateStars();

    void drawGrid(const std::array<float, 16>& mvp);
    void drawStars(const std::array<float, 16>& mvp);
    void drawLineStrip(
        const std::vector<Vec3>& points,
        const std::array<float, 3>& color,
        const std::array<float, 16>& mvp);
    void drawPoints(
        const std::array<Vec3, 3>& points,
        const std::array<float, 3>& color,
        const std::array<float, 16>& mvp,
        float point_size);
    void drawSuns(
        const std::array<Vec3, 3>& positions,
        const std::array<float, 3>& color,
        const std::array<float, 16>& mvp,
        bool is_focused);

private:
    GLFWwindow* window_{nullptr};

    unsigned int shader_program_{0};
    unsigned int vao_{0};
    unsigned int vbo_{0};

    int u_mvp_{-1};
    int u_color_{-1};
    int u_point_size_{-1};
    int u_sun_mode_{-1};

    bool show_grid_{false};
    std::vector<Vec3> stars_near_{};
    std::vector<Vec3> stars_far_{};
};

} // namespace tableau::demos
