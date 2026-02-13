#pragma once

#include "black_hole_physics.h"
#include "opengl_renderer.h"

#include <array>
#include <string>
#include <vector>

struct GLFWwindow;

namespace tableau::demos {

struct BlackHoleRenderData {
    bool visible{true};
    std::array<float, 3> color{0.9F, 0.9F, 0.9F};
    std::vector<Vec3> particles{};
    std::vector<std::vector<Vec3>> trails{};
};

class BlackHoleRenderer {
public:
    BlackHoleRenderer() = default;
    ~BlackHoleRenderer();

    bool initialize(int width, int height, const char* title);
    void shutdown();

    bool shouldClose() const;
    void pollEvents() const;

    void beginFrame();
    void drawFrame(
        const std::array<BlackHoleRenderData, 3>& clouds,
        const CameraState& camera,
        int focused_integrator,
        const BlackHoleParams& params,
        double sim_time);
    void endFrame();

    void setWindowTitle(const std::string& title) const;
    GLFWwindow* window() const { return window_; }

    void setShowGrid(bool enabled) { show_grid_ = enabled; }
    void setShowAccretionDisk(bool enabled) { show_disk_ = enabled; }
    void setShowLensing(bool enabled) { show_lensing_ = enabled; }
    void setLightBackground(bool enabled) { light_background_ = enabled; }
    bool lightBackground() const { return light_background_; }

private:
    bool createProgram();
    void destroyProgram();
    void ensureBuffers();
    void generateStars();

    void drawStars(const std::array<float, 16>& mvp);
    void drawGrid(const std::array<float, 16>& mvp);
    void drawCloud(
        const BlackHoleRenderData& cloud,
        const std::array<float, 16>& mvp,
        bool focused,
        float lensing_radius,
        bool lensing_enabled);
    void drawAccretionDisk(const std::array<float, 16>& mvp, const BlackHoleParams& params, double sim_time);
    void drawHorizon(const std::array<float, 16>& mvp, const BlackHoleParams& params);

private:
    GLFWwindow* window_{nullptr};

    unsigned int shader_program_{0};
    unsigned int vao_{0};
    unsigned int vbo_{0};

    int u_mvp_{-1};
    int u_color_{-1};
    int u_point_size_{-1};
    int u_mode_{-1};
    int u_lensing_enabled_{-1};
    int u_lensing_radius_{-1};

    bool show_grid_{false};
    bool show_disk_{true};
    bool show_lensing_{true};
    bool light_background_{false};

    std::vector<Vec3> stars_near_{};
    std::vector<Vec3> stars_far_{};
};

} // namespace tableau::demos
