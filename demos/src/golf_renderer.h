#pragma once

#include "golf_physics.h"

#include <array>
#include <string>
#include <vector>

struct GLFWwindow;

namespace tableau::demos {

struct GolfCameraState {
    float yaw_degrees{45.0F};
    float pitch_degrees{35.0F};
    float distance{55.0F};
    Vec3 target{};
};

struct GolfTrailData {
    bool visible{true};
    std::array<float, 3> color{0.9F, 0.9F, 0.9F};
    std::vector<Vec3> trail{};
};

struct GolfRenderData {
    std::array<GolfTrailData, 3> trails{};
    bool show_aim{false};
    std::vector<Vec3> aim_line{};
    bool ball_visible{false};
    Vec3 ball_pos{};
    std::array<float, 3> ball_color{1.0F, 1.0F, 1.0F};
    Vec3 hole_pos{};
    float hole_radius{1.0F};
};

class GolfRenderer {
public:
    GolfRenderer() = default;
    ~GolfRenderer();

    bool initialize(int width, int height, const char* title);
    void shutdown();

    bool shouldClose() const;
    void pollEvents() const;

    void beginFrame();
    void drawFrame(const GolfRenderData& data, const GolfCameraState& camera_state);
    void endFrame();

    void setWindowTitle(const std::string& title) const;
    GLFWwindow* window() const { return window_; }

private:
    bool createProgram();
    bool createSphereProgram();
    void destroyProgram();
    void destroySphereProgram();
    void ensureBuffers();
    void ensureSphereBuffers();
    void buildGrid();
    void buildSphereMesh();

    void drawGround(const std::array<float, 16>& mvp);
    void drawGrid(const std::array<float, 16>& mvp);
    void drawHole(const std::array<float, 16>& mvp, const Vec3& center, float radius);
    void drawTrail(const std::vector<Vec3>& points, const std::array<float, 3>& color, const std::array<float, 16>& mvp);
    void drawLine(const std::vector<Vec3>& points, const std::array<float, 3>& color, float alpha, const std::array<float, 16>& mvp);
    void drawSphere(const Vec3& center, float radius, const std::array<float, 16>& mvp);

private:
    GLFWwindow* window_{nullptr};

    unsigned int shader_program_{0};
    unsigned int vao_{0};
    unsigned int vbo_{0};
    unsigned int sphere_program_{0};
    unsigned int sphere_vao_{0};
    unsigned int sphere_vbo_{0};
    unsigned int sphere_ebo_{0};

    int u_mvp_{-1};
    int u_color_{-1};
    int u_point_size_{-1};
    int u_mode_{-1};
    int su_mvp_{-1};
    int su_color_{-1};
    int su_light_dir_{-1};

    std::vector<float> grid_vertices_{};
    std::vector<float> ground_vertices_{};
    std::vector<float> sphere_vertices_{}; // xyz + nx ny nz
    std::vector<unsigned int> sphere_indices_{};
};

} // namespace tableau::demos
