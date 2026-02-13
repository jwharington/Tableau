#include "golf_physics.h"
#include "golf_renderer.h"
#include "golf_runner.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace tableau::demos {
namespace {

constexpr double kPi = 3.14159265358979323846;

class KeyEdgeTracker {
public:
    KeyEdgeTracker() {
        states_.fill(false);
    }

    bool pressed(GLFWwindow* window, int key) {
        if (key < 0 || key >= static_cast<int>(states_.size())) {
            return false;
        }

        const bool is_down = glfwGetKey(window, key) == GLFW_PRESS;
        const bool fire = is_down && !states_[static_cast<std::size_t>(key)];
        states_[static_cast<std::size_t>(key)] = is_down;
        return fire;
    }

private:
    std::array<bool, GLFW_KEY_LAST + 1> states_{};
};

class MouseEdgeTracker {
public:
    void update(GLFWwindow* window) {
        prev_left_ = left_down_;
        left_down_ = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    }

    bool leftPressed() const { return left_down_ && !prev_left_; }
    bool leftReleased() const { return !left_down_ && prev_left_; }
    bool leftDown() const { return left_down_; }

private:
    bool left_down_{false};
    bool prev_left_{false};
};

struct ScrollState {
    double yoffset{0.0};
};

double radians(double degrees) {
    return degrees * (kPi / 180.0);
}

Vec3 crossVec(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

Vec3 normalizeVec(const Vec3& v) {
    const double n = norm(v);
    if (n <= 1e-12) {
        return Vec3{};
    }
    return v * (1.0 / n);
}

void cameraBasis(const GolfCameraState& camera, Vec3& forward, Vec3& right, Vec3& up) {
    const double yaw = radians(static_cast<double>(camera.yaw_degrees));
    const double pitch = radians(static_cast<double>(camera.pitch_degrees));

    const Vec3 eye{
        camera.target.x + camera.distance * std::cos(pitch) * std::cos(yaw),
        camera.target.y + camera.distance * std::cos(pitch) * std::sin(yaw),
        camera.target.z + camera.distance * std::sin(pitch)};

    forward = normalizeVec(camera.target - eye);
    right = normalizeVec(crossVec(forward, Vec3{0.0, 0.0, 1.0}));
    up = normalizeVec(crossVec(right, forward));
}

bool screenToPlane(
    double mouse_x,
    double mouse_y,
    int width,
    int height,
    const GolfCameraState& camera,
    Vec3& out_point) {

    if (width <= 0 || height <= 0) {
        return false;
    }

    const double aspect = static_cast<double>(width) / static_cast<double>(height);
    const double fov = radians(50.0);
    const double tan_half = std::tan(fov * 0.5);

    const double ndc_x = (2.0 * mouse_x / static_cast<double>(width)) - 1.0;
    const double ndc_y = 1.0 - (2.0 * mouse_y / static_cast<double>(height));

    Vec3 forward{};
    Vec3 right{};
    Vec3 up{};
    cameraBasis(camera, forward, right, up);

    const Vec3 view_dir{
        ndc_x * tan_half * aspect,
        ndc_y * tan_half,
        -1.0};

    Vec3 ray_dir = normalizeVec(right * view_dir.x + up * view_dir.y + forward * (-view_dir.z));

    const double yaw = radians(static_cast<double>(camera.yaw_degrees));
    const double pitch = radians(static_cast<double>(camera.pitch_degrees));

    const Vec3 eye{
        camera.target.x + camera.distance * std::cos(pitch) * std::cos(yaw),
        camera.target.y + camera.distance * std::cos(pitch) * std::sin(yaw),
        camera.target.z + camera.distance * std::sin(pitch)};

    if (std::abs(ray_dir.z) <= 1e-8) {
        return false;
    }

    const double t = -eye.z / ray_dir.z;
    if (t <= 0.0) {
        return false;
    }

    out_point = eye + ray_dir * t;
    out_point.z = 0.0;
    return true;
}

std::string buildHudTitle(
    double fps,
    const char* club_name,
    int strokes,
    double distance_to_hole,
    double last_dt,
    bool paused,
    bool in_hole) {

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "golf_demo | FPS " << fps;
    oss << " | club " << club_name;
    oss << " | strokes " << strokes;
    oss << " | dist " << distance_to_hole;
    oss << " | dt " << std::scientific << std::setprecision(2) << last_dt;
    oss << " | " << (paused ? "PAUSED" : "RUNNING");
    if (in_hole) {
        oss << " | IN HOLE";
    }
    return oss.str();
}

} // namespace
} // namespace tableau::demos

int main() {
    using namespace tableau::demos;

    GolfRenderer renderer;
    if (!renderer.initialize(1400, 900, "golf_demo")) {
        return 1;
    }

    std::cout << "Controls:\n"
              << "  Left Drag: aim + power\n"
              << "  Right Drag: rotate camera\n"
              << "  Release: shoot\n"
              << "  1/2/3: select club (integrator)\n"
              << "  R: reset (tee + clear trails)\n"
              << "  Space: pause\n"
              << "  Mouse Wheel: zoom\n"
              << "  Esc: exit\n";

    GolfParams params{};

    const Vec3 tee_pos{0.0, 0.0, 0.0};
    const Vec3 hole_pos{42.0, 18.0, 0.0};
    constexpr double kHoleRadius = 1.6;

    const std::array<std::array<float, 3>, 3> club_colors = {
        std::array<float, 3>{0.82F, 0.58F, 0.46F},
        std::array<float, 3>{0.56F, 0.72F, 0.56F},
        std::array<float, 3>{0.56F, 0.68F, 0.84F},
    };

    GolfState initial_state{};
    setGolfPosition(initial_state, tee_pos);

    std::array<GolfIntegratorRunner, 3> runners = {
        GolfIntegratorRunner{GolfIntegratorKind::RK4, initial_state, params},
        GolfIntegratorRunner{GolfIntegratorKind::RKF45, initial_state, params},
        GolfIntegratorRunner{GolfIntegratorKind::DOP853, initial_state, params},
    };

    std::array<std::vector<Vec3>, 3> trails;

    bool paused = false;
    bool dragging = false;
    bool shot_active = false;
    bool ball_in_hole = false;
    int strokes = 0;

    int selected_club = 0;
    int active_club = -1;

    Vec3 ball_rest = tee_pos;
    Vec3 aim_point{};
    bool aim_valid = false;

    constexpr double kTick = 1.0 / 120.0;
    constexpr std::size_t kTrailLimit = 1200;
    constexpr double kMinDrag = 0.8;
    constexpr double kMaxPower = 55.0;
    constexpr double kPowerScale = 0.85;
    constexpr double kElevation = 0.52; // radians

    GolfCameraState camera;
    camera.yaw_degrees = 45.0F;
    camera.pitch_degrees = 35.0F;
    camera.distance = 62.0F;
    camera.target = ball_rest;

    KeyEdgeTracker keys;
    MouseEdgeTracker mouse_edges;
    ScrollState scroll_state{};

    glfwSetWindowUserPointer(renderer.window(), &scroll_state);
    glfwSetScrollCallback(renderer.window(), [](GLFWwindow* window, double /*xoffset*/, double yoffset) {
        auto* state = static_cast<ScrollState*>(glfwGetWindowUserPointer(window));
        if (state != nullptr) {
            state->yoffset += yoffset;
        }
    });

    double last_wall_time = glfwGetTime();
    double fps_smoothed = 0.0;
    double sim_accumulator = 0.0;
    double title_refresh_accum = 0.0;
    bool right_drag_active = false;
    double right_prev_x = 0.0;
    double right_prev_y = 0.0;

    while (!renderer.shouldClose()) {
        renderer.pollEvents();
        GLFWwindow* window = renderer.window();

        const double now = glfwGetTime();
        double wall_dt = now - last_wall_time;
        last_wall_time = now;
        wall_dt = std::clamp(wall_dt, 0.0, 0.25);

        const double inst_fps = (wall_dt > 1e-8) ? (1.0 / wall_dt) : 0.0;
        if (fps_smoothed <= 0.0) {
            fps_smoothed = inst_fps;
        } else {
            fps_smoothed = 0.90 * fps_smoothed + 0.10 * inst_fps;
        }

        if (keys.pressed(window, GLFW_KEY_ESCAPE)) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (keys.pressed(window, GLFW_KEY_SPACE)) {
            paused = !paused;
        }
        if (keys.pressed(window, GLFW_KEY_R)) {
            ball_rest = tee_pos;
            shot_active = false;
            ball_in_hole = false;
            active_club = -1;
            strokes = 0;
            for (auto& trail : trails) {
                trail.clear();
            }
        }

        if (keys.pressed(window, GLFW_KEY_1)) {
            selected_club = 0;
        }
        if (keys.pressed(window, GLFW_KEY_2)) {
            selected_club = 1;
        }
        if (keys.pressed(window, GLFW_KEY_3)) {
            selected_club = 2;
        }

        if (std::abs(scroll_state.yoffset) > 1e-9) {
            camera.distance = static_cast<float>(camera.distance * std::pow(0.88, scroll_state.yoffset));
            scroll_state.yoffset = 0.0;
        }
        camera.distance = std::clamp(camera.distance, 18.0F, 180.0F);

        double cursor_x = 0.0;
        double cursor_y = 0.0;
        glfwGetCursorPos(window, &cursor_x, &cursor_y);

        const bool right_down = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        if (right_down) {
            if (!right_drag_active) {
                right_drag_active = true;
                right_prev_x = cursor_x;
                right_prev_y = cursor_y;
            } else {
                const double dx = cursor_x - right_prev_x;
                const double dy = cursor_y - right_prev_y;
                right_prev_x = cursor_x;
                right_prev_y = cursor_y;

                camera.yaw_degrees -= static_cast<float>(dx * 0.22);
                camera.pitch_degrees -= static_cast<float>(dy * 0.16);
            }
        } else {
            right_drag_active = false;
        }

        camera.pitch_degrees = std::clamp(camera.pitch_degrees, 8.0F, 80.0F);

        mouse_edges.update(window);
        if (mouse_edges.leftPressed() && !shot_active) {
            dragging = true;
        }

        int fb_width = 0;
        int fb_height = 0;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);

        if (dragging) {
            aim_valid = screenToPlane(cursor_x, cursor_y, fb_width, fb_height, camera, aim_point);
        } else {
            aim_valid = false;
        }

        if (mouse_edges.leftReleased() && dragging) {
            dragging = false;

            if (aim_valid && !shot_active) {
                Vec3 dir = aim_point - ball_rest;
                dir.z = 0.0;
                const double dist = norm(dir);

                if (dist >= kMinDrag) {
                    const Vec3 dir_n = dir * (1.0 / dist);
                    const double power = std::clamp(dist * kPowerScale, 1.0, kMaxPower);
                    const double vxy = power * std::cos(kElevation);
                    const double vz = power * std::sin(kElevation);

                    Vec3 vel = dir_n * vxy;
                    vel.z = vz;

                    GolfState shot_state{};
                    setGolfPosition(shot_state, ball_rest);
                    setGolfVelocity(shot_state, vel);

                    runners[static_cast<std::size_t>(selected_club)].reset(shot_state, 0.0);
                    trails[static_cast<std::size_t>(selected_club)].clear();
                    trails[static_cast<std::size_t>(selected_club)].push_back(ball_rest);

                    active_club = selected_club;
                    shot_active = true;
                    ball_in_hole = false;
                    strokes += 1;
                }
            }
        }

        Vec3 current_ball = ball_rest;
        double last_dt = 0.0;
        if (shot_active && active_club >= 0) {
            if (!paused) {
                sim_accumulator += wall_dt;
                sim_accumulator = std::min(sim_accumulator, 0.5);

                auto& runner = runners[static_cast<std::size_t>(active_club)];

                while (sim_accumulator >= kTick) {
                    const bool ok = runner.stepTo(runner.time() + kTick);
                    if (!ok) {
                        paused = true;
                        std::cerr << "golf_demo: integration failed\n";
                        break;
                    }

                    const Vec3 pos = golfPosition(runner.state());
                    trails[static_cast<std::size_t>(active_club)].push_back(pos);
                    if (trails[static_cast<std::size_t>(active_club)].size() > kTrailLimit) {
                        trails[static_cast<std::size_t>(active_club)].erase(
                            trails[static_cast<std::size_t>(active_club)].begin(),
                            trails[static_cast<std::size_t>(active_club)].begin() +
                                (trails[static_cast<std::size_t>(active_club)].size() - kTrailLimit));
                    }

                    if (isStopped(runner.state())) {
                        shot_active = false;
                        ball_rest = pos;
                        const Vec3 diff = Vec3{ball_rest.x - hole_pos.x, ball_rest.y - hole_pos.y, 0.0};
                        ball_in_hole = (norm(diff) <= kHoleRadius);
                        sim_accumulator = 0.0;
                        break;
                    }

                    sim_accumulator -= kTick;
                }
            }

            current_ball = golfPosition(runners[static_cast<std::size_t>(active_club)].state());
            last_dt = runners[static_cast<std::size_t>(active_club)].lastDt();
        } else {
            sim_accumulator = 0.0;
        }

        camera.target = current_ball;

        const Vec3 diff = Vec3{current_ball.x - hole_pos.x, current_ball.y - hole_pos.y, 0.0};
        const double distance_to_hole = norm(diff);

        GolfRenderData render_data{};
        for (std::size_t i = 0; i < trails.size(); ++i) {
            render_data.trails[i].visible = !trails[i].empty();
            render_data.trails[i].color = club_colors[i];
            render_data.trails[i].trail = trails[i];
        }

        render_data.ball_visible = true;
        render_data.ball_pos = current_ball;
        const int color_index = (active_club >= 0) ? active_club : selected_club;
        render_data.ball_color = club_colors[static_cast<std::size_t>(color_index)];
        render_data.hole_pos = hole_pos;
        render_data.hole_radius = static_cast<float>(kHoleRadius);

        if (dragging && aim_valid) {
            render_data.show_aim = true;
            render_data.aim_line = {ball_rest, aim_point};
        }

        renderer.beginFrame();
        renderer.drawFrame(render_data, camera);
        renderer.endFrame();

        title_refresh_accum += wall_dt;
        if (title_refresh_accum >= 0.12) {
            const char* club_name = golfIntegratorName(static_cast<GolfIntegratorKind>(selected_club));
            renderer.setWindowTitle(
                buildHudTitle(
                    fps_smoothed,
                    club_name,
                    strokes,
                    distance_to_hole,
                    last_dt,
                    paused,
                    ball_in_hole));
            title_refresh_accum = 0.0;
        }
    }

    return 0;
}
