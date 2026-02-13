#include "integrator_runner.h"
#include "metrics.h"
#include "opengl_renderer.h"
#include "presets.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
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
        return {0.0, 0.0, 0.0};
    }
    return v * (1.0 / n);
}

void cameraBasis(const CameraState& camera, Vec3& right, Vec3& up) {
    const double yaw = radians(static_cast<double>(camera.yaw_degrees));
    const double pitch = radians(static_cast<double>(camera.pitch_degrees));

    const Vec3 eye{
        camera.target.x + camera.distance * std::cos(pitch) * std::cos(yaw),
        camera.target.y + camera.distance * std::cos(pitch) * std::sin(yaw),
        camera.target.z + camera.distance * std::sin(pitch)};

    const Vec3 forward = normalizeVec(camera.target - eye);
    right = normalizeVec(crossVec(forward, Vec3{0.0, 0.0, 1.0}));
    up = normalizeVec(crossVec(right, forward));
}

struct SimulationEntry {
    IntegratorKind kind;
    std::string name;
    std::array<float, 3> color{};
    bool visible{true};

    IntegratorRunner runner;
    ConservedQuantities baseline{};
    DriftMetrics drift{};

    std::array<std::deque<Vec3>, 3> trails{};

    SimulationEntry(
        IntegratorKind in_kind,
        std::array<float, 3> in_color,
        const State3B& initial_state)
        : kind(in_kind),
          name(integratorName(in_kind)),
          color(in_color),
          visible(true),
          runner(in_kind, initial_state) {}
};

void appendTrailSample(SimulationEntry& sim, std::size_t max_samples) {
    for (std::size_t i = 0; i < State3B::kBodies; ++i) {
        sim.trails[i].push_back(position(sim.runner.state(), i));
        if (sim.trails[i].size() > max_samples) {
            sim.trails[i].pop_front();
        }
    }
}

std::string buildHudTitle(
    double fps,
    double sim_speed,
    bool paused,
    bool chaos,
    bool follow_center,
    bool grid,
    const std::array<SimulationEntry, 3>& sims) {

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "three_body_demo | FPS " << fps;
    oss << " | speed x" << sim_speed;
    oss << " | " << (paused ? "PAUSED" : "RUNNING");
    oss << " | chaos " << (chaos ? "ON" : "OFF");
    oss << " | follow " << (follow_center ? "ON" : "OFF");
    oss << " | grid " << (grid ? "ON" : "OFF");

    oss << std::scientific << std::setprecision(2);
    for (const auto& sim : sims) {
        oss << " | " << sim.name
            << " dE=" << sim.drift.relative_energy_drift
            << " dP=" << sim.drift.momentum_drift
            << " dt=" << sim.runner.lastDt();
    }

    return oss.str();
}

} // namespace
} // namespace tableau::demos

int main() {
    using namespace tableau::demos;

    OpenGLRenderer renderer;
    if (!renderer.initialize(1400, 900, "three_body_demo")) {
        return 1;
    }

    std::cout << "Controls:\n"
              << "  Space: pause/resume\n"  
              << "  R: reset\n"
              << "  P: toggle chaos perturbation + reset\n"
              << "  [: speed down\n"
              << "  ]: speed up\n"
              << "  F: toggle camera follow mode\n"
              << "  G: toggle helper grid\n"
              << "  1/2/3: focus integrator; press same key again to toggle visibility\n"
              << "  0: clear focus\n"
              << "  A/D: orbit yaw, W/S: orbit pitch, Q/E: zoom\n"
              << "  Mouse Right Drag: orbit | Shift+Left or Middle Drag: pan | Wheel: zoom\n"
              << "  Esc: exit\n";

    bool paused = false;
    bool chaos_enabled = false;
    double simulation_speed = 1.0;
    int focused_integrator = -1;
    bool follow_center = true;
    bool show_grid = false;

    constexpr std::size_t kTrailLimit = 1400;

    ThreeBodyPreset active_preset = makeFigure8Preset(chaos_enabled);

    std::array<SimulationEntry, 3> sims = {
        SimulationEntry{IntegratorKind::RK4, {0.95F, 0.35F, 0.28F}, active_preset.initial_state},
        SimulationEntry{IntegratorKind::RKF45, {0.29F, 0.80F, 0.45F}, active_preset.initial_state},
        SimulationEntry{IntegratorKind::DOP853, {0.25F, 0.60F, 0.95F}, active_preset.initial_state},
    };

    auto resetAll = [&]() {
        active_preset = makeFigure8Preset(chaos_enabled);
        for (auto& sim : sims) {
            sim.runner.reset(active_preset.initial_state);
            sim.baseline = computeConservedQuantities(active_preset.initial_state);
            sim.drift = DriftMetrics{};
            for (auto& trail : sim.trails) {
                trail.clear();
            }
            appendTrailSample(sim, kTrailLimit);
        }
    };

    resetAll();

    CameraState camera;
    camera.yaw_degrees = 35.0F;
    camera.pitch_degrees = 26.0F;
    camera.distance = 9.0F;

    KeyEdgeTracker keys;
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
    double title_refresh_accum = 0.0;

    while (!renderer.shouldClose()) {
        renderer.pollEvents();

        GLFWwindow* window = renderer.window();

        const double now = glfwGetTime();
        double wall_dt = now - last_wall_time;
        last_wall_time = now;
        wall_dt = std::clamp(wall_dt, 0.0, 0.25);

        const double instant_fps = (wall_dt > 1e-8) ? (1.0 / wall_dt) : 0.0;
        if (fps_smoothed <= 0.0) {
            fps_smoothed = instant_fps;
        } else {
            fps_smoothed = 0.92 * fps_smoothed + 0.08 * instant_fps;
        }

        if (keys.pressed(window, GLFW_KEY_ESCAPE)) {
            glfwSetWindowShouldClose(window, GLFW_TRUE);
        }
        if (keys.pressed(window, GLFW_KEY_SPACE)) {
            paused = !paused;
        }
        if (keys.pressed(window, GLFW_KEY_R)) {
            resetAll();
            paused = false;
            follow_center = true;
        }
        if (keys.pressed(window, GLFW_KEY_P)) {
            chaos_enabled = !chaos_enabled;
            resetAll();
            paused = false;
            follow_center = true;
        }
        if (keys.pressed(window, GLFW_KEY_LEFT_BRACKET)) {
            simulation_speed = std::max(0.05, simulation_speed * 0.8);
        }
        if (keys.pressed(window, GLFW_KEY_RIGHT_BRACKET)) {
            simulation_speed = std::min(50.0, simulation_speed * 1.25);
        }
        if (keys.pressed(window, GLFW_KEY_F)) {
            follow_center = !follow_center;
        }
        if (keys.pressed(window, GLFW_KEY_G)) {
            show_grid = !show_grid;
            renderer.setShowGrid(show_grid);
        }

        auto handleIntegratorKey = [&](int index) {
            if (focused_integrator != index) {
                focused_integrator = index;
                sims[static_cast<std::size_t>(index)].visible = true;
                follow_center = true;
                return;
            }

            sims[static_cast<std::size_t>(index)].visible =
                !sims[static_cast<std::size_t>(index)].visible;

            if (!sims[static_cast<std::size_t>(index)].visible &&
                focused_integrator == index) {
                focused_integrator = -1;
            }

            bool any_visible = false;
            for (const auto& sim : sims) {
                any_visible = any_visible || sim.visible;
            }
            if (!any_visible) {
                sims[static_cast<std::size_t>(index)].visible = true;
            }
        };

        if (keys.pressed(window, GLFW_KEY_1)) {
            handleIntegratorKey(0);
        }
        if (keys.pressed(window, GLFW_KEY_2)) {
            handleIntegratorKey(1);
        }
        if (keys.pressed(window, GLFW_KEY_3)) {
            handleIntegratorKey(2);
        }
        if (keys.pressed(window, GLFW_KEY_0)) {
            focused_integrator = -1;
        }

        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera.yaw_degrees += static_cast<float>(55.0 * wall_dt);
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera.yaw_degrees -= static_cast<float>(55.0 * wall_dt);
        }
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera.pitch_degrees += static_cast<float>(55.0 * wall_dt);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera.pitch_degrees -= static_cast<float>(55.0 * wall_dt);
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            camera.distance += static_cast<float>(4.0 * wall_dt);
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            camera.distance -= static_cast<float>(4.0 * wall_dt);
        }

        const bool shift_down = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) ||
            (glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS);
        const bool left_drag = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        const bool right_drag = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
        const bool middle_drag = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
        const bool pan_drag = middle_drag || (left_drag && shift_down);

        double mouse_x = 0.0;
        double mouse_y = 0.0;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);
        static bool mouse_has_prev = false;
        static double prev_mouse_x = 0.0;
        static double prev_mouse_y = 0.0;

        if (right_drag || pan_drag) {
            if (!mouse_has_prev) {
                prev_mouse_x = mouse_x;
                prev_mouse_y = mouse_y;
                mouse_has_prev = true;
            }

            const double dx = mouse_x - prev_mouse_x;
            const double dy = mouse_y - prev_mouse_y;
            prev_mouse_x = mouse_x;
            prev_mouse_y = mouse_y;

            if (right_drag) {
                camera.yaw_degrees -= static_cast<float>(dx * 0.20);
                camera.pitch_degrees -= static_cast<float>(dy * 0.20);
            }
            if (pan_drag) {
                Vec3 cam_right{};
                Vec3 cam_up{};
                cameraBasis(camera, cam_right, cam_up);
                const double pan_scale = 0.0023 * static_cast<double>(camera.distance);
                camera.target = camera.target - (cam_right * (dx * pan_scale));
                camera.target = camera.target + (cam_up * (dy * pan_scale));
                follow_center = false;
            }
        } else {
            mouse_has_prev = false;
        }

        if (std::abs(scroll_state.yoffset) > 1e-9) {
            camera.distance = static_cast<float>(
                camera.distance * std::pow(0.86, scroll_state.yoffset));
            scroll_state.yoffset = 0.0;
        }

        camera.pitch_degrees = std::clamp(camera.pitch_degrees, -85.0F, 85.0F);
        camera.distance = std::clamp(camera.distance, 1.8F, 90.0F);

        if (!paused) {
            const double sim_dt = wall_dt * simulation_speed;

            for (auto& sim : sims) {
                const double next_time = sim.runner.time() + sim_dt;
                if (!sim.runner.stepTo(next_time)) {
                    paused = true;
                    std::cerr << "Simulation halted for " << sim.name << " at t="
                              << sim.runner.time() << "\n";
                    break;
                }

                const auto current = computeConservedQuantities(sim.runner.state());
                sim.drift = computeDriftMetrics(sim.baseline, current);
                appendTrailSample(sim, kTrailLimit);
            }
        }

        if (follow_center) {
            if (focused_integrator >= 0 &&
                sims[static_cast<std::size_t>(focused_integrator)].visible) {
                camera.target = centerOfMass(
                    sims[static_cast<std::size_t>(focused_integrator)].runner.state());
            } else {
                Vec3 mean{};
                int count = 0;
                for (const auto& sim : sims) {
                    if (!sim.visible) {
                        continue;
                    }
                    mean = mean + centerOfMass(sim.runner.state());
                    ++count;
                }
                if (count > 0) {
                    camera.target = mean * (1.0 / static_cast<double>(count));
                }
            }
        }

        std::array<IntegratorRenderData, 3> render_data{};
        for (std::size_t i = 0; i < sims.size(); ++i) {
            render_data[i].visible = sims[i].visible;
            render_data[i].color = sims[i].color;

            for (std::size_t body = 0; body < State3B::kBodies; ++body) {
                render_data[i].body_positions[body] = position(sims[i].runner.state(), body);
                render_data[i].trails[body] = std::vector<Vec3>(
                    sims[i].trails[body].begin(),
                    sims[i].trails[body].end());
            }
        }

        renderer.beginFrame();
        renderer.drawFrame(render_data, camera, focused_integrator);
        renderer.endFrame();

        title_refresh_accum += wall_dt;
        if (title_refresh_accum >= 0.15) {
            renderer.setWindowTitle(
                buildHudTitle(
                    fps_smoothed,
                    simulation_speed,
                    paused,
                    chaos_enabled,
                    follow_center,
                    show_grid,
                    sims));
            title_refresh_accum = 0.0;
        }
    }

    return 0;
}
