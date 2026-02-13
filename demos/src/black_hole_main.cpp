#include "black_hole_metrics.h"
#include "black_hole_renderer.h"
#include "black_hole_runner.h"

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

        const bool down = glfwGetKey(window, key) == GLFW_PRESS;
        const bool fire = down && !states_[static_cast<std::size_t>(key)];
        states_[static_cast<std::size_t>(key)] = down;
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
        return Vec3{};
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

std::vector<std::size_t> makeTrailIndices(std::size_t particle_count, std::size_t target_count) {
    std::vector<std::size_t> out;
    if (particle_count == 0 || target_count == 0) {
        return out;
    }

    const std::size_t step = std::max<std::size_t>(1, particle_count / target_count);
    out.reserve(target_count);

    std::size_t idx = 0;
    while (idx < particle_count && out.size() < target_count) {
        out.push_back(idx);
        idx += step;
    }

    if (out.empty()) {
        out.push_back(0);
    }

    return out;
}

struct CloudEntry {
    BlackHoleIntegratorKind kind;
    std::string name;
    std::array<float, 3> color{};
    bool visible{true};

    BlackHoleSimulation simulation;
    BlackHoleConservedQuantities baseline{};
    BlackHoleDriftMetrics drift{};

    double captures_per_second_ema{0.0};

    std::vector<std::size_t> trail_indices{};
    std::vector<std::deque<Vec3>> trails{};

    CloudEntry(
        BlackHoleIntegratorKind in_kind,
        std::array<float, 3> in_color,
        const ParticleCloudState& initial,
        const BlackHoleParams& params,
        std::uint64_t seed,
        std::size_t trail_tracked)
        : kind(in_kind),
          name(blackHoleIntegratorName(in_kind)),
          color(in_color),
          visible(true),
          simulation(in_kind, initial, params, seed) {
        trail_indices = makeTrailIndices(initial.particleCount(), trail_tracked);
        trails.assign(trail_indices.size(), std::deque<Vec3>{});
    }
};

void appendTrailSamples(CloudEntry& cloud, std::size_t history_limit) {
    const auto& state = cloud.simulation.state();
    for (std::size_t i = 0; i < cloud.trail_indices.size(); ++i) {
        const std::size_t idx = cloud.trail_indices[i];
        if (idx >= state.particleCount()) {
            continue;
        }

        cloud.trails[i].push_back(particlePosition(state, idx));
        if (cloud.trails[i].size() > history_limit) {
            cloud.trails[i].pop_front();
        }
    }
}

std::string buildHudTitle(
    double fps,
    double sim_speed,
    double effective_speed,
    bool paused,
    bool follow,
    bool grid,
    bool lensing,
    bool disk,
    bool light_background,
    double throttle,
    const std::array<CloudEntry, 3>& clouds) {

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "black_hole_demo | FPS " << fps;
    oss << " | speed x" << sim_speed;
    oss << " (eff x" << effective_speed << ")";
    oss << " | throttle " << throttle;
    oss << " | " << (paused ? "PAUSED" : "RUNNING");
    oss << " | follow " << (follow ? "ON" : "OFF");
    oss << " | grid " << (grid ? "ON" : "OFF");
    oss << " | lens " << (lensing ? "ON" : "OFF");
    oss << " | disk " << (disk ? "ON" : "OFF");
    oss << " | bg " << (light_background ? "LIGHT" : "DARK");

    oss << std::scientific << std::setprecision(2);
    for (const auto& c : clouds) {
        oss << " | " << c.name
            << " dE=" << c.drift.relative_energy_drift
            << " dP=" << c.drift.momentum_drift
            << " dt=" << c.simulation.lastDt();

        oss << std::fixed << std::setprecision(1)
            << " cap/s=" << c.captures_per_second_ema
            << std::scientific << std::setprecision(2);
    }

    return oss.str();
}

} // namespace
} // namespace tableau::demos

int main() {
    using namespace tableau::demos;

    BlackHoleRenderer renderer;
    if (!renderer.initialize(1480, 920, "black_hole_demo")) {
        return 1;
    }

    std::cout << "Controls:\n"
              << "  Space: pause/resume\n"
              << "  R: reset simulation\n"
              << "  P: toggle respawn perturbation mode + reset\n"
              << "  [: speed down\n"
              << "  ]: speed up\n"
              << "  F: toggle camera follow (BH center)\n"
              << "  G: toggle helper grid\n"
              << "  L: toggle lensing fake\n"
              << "  B: toggle accretion disk\n"
              << "  V: toggle dark/light background\n"
              << "  1/2/3: focus integrator; press same key to toggle visibility\n"
              << "  0: clear focus\n"
              << "  A/D: orbit yaw, W/S: orbit pitch, Q/E: zoom\n"
              << "  Mouse Right Drag: orbit | Shift+Left or Middle Drag: pan | Wheel: zoom\n"
              << "  Esc: exit\n";

    BlackHoleParams params{};

    constexpr std::size_t kParticlesPerIntegrator = 4000;
    constexpr std::size_t kTrackedTrails = 256;
    constexpr std::size_t kTrailHistory = 72;

    bool paused = false;
    bool follow_center = true;
    bool show_grid = false;
    bool show_lensing = true;
    bool show_disk = true;
    bool light_background = false;
    bool respawn_perturbation = false;

    double simulation_speed = 1.0;
    int focused_integrator = -1;

    ParticleCloudState initial = makeParticleCloud(kParticlesPerIntegrator, params, 424242, respawn_perturbation);

    std::array<CloudEntry, 3> clouds = {
        CloudEntry{BlackHoleIntegratorKind::RK4, {1.00F, 0.45F, 0.28F}, initial, params, 1001, kTrackedTrails},
        CloudEntry{BlackHoleIntegratorKind::RKF45, {0.30F, 1.00F, 0.42F}, initial, params, 2002, kTrackedTrails},
        CloudEntry{BlackHoleIntegratorKind::DOP853, {0.38F, 0.76F, 1.00F}, initial, params, 3003, kTrackedTrails},
    };

    auto resetAll = [&]() {
        initial = makeParticleCloud(kParticlesPerIntegrator, params, 424242, respawn_perturbation);
        for (auto& cloud : clouds) {
            cloud.simulation.setRespawnPerturbation(respawn_perturbation);
            cloud.simulation.reset(initial, 0.0);
            cloud.baseline = computeBlackHoleConservedQuantities(initial, params);
            cloud.drift = BlackHoleDriftMetrics{};
            cloud.captures_per_second_ema = 0.0;
            for (auto& trail : cloud.trails) {
                trail.clear();
            }
            appendTrailSamples(cloud, kTrailHistory);
        }
    };

    resetAll();

    renderer.setShowGrid(show_grid);
    renderer.setShowLensing(show_lensing);
    renderer.setShowAccretionDisk(show_disk);
    renderer.setLightBackground(light_background);

    CameraState camera;
    camera.yaw_degrees = 26.0F;
    camera.pitch_degrees = 30.0F;
    camera.distance = 95.0F;
    camera.target = Vec3{};

    KeyEdgeTracker keys;
    ScrollState scroll_state{};

    glfwSetWindowUserPointer(renderer.window(), &scroll_state);
    glfwSetScrollCallback(renderer.window(), [](GLFWwindow* window, double /*xoff*/, double yoff) {
        auto* state = static_cast<ScrollState*>(glfwGetWindowUserPointer(window));
        if (state != nullptr) {
            state->yoffset += yoff;
        }
    });

    double last_wall_time = glfwGetTime();
    double fps_ema = 0.0;
    double effective_speed_ema = 0.0;
    double title_refresh_accum = 0.0;

    constexpr double kTick = 1.0 / 120.0;
    double sim_accumulator = 0.0;
    double sim_rate_scale = 1.0;
    int max_ticks_per_frame = 5;

    while (!renderer.shouldClose()) {
        renderer.pollEvents();
        GLFWwindow* window = renderer.window();

        const double now = glfwGetTime();
        double wall_dt = now - last_wall_time;
        last_wall_time = now;
        wall_dt = std::clamp(wall_dt, 0.0, 0.25);

        const double inst_fps = (wall_dt > 1e-8) ? (1.0 / wall_dt) : 0.0;
        if (fps_ema <= 0.0) {
            fps_ema = inst_fps;
        } else {
            fps_ema = 0.92 * fps_ema + 0.08 * inst_fps;
        }

        if (fps_ema >= 58.0) {
            max_ticks_per_frame = 5;
            sim_rate_scale = 1.0;
        } else if (fps_ema >= 45.0) {
            max_ticks_per_frame = 3;
            sim_rate_scale = 0.82;
        } else {
            max_ticks_per_frame = 2;
            sim_rate_scale = 0.60;
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
            respawn_perturbation = !respawn_perturbation;
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
        if (keys.pressed(window, GLFW_KEY_L)) {
            show_lensing = !show_lensing;
            renderer.setShowLensing(show_lensing);
        }
        if (keys.pressed(window, GLFW_KEY_B)) {
            show_disk = !show_disk;
            renderer.setShowAccretionDisk(show_disk);
        }
        if (keys.pressed(window, GLFW_KEY_V)) {
            light_background = !light_background;
            renderer.setLightBackground(light_background);
        }

        auto handleIntegratorKey = [&](int idx) {
            if (focused_integrator != idx) {
                focused_integrator = idx;
                clouds[static_cast<std::size_t>(idx)].visible = true;
                return;
            }

            clouds[static_cast<std::size_t>(idx)].visible =
                !clouds[static_cast<std::size_t>(idx)].visible;

            if (!clouds[static_cast<std::size_t>(idx)].visible && focused_integrator == idx) {
                focused_integrator = -1;
            }

            bool any_visible = false;
            for (const auto& c : clouds) {
                any_visible = any_visible || c.visible;
            }
            if (!any_visible) {
                clouds[static_cast<std::size_t>(idx)].visible = true;
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
            camera.distance += static_cast<float>(16.0 * wall_dt);
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            camera.distance -= static_cast<float>(16.0 * wall_dt);
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
        static double prev_x = 0.0;
        static double prev_y = 0.0;

        if (right_drag || pan_drag) {
            if (!mouse_has_prev) {
                mouse_has_prev = true;
                prev_x = mouse_x;
                prev_y = mouse_y;
            }

            const double dx = mouse_x - prev_x;
            const double dy = mouse_y - prev_y;
            prev_x = mouse_x;
            prev_y = mouse_y;

            if (right_drag) {
                camera.yaw_degrees -= static_cast<float>(dx * 0.18);
                camera.pitch_degrees -= static_cast<float>(dy * 0.18);
            }
            if (pan_drag) {
                Vec3 cr{};
                Vec3 cu{};
                cameraBasis(camera, cr, cu);
                const double pan_scale = 0.0035 * static_cast<double>(camera.distance);
                camera.target = camera.target - cr * (dx * pan_scale);
                camera.target = camera.target + cu * (dy * pan_scale);
                follow_center = false;
            }
        } else {
            mouse_has_prev = false;
        }

        if (std::abs(scroll_state.yoffset) > 1e-9) {
            camera.distance = static_cast<float>(camera.distance * std::pow(0.88, scroll_state.yoffset));
            scroll_state.yoffset = 0.0;
        }

        camera.pitch_degrees = std::clamp(camera.pitch_degrees, -85.0F, 85.0F);
        camera.distance = std::clamp(camera.distance, 2.0F, 260.0F);

        if (follow_center) {
            camera.target = Vec3{};
        }

        std::array<std::size_t, 3> frame_captures{0, 0, 0};
        double simulated_dt_this_frame = 0.0;

        if (!paused) {
            sim_accumulator += wall_dt * simulation_speed * sim_rate_scale;
            sim_accumulator = std::min(sim_accumulator, kTick * 240.0);

            int ticks = 0;
            while (sim_accumulator >= kTick && ticks < max_ticks_per_frame) {
                const int remaining_slots = std::max(1, max_ticks_per_frame - ticks);
                double step_target = std::max(kTick, sim_accumulator / static_cast<double>(remaining_slots));
                step_target = std::clamp(step_target, kTick, kTick * 32.0);

                bool ok = true;
                for (std::size_t i = 0; i < clouds.size(); ++i) {
                    ok = ok && clouds[i].simulation.stepTo(clouds[i].simulation.time() + step_target);
                    frame_captures[i] += clouds[i].simulation.consumeCapturedRecent();
                }

                if (!ok) {
                    paused = true;
                    std::cerr << "black_hole_demo: simulation halted due to integration failure\n";
                    break;
                }

                sim_accumulator -= step_target;
                simulated_dt_this_frame += step_target;
                ++ticks;
            }
        }

        const double instant_effective_speed = paused ? 0.0 : (simulated_dt_this_frame / std::max(wall_dt, 1e-6));
        if (effective_speed_ema <= 0.0) {
            effective_speed_ema = instant_effective_speed;
        } else {
            effective_speed_ema = 0.90 * effective_speed_ema + 0.10 * instant_effective_speed;
        }

        for (std::size_t i = 0; i < clouds.size(); ++i) {
            const auto current = computeBlackHoleConservedQuantities(clouds[i].simulation.state(), params);
            clouds[i].drift = computeBlackHoleDriftMetrics(clouds[i].baseline, current);

            const double instant_cps = frame_captures[i] / std::max(wall_dt, 1e-6);
            if (clouds[i].captures_per_second_ema <= 0.0) {
                clouds[i].captures_per_second_ema = instant_cps;
            } else {
                clouds[i].captures_per_second_ema =
                    0.88 * clouds[i].captures_per_second_ema + 0.12 * instant_cps;
            }

            appendTrailSamples(clouds[i], kTrailHistory);
        }

        std::array<BlackHoleRenderData, 3> render_data{};
        for (std::size_t i = 0; i < clouds.size(); ++i) {
            render_data[i].visible = clouds[i].visible;
            render_data[i].color = clouds[i].color;

            const auto& s = clouds[i].simulation.state();
            render_data[i].particles.reserve(s.particleCount());
            for (std::size_t p = 0; p < s.particleCount(); ++p) {
                render_data[i].particles.push_back(particlePosition(s, p));
            }

            render_data[i].trails.reserve(clouds[i].trails.size());
            for (const auto& tr : clouds[i].trails) {
                render_data[i].trails.emplace_back(tr.begin(), tr.end());
            }
        }

        renderer.beginFrame();
        renderer.drawFrame(render_data, camera, focused_integrator, params, now);
        renderer.endFrame();

        title_refresh_accum += wall_dt;
        if (title_refresh_accum >= 0.12) {
            renderer.setWindowTitle(
                buildHudTitle(
                    fps_ema,
                    simulation_speed,
                    effective_speed_ema,
                    paused,
                    follow_center,
                    show_grid,
                    show_lensing,
                    show_disk,
                    light_background,
                    sim_rate_scale,
                    clouds));
            title_refresh_accum = 0.0;
        }
    }

    return 0;
}
