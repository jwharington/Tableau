#include "black_hole_renderer.h"

#if TABLEAU_HAS_GLAD
#if __has_include(<glad/gl.h>)
    #include <glad/gl.h>
    #define TABLEAU_GLAD_V2 1
#elif __has_include(<glad/glad.h>)
    #include <glad/glad.h>
    #define TABLEAU_GLAD_V2 0
#else
#error "TABLEAU_HAS_GLAD=1 but GLAD headers were not found"
#endif
#else
#if defined(__APPLE__)
    #include <OpenGL/gl3.h>
#else
    #include <GL/gl.h>
#endif
#endif
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace tableau::demos {
namespace {

constexpr double kPi = 3.14159265358979323846;

Vec3 crossLocal(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

Vec3 normalize(const Vec3& v) {
    const double n = norm(v);
    if (n <= 1e-12) {
        return Vec3{};
    }
    return v * (1.0 / n);
}

std::array<float, 16> identity() {
    return {
        1.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F,
        0.0F, 0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 0.0F, 1.0F};
}

std::array<float, 16> multiply(const std::array<float, 16>& a, const std::array<float, 16>& b) {
    std::array<float, 16> out{};
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            out[col * 4 + row] =
                a[0 * 4 + row] * b[col * 4 + 0] +
                a[1 * 4 + row] * b[col * 4 + 1] +
                a[2 * 4 + row] * b[col * 4 + 2] +
                a[3 * 4 + row] * b[col * 4 + 3];
        }
    }
    return out;
}

std::array<float, 16> perspective(float fovy_rad, float aspect, float near_plane, float far_plane) {
    const float f = 1.0F / std::tan(fovy_rad * 0.5F);
    std::array<float, 16> out{};
    out[0] = f / aspect;
    out[5] = f;
    out[10] = (far_plane + near_plane) / (near_plane - far_plane);
    out[11] = -1.0F;
    out[14] = (2.0F * far_plane * near_plane) / (near_plane - far_plane);
    return out;
}

std::array<float, 16> lookAt(const Vec3& eye, const Vec3& target, const Vec3& up) {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(crossLocal(f, up));
    const Vec3 u = crossLocal(s, f);

    std::array<float, 16> out = identity();
    out[0] = static_cast<float>(s.x);
    out[1] = static_cast<float>(u.x);
    out[2] = static_cast<float>(-f.x);

    out[4] = static_cast<float>(s.y);
    out[5] = static_cast<float>(u.y);
    out[6] = static_cast<float>(-f.y);

    out[8] = static_cast<float>(s.z);
    out[9] = static_cast<float>(u.z);
    out[10] = static_cast<float>(-f.z);

    out[12] = static_cast<float>(-dot(s, eye));
    out[13] = static_cast<float>(-dot(u, eye));
    out[14] = static_cast<float>(dot(f, eye));
    return out;
}

unsigned int compileShader(unsigned int type, const char* src) {
    const unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    int ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (ok == GL_FALSE) {
        int length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetShaderInfoLog(shader, length, nullptr, log.data());
        std::cerr << "OpenGL shader compile error: " << log << "\n";
        glDeleteShader(shader);
        return 0;
    }
    return shader;
}

} // namespace

BlackHoleRenderer::~BlackHoleRenderer() {
    shutdown();
}

bool BlackHoleRenderer::initialize(int width, int height, const char* title) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    window_ = glfwCreateWindow(width, height, title, nullptr, nullptr);
    if (!window_) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

#if TABLEAU_HAS_GLAD
    #if TABLEAU_GLAD_V2
        const int version = gladLoadGL(glfwGetProcAddress);
        if (version == 0) {
    #else
        if (!gladLoadGLLoader(reinterpret_cast<GLADloadproc>(glfwGetProcAddress))) {
    #endif
            std::cerr << "Failed to initialize GLAD\n";
            shutdown();
            return false;
        }
#endif

    if (!createProgram()) {
        shutdown();
        return false;
    }

    ensureBuffers();
    generateStars();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    return true;
}

void BlackHoleRenderer::shutdown() {
    if (vbo_ != 0U) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (vao_ != 0U) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }

    destroyProgram();

    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    glfwTerminate();
}

bool BlackHoleRenderer::shouldClose() const {
    return (window_ == nullptr) || (glfwWindowShouldClose(window_) != 0);
}

void BlackHoleRenderer::pollEvents() const {
    glfwPollEvents();
}

void BlackHoleRenderer::beginFrame() {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    glViewport(0, 0, width, height);

    if (light_background_) {
        glClearColor(0.92F, 0.91F, 0.89F, 1.0F);
    } else {
        glClearColor(0.02F, 0.02F, 0.04F, 1.0F);
    }
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void BlackHoleRenderer::drawFrame(
    const std::array<BlackHoleRenderData, 3>& clouds,
    const CameraState& camera,
    int focused_integrator,
    const BlackHoleParams& params,
    double sim_time) {

    if (!window_ || shader_program_ == 0U) {
        return;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    const float aspect = (height > 0) ? (static_cast<float>(width) / static_cast<float>(height)) : 1.0F;

    const float yaw = camera.yaw_degrees * static_cast<float>(kPi / 180.0);
    const float pitch = camera.pitch_degrees * static_cast<float>(kPi / 180.0);

    const Vec3 eye{
        camera.target.x + static_cast<double>(camera.distance * std::cos(pitch) * std::cos(yaw)),
        camera.target.y + static_cast<double>(camera.distance * std::cos(pitch) * std::sin(yaw)),
        camera.target.z + static_cast<double>(camera.distance * std::sin(pitch))};

    const auto projection = perspective(52.0F * static_cast<float>(kPi / 180.0), aspect, 0.01F, 1200.0F);
    const auto view = lookAt(eye, camera.target, Vec3{0.0, 0.0, 1.0});
    const auto mvp = multiply(projection, view);

    drawStars(mvp);
    if (show_grid_) {
        drawGrid(mvp);
    }

    if (show_disk_) {
        drawAccretionDisk(mvp, params, sim_time);
    }
    drawHorizon(mvp, params);

    for (std::size_t i = 0; i < clouds.size(); ++i) {
        if (!clouds[i].visible) {
            continue;
        }
        const bool focused = (focused_integrator < 0) || (focused_integrator == static_cast<int>(i));
        drawCloud(clouds[i], mvp, focused, static_cast<float>(params.lensing_radius), show_lensing_);
    }
}

void BlackHoleRenderer::endFrame() {
    if (window_) {
        glfwSwapBuffers(window_);
    }
}

void BlackHoleRenderer::setWindowTitle(const std::string& title) const {
    if (window_) {
        glfwSetWindowTitle(window_, title.c_str());
    }
}

bool BlackHoleRenderer::createProgram() {
    static constexpr const char* kVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_mvp;
uniform float u_point_size;
uniform int u_mode;
uniform int u_lensing_enabled;
uniform float u_lensing_radius;
void main() {
    vec3 p = a_position;
    if (u_mode == 1 && u_lensing_enabled == 1) {
        float r = length(p.xy);
        if (r < u_lensing_radius && r > 1e-5) {
            vec3 tangent = normalize(vec3(-p.y, p.x, 0.0));
            float f = (u_lensing_radius - r) / u_lensing_radius;
            p += tangent * (f * f * 0.45);
        }
    }
    gl_Position = u_mvp * vec4(p, 1.0);
    gl_PointSize = u_point_size;
}
)";

    static constexpr const char* kFragmentShader = R"(
#version 330 core
out vec4 frag_color;
uniform vec4 u_color;
uniform int u_mode;
void main() {
    if (u_mode == 0) {
        frag_color = u_color;
        return;
    }

    vec2 c = gl_PointCoord - vec2(0.5);
    float d = length(c) * 2.0;
    if (d > 1.0) {
        discard;
    }

    if (u_mode == 1) {
        float core = exp(-d * d * 2.8);
        frag_color = vec4(u_color.rgb, core * u_color.a);
    } else if (u_mode == 2) {
        frag_color = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        float halo = exp(-d * d * 1.3);
        frag_color = vec4(u_color.rgb, halo * u_color.a);
    }
}
)";

    const unsigned int vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
    const unsigned int fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    if (vs == 0U || fs == 0U) {
        if (vs != 0U) {
            glDeleteShader(vs);
        }
        if (fs != 0U) {
            glDeleteShader(fs);
        }
        return false;
    }

    shader_program_ = glCreateProgram();
    glAttachShader(shader_program_, vs);
    glAttachShader(shader_program_, fs);
    glLinkProgram(shader_program_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    int ok = 0;
    glGetProgramiv(shader_program_, GL_LINK_STATUS, &ok);
    if (ok == GL_FALSE) {
        int length = 0;
        glGetProgramiv(shader_program_, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(shader_program_, length, nullptr, log.data());
        std::cerr << "OpenGL program link error: " << log << "\n";
        destroyProgram();
        return false;
    }

    u_mvp_ = glGetUniformLocation(shader_program_, "u_mvp");
    u_color_ = glGetUniformLocation(shader_program_, "u_color");
    u_point_size_ = glGetUniformLocation(shader_program_, "u_point_size");
    u_mode_ = glGetUniformLocation(shader_program_, "u_mode");
    u_lensing_enabled_ = glGetUniformLocation(shader_program_, "u_lensing_enabled");
    u_lensing_radius_ = glGetUniformLocation(shader_program_, "u_lensing_radius");

    return true;
}

void BlackHoleRenderer::destroyProgram() {
    if (shader_program_ != 0U) {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
    }
}

void BlackHoleRenderer::ensureBuffers() {
    if (vao_ == 0U) {
        glGenVertexArrays(1, &vao_);
    }
    if (vbo_ == 0U) {
        glGenBuffers(1, &vbo_);
    }

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3, nullptr, GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * static_cast<int>(sizeof(float)), nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void BlackHoleRenderer::drawStars(const std::array<float, 16>& mvp) {
    auto draw_layer = [&](const std::vector<Vec3>& stars, float point_size, float alpha) {
        std::vector<float> vertices;
        vertices.reserve(stars.size() * 3);
        for (const Vec3& p : stars) {
            vertices.push_back(static_cast<float>(p.x));
            vertices.push_back(static_cast<float>(p.y));
            vertices.push_back(static_cast<float>(p.z));
        }

        glDepthMask(GL_FALSE);
        glUseProgram(shader_program_);
        glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
        glUniform1i(u_mode_, 1);
        glUniform1i(u_lensing_enabled_, 0);
        glUniform1f(u_lensing_radius_, 0.0F);
        if (light_background_) {
            glUniform4f(u_color_, 0.35F, 0.32F, 0.28F, alpha * 0.70F);
        } else {
            glUniform4f(u_color_, 0.82F, 0.88F, 1.0F, alpha);
        }
        glUniform1f(u_point_size_, point_size);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(stars.size()));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        glDepthMask(GL_TRUE);
    };

    if (light_background_) {
        draw_layer(stars_far_, 1.2F, 0.16F);
        draw_layer(stars_near_, 2.0F, 0.25F);
    } else {
        draw_layer(stars_far_, 1.2F, 0.22F);
        draw_layer(stars_near_, 2.0F, 0.35F);
    }
}

void BlackHoleRenderer::drawGrid(const std::array<float, 16>& mvp) {
    std::vector<Vec3> lines;
    constexpr int kHalfLines = 14;
    constexpr double kSpacing = 4.0;
    constexpr double kExtent = kHalfLines * kSpacing;

    lines.reserve(static_cast<std::size_t>((kHalfLines * 2 + 1) * 4));
    for (int i = -kHalfLines; i <= kHalfLines; ++i) {
        const double axis = static_cast<double>(i) * kSpacing;
        lines.push_back(Vec3{-kExtent, axis, 0.0});
        lines.push_back(Vec3{kExtent, axis, 0.0});
        lines.push_back(Vec3{axis, -kExtent, 0.0});
        lines.push_back(Vec3{axis, kExtent, 0.0});
    }

    std::vector<float> vertices;
    vertices.reserve(lines.size() * 3);
    for (const Vec3& p : lines) {
        vertices.push_back(static_cast<float>(p.x));
        vertices.push_back(static_cast<float>(p.y));
        vertices.push_back(static_cast<float>(p.z));
    }

    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_mode_, 0);
    glUniform1i(u_lensing_enabled_, 0);
    glUniform1f(u_lensing_radius_, 0.0F);
    if (light_background_) {
        glUniform4f(u_color_, 0.55F, 0.52F, 0.48F, 0.18F);
    } else {
        glUniform4f(u_color_, 0.22F, 0.24F, 0.32F, 0.38F);
    }

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(), GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(lines.size()));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void BlackHoleRenderer::drawCloud(
    const BlackHoleRenderData& cloud,
    const std::array<float, 16>& mvp,
    bool focused,
    float lensing_radius,
    bool lensing_enabled) {

    if (cloud.particles.empty()) {
        return;
    }

    for (const auto& trail : cloud.trails) {
        if (trail.size() < 2) {
            continue;
        }

        std::vector<float> tverts;
        tverts.reserve(trail.size() * 3);
        for (const Vec3& p : trail) {
            tverts.push_back(static_cast<float>(p.x));
            tverts.push_back(static_cast<float>(p.y));
            tverts.push_back(static_cast<float>(p.z));
        }

        glUseProgram(shader_program_);
        glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
        glUniform1i(u_mode_, 0);
        glUniform1i(u_lensing_enabled_, 0);
        glUniform1f(u_lensing_radius_, 0.0F);
        const float trail_alpha = light_background_
            ? (focused ? 0.72F : 0.42F)
            : (focused ? 0.62F : 0.34F);
        const float tr = light_background_ ? std::clamp(cloud.color[0] * 0.50F, 0.0F, 1.0F) : cloud.color[0];
        const float tg = light_background_ ? std::clamp(cloud.color[1] * 0.45F, 0.0F, 1.0F) : cloud.color[1];
        const float tb = light_background_ ? std::clamp(cloud.color[2] * 0.45F, 0.0F, 1.0F) : cloud.color[2];
        glUniform4f(u_color_, tr, tg, tb, trail_alpha);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(tverts.size() * sizeof(float)), tverts.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(trail.size()));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    std::vector<float> vertices;
    vertices.reserve(cloud.particles.size() * 3);
    for (const Vec3& p : cloud.particles) {
        vertices.push_back(static_cast<float>(p.x));
        vertices.push_back(static_cast<float>(p.y));
        vertices.push_back(static_cast<float>(p.z));
    }

    const float r = light_background_
        ? std::clamp(cloud.color[0] * 0.55F, 0.0F, 1.0F)
        : std::clamp(cloud.color[0] * 1.25F + 0.05F, 0.0F, 1.0F);
    const float g = light_background_
        ? std::clamp(cloud.color[1] * 0.50F, 0.0F, 1.0F)
        : std::clamp(cloud.color[1] * 1.25F + 0.05F, 0.0F, 1.0F);
    const float b = light_background_
        ? std::clamp(cloud.color[2] * 0.50F, 0.0F, 1.0F)
        : std::clamp(cloud.color[2] * 1.25F + 0.05F, 0.0F, 1.0F);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(), GL_DYNAMIC_DRAW);

    // Wide glow.
    glBlendFunc(GL_SRC_ALPHA, light_background_ ? GL_ONE_MINUS_SRC_ALPHA : GL_ONE);
    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_mode_, 1);
    glUniform1i(u_lensing_enabled_, lensing_enabled ? 1 : 0);
    glUniform1f(u_lensing_radius_, lensing_radius);
    const float glow_alpha = light_background_
        ? (focused ? 0.40F : 0.28F)
        : (focused ? 0.55F : 0.40F);
    glUniform4f(u_color_, r, g, b, glow_alpha);
    glUniform1f(u_point_size_, focused ? (light_background_ ? 10.0F : 11.0F) : (light_background_ ? 8.0F : 9.0F));
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(cloud.particles.size()));

    // Core points.
    glBlendFunc(GL_SRC_ALPHA, light_background_ ? GL_ONE_MINUS_SRC_ALPHA : GL_ONE);
    glUniform1i(u_mode_, 1);
    const float core_alpha = light_background_
        ? (focused ? 1.0F : 0.92F)
        : (focused ? 0.95F : 0.82F);
    glUniform4f(u_color_, r, g, b, core_alpha);
    glUniform1f(u_point_size_, focused ? (light_background_ ? 5.8F : 7.2F) : (light_background_ ? 4.6F : 5.8F));
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(cloud.particles.size()));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void BlackHoleRenderer::drawAccretionDisk(const std::array<float, 16>& mvp, const BlackHoleParams& params, double sim_time) {
    constexpr int kSegments = 220;

    const double inner = std::max(params.visual_disk_inner, params.captureRadius() * 5.5);
    const double outer = std::max(params.visual_disk_outer, inner + 1.8);

    auto draw_ring = [&](double radius, float alpha, float jitter_amp, float hue_shift) {
        std::vector<float> vertices;
        vertices.reserve(static_cast<std::size_t>((kSegments + 1) * 3));

        for (int i = 0; i <= kSegments; ++i) {
            const double ang = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(kSegments);
            const double jitter = 1.0 + jitter_amp * std::sin(ang * 7.0 + sim_time * (0.4 + hue_shift));
            const double r = radius * jitter;
            vertices.push_back(static_cast<float>(r * std::cos(ang)));
            vertices.push_back(static_cast<float>(r * std::sin(ang)));
            vertices.push_back(static_cast<float>(0.15 * std::sin(ang * 3.0 + sim_time * 0.25)));
        }

        glUseProgram(shader_program_);
        glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
        glUniform1i(u_mode_, 0);
        glUniform1i(u_lensing_enabled_, 0);
        glUniform1f(u_lensing_radius_, 0.0F);
        glUniform4f(u_color_, 1.0F, 0.62F + hue_shift, 0.18F, alpha);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(GL_LINE_STRIP, 0, kSegments + 1);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    };

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    draw_ring(outer, 0.22F, 0.06F, 0.0F);
    draw_ring((inner + outer) * 0.5, 0.34F, 0.08F, 0.04F);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    draw_ring(inner, 0.78F, 0.03F, 0.08F);
}

void BlackHoleRenderer::drawHorizon(const std::array<float, 16>& mvp, const BlackHoleParams& params) {
    const std::vector<float> origin = {0.0F, 0.0F, 0.0F};

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(origin.size() * sizeof(float)), origin.data(), GL_DYNAMIC_DRAW);

    // Soft black halo.
    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_mode_, 3);
    glUniform1i(u_lensing_enabled_, 0);
    glUniform1f(u_lensing_radius_, 0.0F);
    glUniform4f(u_color_, 0.0F, 0.0F, 0.0F, 0.65F);
    glUniform1f(u_point_size_, static_cast<float>(params.visual_horizon_radius * 90.0));
    glDrawArrays(GL_POINTS, 0, 1);

    // Opaque center.
    glUniform1i(u_mode_, 2);
    glUniform1f(u_point_size_, static_cast<float>(params.visual_horizon_radius * 62.0));
    glDrawArrays(GL_POINTS, 0, 1);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void BlackHoleRenderer::generateStars() {
    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> angle_dist(0.0F, 2.0F * static_cast<float>(kPi));
    std::uniform_real_distribution<float> z_dist(-1.0F, 1.0F);

    auto sample = [&](float radius) {
        const float theta = angle_dist(rng);
        const float z = z_dist(rng);
        const float rxy = std::sqrt(std::max(0.0F, 1.0F - z * z));
        return Vec3{
            static_cast<double>(radius * rxy * std::cos(theta)),
            static_cast<double>(radius * rxy * std::sin(theta)),
            static_cast<double>(radius * z)};
    };

    stars_near_.clear();
    stars_far_.clear();
    stars_near_.reserve(1100);
    stars_far_.reserve(1500);

    for (int i = 0; i < 1100; ++i) {
        stars_near_.push_back(sample(220.0F));
    }
    for (int i = 0; i < 1500; ++i) {
        stars_far_.push_back(sample(520.0F));
    }
}

} // namespace tableau::demos
