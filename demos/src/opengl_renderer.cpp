#include "opengl_renderer.h"

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
#include <sstream>
#include <string>
#include <vector>

namespace tableau::demos {
namespace {

constexpr double kPi = 3.14159265358979323846;

std::array<float, 3> clampColor(const std::array<float, 3>& color, float scale) {
    return {
        std::clamp(color[0] * scale, 0.0F, 1.0F),
        std::clamp(color[1] * scale, 0.0F, 1.0F),
        std::clamp(color[2] * scale, 0.0F, 1.0F)};
}

Vec3 cross(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

Vec3 normalize(const Vec3& v) {
    const double n = norm(v);
    if (n <= 1e-12) {
        return {0.0, 0.0, 0.0};
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
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

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

unsigned int compileShader(unsigned int type, const char* source) {
    const unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
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

OpenGLRenderer::~OpenGLRenderer() {
    shutdown();
}

bool OpenGLRenderer::initialize(int width, int height, const char* title) {
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

void OpenGLRenderer::shutdown() {
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

bool OpenGLRenderer::shouldClose() const {
    return (window_ == nullptr) || (glfwWindowShouldClose(window_) != 0);
}

void OpenGLRenderer::pollEvents() const {
    glfwPollEvents();
}

void OpenGLRenderer::beginFrame() {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    glViewport(0, 0, width, height);

    glClearColor(0.01F, 0.01F, 0.03F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void OpenGLRenderer::drawFrame(
    const std::array<IntegratorRenderData, 3>& integrator_data,
    const CameraState& camera_state,
    int focused_integrator) {

    if (!window_ || shader_program_ == 0U) {
        return;
    }

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    const float aspect = (height > 0) ? (static_cast<float>(width) / static_cast<float>(height)) : 1.0F;

    const float yaw = camera_state.yaw_degrees * static_cast<float>(kPi / 180.0);
    const float pitch = camera_state.pitch_degrees * static_cast<float>(kPi / 180.0);

    const Vec3 eye{
        camera_state.target.x + static_cast<double>(camera_state.distance * std::cos(pitch) * std::cos(yaw)),
        camera_state.target.y + static_cast<double>(camera_state.distance * std::cos(pitch) * std::sin(yaw)),
        camera_state.target.z + static_cast<double>(camera_state.distance * std::sin(pitch))};

    const auto projection = perspective(48.0F * static_cast<float>(kPi / 180.0), aspect, 0.01F, 800.0F);
    const auto view = lookAt(eye, camera_state.target, Vec3{0.0, 0.0, 1.0});
    const auto mvp = multiply(projection, view);

    drawStars(mvp);
    if (show_grid_) {
        drawGrid(mvp);
    }

    for (std::size_t i = 0; i < integrator_data.size(); ++i) {
        const auto& entry = integrator_data[i];
        if (!entry.visible) {
            continue;
        }

        const bool is_focused = (focused_integrator < 0) || (focused_integrator == static_cast<int>(i));
        const auto base_color = clampColor(entry.color, is_focused ? 1.0F : 0.35F);

        for (std::size_t body = 0; body < entry.trails.size(); ++body) {
            const float shade = (body == 0) ? 1.0F : ((body == 1) ? 0.82F : 0.65F);
            const auto body_color = clampColor(base_color, shade);
            drawLineStrip(entry.trails[body], body_color, mvp);
        }

        drawSuns(entry.body_positions, base_color, mvp, is_focused);
    }
}

void OpenGLRenderer::endFrame() {
    if (window_) {
        glfwSwapBuffers(window_);
    }
}

void OpenGLRenderer::setWindowTitle(const std::string& title) const {
    if (window_) {
        glfwSetWindowTitle(window_, title.c_str());
    }
}

bool OpenGLRenderer::createProgram() {
    static constexpr const char* kVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 a_position;
uniform mat4 u_mvp;
uniform float u_point_size;
void main() {
    gl_Position = u_mvp * vec4(a_position, 1.0);
    gl_PointSize = u_point_size;
}
)";

    static constexpr const char* kFragmentShader = R"(
#version 330 core
out vec4 frag_color;
uniform vec4 u_color;
uniform int u_sun_mode;
void main() {
    if (u_sun_mode == 0) {
        frag_color = u_color;
    } else {
        vec2 coord = gl_PointCoord - vec2(0.5);
        float dist = length(coord) * 2.0;
        if (u_sun_mode == 1) {
            // Sun core: hot white center fading to body color
            float core = 1.0 - smoothstep(0.0, 1.0, dist);
            vec3 white_hot = vec3(1.0, 1.0, 0.95);
            vec3 col = mix(white_hot, u_color.rgb, smoothstep(0.0, 0.7, dist));
            frag_color = vec4(col, core);
        } else if (u_sun_mode == 2) {
            // Sun glow: soft exponential falloff
            float glow = exp(-dist * dist * 2.5);
            frag_color = vec4(u_color.rgb, glow * u_color.a);
        } else {
            // Corona rays: very soft wide halo
            float halo = exp(-dist * dist * 1.2) * 0.5;
            frag_color = vec4(u_color.rgb, halo * u_color.a);
        }
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
    u_sun_mode_ = glGetUniformLocation(shader_program_, "u_sun_mode");

    return true;
}

void OpenGLRenderer::destroyProgram() {
    if (shader_program_ != 0U) {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
    }
}

void OpenGLRenderer::ensureBuffers() {
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

void OpenGLRenderer::drawGrid(const std::array<float, 16>& mvp) {
    std::vector<Vec3> segments;
    constexpr int kHalfLines = 10;
    constexpr double kSpacing = 0.5;
    constexpr double kExtent = kHalfLines * kSpacing;

    segments.reserve(static_cast<std::size_t>((kHalfLines * 2 + 1) * 4));

    for (int i = -kHalfLines; i <= kHalfLines; ++i) {
        const double axis = static_cast<double>(i) * kSpacing;
        segments.push_back(Vec3{-kExtent, axis, 0.0});
        segments.push_back(Vec3{kExtent, axis, 0.0});
        segments.push_back(Vec3{axis, -kExtent, 0.0});
        segments.push_back(Vec3{axis, kExtent, 0.0});
    }

    std::vector<float> vertices;
    vertices.reserve(segments.size() * 3);
    for (const Vec3& p : segments) {
        vertices.push_back(static_cast<float>(p.x));
        vertices.push_back(static_cast<float>(p.y));
        vertices.push_back(static_cast<float>(p.z));
    }

    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_sun_mode_, 0);
    glUniform4f(u_color_, 0.20F, 0.25F, 0.30F, 0.65F);
    glUniform1f(u_point_size_, 1.0F);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
        vertices.data(),
        GL_DYNAMIC_DRAW);

    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(segments.size()));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::drawLineStrip(
    const std::vector<Vec3>& points,
    const std::array<float, 3>& color,
    const std::array<float, 16>& mvp) {

    if (points.size() < 2) {
        return;
    }

    std::vector<float> vertices;
    vertices.reserve(points.size() * 3);
    for (const Vec3& p : points) {
        vertices.push_back(static_cast<float>(p.x));
        vertices.push_back(static_cast<float>(p.y));
        vertices.push_back(static_cast<float>(p.z));
    }

    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_sun_mode_, 0);
    glUniform4f(u_color_, color[0], color[1], color[2], 0.92F);
    glUniform1f(u_point_size_, 1.0F);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
        vertices.data(),
        GL_DYNAMIC_DRAW);

    glDrawArrays(GL_LINE_STRIP, 0, static_cast<GLsizei>(points.size()));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::drawPoints(
    const std::array<Vec3, 3>& points,
    const std::array<float, 3>& color,
    const std::array<float, 16>& mvp,
    float point_size) {

    std::vector<float> vertices;
    vertices.reserve(points.size() * 3);
    for (const Vec3& p : points) {
        vertices.push_back(static_cast<float>(p.x));
        vertices.push_back(static_cast<float>(p.y));
        vertices.push_back(static_cast<float>(p.z));
    }

    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_sun_mode_, 0);
    const float alpha = (point_size >= 20.0F) ? 0.16F : 1.0F;
    glUniform4f(u_color_, color[0], color[1], color[2], alpha);
    glUniform1f(u_point_size_, point_size);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
        vertices.data(),
        GL_DYNAMIC_DRAW);

    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(points.size()));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::drawSuns(
    const std::array<Vec3, 3>& positions,
    const std::array<float, 3>& color,
    const std::array<float, 16>& mvp,
    bool is_focused) {

    std::vector<float> vertices;
    vertices.reserve(positions.size() * 3);
    for (const Vec3& p : positions) {
        vertices.push_back(static_cast<float>(p.x));
        vertices.push_back(static_cast<float>(p.y));
        vertices.push_back(static_cast<float>(p.z));
    }

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
        vertices.data(),
        GL_DYNAMIC_DRAW);

    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());

    // Layer 1: Wide corona halo (additive blending)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glUniform1i(u_sun_mode_, 3);
    glUniform4f(u_color_, color[0], color[1], color[2], is_focused ? 0.35F : 0.15F);
    glUniform1f(u_point_size_, is_focused ? 96.0F : 64.0F);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));

    // Layer 2: Medium glow (additive blending)
    glUniform1i(u_sun_mode_, 2);
    glUniform4f(u_color_, color[0], color[1], color[2], is_focused ? 0.6F : 0.3F);
    glUniform1f(u_point_size_, is_focused ? 56.0F : 40.0F);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));

    // Layer 3: Bright core (normal blending)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUniform1i(u_sun_mode_, 1);
    glUniform4f(u_color_, color[0], color[1], color[2], 1.0F);
    glUniform1f(u_point_size_, is_focused ? 24.0F : 16.0F);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(positions.size()));

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void OpenGLRenderer::generateStars() {
    std::mt19937 rng(1337U);
    std::uniform_real_distribution<float> angle_dist(0.0F, 2.0F * static_cast<float>(kPi));
    std::uniform_real_distribution<float> z_dist(-1.0F, 1.0F);

    auto sample_on_sphere = [&](float radius) {
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
    stars_near_.reserve(900);
    stars_far_.reserve(1300);

    for (int i = 0; i < 900; ++i) {
        stars_near_.push_back(sample_on_sphere(120.0F));
    }
    for (int i = 0; i < 1300; ++i) {
        stars_far_.push_back(sample_on_sphere(300.0F));
    }
}

void OpenGLRenderer::drawStars(const std::array<float, 16>& mvp) {
    auto draw_star_layer = [&](const std::vector<Vec3>& stars, float point_size, float alpha) {
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
        glUniform1i(u_sun_mode_, 0);
        glUniform4f(u_color_, 0.82F, 0.88F, 1.0F, alpha);
        glUniform1f(u_point_size_, point_size);

        glBindVertexArray(vao_);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glBufferData(
            GL_ARRAY_BUFFER,
            static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
            vertices.data(),
            GL_DYNAMIC_DRAW);

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(stars.size()));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        glDepthMask(GL_TRUE);
    };

    draw_star_layer(stars_far_, 1.4F, 0.30F);
    draw_star_layer(stars_near_, 2.2F, 0.48F);
}

} // namespace tableau::demos
