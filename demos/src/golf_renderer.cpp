#include "golf_renderer.h"

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
#include <string>
#include <vector>

namespace tableau::demos {
namespace {

constexpr double kPi = 3.14159265358979323846;

Vec3 cross(const Vec3& a, const Vec3& b) {
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

GolfRenderer::~GolfRenderer() {
    shutdown();
}

bool GolfRenderer::initialize(int width, int height, const char* title) {
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

    if (!createProgram() || !createSphereProgram()) {
        shutdown();
        return false;
    }

    ensureBuffers();
    buildGrid();
    buildSphereMesh();
    ensureSphereBuffers();

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    return true;
}

void GolfRenderer::shutdown() {
    if (sphere_ebo_ != 0U) {
        glDeleteBuffers(1, &sphere_ebo_);
        sphere_ebo_ = 0;
    }
    if (sphere_vbo_ != 0U) {
        glDeleteBuffers(1, &sphere_vbo_);
        sphere_vbo_ = 0;
    }
    if (sphere_vao_ != 0U) {
        glDeleteVertexArrays(1, &sphere_vao_);
        sphere_vao_ = 0;
    }

    if (vbo_ != 0U) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (vao_ != 0U) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }

    destroyProgram();
    destroySphereProgram();

    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    glfwTerminate();
}

bool GolfRenderer::shouldClose() const {
    return (window_ == nullptr) || (glfwWindowShouldClose(window_) != 0);
}

void GolfRenderer::pollEvents() const {
    glfwPollEvents();
}

void GolfRenderer::beginFrame() {
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    glViewport(0, 0, width, height);

    glClearColor(0.62F, 0.80F, 0.95F, 1.0F);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void GolfRenderer::drawFrame(const GolfRenderData& data, const GolfCameraState& camera_state) {
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

    const auto projection = perspective(50.0F * static_cast<float>(kPi / 180.0), aspect, 0.05F, 800.0F);
    const auto view = lookAt(eye, camera_state.target, Vec3{0.0, 0.0, 1.0});
    const auto mvp = multiply(projection, view);

    drawGround(mvp);
    drawGrid(mvp);
    drawHole(mvp, data.hole_pos, data.hole_radius);

    for (const auto& trail : data.trails) {
        if (!trail.visible) {
            continue;
        }
        drawTrail(trail.trail, trail.color, mvp);
    }

    if (data.show_aim) {
        drawLine(data.aim_line, {0.94F, 0.86F, 0.46F}, 0.8F, mvp);
    }

    if (data.ball_visible) {
        drawSphere(data.ball_pos, 1.25F, mvp);
    }
}

void GolfRenderer::endFrame() {
    if (window_) {
        glfwSwapBuffers(window_);
    }
}

void GolfRenderer::setWindowTitle(const std::string& title) const {
    if (window_) {
        glfwSetWindowTitle(window_, title.c_str());
    }
}

bool GolfRenderer::createProgram() {
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
uniform int u_mode;
void main() {
    if (u_mode == 1) {
        vec2 c = gl_PointCoord - vec2(0.5);
        float d = length(c) * 2.0;
        if (d > 1.0) {
            discard;
        }
        float alpha = exp(-d * d * 2.2);
        frag_color = vec4(u_color.rgb, u_color.a * alpha);
        return;
    }
    frag_color = u_color;
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

    return true;
}

bool GolfRenderer::createSphereProgram() {
    static constexpr const char* kSphereVertexShader = R"(
#version 330 core
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
uniform mat4 u_mvp;
out vec3 v_normal;
void main() {
    v_normal = normalize(a_normal);
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
)";

    static constexpr const char* kSphereFragmentShader = R"(
#version 330 core
in vec3 v_normal;
out vec4 frag_color;
uniform vec3 u_color;
uniform vec3 u_light_dir;
void main() {
    vec3 n = normalize(v_normal);
    vec3 l = normalize(u_light_dir);
    float diffuse = max(dot(n, l), 0.0);
    float lit = 0.30 + 0.70 * diffuse;
    frag_color = vec4(u_color * lit, 1.0);
}
)";

    const unsigned int vs = compileShader(GL_VERTEX_SHADER, kSphereVertexShader);
    const unsigned int fs = compileShader(GL_FRAGMENT_SHADER, kSphereFragmentShader);
    if (vs == 0U || fs == 0U) {
        if (vs != 0U) {
            glDeleteShader(vs);
        }
        if (fs != 0U) {
            glDeleteShader(fs);
        }
        return false;
    }

    sphere_program_ = glCreateProgram();
    glAttachShader(sphere_program_, vs);
    glAttachShader(sphere_program_, fs);
    glLinkProgram(sphere_program_);

    glDeleteShader(vs);
    glDeleteShader(fs);

    int ok = 0;
    glGetProgramiv(sphere_program_, GL_LINK_STATUS, &ok);
    if (ok == GL_FALSE) {
        int length = 0;
        glGetProgramiv(sphere_program_, GL_INFO_LOG_LENGTH, &length);
        std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
        glGetProgramInfoLog(sphere_program_, length, nullptr, log.data());
        std::cerr << "OpenGL sphere program link error: " << log << "\n";
        destroySphereProgram();
        return false;
    }

    su_mvp_ = glGetUniformLocation(sphere_program_, "u_mvp");
    su_color_ = glGetUniformLocation(sphere_program_, "u_color");
    su_light_dir_ = glGetUniformLocation(sphere_program_, "u_light_dir");
    return true;
}

void GolfRenderer::destroyProgram() {
    if (shader_program_ != 0U) {
        glDeleteProgram(shader_program_);
        shader_program_ = 0;
    }
}

void GolfRenderer::destroySphereProgram() {
    if (sphere_program_ != 0U) {
        glDeleteProgram(sphere_program_);
        sphere_program_ = 0;
    }
}

void GolfRenderer::ensureBuffers() {
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

void GolfRenderer::buildGrid() {
    constexpr int kHalfLines = 22;
    constexpr double kSpacing = 5.0;
    constexpr double kExtent = kHalfLines * kSpacing;

    grid_vertices_.clear();
    grid_vertices_.reserve(static_cast<std::size_t>((kHalfLines * 2 + 1) * 12));

    for (int i = -kHalfLines; i <= kHalfLines; ++i) {
        const double axis = static_cast<double>(i) * kSpacing;
        grid_vertices_.push_back(static_cast<float>(-kExtent));
        grid_vertices_.push_back(static_cast<float>(axis));
        grid_vertices_.push_back(0.0F);
        grid_vertices_.push_back(static_cast<float>(kExtent));
        grid_vertices_.push_back(static_cast<float>(axis));
        grid_vertices_.push_back(0.0F);

        grid_vertices_.push_back(static_cast<float>(axis));
        grid_vertices_.push_back(static_cast<float>(-kExtent));
        grid_vertices_.push_back(0.0F);
        grid_vertices_.push_back(static_cast<float>(axis));
        grid_vertices_.push_back(static_cast<float>(kExtent));
        grid_vertices_.push_back(0.0F);
    }

    ground_vertices_ = {
        static_cast<float>(-kExtent), static_cast<float>(-kExtent), 0.0F,
        static_cast<float>(kExtent), static_cast<float>(-kExtent), 0.0F,
        static_cast<float>(-kExtent), static_cast<float>(kExtent), 0.0F,
        static_cast<float>(kExtent), static_cast<float>(kExtent), 0.0F,
    };
}

void GolfRenderer::buildSphereMesh() {
    constexpr int kLatSegments = 18;
    constexpr int kLonSegments = 28;

    sphere_vertices_.clear();
    sphere_indices_.clear();
    sphere_vertices_.reserve(static_cast<std::size_t>((kLatSegments + 1) * (kLonSegments + 1) * 6));
    sphere_indices_.reserve(static_cast<std::size_t>(kLatSegments * kLonSegments * 6));

    for (int lat = 0; lat <= kLatSegments; ++lat) {
        const double theta = kPi * static_cast<double>(lat) / static_cast<double>(kLatSegments);
        const double st = std::sin(theta);
        const double ct = std::cos(theta);

        for (int lon = 0; lon <= kLonSegments; ++lon) {
            const double phi = 2.0 * kPi * static_cast<double>(lon) / static_cast<double>(kLonSegments);
            const double sp = std::sin(phi);
            const double cp = std::cos(phi);

            const float x = static_cast<float>(st * cp);
            const float y = static_cast<float>(st * sp);
            const float z = static_cast<float>(ct);

            sphere_vertices_.push_back(x);
            sphere_vertices_.push_back(y);
            sphere_vertices_.push_back(z);
            sphere_vertices_.push_back(x);
            sphere_vertices_.push_back(y);
            sphere_vertices_.push_back(z);
        }
    }

    for (int lat = 0; lat < kLatSegments; ++lat) {
        for (int lon = 0; lon < kLonSegments; ++lon) {
            const unsigned int row0 =
                static_cast<unsigned int>(lat * (kLonSegments + 1) + lon);
            const unsigned int row1 =
                static_cast<unsigned int>((lat + 1) * (kLonSegments + 1) + lon);

            sphere_indices_.push_back(row0);
            sphere_indices_.push_back(row1);
            sphere_indices_.push_back(row0 + 1U);

            sphere_indices_.push_back(row0 + 1U);
            sphere_indices_.push_back(row1);
            sphere_indices_.push_back(row1 + 1U);
        }
    }
}

void GolfRenderer::ensureSphereBuffers() {
    if (sphere_vao_ == 0U) {
        glGenVertexArrays(1, &sphere_vao_);
    }
    if (sphere_vbo_ == 0U) {
        glGenBuffers(1, &sphere_vbo_);
    }
    if (sphere_ebo_ == 0U) {
        glGenBuffers(1, &sphere_ebo_);
    }

    glBindVertexArray(sphere_vao_);
    glBindBuffer(GL_ARRAY_BUFFER, sphere_vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(sphere_vertices_.size() * sizeof(float)),
        sphere_vertices_.data(),
        GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sphere_ebo_);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(sphere_indices_.size() * sizeof(unsigned int)),
        sphere_indices_.data(),
        GL_STATIC_DRAW);

    constexpr GLsizei stride = static_cast<GLsizei>(6 * sizeof(float));
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        stride,
        reinterpret_cast<const void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

void GolfRenderer::drawGround(const std::array<float, 16>& mvp) {
    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_mode_, 0);
    glUniform4f(u_color_, 0.21F, 0.46F, 0.25F, 1.0F);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(ground_vertices_.size() * sizeof(float)),
        ground_vertices_.data(),
        GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GolfRenderer::drawGrid(const std::array<float, 16>& mvp) {
    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_mode_, 0);
    glUniform4f(u_color_, 0.12F, 0.29F, 0.17F, 0.38F);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(grid_vertices_.size() * sizeof(float)),
        grid_vertices_.data(),
        GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(grid_vertices_.size() / 3));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GolfRenderer::drawHole(const std::array<float, 16>& mvp, const Vec3& center, float radius) {
    constexpr int kSegments = 96;
    std::vector<float> vertices;
    vertices.reserve(static_cast<std::size_t>((kSegments + 1) * 3));

    for (int i = 0; i <= kSegments; ++i) {
        const double ang = (2.0 * kPi * static_cast<double>(i)) / static_cast<double>(kSegments);
        const double x = center.x + radius * std::cos(ang);
        const double y = center.y + radius * std::sin(ang);
        vertices.push_back(static_cast<float>(x));
        vertices.push_back(static_cast<float>(y));
        vertices.push_back(0.02F);
    }

    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_mode_, 0);
    glUniform4f(u_color_, 0.08F, 0.07F, 0.05F, 0.9F);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
        vertices.data(),
        GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINE_STRIP, 0, kSegments + 1);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GolfRenderer::drawTrail(const std::vector<Vec3>& points, const std::array<float, 3>& color, const std::array<float, 16>& mvp) {
    if (points.size() < 2) {
        return;
    }

    std::vector<float> vertices;
    vertices.reserve(points.size() * 3);
    for (const Vec3& p : points) {
        vertices.push_back(static_cast<float>(p.x));
        vertices.push_back(static_cast<float>(p.y));
        vertices.push_back(static_cast<float>(p.z + 0.02));
    }

    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_mode_, 0);
    glUniform4f(u_color_, color[0], color[1], color[2], 0.78F);

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

void GolfRenderer::drawLine(
    const std::vector<Vec3>& points,
    const std::array<float, 3>& color,
    float alpha,
    const std::array<float, 16>& mvp) {
    if (points.size() < 2) {
        return;
    }

    std::vector<float> vertices;
    vertices.reserve(points.size() * 3);
    for (const Vec3& p : points) {
        vertices.push_back(static_cast<float>(p.x));
        vertices.push_back(static_cast<float>(p.y));
        vertices.push_back(static_cast<float>(p.z + 0.02));
    }

    glUseProgram(shader_program_);
    glUniformMatrix4fv(u_mvp_, 1, GL_FALSE, mvp.data());
    glUniform1i(u_mode_, 0);
    glUniform4f(u_color_, color[0], color[1], color[2], alpha);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertices.size() * sizeof(float)),
        vertices.data(),
        GL_DYNAMIC_DRAW);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(points.size()));
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindVertexArray(0);
}

void GolfRenderer::drawSphere(const Vec3& center, float radius, const std::array<float, 16>& mvp) {
    if (sphere_program_ == 0U || sphere_vao_ == 0U || sphere_indices_.empty()) {
        return;
    }

    std::array<float, 16> model = {
        radius, 0.0F, 0.0F, 0.0F,
        0.0F, radius, 0.0F, 0.0F,
        0.0F, 0.0F, radius, 0.0F,
        static_cast<float>(center.x), static_cast<float>(center.y), static_cast<float>(center.z + 1.15), 1.0F};
    const auto sphere_mvp = multiply(mvp, model);

    glUseProgram(sphere_program_);
    glUniformMatrix4fv(su_mvp_, 1, GL_FALSE, sphere_mvp.data());
    glUniform3f(su_color_, 0.96F, 0.96F, 0.95F);
    glUniform3f(su_light_dir_, -0.35F, 0.55F, 0.75F);

    glBindVertexArray(sphere_vao_);
    glDrawElements(
        GL_TRIANGLES,
        static_cast<GLsizei>(sphere_indices_.size()),
        GL_UNSIGNED_INT,
        nullptr);
    glBindVertexArray(0);
}

} // namespace tableau::demos
