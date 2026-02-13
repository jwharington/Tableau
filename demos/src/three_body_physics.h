#pragma once

#include <array>
#include <cmath>
#include <cstddef>

namespace tableau::demos {

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

inline Vec3 operator+(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 operator-(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 operator*(const Vec3& v, double s) {
    return {v.x * s, v.y * s, v.z * s};
}

inline Vec3 operator*(double s, const Vec3& v) {
    return v * s;
}

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline double normSquared(const Vec3& v) {
    return dot(v, v);
}

inline double norm(const Vec3& v) {
    return std::sqrt(normSquared(v));
}

struct State3B {
    static constexpr std::size_t kBodies = 3;
    static constexpr std::size_t kScalarsPerBody = 6; // x,y,z,vx,vy,vz
    static constexpr std::size_t kSize = kBodies * kScalarsPerBody;

    std::array<double, kSize> data{};

    double& operator[](std::size_t idx) { return data[idx]; }
    const double& operator[](std::size_t idx) const { return data[idx]; }
};

inline State3B operator+(const State3B& a, const State3B& b) {
    State3B out;
    for (std::size_t i = 0; i < State3B::kSize; ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

inline State3B operator-(const State3B& a, const State3B& b) {
    State3B out;
    for (std::size_t i = 0; i < State3B::kSize; ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

inline State3B operator*(const State3B& s, double scale) {
    State3B out;
    for (std::size_t i = 0; i < State3B::kSize; ++i) {
        out[i] = s[i] * scale;
    }
    return out;
}

inline State3B operator*(double scale, const State3B& s) {
    return s * scale;
}

inline constexpr double kGravitationalConstant = 1.0;
inline constexpr double kBodyMass = 1.0;
inline constexpr double kSoftening = 1e-6;

inline std::size_t baseIndex(std::size_t body) {
    return body * State3B::kScalarsPerBody;
}

inline Vec3 position(const State3B& state, std::size_t body) {
    const std::size_t b = baseIndex(body);
    return {state[b + 0], state[b + 1], state[b + 2]};
}

inline Vec3 velocity(const State3B& state, std::size_t body) {
    const std::size_t b = baseIndex(body);
    return {state[b + 3], state[b + 4], state[b + 5]};
}

inline void setPosition(State3B& state, std::size_t body, const Vec3& p) {
    const std::size_t b = baseIndex(body);
    state[b + 0] = p.x;
    state[b + 1] = p.y;
    state[b + 2] = p.z;
}

inline void setVelocity(State3B& state, std::size_t body, const Vec3& v) {
    const std::size_t b = baseIndex(body);
    state[b + 3] = v.x;
    state[b + 4] = v.y;
    state[b + 5] = v.z;
}

inline bool isFiniteState(const State3B& state) {
    for (double value : state.data) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

inline double stateL2Norm(const State3B& state) {
    double sum = 0.0;
    for (double value : state.data) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

inline Vec3 centerOfMass(const State3B& state) {
    Vec3 com{};
    for (std::size_t i = 0; i < State3B::kBodies; ++i) {
        com = com + (position(state, i) * kBodyMass);
    }
    return com * (1.0 / (kBodyMass * static_cast<double>(State3B::kBodies)));
}

inline State3B threeBodyDerivative(const State3B& state, double /*t*/) {
    State3B deriv{};

    for (std::size_t i = 0; i < State3B::kBodies; ++i) {
        const Vec3 pi = position(state, i);
        const Vec3 vi = velocity(state, i);

        const std::size_t b = baseIndex(i);
        deriv[b + 0] = vi.x;
        deriv[b + 1] = vi.y;
        deriv[b + 2] = vi.z;

        Vec3 ai{};
        for (std::size_t j = 0; j < State3B::kBodies; ++j) {
            if (i == j) {
                continue;
            }

            const Vec3 pj = position(state, j);
            const Vec3 rij = pj - pi;
            const double r2 = normSquared(rij) + (kSoftening * kSoftening);
            const double inv_r3 = 1.0 / (std::sqrt(r2) * r2);
            ai = ai + (rij * (kGravitationalConstant * kBodyMass * inv_r3));
        }

        deriv[b + 3] = ai.x;
        deriv[b + 4] = ai.y;
        deriv[b + 5] = ai.z;
    }

    return deriv;
}

} // namespace tableau::demos
