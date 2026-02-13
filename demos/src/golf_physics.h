#pragma once

#include "three_body_physics.h"

#include <array>
#include <cmath>
#include <cstddef>

namespace tableau::demos {

struct GolfState {
    static constexpr std::size_t kSize = 6; // x,y,z,vx,vy,vz
    std::array<double, kSize> data{};

    double& operator[](std::size_t idx) { return data[idx]; }
    const double& operator[](std::size_t idx) const { return data[idx]; }
};

inline GolfState operator+(const GolfState& a, const GolfState& b) {
    GolfState out{};
    for (std::size_t i = 0; i < GolfState::kSize; ++i) {
        out[i] = a[i] + b[i];
    }
    return out;
}

inline GolfState operator-(const GolfState& a, const GolfState& b) {
    GolfState out{};
    for (std::size_t i = 0; i < GolfState::kSize; ++i) {
        out[i] = a[i] - b[i];
    }
    return out;
}

inline GolfState operator*(const GolfState& s, double scale) {
    GolfState out{};
    for (std::size_t i = 0; i < GolfState::kSize; ++i) {
        out[i] = s[i] * scale;
    }
    return out;
}

inline GolfState operator*(double scale, const GolfState& s) {
    return s * scale;
}

struct GolfParams {
    double gravity{9.81};
    double rolling_drag{1.85};
    double impact_xy_damping{0.92};
    double stop_speed_epsilon{0.14};
};

inline Vec3 golfPosition(const GolfState& state) {
    return {state[0], state[1], state[2]};
}

inline Vec3 golfVelocity(const GolfState& state) {
    return {state[3], state[4], state[5]};
}

inline void setGolfPosition(GolfState& state, const Vec3& p) {
    state[0] = p.x;
    state[1] = p.y;
    state[2] = p.z;
}

inline void setGolfVelocity(GolfState& state, const Vec3& v) {
    state[3] = v.x;
    state[4] = v.y;
    state[5] = v.z;
}

inline bool isFiniteState(const GolfState& state) {
    for (double value : state.data) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

inline double stateL2Norm(const GolfState& state) {
    double sum = 0.0;
    for (double value : state.data) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

inline bool isStopped(const GolfState& state, double speed_eps = 0.14) {
    const Vec3 v = golfVelocity(state);
    const double speed_xy = std::sqrt(v.x * v.x + v.y * v.y);
    return (state[2] <= 1e-4) && (speed_xy <= speed_eps) && (std::abs(v.z) <= 1e-3);
}

inline void clampToGround(GolfState& state, const GolfParams& params) {
    if (state[2] < 0.0) {
        state[2] = 0.0;
    }

    if (state[2] <= 0.0 && state[5] < 0.0) {
        state[3] *= params.impact_xy_damping;
        state[4] *= params.impact_xy_damping;
        state[2] = 0.0;
        state[5] = 0.0;
    }

    const double speed_xy = std::sqrt(state[3] * state[3] + state[4] * state[4]);
    if (state[2] <= 1e-4 && speed_xy <= params.stop_speed_epsilon) {
        state[3] = 0.0;
        state[4] = 0.0;
        state[5] = 0.0;
    }
}

inline GolfState golfDerivative(const GolfState& state, double /*t*/, const GolfParams& params) {
    GolfState deriv{};
    const bool on_ground = (state[2] <= 1e-4) && (state[5] <= 0.0);

    deriv[0] = state[3];
    deriv[1] = state[4];
    deriv[2] = on_ground ? 0.0 : state[5];

    if (on_ground) {
        deriv[3] = -params.rolling_drag * state[3];
        deriv[4] = -params.rolling_drag * state[4];
        deriv[5] = 0.0;
    } else {
        deriv[3] = 0.0;
        deriv[4] = 0.0;
        deriv[5] = -params.gravity;
    }
    return deriv;
}

} // namespace tableau::demos
