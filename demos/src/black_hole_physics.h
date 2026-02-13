#pragma once

#include "three_body_physics.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace tableau::demos {

struct ParticleCloudState {
    std::vector<double> data{}; // [x,y,z,vx,vy,vz] per particle

    std::size_t particleCount() const {
        return data.size() / 6;
    }

    bool empty() const {
        return data.empty();
    }
};

inline ParticleCloudState operator+(const ParticleCloudState& a, const ParticleCloudState& b) {
    ParticleCloudState out;
    const std::size_t n = std::max(a.data.size(), b.data.size());
    out.data.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double av = (i < a.data.size()) ? a.data[i] : 0.0;
        const double bv = (i < b.data.size()) ? b.data[i] : 0.0;
        out.data[i] = av + bv;
    }
    return out;
}

inline ParticleCloudState operator-(const ParticleCloudState& a, const ParticleCloudState& b) {
    ParticleCloudState out;
    const std::size_t n = std::max(a.data.size(), b.data.size());
    out.data.assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double av = (i < a.data.size()) ? a.data[i] : 0.0;
        const double bv = (i < b.data.size()) ? b.data[i] : 0.0;
        out.data[i] = av - bv;
    }
    return out;
}

inline ParticleCloudState operator*(const ParticleCloudState& s, double scale) {
    ParticleCloudState out;
    out.data.resize(s.data.size());
    for (std::size_t i = 0; i < s.data.size(); ++i) {
        out.data[i] = s.data[i] * scale;
    }
    return out;
}

inline ParticleCloudState operator*(double scale, const ParticleCloudState& s) {
    return s * scale;
}

struct BlackHoleParams {
    double G{1.0};
    double mass{15.0};
    double c_eff{12.0};
    double softening{1e-5};
    double capture_multiplier{1.15};

    double spawn_radius_min{18.0};
    double spawn_radius_max{45.0};
    double spawn_z_sigma{0.35};
    double tangential_jitter{0.08};
    double inward_drift{0.03};
    double vertical_velocity_jitter{0.02};
    double perturbation_scale{0.45};

    // Purely visual parameters.
    double visual_horizon_radius{1.2};
    double visual_disk_inner{1.8};
    double visual_disk_outer{4.4};
    double lensing_radius{5.5};

    double schwarzschildRadius() const {
        return (2.0 * G * mass) / (c_eff * c_eff);
    }

    double captureRadius() const {
        return capture_multiplier * schwarzschildRadius();
    }
};

inline Vec3 cross3(const Vec3& a, const Vec3& b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x};
}

inline std::size_t particleBase(std::size_t idx) {
    return idx * 6;
}

inline Vec3 particlePosition(const ParticleCloudState& state, std::size_t idx) {
    const std::size_t b = particleBase(idx);
    return {state.data[b + 0], state.data[b + 1], state.data[b + 2]};
}

inline Vec3 particleVelocity(const ParticleCloudState& state, std::size_t idx) {
    const std::size_t b = particleBase(idx);
    return {state.data[b + 3], state.data[b + 4], state.data[b + 5]};
}

inline void setParticlePosition(ParticleCloudState& state, std::size_t idx, const Vec3& p) {
    const std::size_t b = particleBase(idx);
    state.data[b + 0] = p.x;
    state.data[b + 1] = p.y;
    state.data[b + 2] = p.z;
}

inline void setParticleVelocity(ParticleCloudState& state, std::size_t idx, const Vec3& v) {
    const std::size_t b = particleBase(idx);
    state.data[b + 3] = v.x;
    state.data[b + 4] = v.y;
    state.data[b + 5] = v.z;
}

inline bool isFiniteState(const ParticleCloudState& state) {
    for (double value : state.data) {
        if (!std::isfinite(value)) {
            return false;
        }
    }
    return true;
}

inline double stateL2Norm(const ParticleCloudState& state) {
    double sum = 0.0;
    for (double value : state.data) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

inline void respawnParticle(
    ParticleCloudState& state,
    std::size_t idx,
    const BlackHoleParams& params,
    std::mt19937_64& rng,
    bool perturbation_mode) {

    const double pscale = perturbation_mode ? (1.0 + params.perturbation_scale) : 1.0;

    std::uniform_real_distribution<double> radius_dist(params.spawn_radius_min, params.spawn_radius_max);
    std::uniform_real_distribution<double> theta_dist(0.0, 2.0 * 3.14159265358979323846);
    std::normal_distribution<double> z_dist(0.0, params.spawn_z_sigma * pscale);
    std::normal_distribution<double> jitter(0.0, 1.0);

    const double r = radius_dist(rng);
    const double theta = theta_dist(rng);

    const Vec3 radial{
        std::cos(theta),
        std::sin(theta),
        0.0};
    const Vec3 tangent{
        -std::sin(theta),
        std::cos(theta),
        0.0};

    Vec3 p = radial * r;
    p.z = z_dist(rng);

    const double v_circ = std::sqrt((params.G * params.mass) / std::max(r, 1e-8));
    const double vt = v_circ * (1.0 + params.tangential_jitter * pscale * jitter(rng));
    const double vr = -params.inward_drift * (1.0 + 0.5 * pscale * jitter(rng)) * v_circ;
    const double vz = params.vertical_velocity_jitter * pscale * jitter(rng);

    Vec3 v = tangent * vt + radial * vr;
    v.z = vz;

    setParticlePosition(state, idx, p);
    setParticleVelocity(state, idx, v);
}

inline ParticleCloudState makeParticleCloud(
    std::size_t particle_count,
    const BlackHoleParams& params,
    std::uint64_t seed,
    bool perturbation_mode) {

    ParticleCloudState state;
    state.data.assign(particle_count * 6, 0.0);

    std::mt19937_64 rng(seed);
    for (std::size_t i = 0; i < particle_count; ++i) {
        respawnParticle(state, i, params, rng, perturbation_mode);
    }

    return state;
}

inline std::size_t respawnCaptured(
    ParticleCloudState& state,
    const BlackHoleParams& params,
    std::mt19937_64& rng,
    bool perturbation_mode) {

    std::size_t captures = 0;
    const double capture_radius = params.captureRadius();

    for (std::size_t i = 0; i < state.particleCount(); ++i) {
        const Vec3 p = particlePosition(state, i);
        const Vec3 v = particleVelocity(state, i);

        const bool invalid =
            (!std::isfinite(p.x)) || (!std::isfinite(p.y)) || (!std::isfinite(p.z)) ||
            (!std::isfinite(v.x)) || (!std::isfinite(v.y)) || (!std::isfinite(v.z));

        const double r = norm(p);
        if (invalid || (r <= capture_radius)) {
            respawnParticle(state, i, params, rng, perturbation_mode);
            ++captures;
        }
    }

    return captures;
}

inline ParticleCloudState blackHoleDerivative(
    const ParticleCloudState& state,
    double /*t*/,
    const BlackHoleParams& params) {

    ParticleCloudState dydt;
    dydt.data.assign(state.data.size(), 0.0);

    if (state.data.empty()) {
        return dydt;
    }

    const double safe_min_r = std::max(params.captureRadius() * 0.5, 1e-4);

    for (std::size_t i = 0; i < state.particleCount(); ++i) {
        const Vec3 p = particlePosition(state, i);
        const Vec3 v = particleVelocity(state, i);

        setParticlePosition(dydt, i, v);

        double r2 = normSquared(p) + params.softening * params.softening;
        double r = std::sqrt(std::max(r2, safe_min_r * safe_min_r));
        r2 = r * r;

        const double inv_r3 = 1.0 / std::max(r2 * r, 1e-12);
        const Vec3 a_newton = p * (-params.G * params.mass * inv_r3);

        const Vec3 h = cross3(p, v);
        const double h2 = normSquared(h);
        const double inv_r5 = 1.0 / std::max(r2 * r2 * r, 1e-12);
        const double rel_coeff = (3.0 * params.G * params.mass * h2) / (params.c_eff * params.c_eff);
        const Vec3 a_rel = p * (rel_coeff * inv_r5);

        Vec3 a = a_newton + a_rel;
        if (!std::isfinite(a.x) || !std::isfinite(a.y) || !std::isfinite(a.z)) {
            a = Vec3{};
        }

        setParticleVelocity(dydt, i, a);
    }

    return dydt;
}

} // namespace tableau::demos
