#pragma once

#include "black_hole_physics.h"

#include <algorithm>
#include <cmath>

namespace tableau::demos {

struct BlackHoleConservedQuantities {
    double total_energy{0.0};
    Vec3 total_momentum{};
};

struct BlackHoleDriftMetrics {
    double relative_energy_drift{0.0};
    double momentum_drift{0.0};
};

inline BlackHoleConservedQuantities computeBlackHoleConservedQuantities(
    const ParticleCloudState& state,
    const BlackHoleParams& params) {

    BlackHoleConservedQuantities out;

    for (std::size_t i = 0; i < state.particleCount(); ++i) {
        const Vec3 p = particlePosition(state, i);
        const Vec3 v = particleVelocity(state, i);

        const double kinetic = 0.5 * normSquared(v);
        const double dist = std::sqrt(normSquared(p) + params.softening * params.softening);
        const double potential = -(params.G * params.mass) / std::max(dist, 1e-8);

        out.total_energy += kinetic + potential;
        out.total_momentum = out.total_momentum + v;
    }

    return out;
}

inline BlackHoleDriftMetrics computeBlackHoleDriftMetrics(
    const BlackHoleConservedQuantities& baseline,
    const BlackHoleConservedQuantities& current) {

    BlackHoleDriftMetrics drift;

    const double denom = std::max(std::abs(baseline.total_energy), 1e-12);
    drift.relative_energy_drift = std::abs(current.total_energy - baseline.total_energy) / denom;

    const Vec3 dp = current.total_momentum - baseline.total_momentum;
    drift.momentum_drift = norm(dp);

    return drift;
}

} // namespace tableau::demos
