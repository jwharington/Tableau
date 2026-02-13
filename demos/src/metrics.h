#pragma once

#include "three_body_physics.h"

#include <algorithm>
#include <cmath>

namespace tableau::demos {

struct ConservedQuantities {
    double total_energy{0.0};
    Vec3 total_momentum{};
};

struct DriftMetrics {
    double relative_energy_drift{0.0};
    double momentum_drift{0.0};
};

inline ConservedQuantities computeConservedQuantities(const State3B& state) {
    ConservedQuantities out;

    double kinetic = 0.0;
    for (std::size_t i = 0; i < State3B::kBodies; ++i) {
        const Vec3 vi = velocity(state, i);
        kinetic += 0.5 * kBodyMass * normSquared(vi);
        out.total_momentum = out.total_momentum + (vi * kBodyMass);
    }

    double potential = 0.0;
    for (std::size_t i = 0; i < State3B::kBodies; ++i) {
        for (std::size_t j = i + 1; j < State3B::kBodies; ++j) {
            const Vec3 rij = position(state, j) - position(state, i);
            const double dist = std::sqrt(normSquared(rij) + (kSoftening * kSoftening));
            potential += -kGravitationalConstant * kBodyMass * kBodyMass / dist;
        }
    }

    out.total_energy = kinetic + potential;
    return out;
}

inline DriftMetrics computeDriftMetrics(
    const ConservedQuantities& baseline,
    const ConservedQuantities& current) {

    DriftMetrics drift;

    const double denom = std::max(std::abs(baseline.total_energy), 1e-12);
    drift.relative_energy_drift = std::abs(current.total_energy - baseline.total_energy) / denom;

    const Vec3 dp = current.total_momentum - baseline.total_momentum;
    drift.momentum_drift = norm(dp);

    return drift;
}

} // namespace tableau::demos
