#include "black_hole_metrics.h"
#include "black_hole_runner.h"

#include <cmath>
#include <iostream>

using tableau::demos::BlackHoleConservedQuantities;
using tableau::demos::BlackHoleIntegratorKind;
using tableau::demos::BlackHoleDriftMetrics;
using tableau::demos::BlackHoleParams;
using tableau::demos::BlackHoleSimulation;
using tableau::demos::ParticleCloudState;
using tableau::demos::computeBlackHoleConservedQuantities;
using tableau::demos::computeBlackHoleDriftMetrics;
using tableau::demos::isFiniteState;
using tableau::demos::makeParticleCloud;
using tableau::demos::setParticlePosition;
using tableau::demos::setParticleVelocity;
using tableau::demos::Vec3;

namespace {

int g_failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++g_failures;
    }
}

struct Limits {
    double max_rel_energy_drift;
    double max_momentum_drift;
};

void runCase(BlackHoleIntegratorKind kind, const Limits& lim, std::uint64_t seed) {
    BlackHoleParams params;
    params.spawn_radius_min = params.captureRadius() * 1.05;
    params.spawn_radius_max = 0.8;
    params.inward_drift = 0.90;
    params.tangential_jitter = 0.05;
    params.vertical_velocity_jitter = 0.01;

    ParticleCloudState init = makeParticleCloud(64, params, seed, false);
    // Force at least one capture/respawn event deterministically.
    setParticlePosition(init, 0, Vec3{params.captureRadius() * 0.5, 0.0, 0.0});
    setParticleVelocity(init, 0, Vec3{0.0, 0.0, 0.0});
    BlackHoleSimulation sim(kind, init, params, seed + 99);
    sim.setRespawnPerturbation(true);

    const BlackHoleConservedQuantities baseline = computeBlackHoleConservedQuantities(init, params);

    constexpr double kDt = 1.0 / 120.0;
    constexpr int kSteps = 360; // 3 seconds

    for (int i = 0; i < kSteps; ++i) {
        const bool ok = sim.stepTo(sim.time() + kDt);
        expect(ok, "simulation step should succeed");
        if (!ok) {
            return;
        }

        expect(isFiniteState(sim.state()), "state should remain finite");
        if (!isFiniteState(sim.state())) {
            return;
        }
    }

    const BlackHoleConservedQuantities current =
        computeBlackHoleConservedQuantities(sim.state(), params);
    const BlackHoleDriftMetrics drift = computeBlackHoleDriftMetrics(baseline, current);

    expect(std::isfinite(drift.relative_energy_drift), "energy drift should be finite");
    expect(std::isfinite(drift.momentum_drift), "momentum drift should be finite");

    expect(drift.relative_energy_drift < lim.max_rel_energy_drift, "relative energy drift exceeded threshold");
    expect(drift.momentum_drift < lim.max_momentum_drift, "momentum drift exceeded threshold");

    expect(sim.capturedTotal() > 0, "captured/respawn count should be greater than zero");
}

} // namespace

int main() {
    runCase(BlackHoleIntegratorKind::RK4, Limits{1e4, 2e3}, 101);
    runCase(BlackHoleIntegratorKind::RKF45, Limits{1e4, 2e3}, 202);
    runCase(BlackHoleIntegratorKind::DOP853, Limits{1e4, 2e3}, 303);

    if (g_failures != 0) {
        std::cerr << "Total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "Black-hole core tests passed\n";
    return 0;
}
