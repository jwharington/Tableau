#include "integrator_runner.h"
#include "metrics.h"
#include "presets.h"
#include "three_body_physics.h"

#include <cmath>
#include <iostream>

using tableau::demos::ConservedQuantities;
using tableau::demos::DriftMetrics;
using tableau::demos::IntegratorKind;
using tableau::demos::IntegratorRunner;
using tableau::demos::State3B;
using tableau::demos::computeConservedQuantities;
using tableau::demos::computeDriftMetrics;
using tableau::demos::isFiniteState;
using tableau::demos::makeFigure8Preset;

namespace {

int g_failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++g_failures;
    }
}

struct Thresholds {
    double max_rel_energy_drift;
    double max_momentum_drift;
};

void runSanityCase(IntegratorKind kind, const Thresholds& limits) {
    const auto preset = makeFigure8Preset(false);

    IntegratorRunner runner(kind, preset.initial_state);
    const ConservedQuantities baseline = computeConservedQuantities(preset.initial_state);

    constexpr double kSampleDt = 1e-3;
    constexpr int kSamples = 250;

    for (int i = 0; i < kSamples; ++i) {
        const double t_next = runner.time() + kSampleDt;
        const bool ok = runner.stepTo(t_next);
        expect(ok, "integrator step should succeed");
        if (!ok) {
            return;
        }

        const State3B& state = runner.state();
        expect(isFiniteState(state), "state should remain finite");
        if (!isFiniteState(state)) {
            return;
        }
    }

    const ConservedQuantities current = computeConservedQuantities(runner.state());
    const DriftMetrics drift = computeDriftMetrics(baseline, current);

    expect(
        std::isfinite(drift.relative_energy_drift),
        "energy drift should be finite");
    expect(
        std::isfinite(drift.momentum_drift),
        "momentum drift should be finite");

    expect(
        drift.relative_energy_drift < limits.max_rel_energy_drift,
        "relative energy drift exceeded threshold");
    expect(
        drift.momentum_drift < limits.max_momentum_drift,
        "momentum drift exceeded threshold");
}

} // namespace

int main() {
    runSanityCase(IntegratorKind::RK4, Thresholds{0.08, 0.08});
    runSanityCase(IntegratorKind::RKF45, Thresholds{0.03, 0.03});
    runSanityCase(IntegratorKind::DOP853, Thresholds{0.03, 0.03});

    if (g_failures != 0) {
        std::cerr << "Total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "Three-body core tests passed\n";
    return 0;
}
