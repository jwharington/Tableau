#include "golf_physics.h"
#include "golf_runner.h"

#include <cmath>
#include <iostream>

using tableau::demos::GolfIntegratorKind;
using tableau::demos::GolfIntegratorRunner;
using tableau::demos::GolfParams;
using tableau::demos::GolfState;
using tableau::demos::Vec3;
using tableau::demos::golfPosition;
using tableau::demos::isFiniteState;
using tableau::demos::isStopped;
using tableau::demos::setGolfPosition;
using tableau::demos::setGolfVelocity;

namespace {

int g_failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++g_failures;
    }
}

const char* integratorName(GolfIntegratorKind kind) {
    switch (kind) {
    case GolfIntegratorKind::RK4:
        return "RK4";
    case GolfIntegratorKind::RKF45:
        return "RKF45";
    case GolfIntegratorKind::DOP853:
        return "DOP853";
    }
    return "unknown";
}

void runSanityCase(GolfIntegratorKind kind) {
    GolfParams params{};

    GolfState initial{};
    setGolfPosition(initial, Vec3{0.0, 0.0, 0.0});
    setGolfVelocity(initial, Vec3{18.0, 0.0, 14.0});

    GolfIntegratorRunner runner(kind, initial, params);

    constexpr double kDt = 1.0 / 120.0;
    constexpr int kSteps = 800;

    double max_z = 0.0;
    bool saw_air = false;
    bool touched_ground_after_air = false;
    double x_at_touchdown = 0.0;
    double x_after_touchdown = 0.0;

    for (int i = 0; i < kSteps; ++i) {
        const bool ok = runner.stepTo(runner.time() + kDt);
        if (!ok) {
            const GolfState& s = runner.state();
            std::cerr << "step failure in " << integratorName(kind)
                      << " at iteration " << i
                      << " | z=" << s[2]
                      << " vz=" << s[5]
                      << " vxy=" << std::sqrt(s[3] * s[3] + s[4] * s[4])
                      << "\n";
        }
        expect(ok, "integrator step should succeed");
        if (!ok) {
            return;
        }

        const GolfState& state = runner.state();
        expect(isFiniteState(state), "state should remain finite");
        if (!isFiniteState(state)) {
            return;
        }

        const Vec3 pos = golfPosition(state);
        max_z = std::max(max_z, pos.z);
        if (pos.z > 0.1) {
            saw_air = true;
        }

        if (saw_air && !touched_ground_after_air && pos.z <= 1e-4) {
            touched_ground_after_air = true;
            x_at_touchdown = pos.x;
        }
        if (touched_ground_after_air) {
            x_after_touchdown = pos.x;
        }

        if (isStopped(state)) {
            break;
        }
    }

    expect(saw_air, "ball should leave the ground");
    expect(max_z > 1.0, "ball should reach a sensible height");
    expect(touched_ground_after_air, "ball should touch ground after flight");
    expect((x_after_touchdown - x_at_touchdown) > 0.5, "ball should roll after touching ground");
    expect(isStopped(runner.state()), "ball should stop on the ground");
}

} // namespace

int main() {
    runSanityCase(GolfIntegratorKind::RK4);
    runSanityCase(GolfIntegratorKind::RKF45);
    runSanityCase(GolfIntegratorKind::DOP853);

    if (g_failures != 0) {
        std::cerr << "Total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "Golf core tests passed\n";
    return 0;
}
