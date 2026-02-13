#include "RK4.h"
#include "Integration.h"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>

using tableau::integration::DerivativeFunction;
using tableau::integration::IntegrationResult;
using tableau::integration::RK4Config;
using tableau::integration::RK4Integrator;
using tableau::integration::integrate;

namespace {

int g_failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++g_failures;
    }
}

// --------------------------------------------------------------------------
// Custom vector state for multi-dimensional tests
// --------------------------------------------------------------------------

struct Vec2 {
    double x{0.0};
    double y{0.0};
};

Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.x + b.x, a.y + b.y}; }
Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.x - b.x, a.y - b.y}; }
Vec2 operator*(const Vec2& v, double s) { return {v.x * s, v.y * s}; }
Vec2 operator*(double s, const Vec2& v) { return v * s; }

double vec2Norm(const Vec2& v) { return std::sqrt(v.x * v.x + v.y * v.y); }

// --------------------------------------------------------------------------
// Scalar: dy/dt = y  =>  y(t) = exp(t)
// --------------------------------------------------------------------------

void test_scalar_exponential() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.01, deriv);

    expect(result.success, "scalar step should succeed");
    expect(result.method_used == "rk4", "method should be rk4");
    expect(result.time_step_used == 0.01, "dt should match requested step");

    // RK4 is 4th order: for dy/dt=y, single step from y=1 at t=0 with h=0.01
    // should be very close to exp(0.01).
    double expected = std::exp(0.01);
    expect(std::abs(result.state - expected) < 1e-12, "scalar step accuracy check");
}

void test_scalar_exponential_full_integration() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto trajectory = integrate<double>(solver, 1.0, 0.0, 1.0, 0.001, deriv, false);

    expect(trajectory.size() == 2, "integrate without intermediate should return 2 points");
    double final_y = trajectory.back().second;
    double expected = std::exp(1.0);
    // With h=0.001 over [0,1], RK4 global error ~ O(h^4) = 1e-12, but accumulated
    expect(std::abs(final_y - expected) < 1e-8, "full integration should be accurate");
}

void test_scalar_exponential_with_intermediate() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto trajectory = integrate<double>(solver, 1.0, 0.0, 0.1, 0.01, deriv, true);

    expect(trajectory.size() > 2, "intermediate storage should produce many points");

    // Check that first point is the initial condition
    expect(std::abs(trajectory[0].first - 0.0) < 1e-15, "first time should be t0");
    expect(std::abs(trajectory[0].second - 1.0) < 1e-15, "first state should be y0");

    // Check monotonically increasing time
    for (size_t i = 1; i < trajectory.size(); ++i) {
        expect(trajectory[i].first > trajectory[i - 1].first, "time should increase");
    }
}

// --------------------------------------------------------------------------
// Scalar: dy/dt = -y  =>  y(t) = exp(-t)  (decay)
// --------------------------------------------------------------------------

void test_scalar_decay() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return -y; };

    auto trajectory = integrate<double>(solver, 1.0, 0.0, 5.0, 0.01, deriv, false);

    double final_y = trajectory.back().second;
    double expected = std::exp(-5.0);
    expect(std::abs(final_y - expected) < 1e-8, "exponential decay should be accurate");
}

// --------------------------------------------------------------------------
// Scalar: dy/dt = cos(t)  =>  y(t) = sin(t)
// --------------------------------------------------------------------------

void test_scalar_sinusoidal() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double&, double t) { return std::cos(t); };

    auto trajectory = integrate<double>(solver, 0.0, 0.0, 2.0 * M_PI, 0.001, deriv, false);

    double final_y = trajectory.back().second;
    // sin(2*pi) = 0
    expect(std::abs(final_y) < 1e-8, "sinusoidal should return to zero after full period");
}

// --------------------------------------------------------------------------
// Harmonic oscillator: y'' = -y  =>  y(t) = cos(t)
// --------------------------------------------------------------------------

void test_harmonic_oscillator() {
    RK4Config<Vec2> cfg;
    cfg.norm = vec2Norm;
    RK4Integrator<Vec2> solver(cfg);

    DerivativeFunction<Vec2> deriv = [](const Vec2& s, double) {
        return Vec2{s.y, -s.x};
    };

    Vec2 y0{1.0, 0.0}; // cos(0) = 1, sin(0) = 0
    auto trajectory = integrate<Vec2>(solver, y0, 0.0, 2.0 * M_PI, 0.001, deriv, false);

    Vec2 final_state = trajectory.back().second;
    // After one full period, should return to (1, 0)
    expect(std::abs(final_state.x - 1.0) < 1e-5, "harmonic x should return to 1");
    expect(std::abs(final_state.y) < 1e-5, "harmonic y should return to 0");
}

void test_harmonic_energy_conservation() {
    RK4Config<Vec2> cfg;
    cfg.norm = vec2Norm;
    RK4Integrator<Vec2> solver(cfg);

    DerivativeFunction<Vec2> deriv = [](const Vec2& s, double) {
        return Vec2{s.y, -s.x};
    };

    Vec2 y0{1.0, 0.0};
    auto trajectory = integrate<Vec2>(solver, y0, 0.0, 10.0, 0.001, deriv, true);

    double initial_energy = 0.5 * (y0.x * y0.x + y0.y * y0.y);
    double max_energy_drift = 0.0;

    for (const auto& [t, s] : trajectory) {
        double energy = 0.5 * (s.x * s.x + s.y * s.y);
        max_energy_drift = std::max(max_energy_drift, std::abs(energy - initial_energy));
    }

    expect(max_energy_drift < 1e-6, "harmonic energy should be approximately conserved");
}

// --------------------------------------------------------------------------
// Edge cases and validation
// --------------------------------------------------------------------------

void test_zero_step_size_rejected() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.0, deriv);
    expect(!result.success, "zero step size should fail");
}

void test_negative_step_size_rejected() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, -0.01, deriv);
    expect(!result.success, "negative step size should fail");
}

void test_step_exceeding_max_rejected() {
    RK4Config<double> cfg;
    cfg.max_time_step = 0.5;
    RK4Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.6, deriv);
    expect(!result.success, "step exceeding max should fail");
}

void test_validator_rejects_invalid_state() {
    RK4Config<double> cfg;
    cfg.enable_validation = true;
    cfg.validator = [](const double& y) { return std::isfinite(y); };

    RK4Integrator<double> solver(cfg);
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(std::numeric_limits<double>::infinity(), 0.0, 0.01, deriv);
    expect(!result.success, "validator should reject infinite input state");
}

void test_validator_rejects_invalid_derivative() {
    RK4Config<double> cfg;
    cfg.enable_validation = true;
    cfg.validator = [](const double& y) { return std::isfinite(y); };

    RK4Integrator<double> solver(cfg);
    DerivativeFunction<double> deriv = [](const double&, double) {
        return std::numeric_limits<double>::infinity();
    };

    auto result = solver.step(1.0, 0.0, 0.01, deriv);
    expect(!result.success, "validator should reject infinite derivative");
}

void test_error_estimate_with_norm() {
    RK4Config<double> cfg;
    cfg.norm = [](const double& s) { return std::abs(s); };
    RK4Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.1, deriv);
    expect(result.success, "step should succeed");
    expect(result.estimated_error > 0.0, "error estimate should be positive with norm");
}

void test_error_estimate_without_norm() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.1, deriv);
    expect(result.success, "step should succeed without norm");
    expect(result.estimated_error == 0.0, "error estimate should be zero without norm");
}

// --------------------------------------------------------------------------
// Metadata
// --------------------------------------------------------------------------

void test_type_and_order() {
    RK4Integrator<double> solver;
    expect(solver.getType() == "rk4", "type should be rk4");
    expect(solver.getOrder() == 4, "order should be 4");
    expect(!solver.supportsAdaptiveStep(), "RK4 should not support adaptive step");
}

// --------------------------------------------------------------------------
// Integration edge cases
// --------------------------------------------------------------------------

void test_integrate_invalid_interval() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = integrate<double>(solver, 1.0, 1.0, 0.0, 0.01, deriv, false);
    expect(result.empty(), "inverted interval should return empty");
}

void test_integrate_last_step_truncated() {
    RK4Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    // Integrate from 0 to 0.15 with dt=0.1: two steps, second is 0.05
    auto result = integrate<double>(solver, 1.0, 0.0, 0.15, 0.1, deriv, true);
    expect(result.size() >= 3, "should have at least 3 points (initial + 2 steps)");

    double final_t = result.back().first;
    expect(std::abs(final_t - 0.15) < 1e-12, "final time should match endpoint");
}

// --------------------------------------------------------------------------
// Convergence order verification
// --------------------------------------------------------------------------

void test_fourth_order_convergence() {
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    double prev_error = 0.0;
    double prev_h = 0.0;

    for (double h : {0.1, 0.05, 0.025, 0.0125}) {
        RK4Integrator<double> solver;
        auto traj = integrate<double>(solver, 1.0, 0.0, 1.0, h, deriv, false);
        double error = std::abs(traj.back().second - std::exp(1.0));

        if (prev_h > 0.0 && error > 1e-15 && prev_error > 1e-15) {
            double ratio = std::log(prev_error / error) / std::log(prev_h / h);
            // Expect ratio close to 4 (4th order method)
            expect(ratio > 3.5 && ratio < 4.5, "convergence order should be ~4");
        }

        prev_error = error;
        prev_h = h;
    }
}

} // namespace

int main() {
    // Scalar accuracy
    test_scalar_exponential();
    test_scalar_exponential_full_integration();
    test_scalar_exponential_with_intermediate();
    test_scalar_decay();
    test_scalar_sinusoidal();

    // Vector accuracy
    test_harmonic_oscillator();
    test_harmonic_energy_conservation();

    // Edge cases
    test_zero_step_size_rejected();
    test_negative_step_size_rejected();
    test_step_exceeding_max_rejected();
    test_validator_rejects_invalid_state();
    test_validator_rejects_invalid_derivative();
    test_error_estimate_with_norm();
    test_error_estimate_without_norm();

    // Metadata
    test_type_and_order();

    // Integration edge cases
    test_integrate_invalid_interval();
    test_integrate_last_step_truncated();

    // Convergence
    test_fourth_order_convergence();

    if (g_failures != 0) {
        std::cerr << "Total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "All RK4 tests passed\n";
    return 0;
}
