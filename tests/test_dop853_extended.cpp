#include "DOP853.h"
#include "Integration.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using tableau::integration::DerivativeFunction;
using tableau::integration::DOP853Config;
using tableau::integration::DOP853DenseOutput;
using tableau::integration::DOP853Integrator;
using tableau::integration::DOP853OutputMode;
using tableau::integration::DOP853RunResult;
using tableau::integration::DOP853Stats;
using tableau::integration::DOP853Status;
using tableau::integration::DOP853StepEvent;
using tableau::integration::DOP853Tolerance;

namespace {

int g_failures = 0;

void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        ++g_failures;
    }
}

using Vec = std::vector<double>;

// ============================================================================
// SCALAR STATE (non-vector path)
// ============================================================================

void test_scalar_step() {
    DOP853Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.01, deriv);

    expect(result.success, "scalar step should succeed");
    expect(result.method_used == "dop853", "method should be dop853");
    expect(std::abs(result.state - std::exp(0.01)) < 1e-14,
           "8th order should be extremely accurate for single step");
}

void test_scalar_adaptive_step() {
    DOP853Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.adaptiveStep(1.0, 0.0, 0.05, 1e-8, deriv);

    expect(result.success, "scalar adaptive step should succeed");
    expect(result.estimated_error <= 1e-8 || result.estimated_error == 0.0,
           "error should be within target");
}

void test_scalar_type_and_order() {
    DOP853Integrator<double> solver;
    expect(solver.getType() == "dop853", "type should be dop853");
    expect(solver.getOrder() == 8, "order should be 8");
    expect(solver.supportsAdaptiveStep(), "DOP853 should support adaptive step");
}

// ============================================================================
// FULL INTEGRATION API (std::vector<double> path)
// ============================================================================

void test_lorenz_finite_solution() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& s, double) -> Vec {
        constexpr double sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;
        return {sigma * (s[1] - s[0]),
                s[0] * (rho - s[2]) - s[1],
                s[0] * s[1] - beta * s[2]};
    };
    cfg.output_mode = DOP853OutputMode::None;
    cfg.max_step = 1.0;

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{1.0, 1.0, 1.0}, 20.0,
                                   DOP853Tolerance::scalar(1e-10, 1e-10));

    expect(result.status == DOP853Status::Success, "Lorenz should complete");
    expect(result.y.size() == 3, "output should have 3 components");

    for (double v : result.y) {
        expect(std::isfinite(v), "Lorenz solution should stay finite");
    }

    // Lorenz attractor is bounded: |x|, |y| < 50, |z| < 60 typically
    expect(std::abs(result.y[0]) < 100.0, "x should be bounded");
    expect(std::abs(result.y[1]) < 100.0, "y should be bounded");
    expect(std::abs(result.y[2]) < 100.0, "z should be bounded");
}

void test_stats_populated() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::None;

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{1.0}, 1.0,
                                   DOP853Tolerance::scalar(1e-8, 1e-8));

    expect(result.status == DOP853Status::Success, "should succeed");
    expect(result.stats.nfcn > 0, "nfcn should be positive");
    expect(result.stats.nstep > 0, "nstep should be positive");
    expect(result.stats.naccpt > 0, "naccpt should be positive");
    expect(result.stats.naccpt <= result.stats.nstep, "accepted <= total steps");
}

void test_suggested_step_returned() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::None;

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{1.0}, 1.0,
                                   DOP853Tolerance::scalar(1e-8, 1e-8));

    expect(result.status == DOP853Status::Success, "should succeed");
    expect(result.suggested_h > 0.0, "suggested h should be positive");
}

// ============================================================================
// TOLERANCE SENSITIVITY
// ============================================================================

void test_tighter_tolerance_more_steps() {
    auto make_solver = [](double tol) {
        DOP853Config<Vec> cfg;
        cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
        cfg.output_mode = DOP853OutputMode::None;
        cfg.max_step = 1.0;
        DOP853Integrator<Vec> solver(cfg);
        return solver.integrate(0.0, Vec{1.0}, 1.0,
                                DOP853Tolerance::scalar(tol, tol));
    };

    auto loose = make_solver(1e-4);
    auto tight = make_solver(1e-12);

    expect(loose.status == DOP853Status::Success, "loose should succeed");
    expect(tight.status == DOP853Status::Success, "tight should succeed");
    expect(tight.stats.nstep >= loose.stats.nstep,
           "tighter tolerance should require more steps");
}

void test_tighter_tolerance_better_accuracy() {
    auto run = [](double tol) {
        DOP853Config<Vec> cfg;
        cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
        cfg.output_mode = DOP853OutputMode::None;
        cfg.max_step = 1.0;
        DOP853Integrator<Vec> solver(cfg);
        return solver.integrate(0.0, Vec{1.0}, 1.0,
                                DOP853Tolerance::scalar(tol, tol));
    };

    auto loose = run(1e-4);
    auto tight = run(1e-12);

    double err_loose = std::abs(loose.y[0] - std::exp(1.0));
    double err_tight = std::abs(tight.y[0] - std::exp(1.0));

    expect(err_tight < err_loose, "tighter tolerance should give better accuracy");
}

// ============================================================================
// PER-COMPONENT TOLERANCES
// ============================================================================

void test_per_component_tolerance_accuracy() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec {
        return {y[0], -2.0 * y[1], 3.0 * y[2]};
    };
    cfg.output_mode = DOP853OutputMode::None;

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(
        0.0,
        Vec{1.0, 1.0, 1.0},
        1.0,
        DOP853Tolerance::perComponent(
            {1e-10, 1e-10, 1e-10},
            {1e-10, 1e-10, 1e-10}));

    expect(result.status == DOP853Status::Success, "per-component should succeed");
    expect(std::abs(result.y[0] - std::exp(1.0)) < 1e-8, "component 0");
    expect(std::abs(result.y[1] - std::exp(-2.0)) < 1e-8, "component 1");
    expect(std::abs(result.y[2] - std::exp(3.0)) < 1e-6, "component 2");
}

void test_per_component_tolerance_mismatched_atol_size() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0], y[1]}; };

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(
        0.0, Vec{1.0, 1.0}, 1.0,
        DOP853Tolerance::perComponent({1e-8, 1e-8, 1e-8}, {1e-8}));

    expect(result.status == DOP853Status::InvalidInput,
           "mismatched atol size should fail");
}

void test_per_component_tolerance_mismatched_rtol_size() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(
        0.0, Vec{1.0}, 1.0,
        DOP853Tolerance::perComponent({1e-8, 1e-8}, {1e-8}));

    expect(result.status == DOP853Status::InvalidInput,
           "mismatched rtol size should fail");
}

// ============================================================================
// DENSE OUTPUT
// ============================================================================

void test_dense_output_continuity() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::DenseEveryStep;
    cfg.dense_components = {0};
    cfg.max_step = 0.3;

    double max_discontinuity = 0.0;
    double prev_y_end = 1.0;

    cfg.solout = [&](const DOP853StepEvent& event, double&) {
        if (event.dense_output != nullptr) {
            // Evaluate at the left boundary of this step
            double y_at_xold = event.dense_output->evaluate(0, event.x_old);
            max_discontinuity = std::max(max_discontinuity,
                                         std::abs(y_at_xold - prev_y_end));
            // Evaluate at the right boundary
            prev_y_end = event.dense_output->evaluate(0, event.x);
        }
        return 1;
    };

    DOP853Integrator<Vec> solver(cfg);
    solver.integrate(0.0, Vec{1.0}, 2.0, DOP853Tolerance::scalar(1e-10, 1e-10));

    expect(max_discontinuity < 1e-10,
           "dense output should be continuous across step boundaries");
}

void test_dense_output_accuracy_at_midpoints() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::DenseEveryStep;
    cfg.dense_components = {0};
    cfg.max_step = 0.5;

    double max_error = 0.0;

    cfg.solout = [&](const DOP853StepEvent& event, double&) {
        if (event.dense_output != nullptr) {
            // Sample 10 points within the step
            for (int k = 1; k <= 10; ++k) {
                double frac = static_cast<double>(k) / 11.0;
                double x = event.x_old + frac * (event.x - event.x_old);
                double y_interp = event.dense_output->evaluate(0, x);
                double y_exact = std::exp(x);
                max_error = std::max(max_error, std::abs(y_interp - y_exact));
            }
        }
        return 1;
    };

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{1.0}, 2.0,
                                   DOP853Tolerance::scalar(1e-11, 1e-11));

    expect(result.status == DOP853Status::Success, "should succeed");
    expect(max_error < 1e-7, "dense output midpoint accuracy");
}

void test_dense_output_invalid_component_throws() {
    DOP853DenseOutput dense;
    dense.components = {0, 1};
    dense.cont.resize(8 * 2, 0.1);
    dense.xold = 0.0;
    dense.hout = 0.1;

    bool threw = false;
    try {
        dense.evaluate(5, 0.05); // component 5 not available
    } catch (const std::out_of_range&) {
        threw = true;
    }
    expect(threw, "evaluating unavailable component should throw");
}

void test_dense_output_zero_step_throws() {
    DOP853DenseOutput dense;
    dense.components = {0};
    dense.cont.resize(8, 0.1);
    dense.xold = 0.0;
    dense.hout = 0.0;

    bool threw = false;
    try {
        dense.evaluate(0, 0.0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    expect(threw, "evaluating with zero step should throw");
}

// ============================================================================
// CALLBACK CONTROL
// ============================================================================

void test_callback_receives_step_info() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::EveryAcceptedStep;

    bool saw_valid_event = false;
    cfg.solout = [&](const DOP853StepEvent& event, double&) {
        if (event.step_number > 0 &&
            event.x > event.x_old &&
            event.step_size > 0.0 &&
            event.y != nullptr &&
            event.y->size() == 1) {
            saw_valid_event = true;
        }
        return 1;
    };

    DOP853Integrator<Vec> solver(cfg);
    solver.integrate(0.0, Vec{1.0}, 1.0, DOP853Tolerance::scalar(1e-8, 1e-8));

    expect(saw_valid_event, "callback should receive valid step events");
}

void test_callback_interrupts_at_specific_time() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::EveryAcceptedStep;

    double interrupt_x = 0.0;
    cfg.solout = [&](const DOP853StepEvent& event, double&) {
        if (event.x >= 0.5) {
            interrupt_x = event.x;
            return -1;
        }
        return 1;
    };

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{1.0}, 10.0,
                                   DOP853Tolerance::scalar(1e-8, 1e-8));

    expect(result.status == DOP853Status::Interrupted, "should be interrupted");
    expect(interrupt_x >= 0.5 && interrupt_x < 2.0,
           "should interrupt near requested time");
}

void test_no_callback_mode_none() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::None;

    int callback_count = 0;
    cfg.solout = [&](const DOP853StepEvent&, double&) {
        ++callback_count;
        return 1;
    };

    DOP853Integrator<Vec> solver(cfg);
    solver.integrate(0.0, Vec{1.0}, 1.0, DOP853Tolerance::scalar(1e-8, 1e-8));

    expect(callback_count == 0, "OutputMode::None should not trigger callbacks");
}

// ============================================================================
// STEP SIZE LIMITS
// ============================================================================

void test_max_steps_exceeded() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::None;
    cfg.nmax = 5;
    cfg.max_step = 0.001; // Force many tiny steps

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{1.0}, 100.0,
                                   DOP853Tolerance::scalar(1e-12, 1e-12));

    expect(result.status == DOP853Status::TooManySteps,
           "should hit TooManySteps with low nmax");
}

// ============================================================================
// MULTI-COMPONENT DENSE OUTPUT
// ============================================================================

void test_multi_component_dense() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec {
        return {y[0], -y[1]};
    };
    cfg.output_mode = DOP853OutputMode::DenseEveryStep;
    cfg.dense_components = {0, 1};
    cfg.max_step = 0.4;

    double max_err_0 = 0.0;
    double max_err_1 = 0.0;

    cfg.solout = [&](const DOP853StepEvent& event, double&) {
        if (event.dense_output != nullptr) {
            double xmid = 0.5 * (event.x_old + event.x);
            double y0 = event.dense_output->evaluate(0, xmid);
            double y1 = event.dense_output->evaluate(1, xmid);
            max_err_0 = std::max(max_err_0, std::abs(y0 - std::exp(xmid)));
            max_err_1 = std::max(max_err_1, std::abs(y1 - std::exp(-xmid)));
        }
        return 1;
    };

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{1.0, 1.0}, 2.0,
                                   DOP853Tolerance::scalar(1e-11, 1e-11));

    expect(result.status == DOP853Status::Success, "multi-component should succeed");
    expect(max_err_0 < 1e-7, "component 0 dense accuracy");
    expect(max_err_1 < 1e-7, "component 1 dense accuracy");
}

// ============================================================================
// COEFFICIENT SNAPSHOT
// ============================================================================

void test_coefficient_snapshot() {
    DOP853Integrator<Vec> solver;
    auto coeffs = solver.coefficients();

    // Verify some known DOP853 nodes (c values)
    expect(coeffs.c[0] == 0.0, "c[0] should be 0");
    expect(coeffs.c[11] == 1.0, "c[11] should be 1 (last node)");

    // bhh coefficients for the 3rd-order error estimate
    expect(coeffs.bhh1 != 0.0, "bhh1 should be nonzero");
    expect(coeffs.bhh2 != 0.0, "bhh2 should be nonzero");
    expect(coeffs.bhh3 != 0.0, "bhh3 should be nonzero");
}

// ============================================================================
// VAN DER POL OSCILLATOR (nonlinear, stiffness boundary)
// ============================================================================

void test_van_der_pol() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& s, double) -> Vec {
        constexpr double mu = 1.0;
        return {s[1], mu * (1.0 - s[0] * s[0]) * s[1] - s[0]};
    };
    cfg.output_mode = DOP853OutputMode::None;
    cfg.max_step = 1.0;
    cfg.nmax = 200000;

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{2.0, 0.0}, 20.0,
                                   DOP853Tolerance::scalar(1e-10, 1e-10));

    expect(result.status == DOP853Status::Success, "Van der Pol should succeed with mu=1");
    expect(result.y.size() == 2, "should have 2 components");

    // The limit cycle amplitude is approximately 2 for mu=1
    double amplitude = std::sqrt(result.y[0] * result.y[0] + result.y[1] * result.y[1]);
    expect(amplitude < 10.0, "Van der Pol should stay bounded");
}

// ============================================================================
// KEPLER PROBLEM (energy conservation test)
// ============================================================================

void test_kepler_energy_conservation() {
    DOP853Config<Vec> cfg;
    // State: [x, y, vx, vy], circular orbit at r=1, v=1
    cfg.derivative = [](const Vec& s, double) -> Vec {
        double r3 = std::pow(s[0] * s[0] + s[1] * s[1], 1.5);
        return {s[2], s[3], -s[0] / r3, -s[1] / r3};
    };
    cfg.output_mode = DOP853OutputMode::None;
    cfg.max_step = 1.0;

    DOP853Integrator<Vec> solver(cfg);
    Vec y0 = {1.0, 0.0, 0.0, 1.0}; // circular orbit

    auto result = solver.integrate(0.0, y0, 2.0 * M_PI,
                                   DOP853Tolerance::scalar(1e-12, 1e-12));

    expect(result.status == DOP853Status::Success, "Kepler should succeed");

    // Energy = 0.5*v^2 - 1/r
    auto energy = [](const Vec& s) {
        double r = std::sqrt(s[0] * s[0] + s[1] * s[1]);
        double v2 = s[2] * s[2] + s[3] * s[3];
        return 0.5 * v2 - 1.0 / r;
    };

    double E0 = energy(y0);
    double E1 = energy(result.y);
    expect(std::abs(E1 - E0) < 1e-10, "Kepler energy should be conserved to high precision");

    // After one period of circular orbit, should return close to (1, 0, 0, 1)
    expect(std::abs(result.y[0] - 1.0) < 1e-8, "x should return to 1");
    expect(std::abs(result.y[1]) < 1e-8, "y should return to 0");
}

// ============================================================================
// EMPTY / DEGENERATE INPUTS
// ============================================================================

void test_zero_length_integration() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0]}; };
    cfg.output_mode = DOP853OutputMode::None;

    DOP853Integrator<Vec> solver(cfg);
    auto result = solver.integrate(0.0, Vec{1.0}, 0.0,
                                   DOP853Tolerance::scalar(1e-8, 1e-8));

    // Integrating from 0 to 0 should either succeed trivially or return input
    // The important thing is it doesn't crash
    expect(result.y.size() == 1, "should return a valid vector");
}

void test_vector_step_api() {
    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double) -> Vec { return {y[0], -y[1]}; };

    DOP853Integrator<Vec> solver(cfg);

    auto result = solver.step(Vec{1.0, 1.0}, 0.0, 0.01, cfg.derivative);
    expect(result.success, "vector step should succeed");
    expect(result.state.size() == 2, "output should have 2 components");

    expect(std::abs(result.state[0] - std::exp(0.01)) < 1e-12, "component 0 accuracy");
    expect(std::abs(result.state[1] - std::exp(-0.01)) < 1e-12, "component 1 accuracy");
}

void test_vector_adaptive_step_api() {
    DOP853Integrator<Vec> solver;
    DerivativeFunction<Vec> deriv = [](const Vec& y, double) -> Vec { return {y[0]}; };

    auto result = solver.adaptiveStep(Vec{1.0}, 0.0, 0.05, 1e-8, deriv);
    expect(result.success, "vector adaptive step should succeed");
}

} // namespace

int main() {
    // Scalar path
    test_scalar_step();
    test_scalar_adaptive_step();
    test_scalar_type_and_order();

    // Full integration
    test_lorenz_finite_solution();
    test_stats_populated();
    test_suggested_step_returned();

    // Tolerance
    test_tighter_tolerance_more_steps();
    test_tighter_tolerance_better_accuracy();
    test_per_component_tolerance_accuracy();
    test_per_component_tolerance_mismatched_atol_size();
    test_per_component_tolerance_mismatched_rtol_size();

    // Dense output
    test_dense_output_continuity();
    test_dense_output_accuracy_at_midpoints();
    test_dense_output_invalid_component_throws();
    test_dense_output_zero_step_throws();
    test_multi_component_dense();

    // Callback control
    test_callback_receives_step_info();
    test_callback_interrupts_at_specific_time();
    test_no_callback_mode_none();

    // Limits
    test_max_steps_exceeded();

    // Coefficients
    test_coefficient_snapshot();

    // Nonlinear systems
    test_van_der_pol();
    test_kepler_energy_conservation();

    // Edge cases
    test_zero_length_integration();
    test_vector_step_api();
    test_vector_adaptive_step_api();

    if (g_failures != 0) {
        std::cerr << "Total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "All DOP853 extended tests passed\n";
    return 0;
}
