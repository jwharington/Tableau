#include "DOP853.h"

#include <cmath>
#include <iostream>
#include <vector>

using tableau::integration::DOP853Config;
using tableau::integration::DOP853Integrator;
using tableau::integration::DOP853OutputMode;
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

void test_scalar_tolerance() {
    using Vec = std::vector<double>;

    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double /*t*/) {
        return Vec{y[0]};
    };
    cfg.output_mode = DOP853OutputMode::None;
    cfg.hinitial = 0.0;
    cfg.max_step = 0.25;
    cfg.min_step = 1e-14;
    cfg.nmax = 100000;

    DOP853Integrator<Vec> integrator(cfg);
    const auto result = integrator.integrate(0.0, Vec{1.0}, 1.0, DOP853Tolerance::scalar(1e-12, 1e-12));

    expect(result.status == DOP853Status::Success, "scalar tolerance run should succeed");
    expect(result.stats.naccpt > 0, "scalar tolerance run should accept at least one step");
    expect(result.stats.nfcn > 0, "scalar tolerance run should evaluate derivative");

    const double expected = std::exp(1.0);
    expect(std::abs(result.y[0] - expected) < 1e-9, "scalar tolerance solution should match exp(1)");
}

void test_vector_tolerance() {
    using Vec = std::vector<double>;

    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double /*t*/) {
        return Vec{y[0], -2.0 * y[1]};
    };
    cfg.output_mode = DOP853OutputMode::None;
    cfg.max_step = 0.2;

    DOP853Integrator<Vec> integrator(cfg);
    const auto result = integrator.integrate(
        0.0,
        Vec{1.0, 1.0},
        1.0,
        DOP853Tolerance::perComponent({1e-11, 1e-11}, {1e-11, 1e-11}));

    expect(result.status == DOP853Status::Success, "vector tolerance run should succeed");
    expect(std::abs(result.y[0] - std::exp(1.0)) < 1e-8, "first component should match exp(1)");
    expect(std::abs(result.y[1] - std::exp(-2.0)) < 1e-8, "second component should match exp(-2)");
}

void test_dense_output_and_callback() {
    using Vec = std::vector<double>;

    int callback_count = 0;
    double dense_max_error = 0.0;

    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double /*t*/) {
        return Vec{y[0]};
    };
    cfg.output_mode = DOP853OutputMode::DenseEveryStep;
    cfg.dense_components = {0};
    cfg.max_step = 0.3;
    cfg.solout = [&](const DOP853StepEvent& event, double& /*xout*/) {
        ++callback_count;

        if (event.dense_output != nullptr) {
            const double xmid = 0.5 * (event.x_old + event.x);
            const double ymid = event.dense_output->evaluate(0, xmid);
            const double expected = std::exp(xmid);
            dense_max_error = std::max(dense_max_error, std::abs(ymid - expected));
        }

        return 1;
    };

    DOP853Integrator<Vec> integrator(cfg);
    const auto result = integrator.integrate(0.0, Vec{1.0}, 1.0, DOP853Tolerance::scalar(1e-11, 1e-11));

    expect(result.status == DOP853Status::Success, "dense output run should succeed");
    expect(callback_count > 0, "dense callback should be called");
    expect(dense_max_error < 5e-8, "dense interpolation should be accurate");
    expect(integrator.lastDenseOutput() != nullptr, "lastDenseOutput should be available after dense run");
}

void test_dense_sparse_output_mode() {
    using Vec = std::vector<double>;

    int sparse_calls = 0;
    double max_midpoint_error = 0.0;
    double next_xout = 0.2;

    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double /*t*/) {
        return Vec{y[0]};
    };
    cfg.output_mode = DOP853OutputMode::DenseSparse;
    cfg.dense_components = {0};
    cfg.max_step = 0.15;
    cfg.solout = [&](const DOP853StepEvent& event, double& xout) {
        if (event.step_number == 1) {
            xout = next_xout;
            return 1;
        }

        ++sparse_calls;
        expect(event.dense_output != nullptr, "DenseSparse callback should receive dense output");
        expect(event.step_size > 0.0, "DenseSparse callback should include step_size");

        const double xmid = 0.5 * (event.x_old + event.x);
        const double ymid = event.dense_output->evaluate(0, xmid);
        max_midpoint_error = std::max(max_midpoint_error, std::abs(ymid - std::exp(xmid)));

        next_xout += 0.2;
        xout = next_xout;
        return 1;
    };

    DOP853Integrator<Vec> integrator(cfg);
    const auto result = integrator.integrate(0.0, Vec{1.0}, 1.0, DOP853Tolerance::scalar(1e-11, 1e-11));

    expect(result.status == DOP853Status::Success, "DenseSparse run should succeed");
    expect(sparse_calls > 0, "DenseSparse callback should be triggered");
    expect(max_midpoint_error < 8e-8, "DenseSparse interpolation should stay accurate");
}

void test_interrupted_run() {
    using Vec = std::vector<double>;

    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double /*t*/) {
        return Vec{y[0]};
    };
    cfg.output_mode = DOP853OutputMode::EveryAcceptedStep;
    cfg.max_step = 0.1;
    cfg.solout = [](const DOP853StepEvent& event, double& /*xout*/) {
        if (event.step_number >= 4) {
            return -1;
        }
        return 1;
    };

    DOP853Integrator<Vec> integrator(cfg);
    const auto result = integrator.integrate(0.0, Vec{1.0}, 10.0, DOP853Tolerance::scalar(1e-9, 1e-9));

    expect(result.status == DOP853Status::Interrupted, "callback should interrupt run");
}

void test_probably_stiff_detection() {
    using Vec = std::vector<double>;

    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double t) {
        // Very stiff test problem with known bounded solution.
        return Vec{-1000.0 * (y[0] - std::cos(t)) - std::sin(t)};
    };
    cfg.output_mode = DOP853OutputMode::None;
    cfg.nstiff = 1;
    cfg.max_step = 1.0;
    cfg.nmax = 200000;

    DOP853Integrator<Vec> integrator(cfg);
    const auto result = integrator.integrate(0.0, Vec{1.0}, 20.0, DOP853Tolerance::scalar(1e-6, 1e-6));

    expect(result.status == DOP853Status::ProbablyStiff, "stiff detector should trigger ProbablyStiff");
}

void test_invalid_tolerance_dimensions() {
    using Vec = std::vector<double>;

    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double /*t*/) {
        return Vec{y[0], y[1]};
    };

    DOP853Integrator<Vec> integrator(cfg);
    const auto result = integrator.integrate(
        0.0,
        Vec{1.0, 1.0},
        1.0,
        DOP853Tolerance::perComponent({1e-8}, {1e-8, 1e-8}));

    expect(result.status == DOP853Status::InvalidInput, "invalid per-component tolerance sizes should fail");
}

void test_step_api_smoke() {
    using Vec = std::vector<double>;

    DOP853Config<Vec> cfg;
    cfg.derivative = [](const Vec& y, double /*t*/) {
        return Vec{y[0]};
    };

    DOP853Integrator<Vec> integrator(cfg);
    const auto step_result = integrator.step(Vec{1.0}, 0.0, 0.01, cfg.derivative);
    expect(step_result.success, "step API should produce a successful trial step");

    const auto adaptive = integrator.adaptiveStep(Vec{1.0}, 0.0, 0.05, 1.0, cfg.derivative);
    expect(adaptive.success, "adaptiveStep API should find an acceptable step");
}

} // namespace

int main() {
    test_scalar_tolerance();
    test_vector_tolerance();
    test_dense_output_and_callback();
    test_dense_sparse_output_mode();
    test_interrupted_run();
    test_probably_stiff_detection();
    test_invalid_tolerance_dimensions();
    test_step_api_smoke();

    if (g_failures != 0) {
        std::cerr << "Total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "All DOP853 tests passed\n";
    return 0;
}
