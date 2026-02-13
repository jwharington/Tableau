#include "RKF45.h"
#include "Integration.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

using tableau::integration::DerivativeFunction;
using tableau::integration::IntegrationResult;
using tableau::integration::RKF45Config;
using tableau::integration::RKF45Integrator;
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
// Custom vector state
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

// ============================================================================
// FIXED STEP TESTS
// ============================================================================

void test_scalar_step() {
    RKF45Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.01, deriv);

    expect(result.success, "scalar fixed step should succeed");
    expect(result.method_used == "rkf45", "method should be rkf45");
    expect(result.time_step_used == 0.01, "dt should match");

    double expected = std::exp(0.01);
    expect(std::abs(result.state - expected) < 1e-10, "scalar step accuracy");
}

void test_step_returns_error_estimate() {
    RKF45Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.01, deriv);
    expect(result.success, "step should succeed");
    expect(result.estimated_error > 0.0, "error estimate should be positive");
}

void test_step_zero_dt_rejected() {
    RKF45Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.0, deriv);
    expect(!result.success, "zero dt should fail");
}

void test_step_negative_dt_rejected() {
    RKF45Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, -0.01, deriv);
    expect(!result.success, "negative dt should fail");
}

void test_step_exceeds_max_rejected() {
    RKF45Config<double> cfg;
    cfg.max_step = 0.05;
    RKF45Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 0.1, deriv);
    expect(!result.success, "step exceeding max should fail");
}

void test_step_below_min_rejected() {
    RKF45Config<double> cfg;
    cfg.min_step = 1e-6;
    RKF45Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(1.0, 0.0, 1e-8, deriv);
    expect(!result.success, "step below min should fail");
}

// ============================================================================
// ADAPTIVE STEP TESTS
// ============================================================================

void test_adaptive_scalar_exponential() {
    RKF45Integrator<double> solver;
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.adaptiveStep(1.0, 0.0, 0.05, 1e-6, deriv);

    expect(result.success, "adaptive step should succeed");
    expect(result.estimated_error <= 1e-6 || result.estimated_error == 0.0,
           "error should be within tolerance");
}

void test_adaptive_step_reduces_on_stiff_problem() {
    RKF45Config<double> cfg;
    cfg.max_step = 1.0;
    cfg.min_step = 1e-14;
    cfg.max_rejections = 50;
    RKF45Integrator<double> solver(cfg);

    // Mildly stiff: dy/dt = -50*y
    DerivativeFunction<double> deriv = [](const double& y, double) { return -50.0 * y; };

    auto result = solver.adaptiveStep(1.0, 0.0, 0.1, 1e-6, deriv);

    expect(result.success, "adaptive should eventually find acceptable step");
    expect(result.time_step_used <= 0.1, "step should have been reduced");
}

void test_adaptive_exhausts_rejections() {
    RKF45Config<double> cfg;
    cfg.max_rejections = 2;
    cfg.min_step = 0.01; // Large min_step forces failure
    cfg.max_step = 0.1;
    RKF45Integrator<double> solver(cfg);

    // Very rapidly growing derivative forces repeated rejection
    DerivativeFunction<double> deriv = [](const double& y, double) { return 1e6 * y; };

    auto result = solver.adaptiveStep(1.0, 0.0, 0.1, 1e-12, deriv);
    expect(!result.success, "should fail after exhausting rejections");
}

// ============================================================================
// ACCURACY ON KNOWN SOLUTIONS
// ============================================================================

void test_exponential_decay_accuracy() {
    RKF45Config<double> cfg;
    cfg.max_step = 1.0;
    RKF45Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double& y, double) { return -y; };

    // Integrate via adaptive steps manually
    double y = 1.0;
    double t = 0.0;
    double dt = 0.01;

    while (t < 3.0) {
        double remaining = 3.0 - t;
        double step = std::min(dt, remaining);
        auto result = solver.adaptiveStep(y, t, step, 1e-8, deriv);
        if (!result.success) break;
        y = result.state;
        t += result.time_step_used;
        dt = result.time_step_used * 1.2; // grow step
    }

    expect(std::abs(y - std::exp(-3.0)) < 1e-6, "decay solution accuracy");
}

void test_harmonic_oscillator_adaptive() {
    RKF45Config<Vec2> cfg;
    cfg.norm = vec2Norm;
    cfg.max_step = 1.0;
    cfg.min_step = 1e-10;
    RKF45Integrator<Vec2> solver(cfg);

    DerivativeFunction<Vec2> deriv = [](const Vec2& s, double) {
        return Vec2{s.y, -s.x};
    };

    Vec2 y{1.0, 0.0};
    double t = 0.0;
    double dt = 0.05;

    while (t < 2.0 * M_PI) {
        double remaining = 2.0 * M_PI - t;
        double step = std::min(dt, remaining);
        auto result = solver.adaptiveStep(y, t, step, 1e-8, deriv);
        if (!result.success) break;
        y = result.state;
        t += result.time_step_used;
        dt = result.time_step_used;
    }

    // After one period, should return close to (1, 0)
    expect(std::abs(y.x - 1.0) < 1e-4, "harmonic x after one period");
    expect(std::abs(y.y) < 1e-4, "harmonic y after one period");
}

// ============================================================================
// FULL INTEGRATION (integrate<> free function with RKF45)
// ============================================================================

void test_full_integration_scalar() {
    RKF45Config<double> cfg;
    cfg.max_step = 0.5;
    RKF45Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto traj = integrate<double>(solver, 1.0, 0.0, 1.0, 0.01, deriv, false);

    expect(traj.size() >= 2, "should have at least start and end");
    double final_y = traj.back().second;
    expect(std::abs(final_y - std::exp(1.0)) < 1e-6, "full integration accuracy");
}

// ============================================================================
// VALIDATOR TESTS
// ============================================================================

void test_validator_rejects_nan_state() {
    RKF45Config<double> cfg;
    cfg.validator = [](const double& y) { return std::isfinite(y); };
    RKF45Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    auto result = solver.step(std::numeric_limits<double>::quiet_NaN(), 0.0, 0.01, deriv);
    expect(!result.success, "NaN input should be rejected by validator");
}

void test_validator_rejects_nan_derivative() {
    RKF45Config<double> cfg;
    cfg.validator = [](const double& y) { return std::isfinite(y); };
    RKF45Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double&, double) {
        return std::numeric_limits<double>::quiet_NaN();
    };

    auto result = solver.step(1.0, 0.0, 0.01, deriv);
    expect(!result.success, "NaN derivative should be rejected by validator");
}

// ============================================================================
// NON-SCALAR STATE REQUIRING NORM
// ============================================================================

void test_non_scalar_without_norm_throws() {
    RKF45Config<Vec2> cfg;
    // No norm provided for non-arithmetic type
    bool threw = false;
    try {
        RKF45Integrator<Vec2> solver(cfg);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect(threw, "constructing without norm for custom type should throw");
}

// ============================================================================
// METADATA
// ============================================================================

void test_type_and_order() {
    RKF45Integrator<double> solver;
    expect(solver.getType() == "rkf45", "type should be rkf45");
    expect(solver.getOrder() == 4, "order should be 4");
    expect(solver.supportsAdaptiveStep(), "RKF45 should support adaptive step");
}

// ============================================================================
// CONVERGENCE: error decreases with tolerance
// ============================================================================

void test_tighter_tolerance_gives_less_error() {
    DerivativeFunction<double> deriv = [](const double& y, double) { return y; };

    double error_loose = 0.0;
    double error_tight = 0.0;

    {
        RKF45Config<double> cfg;
        cfg.max_step = 1.0;
        RKF45Integrator<double> solver(cfg);
        auto result = solver.adaptiveStep(1.0, 0.0, 0.1, 1e-3, deriv);
        if (result.success) error_loose = result.estimated_error;
    }
    {
        RKF45Config<double> cfg;
        cfg.max_step = 1.0;
        RKF45Integrator<double> solver(cfg);
        auto result = solver.adaptiveStep(1.0, 0.0, 0.1, 1e-9, deriv);
        if (result.success) error_tight = result.estimated_error;
    }

    expect(error_tight <= error_loose, "tighter tolerance should give equal or less error");
}

// ============================================================================
// POLYNOMIAL: dy/dt = 3t^2  => y(t) = t^3 (RKF45 should solve exactly for low degree)
// ============================================================================

void test_polynomial_exact() {
    RKF45Config<double> cfg;
    cfg.max_step = 1.0;
    RKF45Integrator<double> solver(cfg);

    DerivativeFunction<double> deriv = [](const double&, double t) { return 3.0 * t * t; };

    // y(0) = 0, dy/dt = 3t^2 => y(t) = t^3
    // RKF45 (order 4/5) should integrate a cubic polynomial exactly in a single step
    auto result = solver.step(0.0, 0.0, 1.0, deriv);

    expect(result.success, "polynomial step should succeed");
    expect(std::abs(result.state - 1.0) < 1e-12, "cubic polynomial should be integrated exactly");
}

// ============================================================================
// LARGE SYSTEM: std::vector<double> state
// ============================================================================

struct VecState {
    std::vector<double> d;
    VecState() = default;
    explicit VecState(std::vector<double> v) : d(std::move(v)) {}
};

VecState operator+(const VecState& a, const VecState& b) {
    if (a.d.empty()) return b;
    if (b.d.empty()) return a;
    VecState out;
    out.d.resize(a.d.size());
    for (size_t i = 0; i < a.d.size(); ++i) out.d[i] = a.d[i] + b.d[i];
    return out;
}
VecState operator-(const VecState& a, const VecState& b) {
    if (a.d.empty()) return b;
    if (b.d.empty()) return a;
    VecState out;
    out.d.resize(a.d.size());
    for (size_t i = 0; i < a.d.size(); ++i) out.d[i] = a.d[i] - b.d[i];
    return out;
}
VecState operator*(const VecState& a, double s) {
    VecState out;
    out.d.resize(a.d.size());
    for (size_t i = 0; i < a.d.size(); ++i) out.d[i] = a.d[i] * s;
    return out;
}
VecState operator*(double s, const VecState& a) { return a * s; }

double vecStateNorm(const VecState& s) {
    double sum = 0.0;
    for (double v : s.d) sum += v * v;
    return std::sqrt(sum);
}

void test_high_dimensional_system() {
    constexpr int N = 50;

    RKF45Config<VecState> cfg;
    cfg.norm = vecStateNorm;
    cfg.max_step = 0.5;
    RKF45Integrator<VecState> solver(cfg);

    // Each component: dy_i/dt = -i * y_i => y_i(t) = exp(-i*t)
    DerivativeFunction<VecState> deriv = [](const VecState& y, double) {
        VecState out;
        out.d.resize(y.d.size());
        for (size_t i = 0; i < y.d.size(); ++i) {
            out.d[i] = -static_cast<double>(i + 1) * y.d[i];
        }
        return out;
    };

    VecState y0;
    y0.d.assign(N, 1.0);

    auto traj = integrate<VecState>(solver, y0, 0.0, 1.0, 0.01, deriv, false);
    const VecState& final_state = traj.back().second;

    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        double expected = std::exp(-static_cast<double>(i + 1));
        max_err = std::max(max_err, std::abs(final_state.d[i] - expected));
    }

    expect(max_err < 1e-4, "high-dimensional decoupled system should be accurate");
}

} // namespace

int main() {
    // Fixed step
    test_scalar_step();
    test_step_returns_error_estimate();
    test_step_zero_dt_rejected();
    test_step_negative_dt_rejected();
    test_step_exceeds_max_rejected();
    test_step_below_min_rejected();

    // Adaptive step
    test_adaptive_scalar_exponential();
    test_adaptive_step_reduces_on_stiff_problem();
    test_adaptive_exhausts_rejections();

    // Accuracy
    test_exponential_decay_accuracy();
    test_harmonic_oscillator_adaptive();
    test_full_integration_scalar();
    test_polynomial_exact();

    // Validator
    test_validator_rejects_nan_state();
    test_validator_rejects_nan_derivative();
    test_non_scalar_without_norm_throws();

    // Metadata
    test_type_and_order();

    // Convergence
    test_tighter_tolerance_gives_less_error();

    // High-dimensional
    test_high_dimensional_system();

    if (g_failures != 0) {
        std::cerr << "Total failures: " << g_failures << "\n";
        return 1;
    }

    std::cout << "All RKF45 tests passed\n";
    return 0;
}
