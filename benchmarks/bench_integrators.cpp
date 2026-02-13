#include <benchmark/benchmark.h>

#include "Integration.h"
#include "RK4.h"
#include "RKF45.h"
#include "DOP853.h"

#include <array>
#include <cmath>
#include <vector>

using namespace tableau::integration;

// ============================================================================
// State type with arithmetic operators (wraps std::vector<double>)
// ============================================================================

struct VecState {
    std::vector<double> d;
    VecState() = default;
    explicit VecState(std::vector<double> v) : d(std::move(v)) {}
    VecState(std::initializer_list<double> il) : d(il) {}
    std::size_t size() const { return d.size(); }
    double& operator[](std::size_t i) { return d[i]; }
    const double& operator[](std::size_t i) const { return d[i]; }
};

inline VecState operator+(const VecState& a, const VecState& b) {
    if (a.d.empty()) return b;
    if (b.d.empty()) return a;
    VecState out;
    out.d.resize(a.d.size());
    for (std::size_t i = 0; i < a.d.size(); ++i) out.d[i] = a.d[i] + b.d[i];
    return out;
}
inline VecState operator-(const VecState& a, const VecState& b) {
    if (a.d.empty()) return b;
    if (b.d.empty()) return a;
    VecState out;
    out.d.resize(a.d.size());
    for (std::size_t i = 0; i < a.d.size(); ++i) out.d[i] = a.d[i] - b.d[i];
    return out;
}
inline VecState operator*(const VecState& a, double s) {
    VecState out;
    out.d.resize(a.d.size());
    for (std::size_t i = 0; i < a.d.size(); ++i) out.d[i] = a.d[i] * s;
    return out;
}
inline VecState operator*(double s, const VecState& a) { return a * s; }

inline double vecNorm(const VecState& s) {
    double sum = 0.0;
    for (double v : s.d) sum += v * v;
    return std::sqrt(sum);
}

// ============================================================================
// Three-body state (stack-allocated, 18 doubles)
// ============================================================================

struct State3B {
    static constexpr std::size_t kSize = 18;
    std::array<double, kSize> data{};
    double& operator[](std::size_t i) { return data[i]; }
    const double& operator[](std::size_t i) const { return data[i]; }
};

inline State3B operator+(const State3B& a, const State3B& b) {
    State3B out;
    for (std::size_t i = 0; i < State3B::kSize; ++i) out[i] = a[i] + b[i];
    return out;
}
inline State3B operator-(const State3B& a, const State3B& b) {
    State3B out;
    for (std::size_t i = 0; i < State3B::kSize; ++i) out[i] = a[i] - b[i];
    return out;
}
inline State3B operator*(const State3B& s, double c) {
    State3B out;
    for (std::size_t i = 0; i < State3B::kSize; ++i) out[i] = s[i] * c;
    return out;
}
inline State3B operator*(double c, const State3B& s) { return s * c; }

inline double state3bNorm(const State3B& s) {
    double sum = 0.0;
    for (double v : s.data) sum += v * v;
    return std::sqrt(sum);
}

// ============================================================================
// ODE problems
// ============================================================================

// Scalar: dy/dt = y  (exponential growth)
static const DerivativeFunction<double> expDeriv =
    [](const double& y, double) -> double { return y; };

// Harmonic oscillator: y'' = -y
static const DerivativeFunction<VecState> harmonicDeriv =
    [](const VecState& s, double) -> VecState { return {s[1], -s[0]}; };

// Lorenz attractor (chaotic 3D)
static const DerivativeFunction<VecState> lorenzDeriv =
    [](const VecState& s, double) -> VecState {
    constexpr double sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;
    return {sigma * (s[1] - s[0]),
            s[0] * (rho - s[2]) - s[1],
            s[0] * s[1] - beta * s[2]};
};

// Lorenz for std::vector<double> (DOP853 full-run API)
static const DerivativeFunction<std::vector<double>> lorenzDerivVec =
    [](const std::vector<double>& s, double) -> std::vector<double> {
    constexpr double sigma = 10.0, rho = 28.0, beta = 8.0 / 3.0;
    return {sigma * (s[1] - s[0]),
            s[0] * (rho - s[2]) - s[1],
            s[0] * s[1] - beta * s[2]};
};

// Van der Pol oscillator
static const DerivativeFunction<VecState> vanDerPolDeriv =
    [](const VecState& s, double) -> VecState {
    constexpr double mu = 1.0;
    return {s[1], mu * (1.0 - s[0] * s[0]) * s[1] - s[0]};
};

static const DerivativeFunction<std::vector<double>> vanDerPolDerivVec =
    [](const std::vector<double>& s, double) -> std::vector<double> {
    constexpr double mu = 1.0;
    return {s[1], mu * (1.0 - s[0] * s[0]) * s[1] - s[0]};
};

// Harmonic oscillator for std::vector<double>
static const DerivativeFunction<std::vector<double>> harmonicDerivVec =
    [](const std::vector<double>& s, double) -> std::vector<double> {
    return {s[1], -s[0]};
};

// Three-body problem (custom state)
static const DerivativeFunction<State3B> threeBodyDeriv =
    [](const State3B& state, double) -> State3B {
    constexpr double G = 1.0, M = 1.0, soft2 = 1e-12;
    State3B deriv{};
    for (int i = 0; i < 3; ++i) {
        int bi = i * 6;
        deriv[bi + 0] = state[bi + 3];
        deriv[bi + 1] = state[bi + 4];
        deriv[bi + 2] = state[bi + 5];
        double ax = 0, ay = 0, az = 0;
        for (int j = 0; j < 3; ++j) {
            if (i == j) continue;
            int bj = j * 6;
            double dx = state[bj] - state[bi];
            double dy = state[bj + 1] - state[bi + 1];
            double dz = state[bj + 2] - state[bi + 2];
            double r2 = dx * dx + dy * dy + dz * dz + soft2;
            double inv_r3 = 1.0 / (std::sqrt(r2) * r2);
            ax += G * M * dx * inv_r3;
            ay += G * M * dy * inv_r3;
            az += G * M * dz * inv_r3;
        }
        deriv[bi + 3] = ax;
        deriv[bi + 4] = ay;
        deriv[bi + 5] = az;
    }
    return deriv;
};

static State3B makeThreeBodyIC() {
    State3B s{};
    s[0] = 0.97000436;  s[1] = -0.24308753; s[2] = 0.0;
    s[3] = 0.4662036850; s[4] = 0.4323657300; s[5] = 0.0;
    s[6] = -0.97000436; s[7] = 0.24308753;  s[8] = 0.0;
    s[9] = 0.4662036850; s[10] = 0.4323657300; s[11] = 0.0;
    s[12] = 0.0; s[13] = 0.0; s[14] = 0.0;
    s[15] = -0.93240737; s[16] = -0.86473146; s[17] = 0.0;
    return s;
}

// N-body derivative via VecState (parametric body count)
static const DerivativeFunction<VecState> nBodyDeriv =
    [](const VecState& state, double) -> VecState {
    constexpr double G = 1.0, M = 1.0, soft2 = 1e-12;
    const std::size_t n = state.size() / 6;
    VecState d;
    d.d.assign(state.size(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t bi = i * 6;
        d[bi + 0] = state[bi + 3];
        d[bi + 1] = state[bi + 4];
        d[bi + 2] = state[bi + 5];
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            std::size_t bj = j * 6;
            double dx = state[bj] - state[bi];
            double dy = state[bj + 1] - state[bi + 1];
            double dz = state[bj + 2] - state[bi + 2];
            double r2 = dx * dx + dy * dy + dz * dz + soft2;
            double inv_r3 = 1.0 / (std::sqrt(r2) * r2);
            d[bi + 3] += G * M * dx * inv_r3;
            d[bi + 4] += G * M * dy * inv_r3;
            d[bi + 5] += G * M * dz * inv_r3;
        }
    }
    return d;
};

// Same for std::vector<double> (DOP853 full-run)
static const DerivativeFunction<std::vector<double>> nBodyDerivVec =
    [](const std::vector<double>& state, double) -> std::vector<double> {
    constexpr double G = 1.0, M = 1.0, soft2 = 1e-12;
    const std::size_t n = state.size() / 6;
    std::vector<double> d(state.size(), 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::size_t bi = i * 6;
        d[bi + 0] = state[bi + 3];
        d[bi + 1] = state[bi + 4];
        d[bi + 2] = state[bi + 5];
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) continue;
            std::size_t bj = j * 6;
            double dx = state[bj] - state[bi];
            double dy = state[bj + 1] - state[bi + 1];
            double dz = state[bj + 2] - state[bi + 2];
            double r2 = dx * dx + dy * dy + dz * dz + soft2;
            double inv_r3 = 1.0 / (std::sqrt(r2) * r2);
            d[bi + 3] += G * M * dx * inv_r3;
            d[bi + 4] += G * M * dy * inv_r3;
            d[bi + 5] += G * M * dz * inv_r3;
        }
    }
    return d;
};

static VecState makeNBodyIC(int bodies) {
    VecState s;
    s.d.assign(bodies * 6, 0.0);
    for (int i = 0; i < bodies; ++i) {
        double angle = 2.0 * M_PI * i / bodies;
        double r = 2.0;
        s[i * 6 + 0] = r * std::cos(angle);
        s[i * 6 + 1] = r * std::sin(angle);
        double v = std::sqrt(1.0 / r);
        s[i * 6 + 3] = -v * std::sin(angle);
        s[i * 6 + 4] = v * std::cos(angle);
    }
    return s;
}

static std::vector<double> makeNBodyICVec(int bodies) {
    std::vector<double> s(bodies * 6, 0.0);
    for (int i = 0; i < bodies; ++i) {
        double angle = 2.0 * M_PI * i / bodies;
        double r = 2.0;
        s[i * 6 + 0] = r * std::cos(angle);
        s[i * 6 + 1] = r * std::sin(angle);
        double v = std::sqrt(1.0 / r);
        s[i * 6 + 3] = -v * std::sin(angle);
        s[i * 6 + 4] = v * std::cos(angle);
    }
    return s;
}

// ============================================================================
// SINGLE STEP BENCHMARKS
// ============================================================================

// --- RK4 ---

static void BM_RK4_Step_Scalar(benchmark::State& st) {
    RK4Integrator<double> integrator;
    for (auto _ : st) {
        auto res = integrator.step(1.0, 0.0, 0.01, expDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RK4_Step_Scalar);

static void BM_RK4_Step_HarmonicOscillator(benchmark::State& st) {
    RK4Config<VecState> cfg;
    cfg.norm = vecNorm;
    RK4Integrator<VecState> integrator(cfg);
    VecState y = {1.0, 0.0};
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.01, harmonicDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RK4_Step_HarmonicOscillator);

static void BM_RK4_Step_Lorenz(benchmark::State& st) {
    RK4Config<VecState> cfg;
    cfg.norm = vecNorm;
    RK4Integrator<VecState> integrator(cfg);
    VecState y = {1.0, 1.0, 1.0};
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, lorenzDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RK4_Step_Lorenz);

static void BM_RK4_Step_ThreeBody(benchmark::State& st) {
    RK4Config<State3B> cfg;
    cfg.norm = state3bNorm;
    RK4Integrator<State3B> integrator(cfg);
    State3B y = makeThreeBodyIC();
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, threeBodyDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RK4_Step_ThreeBody);

// --- RKF45 ---

static void BM_RKF45_Step_Scalar(benchmark::State& st) {
    RKF45Integrator<double> integrator;
    for (auto _ : st) {
        auto res = integrator.step(1.0, 0.0, 0.01, expDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RKF45_Step_Scalar);

static void BM_RKF45_Step_HarmonicOscillator(benchmark::State& st) {
    RKF45Config<VecState> cfg;
    cfg.norm = vecNorm;
    RKF45Integrator<VecState> integrator(cfg);
    VecState y = {1.0, 0.0};
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.01, harmonicDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RKF45_Step_HarmonicOscillator);

static void BM_RKF45_Step_Lorenz(benchmark::State& st) {
    RKF45Config<VecState> cfg;
    cfg.norm = vecNorm;
    RKF45Integrator<VecState> integrator(cfg);
    VecState y = {1.0, 1.0, 1.0};
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, lorenzDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RKF45_Step_Lorenz);

static void BM_RKF45_Step_ThreeBody(benchmark::State& st) {
    RKF45Config<State3B> cfg;
    cfg.norm = state3bNorm;
    RKF45Integrator<State3B> integrator(cfg);
    State3B y = makeThreeBodyIC();
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, threeBodyDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RKF45_Step_ThreeBody);

// --- DOP853 ---

static void BM_DOP853_Step_Scalar(benchmark::State& st) {
    DOP853Integrator<double> integrator;
    for (auto _ : st) {
        auto res = integrator.step(1.0, 0.0, 0.01, expDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_Step_Scalar);

static void BM_DOP853_Step_HarmonicOscillator(benchmark::State& st) {
    DOP853Integrator<std::vector<double>> integrator;
    std::vector<double> y = {1.0, 0.0};
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.01, harmonicDerivVec);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_Step_HarmonicOscillator);

static void BM_DOP853_Step_Lorenz(benchmark::State& st) {
    DOP853Integrator<std::vector<double>> integrator;
    std::vector<double> y = {1.0, 1.0, 1.0};
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, lorenzDerivVec);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_Step_Lorenz);

static void BM_DOP853_Step_ThreeBody(benchmark::State& st) {
    DOP853Config<State3B> cfg;
    cfg.norm = state3bNorm;
    DOP853Integrator<State3B> integrator(cfg);
    State3B y = makeThreeBodyIC();
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, threeBodyDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_Step_ThreeBody);

// ============================================================================
// ADAPTIVE STEP BENCHMARKS
// ============================================================================

static void BM_RKF45_AdaptiveStep_Lorenz(benchmark::State& st) {
    RKF45Config<VecState> cfg;
    cfg.norm = vecNorm;
    cfg.max_step = 1.0;
    RKF45Integrator<VecState> integrator(cfg);
    VecState y = {1.0, 1.0, 1.0};
    for (auto _ : st) {
        auto res = integrator.adaptiveStep(y, 0.0, 0.01, 1e-6, lorenzDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RKF45_AdaptiveStep_Lorenz);

static void BM_DOP853_AdaptiveStep_Lorenz(benchmark::State& st) {
    DOP853Integrator<std::vector<double>> integrator;
    std::vector<double> y = {1.0, 1.0, 1.0};
    for (auto _ : st) {
        auto res = integrator.adaptiveStep(y, 0.0, 0.01, 1e-8, lorenzDerivVec);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_AdaptiveStep_Lorenz);

static void BM_RKF45_AdaptiveStep_VanDerPol(benchmark::State& st) {
    RKF45Config<VecState> cfg;
    cfg.norm = vecNorm;
    cfg.max_step = 1.0;
    RKF45Integrator<VecState> integrator(cfg);
    VecState y = {2.0, 0.0};
    for (auto _ : st) {
        auto res = integrator.adaptiveStep(y, 0.0, 0.01, 1e-6, vanDerPolDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RKF45_AdaptiveStep_VanDerPol);

static void BM_DOP853_AdaptiveStep_VanDerPol(benchmark::State& st) {
    DOP853Integrator<std::vector<double>> integrator;
    std::vector<double> y = {2.0, 0.0};
    for (auto _ : st) {
        auto res = integrator.adaptiveStep(y, 0.0, 0.01, 1e-8, vanDerPolDerivVec);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_AdaptiveStep_VanDerPol);

// ============================================================================
// FULL INTEGRATION (over a time interval)
// ============================================================================

// Harmonic oscillator — 1 full period
static void BM_RK4_Integrate_Harmonic_1Period(benchmark::State& st) {
    RK4Config<VecState> cfg;
    cfg.norm = vecNorm;
    RK4Integrator<VecState> integrator(cfg);
    VecState y0 = {1.0, 0.0};
    for (auto _ : st) {
        auto res = integrate<VecState>(integrator, y0, 0.0, 2.0 * M_PI, 0.01, harmonicDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RK4_Integrate_Harmonic_1Period);

static void BM_RKF45_Integrate_Harmonic_1Period(benchmark::State& st) {
    RKF45Config<VecState> cfg;
    cfg.norm = vecNorm;
    cfg.max_step = 1.0;
    RKF45Integrator<VecState> integrator(cfg);
    VecState y0 = {1.0, 0.0};
    for (auto _ : st) {
        auto res = integrate<VecState>(integrator, y0, 0.0, 2.0 * M_PI, 0.01, harmonicDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RKF45_Integrate_Harmonic_1Period);

static void BM_DOP853_Integrate_Harmonic_1Period(benchmark::State& st) {
    DOP853Config<std::vector<double>> cfg;
    cfg.max_step = 1.0;
    cfg.derivative = harmonicDerivVec;
    DOP853Integrator<std::vector<double>> integrator(cfg);
    std::vector<double> y0 = {1.0, 0.0};
    auto tol = DOP853Tolerance::scalar(1e-8, 1e-8);
    for (auto _ : st) {
        auto res = integrator.integrate(0.0, y0, 2.0 * M_PI, tol);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_Integrate_Harmonic_1Period);

// Lorenz — 10 time units
static void BM_RK4_Integrate_Lorenz_10s(benchmark::State& st) {
    RK4Config<VecState> cfg;
    cfg.norm = vecNorm;
    RK4Integrator<VecState> integrator(cfg);
    VecState y0 = {1.0, 1.0, 1.0};
    for (auto _ : st) {
        auto res = integrate<VecState>(integrator, y0, 0.0, 10.0, 0.001, lorenzDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RK4_Integrate_Lorenz_10s);

static void BM_RKF45_Integrate_Lorenz_10s(benchmark::State& st) {
    RKF45Config<VecState> cfg;
    cfg.norm = vecNorm;
    cfg.max_step = 1.0;
    RKF45Integrator<VecState> integrator(cfg);
    VecState y0 = {1.0, 1.0, 1.0};
    for (auto _ : st) {
        auto res = integrate<VecState>(integrator, y0, 0.0, 10.0, 0.01, lorenzDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_RKF45_Integrate_Lorenz_10s);

static void BM_DOP853_Integrate_Lorenz_10s(benchmark::State& st) {
    DOP853Config<std::vector<double>> cfg;
    cfg.max_step = 1.0;
    cfg.derivative = lorenzDerivVec;
    DOP853Integrator<std::vector<double>> integrator(cfg);
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    auto tol = DOP853Tolerance::scalar(1e-8, 1e-8);
    for (auto _ : st) {
        auto res = integrator.integrate(0.0, y0, 10.0, tol);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_Integrate_Lorenz_10s);

// ============================================================================
// DOP853 FULL-RUN SPECIFICS
// ============================================================================

// Tolerance sweep
static void BM_DOP853_FullRun_Lorenz_TolSweep(benchmark::State& st) {
    double tol_exp = static_cast<double>(st.range(0));
    double tol_val = std::pow(10.0, -tol_exp);

    DOP853Config<std::vector<double>> cfg;
    cfg.max_step = 1.0;
    cfg.derivative = lorenzDerivVec;
    DOP853Integrator<std::vector<double>> integrator(cfg);
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    auto tol = DOP853Tolerance::scalar(tol_val, tol_val);
    for (auto _ : st) {
        auto res = integrator.integrate(0.0, y0, 10.0, tol);
        benchmark::DoNotOptimize(res);
    }
    st.SetLabel("tol=1e-" + std::to_string(st.range(0)));
}
BENCHMARK(BM_DOP853_FullRun_Lorenz_TolSweep)->DenseRange(4, 12, 2);

// Dense output overhead
static void BM_DOP853_FullRun_Lorenz_NoDense(benchmark::State& st) {
    DOP853Config<std::vector<double>> cfg;
    cfg.max_step = 1.0;
    cfg.derivative = lorenzDerivVec;
    cfg.output_mode = DOP853OutputMode::None;
    DOP853Integrator<std::vector<double>> integrator(cfg);
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    auto tol = DOP853Tolerance::scalar(1e-8, 1e-8);
    for (auto _ : st) {
        auto res = integrator.integrate(0.0, y0, 10.0, tol);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_FullRun_Lorenz_NoDense);

static void BM_DOP853_FullRun_Lorenz_WithDense(benchmark::State& st) {
    int step_count = 0;
    DOP853Config<std::vector<double>> cfg;
    cfg.max_step = 1.0;
    cfg.derivative = lorenzDerivVec;
    cfg.output_mode = DOP853OutputMode::DenseEveryStep;
    cfg.solout = [&step_count](const DOP853StepEvent&, double&) -> int {
        ++step_count;
        return 0;
    };
    DOP853Integrator<std::vector<double>> integrator(cfg);
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    auto tol = DOP853Tolerance::scalar(1e-8, 1e-8);
    for (auto _ : st) {
        step_count = 0;
        auto res = integrator.integrate(0.0, y0, 10.0, tol);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_FullRun_Lorenz_WithDense);

// Per-component vs scalar tolerance
static void BM_DOP853_FullRun_Lorenz_ScalarTol(benchmark::State& st) {
    DOP853Config<std::vector<double>> cfg;
    cfg.max_step = 1.0;
    cfg.derivative = lorenzDerivVec;
    DOP853Integrator<std::vector<double>> integrator(cfg);
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    auto tol = DOP853Tolerance::scalar(1e-8, 1e-8);
    for (auto _ : st) {
        auto res = integrator.integrate(0.0, y0, 10.0, tol);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_FullRun_Lorenz_ScalarTol);

static void BM_DOP853_FullRun_Lorenz_PerComponentTol(benchmark::State& st) {
    DOP853Config<std::vector<double>> cfg;
    cfg.max_step = 1.0;
    cfg.derivative = lorenzDerivVec;
    DOP853Integrator<std::vector<double>> integrator(cfg);
    std::vector<double> y0 = {1.0, 1.0, 1.0};
    auto tol = DOP853Tolerance::perComponent({1e-8, 1e-8, 1e-8}, {1e-8, 1e-8, 1e-8});
    for (auto _ : st) {
        auto res = integrator.integrate(0.0, y0, 10.0, tol);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_DOP853_FullRun_Lorenz_PerComponentTol);

// ============================================================================
// SCALING: state dimension (N-body, O(N^2) derivative)
// ============================================================================

static void BM_RK4_Step_NBody(benchmark::State& st) {
    int bodies = st.range(0);
    auto y = makeNBodyIC(bodies);
    RK4Config<VecState> cfg;
    cfg.norm = vecNorm;
    RK4Integrator<VecState> integrator(cfg);
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, nBodyDeriv);
        benchmark::DoNotOptimize(res);
    }
    st.SetComplexityN(bodies);
}
BENCHMARK(BM_RK4_Step_NBody)->RangeMultiplier(2)->Range(2, 128)->Complexity();

static void BM_RKF45_Step_NBody(benchmark::State& st) {
    int bodies = st.range(0);
    auto y = makeNBodyIC(bodies);
    RKF45Config<VecState> cfg;
    cfg.norm = vecNorm;
    RKF45Integrator<VecState> integrator(cfg);
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, nBodyDeriv);
        benchmark::DoNotOptimize(res);
    }
    st.SetComplexityN(bodies);
}
BENCHMARK(BM_RKF45_Step_NBody)->RangeMultiplier(2)->Range(2, 128)->Complexity();

static void BM_DOP853_Step_NBody(benchmark::State& st) {
    int bodies = st.range(0);
    auto y = makeNBodyICVec(bodies);
    DOP853Integrator<std::vector<double>> integrator;
    for (auto _ : st) {
        auto res = integrator.step(y, 0.0, 0.001, nBodyDerivVec);
        benchmark::DoNotOptimize(res);
    }
    st.SetComplexityN(bodies);
}
BENCHMARK(BM_DOP853_Step_NBody)->RangeMultiplier(2)->Range(2, 128)->Complexity();

// ============================================================================
// HEAD-TO-HEAD: three-body integration, 1 time unit
// ============================================================================

static void BM_HeadToHead_ThreeBody_RK4(benchmark::State& st) {
    RK4Config<State3B> cfg;
    cfg.norm = state3bNorm;
    RK4Integrator<State3B> integrator(cfg);
    State3B y0 = makeThreeBodyIC();
    for (auto _ : st) {
        auto res = integrate<State3B>(integrator, y0, 0.0, 1.0, 0.001, threeBodyDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_HeadToHead_ThreeBody_RK4);

static void BM_HeadToHead_ThreeBody_RKF45(benchmark::State& st) {
    RKF45Config<State3B> cfg;
    cfg.norm = state3bNorm;
    cfg.max_step = 1.0;
    RKF45Integrator<State3B> integrator(cfg);
    State3B y0 = makeThreeBodyIC();
    for (auto _ : st) {
        auto res = integrate<State3B>(integrator, y0, 0.0, 1.0, 0.01, threeBodyDeriv);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_HeadToHead_ThreeBody_RKF45);

static void BM_HeadToHead_ThreeBody_DOP853(benchmark::State& st) {
    DOP853Config<std::vector<double>> cfg;
    cfg.max_step = 1.0;
    cfg.derivative = nBodyDerivVec;
    DOP853Integrator<std::vector<double>> integrator(cfg);
    auto y0 = makeNBodyICVec(3);
    auto tol = DOP853Tolerance::scalar(1e-8, 1e-8);
    for (auto _ : st) {
        auto res = integrator.integrate(0.0, y0, 1.0, tol);
        benchmark::DoNotOptimize(res);
    }
}
BENCHMARK(BM_HeadToHead_ThreeBody_DOP853);

// ============================================================================
// DENSE OUTPUT INTERPOLATION COST
// ============================================================================

static void BM_DOP853_DenseOutput_Evaluate(benchmark::State& st) {
    DOP853DenseOutput dense;
    dense.components = {0, 1, 2};
    dense.cont.resize(8 * 3);
    for (std::size_t i = 0; i < dense.cont.size(); ++i) {
        dense.cont[i] = 0.1 * static_cast<double>(i + 1);
    }
    dense.xold = 0.0;
    dense.hout = 0.1;

    double x = 0.05;
    for (auto _ : st) {
        double v = dense.evaluate(0, x) + dense.evaluate(1, x) + dense.evaluate(2, x);
        benchmark::DoNotOptimize(v);
    }
}
BENCHMARK(BM_DOP853_DenseOutput_Evaluate);

// ============================================================================
// CONSTRUCTION COST
// ============================================================================

static void BM_Construct_RK4(benchmark::State& st) {
    for (auto _ : st) {
        RK4Integrator<double> integrator;
        benchmark::DoNotOptimize(integrator);
    }
}
BENCHMARK(BM_Construct_RK4);

static void BM_Construct_RKF45(benchmark::State& st) {
    for (auto _ : st) {
        RKF45Integrator<double> integrator;
        benchmark::DoNotOptimize(integrator);
    }
}
BENCHMARK(BM_Construct_RKF45);

static void BM_Construct_DOP853(benchmark::State& st) {
    for (auto _ : st) {
        DOP853Integrator<double> integrator;
        benchmark::DoNotOptimize(integrator);
    }
}
BENCHMARK(BM_Construct_DOP853);

static void BM_Construct_DOP853_FullConfig(benchmark::State& st) {
    for (auto _ : st) {
        DOP853Config<std::vector<double>> cfg;
        cfg.derivative = lorenzDerivVec;
        cfg.output_mode = DOP853OutputMode::DenseEveryStep;
        cfg.solout = [](const DOP853StepEvent&, double&) { return 0; };
        DOP853Integrator<std::vector<double>> integrator(cfg);
        benchmark::DoNotOptimize(integrator);
    }
}
BENCHMARK(BM_Construct_DOP853_FullConfig);

BENCHMARK_MAIN();
