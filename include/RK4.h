#pragma once

#include "Integration.h"
#include <array>
#include <cmath>

namespace tableau::integration {

template <typename State>
struct RK4Config {
    double max_time_step = 1.0;
    bool enable_validation = false;
    StateValidator<State> validator{};
    StateNorm<State> norm{};
};

/**
 * @brief Classic 4th-order Runge-Kutta integrator (fixed step).
 *
 * Requirements for State:
 *  - Default constructible to represent a zero value.
 *  - Supports addition/subtraction and scalar multiplication by double.
 */
template <typename State>
class RK4Integrator : public Integrator<State> {
public:
    RK4Integrator() = default;
    explicit RK4Integrator(const RK4Config<State>& config)
        : max_time_step_(config.max_time_step),
          validate_(config.enable_validation),
          validator_(config.validator),
          norm_(config.norm) {}

    IntegrationResult<State> step(
        const State& current_state,
        double current_time,
        double time_step,
        const DerivativeFunction<State>& derivative_func) const override {

        if (time_step <= 0.0 || time_step > max_time_step_) {
            return {};
        }
        if (validate_ && validator_ && !validator_(current_state)) {
            return {};
        }

        std::array<State, NUM_STAGES> k{};

        // Stage 0
        k[0] = derivative_func(current_state, current_time);
        if (validate_ && validator_ && !validator_(k[0])) {
            return {};
        }

        // Stages 1..3
        for (int i = 1; i < NUM_STAGES; ++i) {
            State sum{};
            for (int j = 0; j < i; ++j) {
                sum = sum + k[j] * A[i][j];
            }

            State stage_state = current_state + sum * time_step;
            double stage_time = current_time + C[i] * time_step;

            k[i] = derivative_func(stage_state, stage_time);
            if (validate_ && validator_ && !validator_(k[i])) {
                return {};
            }
        }

        State weighted_sum{};
        for (int i = 0; i < NUM_STAGES; ++i) {
            weighted_sum = weighted_sum + k[i] * B[i];
        }

        State next_state = current_state + weighted_sum * time_step;

        IntegrationResult<State> result;
        result.state = next_state;
        result.time_step_used = time_step;
        result.estimated_error = estimateLocalError(k, time_step);
        result.success = true;
        result.method_used = "rk4";
        return result;
    }

    std::string getType() const override { return "rk4"; }
    int getOrder() const override { return 4; }

private:
    static constexpr int NUM_STAGES = 4;

    // Butcher tableau coefficients
    static constexpr double C[NUM_STAGES] = {0.0, 0.5, 0.5, 1.0};
    static constexpr double A[NUM_STAGES][NUM_STAGES] = {
        {0.0, 0.0, 0.0, 0.0},
        {0.5, 0.0, 0.0, 0.0},
        {0.0, 0.5, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0}};
    static constexpr double B[NUM_STAGES] = {
        1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0};

    double max_time_step_{1.0};
    bool validate_{false};
    StateValidator<State> validator_{};
    StateNorm<State> norm_{};

    double estimateLocalError(const std::array<State, NUM_STAGES>& k, double time_step) const {
        if (!norm_) {
            return 0.0;
        }

        State average{};
        for (int i = 0; i < NUM_STAGES; ++i) {
            average = average + k[i] * B[i];
        }

        double max_deviation = 0.0;
        for (int i = 0; i < NUM_STAGES; ++i) {
            double dev = norm_(k[i] - average);
            if (dev > max_deviation) {
                max_deviation = dev;
            }
        }

        return max_deviation * std::pow(time_step, 5) / 24.0;
    }
};

template <typename State>
constexpr double RK4Integrator<State>::C[NUM_STAGES];
template <typename State>
constexpr double RK4Integrator<State>::A[NUM_STAGES][NUM_STAGES];
template <typename State>
constexpr double RK4Integrator<State>::B[NUM_STAGES];

} // namespace tableau::integration

