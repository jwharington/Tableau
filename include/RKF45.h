#pragma once

#include "Integration.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace tableau::integration {

template <typename State>
struct RKF45Config {
    double tolerance = 1e-6;
    double min_step = 1e-12;
    double max_step = 0.1;
    double safety_factor = 0.9;
    double absolute_tolerance = 1e-6;
    double relative_tolerance = 1e-6;
    int max_rejections = 20;
    StateValidator<State> validator{};
    StateNorm<State> norm{};
};

/**
 * @brief Runge-Kutta-Fehlberg 4(5) integrator with adaptive step control.
 *
 * Requirements for State:
 *  - Default constructible to represent a zero value.
 *  - Supports addition/subtraction and scalar multiplication by double.
 */
template <typename State>
class RKF45Integrator : public AdaptiveIntegrator<State> {
public:
    RKF45Integrator() : RKF45Integrator(RKF45Config<State>{}) {}

    explicit RKF45Integrator(const RKF45Config<State>& config)
        : tolerance_(config.tolerance),
          min_step_(config.min_step),
          max_step_(config.max_step),
          safety_factor_(config.safety_factor),
          absolute_tolerance_(config.absolute_tolerance),
          relative_tolerance_(config.relative_tolerance),
          max_rejections_(config.max_rejections),
          validator_(config.validator),
          norm_(config.norm) {
        if (!norm_) {
            if constexpr (std::is_arithmetic_v<State>) {
                norm_ = [](const State& s) { return std::abs(s); };
            } else {
                throw std::invalid_argument("RKF45Integrator requires a norm for non-scalar states");
            }
        }
    }

    IntegrationResult<State> step(
        const State& current_state,
        double current_time,
        double time_step,
        const DerivativeFunction<State>& derivative_func) const override {

        if (time_step <= 0.0 || time_step < min_step_ || time_step > max_step_) {
            return {};
        }
        if (validator_ && !validator_(current_state)) {
            return {};
        }

        if (!computeAllStages(current_state, current_time, time_step, derivative_func)) {
            return {};
        }

        State y4 = current_state + computeSolution(time_step, false);
        State y5 = current_state + computeSolution(time_step, true);

        State err_state = y5 - y4;
        double error = estimateError(err_state, current_state, y5);

        IntegrationResult<State> result;
        result.state = y4;
        result.time_step_used = time_step;
        result.estimated_error = error;
        result.success = true;
        result.method_used = "rkf45";
        return result;
    }

    IntegrationResult<State> adaptiveStep(
        const State& current_state,
        double current_time,
        double initial_time_step,
        double target_error,
        const DerivativeFunction<State>& derivative_func) const override {

        double h = initial_time_step;
        int rejections = 0;

        while (rejections <= max_rejections_) {
            if (h < min_step_ || h > max_step_) {
                return {};
            }

            if (!computeAllStages(current_state, current_time, h, derivative_func)) {
                h *= 0.5;
                ++rejections;
                continue;
            }

            State y4 = current_state + computeSolution(h, false);
            State y5 = current_state + computeSolution(h, true);
            State err_state = y5 - y4;
            double error = estimateError(err_state, current_state, y5);

            if (error <= target_error || error == 0.0) {
                IntegrationResult<State> result;
                result.state = y4;
                result.time_step_used = h;
                result.estimated_error = error;
                result.success = true;
                result.method_used = "rkf45";
                return result;
            }

            h = calculateOptimalStep(h, error, target_error);
            ++rejections;
        }

        return {};
    }

    std::string getType() const override { return "rkf45"; }
    int getOrder() const override { return 4; }

private:
    static constexpr int NUM_STAGES = 6;

    struct ButcherTableau {
        static constexpr double C[NUM_STAGES] = {
            0.0,
            0.25,
            0.375,
            12.0 / 13.0,
            1.0,
            0.5};

        static constexpr double A[NUM_STAGES][NUM_STAGES] = {
            {0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            {0.25, 0.0, 0.0, 0.0, 0.0, 0.0},
            {3.0 / 32.0, 9.0 / 32.0, 0.0, 0.0, 0.0, 0.0},
            {1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 0.0, 0.0, 0.0},
            {439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, 0.0, 0.0},
            {-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0, 0.0}};

        static constexpr double B4[NUM_STAGES] = {
            25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4104.0, -1.0 / 5.0, 0.0};

        static constexpr double B5[NUM_STAGES] = {
            16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};
    };

    double tolerance_{1e-6};
    double min_step_{1e-12};
    double max_step_{0.1};
    double safety_factor_{0.9};
    double absolute_tolerance_{1e-6};
    double relative_tolerance_{1e-6};
    int max_rejections_{20};
    StateValidator<State> validator_{};
    StateNorm<State> norm_{};
    mutable std::array<State, NUM_STAGES> k_buffer_{};

    bool computeAllStages(
        const State& current_state,
        double current_time,
        double time_step_h,
        const DerivativeFunction<State>& derivative_func) const {

        k_buffer_[0] = derivative_func(current_state, current_time);
        if (validator_ && !validator_(k_buffer_[0])) {
            return false;
        }

        for (int i = 1; i < NUM_STAGES; ++i) {
            State sum{};
            for (int j = 0; j < i; ++j) {
                sum = sum + k_buffer_[j] * ButcherTableau::A[i][j];
            }

            State stage_state = current_state + sum * time_step_h;
            double stage_time = current_time + ButcherTableau::C[i] * time_step_h;

            k_buffer_[i] = derivative_func(stage_state, stage_time);
            if (validator_ && !validator_(k_buffer_[i])) {
                return false;
            }
        }

        return true;
    }

    State computeSolution(double time_step_h, bool use_fifth_order) const {
        State solution{};
        const double* coeffs = use_fifth_order ? ButcherTableau::B5 : ButcherTableau::B4;

        for (int i = 0; i < NUM_STAGES; ++i) {
            solution = solution + k_buffer_[i] * (coeffs[i] * time_step_h);
        }

        return solution;
    }

    double estimateError(
        const State& err_state,
        const State& current_state,
        const State& next_state) const {

        double scale = absolute_tolerance_ +
            relative_tolerance_ * std::max(norm_(current_state), norm_(next_state));

        if (scale <= std::numeric_limits<double>::min()) {
            scale = absolute_tolerance_;
        }

        double err_norm = norm_(err_state);
        return err_norm / scale;
    }

    double calculateOptimalStep(double current_step, double error, double target_error) const {
        constexpr double EXPO = 0.2; // 1/(order+1) for order 5
        double factor = safety_factor_ * std::pow(target_error / std::max(error, 1e-16), EXPO);
        factor = std::clamp(factor, 0.1, 5.0);
        double proposed = current_step * factor;
        if (proposed < min_step_) {
            proposed = min_step_;
        }
        if (proposed > max_step_) {
            proposed = max_step_;
        }
        return proposed;
    }
};

template <typename State>
constexpr double RKF45Integrator<State>::ButcherTableau::C[NUM_STAGES];
template <typename State>
constexpr double RKF45Integrator<State>::ButcherTableau::A[NUM_STAGES][NUM_STAGES];
template <typename State>
constexpr double RKF45Integrator<State>::ButcherTableau::B4[NUM_STAGES];
template <typename State>
constexpr double RKF45Integrator<State>::ButcherTableau::B5[NUM_STAGES];

} // namespace tableau::integration
