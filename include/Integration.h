#pragma once

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace tableau::integration {

/**
 * @brief Function type for state derivative calculation.
 *
 * The function receives the current state and time, and returns the state
 * derivative (dy/dt) for that point.
 */
template <typename State>
using DerivativeFunction = std::function<State(const State&, double)>;

/**
 * @brief Optional state validator hook.
 *
 * When provided, integrators will call this before accepting intermediate
 * states or derivatives. Returning false marks the step as invalid.
 */
template <typename State>
using StateValidator = std::function<bool(const State&)>;

/**
 * @brief Norm function used for error control in adaptive methods.
 *
 * The norm should be inexpensive and monotonic with respect to the magnitude
 * of the state (e.g., L2 norm, max component, or absolute value for scalars).
 */
template <typename State>
using StateNorm = std::function<double(const State&)>;

/**
 * @brief Encapsulates the result of an integration step.
 */
template <typename State>
struct IntegrationResult {
    State state{};                  ///< Resulting state after integration
    double time_step_used{0.0};     ///< Actual step size taken
    double estimated_error{0.0};    ///< Error estimate (adaptive methods)
    bool success{false};            ///< True when the step succeeded
    std::string method_used{"unknown"}; ///< Identifier of the integrator

    bool isValid() const { return success; }

    std::string toString() const {
        std::ostringstream oss;
        oss << "IntegrationResult{method=" << method_used
            << ", dt=" << time_step_used
            << ", error=" << estimated_error
            << ", success=" << (success ? "true" : "false")
            << "}";
        return oss.str();
    }
};

/**
 * @brief Abstract base class for fixed-step integrators.
 */
template <typename State>
class Integrator {
public:
    virtual ~Integrator() = default;

    virtual IntegrationResult<State> step(
        const State& current_state,
        double current_time,
        double time_step,
        const DerivativeFunction<State>& derivative_func) const = 0;

    virtual std::string getType() const = 0;
    virtual int getOrder() const = 0;
    virtual bool supportsAdaptiveStep() const { return false; }

    /**
     * @brief Optional heuristic for picking a first time step.
     *
     * Default implementation simply returns a small fixed step.
     */
    virtual double estimateOptimalTimeStep(
        const State& /*current_state*/,
        const DerivativeFunction<State>& /*derivative_func*/,
        double /*target_error*/ = 1e-6) const {
        return 1e-3;
    }
};

/**
 * @brief Base class for adaptive integrators.
 */
template <typename State>
class AdaptiveIntegrator : public Integrator<State> {
public:
    virtual IntegrationResult<State> adaptiveStep(
        const State& current_state,
        double current_time,
        double initial_time_step,
        double target_error,
        const DerivativeFunction<State>& derivative_func) const = 0;

    bool supportsAdaptiveStep() const override { return true; }
};

/**
 * @brief Integrate across a time interval using fixed steps.
 */
template <typename State>
std::vector<std::pair<double, State>> integrate(
    const Integrator<State>& integrator,
    const State& initial_state,
    double initial_time,
    double final_time,
    double time_step,
    const DerivativeFunction<State>& derivative_func,
    bool store_intermediate = false) {

    std::vector<std::pair<double, State>> results;

    if (final_time <= initial_time || time_step <= 0.0) {
        return results;
    }

    results.emplace_back(initial_time, initial_state);

    State current_state = initial_state;
    double current_time = initial_time;

    while (current_time < final_time) {
        double actual_step = std::min(time_step, final_time - current_time);
        IntegrationResult<State> result = integrator.step(
            current_state, current_time, actual_step, derivative_func);

        if (!result.success) {
            break;
        }

        current_state = result.state;
        current_time += result.time_step_used;

        if (store_intermediate) {
            results.emplace_back(current_time, current_state);
        }
    }

    if (!store_intermediate && results.size() == 1) {
        results.emplace_back(current_time, current_state);
    }

    return results;
}

} // namespace tableau::integration
