#pragma once

#include "DOP853.h"
#include "RK4.h"
#include "RKF45.h"
#include "three_body_physics.h"

#include <algorithm>
#include <string>

namespace tableau::demos {

enum class IntegratorKind {
    RK4 = 0,
    RKF45 = 1,
    DOP853 = 2,
};

struct RunnerTuning {
    double rk4_step{1e-3};
    double initial_step_guess{5e-3};
    double min_step{1e-6};
    double max_step{2e-2};
    double target_error{1.0};
    int max_substeps{500000};
};

class IntegratorRunner {
public:
    IntegratorRunner(
        IntegratorKind kind,
        const State3B& initial_state,
        const RunnerTuning& tuning = RunnerTuning{})
        : kind_(kind),
          state_(initial_state),
          time_(0.0),
          last_dt_(0.0),
          next_step_guess_(tuning.initial_step_guess),
          tuning_(tuning),
          rk4_([&] {
              tableau::integration::RK4Config<State3B> cfg;
              cfg.max_time_step = 0.1;
              cfg.enable_validation = true;
              cfg.validator = isFiniteState;
              cfg.norm = stateL2Norm;
              return tableau::integration::RK4Integrator<State3B>(cfg);
          }()),
          rkf45_([&] {
              tableau::integration::RKF45Config<State3B> cfg;
              cfg.min_step = tuning_.min_step;
              cfg.max_step = tuning_.max_step;
              cfg.absolute_tolerance = 1e-8;
              cfg.relative_tolerance = 1e-8;
              cfg.validator = isFiniteState;
              cfg.norm = stateL2Norm;
              return tableau::integration::RKF45Integrator<State3B>(cfg);
          }()),
          dop853_([&] {
              tableau::integration::DOP853Config<State3B> cfg;
              cfg.min_step = tuning_.min_step;
              cfg.max_step = tuning_.max_step;
              cfg.absolute_tolerance = 1e-9;
              cfg.relative_tolerance = 1e-9;
              cfg.validator = isFiniteState;
              cfg.norm = stateL2Norm;
              return tableau::integration::DOP853Integrator<State3B>(cfg);
          }()) {}

    IntegratorKind kind() const { return kind_; }

    const State3B& state() const { return state_; }

    double time() const { return time_; }

    double lastDt() const { return last_dt_; }

    void reset(const State3B& initial_state, double initial_time = 0.0) {
        state_ = initial_state;
        time_ = initial_time;
        last_dt_ = 0.0;
        next_step_guess_ = tuning_.initial_step_guess;
    }

    bool stepTo(double target_time) {
        if (target_time < time_) {
            return false;
        }

        int guard = 0;
        while ((time_ + 1e-14) < target_time) {
            if (guard++ > tuning_.max_substeps) {
                return false;
            }

            const double remaining = target_time - time_;
            if (remaining <= 0.0) {
                break;
            }

            if (kind_ == IntegratorKind::RK4) {
                const double dt = std::min(tuning_.rk4_step, remaining);
                const auto result = rk4_.step(state_, time_, dt, threeBodyDerivative);
                if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                    return false;
                }

                state_ = result.state;
                last_dt_ = result.time_step_used;
                time_ += result.time_step_used;
                continue;
            }

            const double seed = std::clamp(
                std::min(next_step_guess_, remaining),
                tuning_.min_step,
                tuning_.max_step);

            if (kind_ == IntegratorKind::RKF45) {
                const auto result = rkf45_.adaptiveStep(
                    state_,
                    time_,
                    seed,
                    tuning_.target_error,
                    threeBodyDerivative);

                if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                    return false;
                }

                state_ = result.state;
                last_dt_ = result.time_step_used;
                time_ += result.time_step_used;
                next_step_guess_ = std::clamp(last_dt_ * 1.35, tuning_.min_step, tuning_.max_step);
                continue;
            }

            const auto result = dop853_.adaptiveStep(
                state_,
                time_,
                seed,
                tuning_.target_error,
                threeBodyDerivative);

            if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                return false;
            }

            state_ = result.state;
            last_dt_ = result.time_step_used;
            time_ += result.time_step_used;
            next_step_guess_ = std::clamp(last_dt_ * 1.35, tuning_.min_step, tuning_.max_step);
        }

        return true;
    }

private:
    IntegratorKind kind_;
    State3B state_;
    double time_{0.0};
    double last_dt_{0.0};
    double next_step_guess_{0.0};
    RunnerTuning tuning_{};

    tableau::integration::RK4Integrator<State3B> rk4_;
    tableau::integration::RKF45Integrator<State3B> rkf45_;
    tableau::integration::DOP853Integrator<State3B> dop853_;
};

inline const char* integratorName(IntegratorKind kind) {
    switch (kind) {
    case IntegratorKind::RK4:
        return "RK4";
    case IntegratorKind::RKF45:
        return "RKF45";
    case IntegratorKind::DOP853:
        return "DOP853";
    }
    return "Unknown";
}

} // namespace tableau::demos
