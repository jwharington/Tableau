#pragma once

#include "DOP853.h"
#include "RK4.h"
#include "RKF45.h"
#include "golf_physics.h"

#include <algorithm>
#include <cmath>

namespace tableau::demos {

enum class GolfIntegratorKind {
    RK4 = 0,
    RKF45 = 1,
    DOP853 = 2,
};

inline const char* golfIntegratorName(GolfIntegratorKind kind) {
    switch (kind) {
    case GolfIntegratorKind::RK4:
        return "RK4";
    case GolfIntegratorKind::RKF45:
        return "RKF45";
    case GolfIntegratorKind::DOP853:
        return "DOP853";
    }
    return "Unknown";
}

struct GolfRunnerTuning {
    double rk4_step{1.0 / 240.0};
    double initial_step_guess{1.0 / 180.0};
    double min_step{1e-6};
    double max_step{1.0 / 30.0};
    double target_error{1.0};
    int max_substeps{200000};
};

class GolfIntegratorRunner {
public:
    GolfIntegratorRunner(
        GolfIntegratorKind kind,
        const GolfState& initial_state,
        const GolfParams& params,
        const GolfRunnerTuning& tuning = GolfRunnerTuning{})
        : kind_(kind),
          state_(initial_state),
          params_(params),
          tuning_(tuning),
          rk4_([&] {
              tableau::integration::RK4Config<GolfState> cfg;
              cfg.max_time_step = 0.1;
              cfg.enable_validation = true;
              cfg.validator = [](const GolfState& s) { return isFiniteState(s); };
              cfg.norm = [](const GolfState& s) { return stateL2Norm(s); };
              return tableau::integration::RK4Integrator<GolfState>(cfg);
          }()),
          rkf45_([&] {
              tableau::integration::RKF45Config<GolfState> cfg;
              cfg.min_step = tuning_.min_step;
              cfg.max_step = tuning_.max_step;
              cfg.absolute_tolerance = 1e-7;
              cfg.relative_tolerance = 1e-7;
              cfg.validator = [](const GolfState& s) { return isFiniteState(s); };
              cfg.norm = [](const GolfState& s) { return stateL2Norm(s); };
              return tableau::integration::RKF45Integrator<GolfState>(cfg);
          }()),
          dop853_([&] {
              tableau::integration::DOP853Config<GolfState> cfg;
              cfg.min_step = tuning_.min_step;
              cfg.max_step = tuning_.max_step;
              cfg.absolute_tolerance = 1e-8;
              cfg.relative_tolerance = 1e-8;
              cfg.validator = [](const GolfState& s) { return isFiniteState(s); };
              cfg.norm = [](const GolfState& s) { return stateL2Norm(s); };
              return tableau::integration::DOP853Integrator<GolfState>(cfg);
          }()) {}

    GolfIntegratorKind kind() const { return kind_; }
    const GolfState& state() const { return state_; }
    const GolfParams& params() const { return params_; }

    double time() const { return time_; }
    double lastDt() const { return last_dt_; }

    void reset(const GolfState& initial_state, double initial_time = 0.0) {
        state_ = initial_state;
        time_ = initial_time;
        last_dt_ = 0.0;
        next_step_guess_ = tuning_.initial_step_guess;
    }

    bool stepTo(double target_time) {
        if (target_time < time_) {
            return false;
        }

        if (isStopped(state_, params_.stop_speed_epsilon)) {
            time_ = target_time;
            return true;
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

            const bool on_ground_rolling =
                (state_[2] <= 1e-4) &&
                (state_[5] <= 1e-6) &&
                ((std::abs(state_[3]) > params_.stop_speed_epsilon) ||
                    (std::abs(state_[4]) > params_.stop_speed_epsilon));
            if (on_ground_rolling) {
                const double dt = remaining;
                const double k = std::max(1e-9, params_.rolling_drag);
                const double factor = std::exp(-k * dt);
                const double inv_k = 1.0 / k;

                state_[0] += state_[3] * (1.0 - factor) * inv_k;
                state_[1] += state_[4] * (1.0 - factor) * inv_k;
                state_[3] *= factor;
                state_[4] *= factor;
                state_[2] = 0.0;
                state_[5] = 0.0;

                last_dt_ = dt;
                time_ += dt;
                clampToGround(state_, params_);
                continue;
            }

            const bool near_impact = (state_[2] <= 1.0) && (state_[5] < 0.0);
            if (near_impact) {
                const double dt = std::min(tuning_.rk4_step, remaining);
                const auto result = rk4_.step(state_, time_, dt, [this](const GolfState& s, double t) {
                    return golfDerivative(s, t, params_);
                });
                if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                    return false;
                }

                state_ = result.state;
                last_dt_ = result.time_step_used;
                time_ += result.time_step_used;
                clampToGround(state_, params_);
                if (isStopped(state_, params_.stop_speed_epsilon)) {
                    time_ = target_time;
                    return true;
                }
                continue;
            }

            if (kind_ == GolfIntegratorKind::RK4) {
                const double dt = std::min(tuning_.rk4_step, remaining);
                const auto result = rk4_.step(state_, time_, dt, [this](const GolfState& s, double t) {
                    return golfDerivative(s, t, params_);
                });
                if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                    return false;
                }

                state_ = result.state;
                last_dt_ = result.time_step_used;
                time_ += result.time_step_used;
                clampToGround(state_, params_);
                if (isStopped(state_, params_.stop_speed_epsilon)) {
                    time_ = target_time;
                    return true;
                }
                continue;
            }

            const double seed_dt = std::clamp(
                std::min(next_step_guess_, remaining),
                tuning_.min_step,
                tuning_.max_step);

            if (kind_ == GolfIntegratorKind::RKF45) {
                const auto result = rkf45_.adaptiveStep(
                    state_,
                    time_,
                    seed_dt,
                    tuning_.target_error,
                    [this](const GolfState& s, double t) {
                        return golfDerivative(s, t, params_);
                    });

                if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                    return false;
                }

                state_ = result.state;
                last_dt_ = result.time_step_used;
                time_ += result.time_step_used;
                next_step_guess_ = std::clamp(last_dt_ * 1.35, tuning_.min_step, tuning_.max_step);
                clampToGround(state_, params_);
                if (isStopped(state_, params_.stop_speed_epsilon)) {
                    time_ = target_time;
                    return true;
                }
                continue;
            }

            const auto result = dop853_.adaptiveStep(
                state_,
                time_,
                seed_dt,
                tuning_.target_error,
                [this](const GolfState& s, double t) {
                    return golfDerivative(s, t, params_);
                });

            if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                return false;
            }

            state_ = result.state;
            last_dt_ = result.time_step_used;
            time_ += result.time_step_used;
            next_step_guess_ = std::clamp(last_dt_ * 1.35, tuning_.min_step, tuning_.max_step);
            clampToGround(state_, params_);
            if (isStopped(state_, params_.stop_speed_epsilon)) {
                time_ = target_time;
                return true;
            }
        }

        return true;
    }

private:
    GolfIntegratorKind kind_;
    GolfState state_{};
    GolfParams params_{};

    double time_{0.0};
    double last_dt_{0.0};
    double next_step_guess_{0.0};

    GolfRunnerTuning tuning_{};

    tableau::integration::RK4Integrator<GolfState> rk4_;
    tableau::integration::RKF45Integrator<GolfState> rkf45_;
    tableau::integration::DOP853Integrator<GolfState> dop853_;
};

} // namespace tableau::demos
