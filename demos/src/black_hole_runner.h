#pragma once

#include "DOP853.h"
#include "RK4.h"
#include "RKF45.h"
#include "black_hole_physics.h"

#include <algorithm>
#include <cstdint>

namespace tableau::demos {

enum class BlackHoleIntegratorKind {
    RK4 = 0,
    RKF45 = 1,
    DOP853 = 2,
};

inline const char* blackHoleIntegratorName(BlackHoleIntegratorKind kind) {
    switch (kind) {
    case BlackHoleIntegratorKind::RK4:
        return "RK4";
    case BlackHoleIntegratorKind::RKF45:
        return "RKF45";
    case BlackHoleIntegratorKind::DOP853:
        return "DOP853";
    }
    return "Unknown";
}

struct BlackHoleRunnerTuning {
    double rk4_step{1.0 / 240.0};
    double initial_step_guess{1.0 / 180.0};
    double min_step{1e-6};
    double max_step{1.0 / 30.0};
    double target_error{1.0};
    int max_substeps{250000};
};

class BlackHoleSimulation {
public:
    BlackHoleSimulation(
        BlackHoleIntegratorKind kind,
        const ParticleCloudState& initial_state,
        const BlackHoleParams& params,
        std::uint64_t seed,
        const BlackHoleRunnerTuning& tuning = BlackHoleRunnerTuning{})
        : kind_(kind),
          state_(initial_state),
          params_(params),
          seed_(seed),
          rng_(seed),
          tuning_(tuning),
          rk4_([&] {
              tableau::integration::RK4Config<ParticleCloudState> cfg;
              cfg.max_time_step = 0.1;
              cfg.enable_validation = true;
              cfg.validator = [](const ParticleCloudState& s) { return isFiniteState(s); };
              cfg.norm = [](const ParticleCloudState& s) { return stateL2Norm(s); };
              return tableau::integration::RK4Integrator<ParticleCloudState>(cfg);
          }()),
          rkf45_([&] {
              tableau::integration::RKF45Config<ParticleCloudState> cfg;
              cfg.min_step = tuning_.min_step;
              cfg.max_step = tuning_.max_step;
              cfg.absolute_tolerance = 1e-7;
              cfg.relative_tolerance = 1e-7;
              cfg.validator = [](const ParticleCloudState& s) { return isFiniteState(s); };
              cfg.norm = [](const ParticleCloudState& s) { return stateL2Norm(s); };
              return tableau::integration::RKF45Integrator<ParticleCloudState>(cfg);
          }()),
          dop853_([&] {
              tableau::integration::DOP853Config<ParticleCloudState> cfg;
              cfg.min_step = tuning_.min_step;
              cfg.max_step = tuning_.max_step;
              cfg.absolute_tolerance = 1e-8;
              cfg.relative_tolerance = 1e-8;
              cfg.validator = [](const ParticleCloudState& s) { return isFiniteState(s); };
              cfg.norm = [](const ParticleCloudState& s) { return stateL2Norm(s); };
              return tableau::integration::DOP853Integrator<ParticleCloudState>(cfg);
          }()) {}

    BlackHoleIntegratorKind kind() const { return kind_; }
    const ParticleCloudState& state() const { return state_; }
    const BlackHoleParams& params() const { return params_; }

    double time() const { return time_; }
    double lastDt() const { return last_dt_; }

    std::size_t capturedTotal() const { return captured_total_; }
    std::size_t consumeCapturedRecent() {
        const std::size_t v = captured_recent_;
        captured_recent_ = 0;
        return v;
    }

    void setRespawnPerturbation(bool enabled) {
        respawn_perturbation_ = enabled;
    }

    void reset(const ParticleCloudState& initial_state, double initial_time = 0.0) {
        state_ = initial_state;
        time_ = initial_time;
        last_dt_ = 0.0;
        next_step_guess_ = tuning_.initial_step_guess;
        captured_total_ = 0;
        captured_recent_ = 0;
        rng_.seed(seed_);
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

            if (kind_ == BlackHoleIntegratorKind::RK4) {
                const double dt = std::min(tuning_.rk4_step, remaining);
                const auto result = rk4_.step(state_, time_, dt, [this](const ParticleCloudState& s, double t) {
                    return blackHoleDerivative(s, t, params_);
                });
                if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                    return false;
                }

                state_ = result.state;
                last_dt_ = result.time_step_used;
                time_ += result.time_step_used;
                handleCaptures();
                continue;
            }

            const double seed_dt = std::clamp(
                std::min(next_step_guess_, remaining),
                tuning_.min_step,
                tuning_.max_step);

            if (kind_ == BlackHoleIntegratorKind::RKF45) {
                const auto result = rkf45_.adaptiveStep(
                    state_,
                    time_,
                    seed_dt,
                    tuning_.target_error,
                    [this](const ParticleCloudState& s, double t) {
                        return blackHoleDerivative(s, t, params_);
                    });

                if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                    return false;
                }

                state_ = result.state;
                last_dt_ = result.time_step_used;
                time_ += result.time_step_used;
                next_step_guess_ = std::clamp(last_dt_ * 1.35, tuning_.min_step, tuning_.max_step);
                handleCaptures();
                continue;
            }

            const auto result = dop853_.adaptiveStep(
                state_,
                time_,
                seed_dt,
                tuning_.target_error,
                [this](const ParticleCloudState& s, double t) {
                    return blackHoleDerivative(s, t, params_);
                });

            if (!result.success || !isFiniteState(result.state) || result.time_step_used <= 0.0) {
                return false;
            }

            state_ = result.state;
            last_dt_ = result.time_step_used;
            time_ += result.time_step_used;
            next_step_guess_ = std::clamp(last_dt_ * 1.35, tuning_.min_step, tuning_.max_step);
            handleCaptures();
        }

        return true;
    }

private:
    void handleCaptures() {
        const std::size_t captures = respawnCaptured(state_, params_, rng_, respawn_perturbation_);
        captured_total_ += captures;
        captured_recent_ += captures;
    }

private:
    BlackHoleIntegratorKind kind_;
    ParticleCloudState state_;
    BlackHoleParams params_{};

    std::uint64_t seed_{0};
    std::mt19937_64 rng_;
    bool respawn_perturbation_{false};

    double time_{0.0};
    double last_dt_{0.0};
    double next_step_guess_{0.0};

    std::size_t captured_total_{0};
    std::size_t captured_recent_{0};

    BlackHoleRunnerTuning tuning_{};

    tableau::integration::RK4Integrator<ParticleCloudState> rk4_;
    tableau::integration::RKF45Integrator<ParticleCloudState> rkf45_;
    tableau::integration::DOP853Integrator<ParticleCloudState> dop853_;
};

} // namespace tableau::demos
