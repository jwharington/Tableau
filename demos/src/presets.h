#pragma once

#include "three_body_physics.h"

namespace tableau::demos {

struct ThreeBodyPreset {
    const char* name{"figure8_3d"};
    State3B initial_state{};
};

inline ThreeBodyPreset makeFigure8Preset(bool chaotic_perturbation) {
    ThreeBodyPreset preset;

    State3B s{};

    // Canonical figure-8 (equal masses), extended with non-zero Z components.
    setPosition(s, 0, Vec3{-0.97000436, 0.24308753, 0.08});
    setPosition(s, 1, Vec3{0.97000436, -0.24308753, -0.08});
    setPosition(s, 2, Vec3{0.0, 0.0, 0.0});

    setVelocity(s, 0, Vec3{0.466203685, 0.43236573, 0.15});
    setVelocity(s, 1, Vec3{0.466203685, 0.43236573, -0.15});
    setVelocity(s, 2, Vec3{-0.93240737, -0.86473146, 0.0});

    if (chaotic_perturbation) {
        // Tiny perturbation to expose chaotic divergence over time.
        s[0] += 1e-5;
        s[5] -= 1e-5;
        s[12] += 1e-5;
    }

    preset.initial_state = s;
    return preset;
}

} // namespace tableau::demos
