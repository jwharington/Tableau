# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-13

Initial stable release.

### Integrators

- **RK4**: Classical 4th-order Runge-Kutta with fixed step size. Original implementation.
- **RKF45**: Runge-Kutta-Fehlberg 4(5) with embedded adaptive step control, configurable absolute and relative tolerances. Original implementation.
- **DOP853**: Dormand-Prince 8th-order method with 5th- and 3rd-order embedded error estimates. Direct port of the original Fortran code by E. Hairer and G. Wanner, reinterpreted in C++. Includes dense output interpolation, per-component tolerances, stiffness detection, execution statistics, and step callback interface. Optimized path for `std::vector<double>`.

### Library

- Header-only distribution with CMake `INTERFACE` target (`tableau::core`).
- Generic state support: any type implementing `+`, `-`, and scalar `*`.
- Structured step results, termination status, and instrumentation across all solvers.

### Demonstration Programs

- **three_body_demo**: Real-time three-body gravitational simulation with simultaneous RK4/RKF45/DOP853 integration and conserved-quantity diagnostics.
- **black_hole_demo**: Particle cloud orbiting a Schwarzschild-like potential with capture/respawn dynamics.
- **golf_demo**: Projectile motion under gravity with rolling drag and ground-impact damping.

### Testing and Benchmarks

- Test suite covering DOP853, three-body core, black hole core, and golf core physics.
- Google Benchmark suite for harmonic oscillator, Lorenz attractor, and three-body problems.
