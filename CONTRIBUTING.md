# Contributing

Contributions to Tableau are welcome. Please read this document before submitting a pull request.

## Automated Submissions Are Prohibited

**Pull requests authored, co-authored, or submitted by bots, AI agents, or any form of automated tooling are strictly prohibited and will be closed without review.**

This includes, but is not limited to:

- GitHub bots and automated PR generators.
- AI coding assistants submitting PRs on behalf of a user.
- Bulk refactoring tools operating without direct human oversight.
- Any workflow where a human did not personally write and review every line of the proposed change.

If there is any indication that a PR was generated or submitted by an automated system, it will be rejected immediately. No exceptions.

## Before You Start

- Open an issue describing the change you want to make before writing code. This prevents wasted effort on changes that may not align with the project's direction.
- For bug reports, include a minimal reproducible example and the output of `cmake --version`, your compiler version, and operating system.

## Code Standards

- C++17. No compiler extensions.
- Follow the existing code style. No reformatting of unrelated code.
- No new external dependencies in the core library. It must remain header-only with zero dependencies beyond the C++17 standard library.
- Changes to the numerical core (RK4, RKF45, DOP853) require corresponding test coverage.
- DOP853 modifications must preserve numerical equivalence with the original Fortran implementation by Hairer and Wanner. Any deviation must be justified and documented.

## Pull Request Process

1. Fork the repository and create a branch from `main`.
2. Make your changes. Keep commits focused and atomic.
3. Ensure all tests pass: `./build.sh && ctest --test-dir build --output-on-failure`.
4. Write a clear PR description explaining what the change does and why.
5. One approval is required before merging.

## Scope

The following are generally in scope:

- Bug fixes with tests.
- Performance improvements with benchmark evidence.
- New solver implementations, provided they follow the existing `Integrator<State>` interface.
- Documentation corrections.

The following are generally out of scope:

- Stylistic changes or reformatting.
- Adding dependencies to the core library.
- Changes that break the public API without prior discussion.
