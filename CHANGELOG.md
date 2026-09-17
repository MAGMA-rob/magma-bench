# Changelog

All notable changes to MAGMA Bench are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0b4] - 2026-09-17

### Added

- Added precise resume incompatibility diagnostics that report every differing
  manifest key and its stored and current values.
- Added the shared `--agent-timeout` override backed by MAGMA Core's top-level
  agent timeout setting.

### Changed

- Made run resumption independent from `--run-name` and compatible across beta
  serials of the same agent and benchmark release.
- Kept benchmark fingerprints independent from the declared benchmark version
  while retaining compatibility with fingerprints written by earlier betas.
- Gave tool calls priority when an agent response also contains a message; the
  message is ignored and the calls are executed.
- Made `--results-path` mandatory and removed the timestamped `--save-dir`
  output mode.

## [2.0.0b3] - 2026-09-17

### Added

- Added `off`, `all`, and `planner-failure` video modes. Failure-only recording
  starts after a planner error and publishes a video only when the episode ends
  with an infrastructure failure.

### Changed

- Reduced planner retry log noise and report terminal infrastructure failures
  through the episode completion warning.

## [2.0.0b2] - 2026-09-15

### Changed

- Published the second v2 beta package metadata and installation guidance.

## [2.0.0b1]

### Added

- Published the first v2 beta.

[Unreleased]: https://github.com/MAGMA-rob/magma-bench/compare/v2.0.0b4...HEAD
[2.0.0b4]: https://github.com/MAGMA-rob/magma-bench/compare/v2.0.0b3...v2.0.0b4
[2.0.0b3]: https://github.com/MAGMA-rob/magma-bench/compare/v2.0.0b2...v2.0.0b3
[2.0.0b2]: https://github.com/MAGMA-rob/magma-bench/compare/v2.0.0b1...v2.0.0b2
[2.0.0b1]: https://github.com/MAGMA-rob/magma-bench/releases/tag/v2.0.0b1
