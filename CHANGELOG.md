# Changelog

All notable changes to MAGMA Bench are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/MAGMA-rob/magma-bench/compare/v2.0.0b3...HEAD
[2.0.0b3]: https://github.com/MAGMA-rob/magma-bench/compare/v2.0.0b2...v2.0.0b3
[2.0.0b2]: https://github.com/MAGMA-rob/magma-bench/compare/v2.0.0b1...v2.0.0b2
[2.0.0b1]: https://github.com/MAGMA-rob/magma-bench/releases/tag/v2.0.0b1
