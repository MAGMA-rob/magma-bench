# MAGMA-BENCH

Evaluate agents on interactive robotic tasks with MAGMA. MAGMA-BENCH loads
compiled benchmark episodes, runs them in simulation, communicates with an agent
through the MAGMA HTTP protocol, and saves detailed results and metrics.

**Version 2.0.0b5 is a beta of v2.** APIs, command-line options, and
result formats may change before the stable release. End-to-end validation is
still in progress.

[Official documentation](https://magma-rob.github.io/docs/intro) ·
[Report an issue](https://github.com/MAGMA-rob/magma-bench/issues)

## Installation

Python 3.12 is required. Simulation installation has been checked on Linux
x86_64. GPU simulation and rendering require compatible system drivers.

```bash
python -m pip install "magma_bench==2.0.0b5"
```

This installs `magma_core[simulation]>=2.0.0b3,<3.0.0` and
`magma_scenarios>=2.0.1,<3.0.0` with their Python dependencies. The benchmark
files and the agent server are separate inputs and are not bundled with this
package.

The version is pinned explicitly because this is a prerelease. To request the
latest version, including prereleases, use:

```bash
python -m pip install --upgrade --pre magma_bench
```

## Usage

Start a compatible MAGMA agent server, then run a compiled benchmark directory:

```bash
magma-bench run \
  --benchmark-root /path/to/compiled-benchmark \
  --results-path /path/to/results/experiment-1 \
  --agent-address http://127.0.0.1:8888 \
  --run-name experiment-1
```

The agent must expose `/health`, `/v1/info`, and `/v1/responses` using protocol
version 2.0. Configuration can be supplied with `--config-path`; command-line
options override its benchmark settings.

Useful options include:

```bash
magma-bench run --help
magma-bench run --benchmark-root /path/to/benchmark --results-path ./eval/coffee --scenarios coffee_comp
magma-bench run --benchmark-root /path/to/benchmark --results-path ./eval/no-judge --skip-judge
magma-bench run --benchmark-root /path/to/benchmark --results-path ./eval/logged --model-logs --videos all
magma-bench run --benchmark-root /path/to/benchmark --results-path ./eval/failures --videos planner-failure
```

Video output uses the optional dependencies:

```bash
python -m pip install "magma_bench[video]==2.0.0b5"
```

Version 2 uses result schema `2.0`. Resume a compatible run with
`--results-path`; older result schemas and agent adapters are unsupported.

See the [official documentation](https://magma-rob.github.io/docs/intro) for
benchmark preparation, configuration, agent setup, metrics, and result formats.

## Install from source

Install the tagged release from GitHub:

```bash
python -m pip install "magma_bench @ git+https://github.com/MAGMA-rob/magma-bench.git@v2.0.0b5"
```

Dependencies are resolved from PyPI. For local development, clone the repository
and run `python -m pip install -e ".[dev,video]"`.

## License

[BSD 2-Clause](https://github.com/MAGMA-rob/magma-bench/blob/main/LICENSE).
