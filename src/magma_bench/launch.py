from magma_core.utils.text_utils import auto_cast
from magma_core.configs import MAGMAConfig

import argparse
from typing import Optional
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", type=str, help="Benchmark agent mode to evaluate.")
    parser.add_argument(
        '--benchmark-root', '--benchmark_root',
        type=Path,
        default=None,
        help="Directory produced by magma-bench-generator build.",
    )
    
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="One or more scenario IDs or names; all their episodes are run.",
    )
    parser.add_argument(
        '--config-path', '--config_path', "-c",
        type=str,
        default=None,
        help="Custom config to pass to MAGMA-GEN."
    )

    parser.add_argument(
        '--verifier-backend', '--verifier_backend', '-vb',
        type=str,
        help="Which backend instance to use for the verifier."
    )
    parser.add_argument(
        '--magma-agent-address', '--magma_agent_address', '-mas',
        type=str,
        help="The address of the magma_agent server to use for this generation."
    )
    parser.add_argument(
        "-b",
        '--sim-backend', "--sim_backend",
        choices=("auto", "cpu", "gpu"),
        help="Simulation backend.",
    )
    parser.add_argument('--save-dir', '--save_dir', type=str, help="where to save videos, log, result of the benchmark")
    parser.add_argument(
        '--results-path', '--results_path',
        type=Path,
        help="Exact result directory to create or resume.",
    )
    parser.add_argument(
        '--deterministic-decoding', '--deterministic_decoding',
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request greedy inference from magma_agent (default: enabled).",
    )
    parser.add_argument("--seed", type=int, help="The default start seed (default = 42)")
    parser.add_argument(
        "--videos",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record one annotated MP4 per executed episode.",
    )
    parser.add_argument(
        "--model-logs",
        "--logs",
        dest="model_logs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save ordered per-episode TSM/Dispatcher inputs and outputs "
            "(TSR agent)."
        ),
    )
    parser.add_argument(
        '--video-fps', '--video_fps',
        type=int,
        help="Annotated video frame rate (default: 20).",
    )
    parser.add_argument(
        '--video-hold-seconds', '--video_hold_seconds',
        type=float,
        help="Duration of non-physical video events (default: 1 second).",
    )
    parser.add_argument(
        '--skip-judge', '--skip_judge',
        action="store_true",
        help="Auto-validate text answers without starting a judge backend.",
    )
    args, unknown = parser.parse_known_args()

    # Parse extra --key value pairs
    extra_args = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown):
                value = auto_cast(unknown[i + 1])
            else:
                value = True
            extra_args[key] = value
            i += 2
        else:
            raise ValueError(f"Unexpected argument format: {unknown[i]}")

    args.extra = extra_args
    return args

def build_override_dict(args):
    excluded = {
        "agent",
        "benchmark_root",
        "config_path",
        "scenarios",
        "skip_judge",
        "extra",
    }
    overrides = {"benchmark":{}}

    for key, value in vars(args).items():
        if key in excluded:
            continue
        if value is None:
            continue
        if key == "magma_agent_address":
            overrides[key] = value
        else:
            overrides["benchmark"][key] = value

    return overrides

def resolve_config_path(path: Optional[str]) -> Optional[Path]:
    """
    Resolve config file with precedence:
    1. CLI path
    2. ./config.yaml
    3. package default
    """

    if path:
        cli_path = Path(path)
        if cli_path.exists():
            return cli_path
        raise TypeError(f"Impossible to find the config at path: {path}")

    cwd_config = Path.cwd() / "config.yaml"
    if cwd_config.exists():
        return cwd_config

    return None

def main(args : argparse.Namespace):
    from .runner import BenchmarkRunner

    print(f"[MAGMA-BENCH] Loading...")

    override_dict = build_override_dict(args)
    default_path = resolve_config_path(args.config_path)
    magma_config = MAGMAConfig.load(
        default_path,
        accept_no_backend=args.skip_judge,
    )
    magma_config.override_with_dict(override_dict)

    runner = BenchmarkRunner(
        args.agent, 
        magma_config=magma_config,
        class_specific_args=args.extra,
        skip_judge=args.skip_judge,
    )
    runner.load_benchmark(args.benchmark_root, args.scenarios)

    runner.run()


if __name__ == "__main__":
    a = parse_args()
    main(a)
