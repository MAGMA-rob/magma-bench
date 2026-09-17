from magma_core.configs import MAGMAConfig

import argparse
import json
from typing import Optional
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a compiled MAGMA benchmark.")
    parser.add_argument(
        "--run-name",
        help="Result directory name; defaults to the agent runtime identity.",
    )
    parser.add_argument(
        "--extra-keys",
        type=json.loads,
        default={},
        help="JSON object of advanced options passed to every agent request.",
    )
    parser.add_argument(
        '--benchmark-root',
        type=Path,
        required=True,
        help="Directory produced by magma-bench-generator build.",
    )
    
    parser.add_argument(
        "--scenarios",
        nargs="+",
        help="One or more scenario IDs or names; all their episodes are run.",
    )
    parser.add_argument(
        '--config-path',
        type=Path,
        help="Benchmark configuration file."
    )

    parser.add_argument(
        '--verifier-backend',
        dest="backend_verifier",
        type=str,
        help="Which backend instance to use for the verifier."
    )
    parser.add_argument(
        '--agent-address',
        type=str,
        help="The address of the protocol-v2 agent server."
    )
    parser.add_argument(
        '--planner-address',
        help="Address of the MAGMA motion planner server.",
    )
    parser.add_argument(
        '--agent-timeout',
        type=float,
        help="Agent HTTP timeout in seconds (default: 360).",
    )
    parser.add_argument(
        '--nb-env',
        type=int,
        help="Maximum number of parallel simulation environments.",
    )
    parser.add_argument(
        '--sim-backend',
        choices=("auto", "cpu", "gpu"),
        help="Simulation backend.",
    )
    parser.add_argument(
        '--results-path',
        type=Path,
        required=True,
        help="Exact result directory to create or resume.",
    )
    parser.add_argument(
        '--deterministic-decoding',
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Request greedy inference from the agent runtime (default: enabled).",
    )
    parser.add_argument("--seed", type=int, help="The default start seed (default = 42)")
    parser.add_argument(
        "--videos",
        choices=("off", "all", "planner-failure"),
        default=None,
        help=(
            "Video recording mode: off, all episodes, or only terminal "
            "planner failures."
        ),
    )
    parser.add_argument(
        "--model-logs",
        dest="model_logs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Save complete agent exchanges and ordered internal steps per episode."
        ),
    )
    parser.add_argument(
        '--video-fps',
        type=int,
        help="Annotated video frame rate (default: 20).",
    )
    parser.add_argument(
        '--video-hold-seconds',
        type=float,
        help="Duration of non-physical video events (default: 1 second).",
    )
    parser.add_argument(
        '--skip-judge',
        action="store_true",
        help="Auto-validate text answers without starting a judge backend.",
    )
    args = parser.parse_args()
    if not isinstance(args.extra_keys, dict):
        parser.error("--extra-keys must be a JSON object")
    if args.agent_timeout is not None and args.agent_timeout <= 0:
        parser.error("--agent-timeout must be strictly positive")
    if args.nb_env is not None and args.nb_env <= 0:
        parser.error("--nb-env must be strictly positive")
    if args.skip_judge and args.backend_verifier is not None:
        parser.error("--verifier-backend cannot be combined with --skip-judge")
    return args

def build_override_dict(args):
    excluded = {
        "run_name",
        "benchmark_root",
        "config_path",
        "scenarios",
        "skip_judge",
        "extra_keys",
        "planner_address",
    }
    overrides = {"benchmark":{}}

    for key, value in vars(args).items():
        if key in excluded:
            continue
        if value is None:
            continue
        if key == "agent_address":
            overrides["magma_agent_address"] = value
        elif key == "agent_timeout":
            overrides["magma_agent_timeout"] = value
        else:
            overrides["benchmark"][key] = value

    return overrides

def resolve_config_path(path: Optional[Path]) -> Optional[Path]:
    """
    Resolve config file with precedence:
    1. CLI path
    2. ./config.yaml
    3. package default
    """

    if path:
        if path.is_file():
            return path
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

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
    if args.planner_address is not None:
        magma_config.magma_planner_address = args.planner_address
    if magma_config.benchmark.get("videos", "off") == "off" and (
        args.video_fps is not None or args.video_hold_seconds is not None
    ):
        raise ValueError(
            "--video-fps and --video-hold-seconds require --videos all, "
            "--videos planner-failure, or a matching configuration value."
        )

    runner = BenchmarkRunner(
        magma_config=magma_config,
        run_name=args.run_name,
        extra_keys=args.extra_keys,
        skip_judge=args.skip_judge,
    )
    runner.load_benchmark(args.benchmark_root, args.scenarios)

    runner.run()


if __name__ == "__main__":
    a = parse_args()
    main(a)
