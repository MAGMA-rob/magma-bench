from .launch import build_override_dict, resolve_config_path
from .runner import BenchmarkRunner
from magma_core.configs import MAGMAConfig
from magma_core.utils.text_utils import auto_cast

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", type=str, help="Scenario name to evaluate.")
    parser.add_argument(
        "--benchmark_root",
        type=str,
        default=None,
        help="Directory produced by magma-bench-build.",
    )
    parser.add_argument(
        "--task_indices",
        nargs="+",
        type=int,
        required=True,
        help="Indices of the scenario tasks to execute.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="task_state_reactive",
        help="Benchmark agent mode to evaluate.",
    )
    parser.add_argument(
        "--config_path", "-c",
        type=str,
        default=None,
        help="Custom config to pass to MAGMA-BENCH.",
    )
    parser.add_argument("--videos", action="store_true", help="If specified, enable video export.")
    parser.add_argument(
        "--skip_judge",
        action="store_true",
        help="If specified, bypass text judge verification for fast local tests.",
    )
    parser.add_argument(
        "--verifier_backend", "-vb",
        type=str,
        help="Which backend instance to use for the verifier.",
    )
    parser.add_argument(
        "--magma_agent_address", "-mas",
        type=str,
        help="The address of the magma_agent server to use for this generation.",
    )
    parser.add_argument("-b", "--sim_backend", type=str, help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--shader", type=str, help="Shader used for rendering.")
    parser.add_argument("--save_dir", type=str, help="Where to save videos and logs.")
    parser.add_argument("--seed", type=int, help="The default start seed (default = 42)")
    args, unknown = parser.parse_known_args()

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
    args.no_metrics = True
    return args


def main(args: argparse.Namespace):
    print("[MAGMA-BENCH] Loading subset benchmark...")

    override_dict = build_override_dict(args)
    override_dict.setdefault("benchmark", {})
    override_dict["benchmark"].pop("scenario", None)
    override_dict["benchmark"].pop("task_indices", None)
    override_dict["benchmark"].pop("no_metrics", None)
    override_dict["benchmark"].pop("skip_judge", None)

    default_path = resolve_config_path(args.config_path)
    magma_config = MAGMAConfig.load(default_path, accept_no_backend=args.skip_judge)
    magma_config.override_with_dict(override_dict)

    runner = BenchmarkRunner(
        args.agent,
        magma_config=magma_config,
        class_specific_args=args.extra,
        skip_backends=args.skip_judge,
    )
    runner.load_benchmark(getattr(args, "benchmark_root", None), [args.scenario])

    if len(runner._scenarios) != 1:
        raise ValueError(
            f"Scenario selection must resolve to exactly one scenario. Got {len(runner._scenarios)}."
        )

    runner.run(args)


if __name__ == "__main__":
    a = parse_args()
    main(a)
