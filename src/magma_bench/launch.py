from .runner import BenchmarkRunner
from magma_core.utils.text_utils import auto_cast
from magma_core.configs import MAGMAConfig

import argparse
from typing import Optional
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("system", type=str, help="The system class name to evaluate. Must be inside magma_bench.system")
    
    group = parser.add_mutually_exclusive_group(required=True)
    possible_criteria = ["multi-steps", "c-reasoning", "lg-memorization", "all"]
    group.add_argument("--criteria", nargs="+", choices=possible_criteria, help="One or more criteria.")
    group.add_argument("--scenarios", nargs="+", help="One or more scenario names.")
    parser.add_argument(
        "--config_path", "-c",
        type=str,
        default=None,
        help="Custom config to pass to MAGMA-GEN."
    )

    parser.add_argument("--logs", action="store_true", help="If specified, save per steps logs data to visualize the full conversation")
    parser.add_argument("--videos", action="store_true", help="If specified, enable the video Wrapper to see videos of the benchmark")
    parser.add_argument(
        "--verifier_backend", '-vb',
        type=str,
        help="Which backend instance to use for the verifier."
    )
    parser.add_argument("-n", "--num_eval", type=int, help="Number of evaluation per instruction.")
    parser.add_argument("-nv", "--num_variations", type=int, help="Number of tools and attributes randomization per task.")
    parser.add_argument(
        "--magma_agent_address", '-mas',
        type=str,
        help="The address of the magma_agent server to use for this generation."
    )
    parser.add_argument("-b", "--sim_backend", type=str, help="Which simulation backend to use. Can be 'auto', 'cpu', 'gpu'")
    parser.add_argument("--shader", type=str, help="Change shader used for rendering. Default is 'default' which is very fast. Can also be 'rt' for ray tracing and generating photo-realistic renders. Can also be 'rt-fast' for a faster but lower quality ray-traced renderer")
    parser.add_argument("--save_dir", type=str, help="where to save videos, log, result of the benchmark")
    parser.add_argument("--seed", type=int, help="The default start seed (default = 42)")
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
    excluded = {"system", "criteria", "scenarios","extra"}
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
    print(f"[MAGMA-BENCH] Loading...")

    override_dict = build_override_dict(args)
    default_path = resolve_config_path(args.config_path)
    magma_config = MAGMAConfig.load(default_path)
    magma_config.override_with_dict(override_dict)

    runner = BenchmarkRunner(
        args.system, 
        magma_config=magma_config,
        class_specific_args=args.extra,
        variations=args.num_variations
    )
    runner.load_benchmark(args.criteria, args.scenarios)

    runner.run(args)


if __name__ == "__main__":
    a = parse_args()
    main(a)