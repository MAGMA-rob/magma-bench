"""Public command-line entry point; implementations load only after dispatch."""

import argparse
from importlib import import_module
from importlib.metadata import version
import sys


COMMANDS: dict[str, tuple[str, str, str]] = {
    "run": (
        "magma_bench.launch", "args", "Evaluate a benchmark",
    ),
}


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="magma-bench",
        description="Available commands:\n" + "\n".join(
            f"  {name:16} {details[2]}" for name, details in COMMANDS.items()
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"magma-bench {version('magma_bench')}")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER, help="Arguments for the selected command")
    if not arguments:
        parser.print_help()
        return 0
    options = parser.parse_args(arguments)
    module_name, mode, _ = COMMANDS[options.command]
    previous_argv = sys.argv
    try:
        sys.argv = [f"{parser.prog} {options.command}", *options.arguments]
        module = import_module(module_name)
        if mode == "args":
            result = module.main(module.parse_args())
        else:
            result = module.main()
        return 0 if result is None else result
    finally:
        sys.argv = previous_argv
