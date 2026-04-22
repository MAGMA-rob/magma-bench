import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "magma-bench-dev" / "src"
sys.path.insert(0, str(SRC_ROOT))


def test_parse_args_sets_skip_judge_flag(monkeypatch):
    magma_bench_pkg = types.ModuleType("magma_bench")
    magma_bench_pkg.__path__ = [str(SRC_ROOT / "magma_bench")]
    sys.modules["magma_bench"] = magma_bench_pkg

    launch_module = types.ModuleType("magma_bench.launch")
    launch_module.build_override_dict = lambda _args: {"benchmark": {}}
    launch_module.resolve_config_path = lambda path: path
    sys.modules["magma_bench.launch"] = launch_module

    runner_module = types.ModuleType("magma_bench.runner")
    runner_module.BenchmarkRunner = object
    sys.modules["magma_bench.runner"] = runner_module

    sys.modules["magma_core"] = types.ModuleType("magma_core")
    configs_module = types.ModuleType("magma_core.configs")
    configs_module.MAGMAConfig = object
    sys.modules["magma_core.configs"] = configs_module
    sys.modules["magma_core.utils"] = types.ModuleType("magma_core.utils")
    text_utils_module = types.ModuleType("magma_core.utils.text_utils")
    text_utils_module.auto_cast = lambda value: value
    sys.modules["magma_core.utils.text_utils"] = text_utils_module

    sys.modules.pop("magma_bench.launch_subset", None)
    launch_subset = importlib.import_module("magma_bench.launch_subset")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "launch_subset.py",
            "demo_scenario",
            "--task_indices",
            "0",
            "--skip_judge",
        ],
    )

    args = launch_subset.parse_args()

    assert args.skip_judge is True


def test_launch_subset_enables_backendless_test_mode_when_skip_judge_is_set():
    magma_bench_pkg = types.ModuleType("magma_bench")
    magma_bench_pkg.__path__ = [str(SRC_ROOT / "magma_bench")]
    sys.modules["magma_bench"] = magma_bench_pkg

    load_calls = {}
    override_calls = {}
    runner_state = {}

    launch_module = types.ModuleType("magma_bench.launch")
    launch_module.build_override_dict = lambda _args: {"benchmark": {"logs": False}}
    launch_module.resolve_config_path = lambda path: path
    sys.modules["magma_bench.launch"] = launch_module

    class FakeConfig:
        def override_with_dict(self, override):
            override_calls["value"] = override

    class FakeMAGMAConfig:
        @staticmethod
        def load(path, accept_no_backend=False):
            load_calls["path"] = path
            load_calls["accept_no_backend"] = accept_no_backend
            return FakeConfig()

    sys.modules["magma_core"] = types.ModuleType("magma_core")
    configs_module = types.ModuleType("magma_core.configs")
    configs_module.MAGMAConfig = FakeMAGMAConfig
    sys.modules["magma_core.configs"] = configs_module
    sys.modules["magma_core.utils"] = types.ModuleType("magma_core.utils")
    text_utils_module = types.ModuleType("magma_core.utils.text_utils")
    text_utils_module.auto_cast = lambda value: value
    sys.modules["magma_core.utils.text_utils"] = text_utils_module

    class FakeRunner:
        def __init__(self, system_name, magma_config, class_specific_args, skip_backends=False):
            runner_state["init"] = (system_name, magma_config, class_specific_args, skip_backends)
            self._benchmark_configs = [SimpleNamespace()]

        def load_benchmark(self, criteria, scenarios):
            runner_state["load_benchmark"] = (criteria, scenarios)

        def run(self, args):
            runner_state["run_args"] = args

    runner_module = types.ModuleType("magma_bench.runner")
    runner_module.BenchmarkRunner = FakeRunner
    sys.modules["magma_bench.runner"] = runner_module

    sys.modules.pop("magma_bench.launch_subset", None)
    launch_subset = importlib.import_module("magma_bench.launch_subset")

    args = SimpleNamespace(
        scenario="demo_scenario",
        system="MagmaSingle",
        config_path="/tmp/config.yaml",
        extra={"temperature": 0},
        skip_judge=True,
        no_metrics=True,
        task_indices=[0],
        videos=False,
        verifier_backend=None,
        num_eval=None,
        magma_agent_address=None,
        sim_backend=None,
        shader=None,
        save_dir=None,
        seed=None,
    )

    launch_subset.main(args)

    assert load_calls == {
        "path": "/tmp/config.yaml",
        "accept_no_backend": True,
    }
    assert override_calls["value"]["benchmark"]["logs"] is True
    assert "skip_judge_verification" not in override_calls["value"]["benchmark"]
    assert runner_state["init"][3] is True
    assert runner_state["load_benchmark"] == (None, ["demo_scenario"])
    assert runner_state["run_args"] is args


def test_launch_subset_keeps_judge_enabled_by_default():
    magma_bench_pkg = types.ModuleType("magma_bench")
    magma_bench_pkg.__path__ = [str(SRC_ROOT / "magma_bench")]
    sys.modules["magma_bench"] = magma_bench_pkg

    load_calls = {}
    override_calls = {}
    runner_state = {}

    launch_module = types.ModuleType("magma_bench.launch")
    launch_module.build_override_dict = lambda _args: {"benchmark": {"logs": False, "skip_judge": False}}
    launch_module.resolve_config_path = lambda path: path
    sys.modules["magma_bench.launch"] = launch_module

    class FakeConfig:
        def override_with_dict(self, override):
            override_calls["value"] = override

    class FakeMAGMAConfig:
        @staticmethod
        def load(path, accept_no_backend=False):
            load_calls["path"] = path
            load_calls["accept_no_backend"] = accept_no_backend
            return FakeConfig()

    sys.modules["magma_core"] = types.ModuleType("magma_core")
    configs_module = types.ModuleType("magma_core.configs")
    configs_module.MAGMAConfig = FakeMAGMAConfig
    sys.modules["magma_core.configs"] = configs_module
    sys.modules["magma_core.utils"] = types.ModuleType("magma_core.utils")
    text_utils_module = types.ModuleType("magma_core.utils.text_utils")
    text_utils_module.auto_cast = lambda value: value
    sys.modules["magma_core.utils.text_utils"] = text_utils_module

    class FakeRunner:
        def __init__(self, system_name, magma_config, class_specific_args, skip_backends=False):
            runner_state["init"] = (system_name, magma_config, class_specific_args, skip_backends)
            self._benchmark_configs = [SimpleNamespace()]

        def load_benchmark(self, criteria, scenarios):
            runner_state["load_benchmark"] = (criteria, scenarios)

        def run(self, args):
            runner_state["run_args"] = args

    runner_module = types.ModuleType("magma_bench.runner")
    runner_module.BenchmarkRunner = FakeRunner
    sys.modules["magma_bench.runner"] = runner_module

    sys.modules.pop("magma_bench.launch_subset", None)
    launch_subset = importlib.import_module("magma_bench.launch_subset")

    args = SimpleNamespace(
        scenario="demo_scenario",
        system="MagmaSingle",
        config_path="/tmp/config.yaml",
        extra={"temperature": 0},
        skip_judge=False,
        no_metrics=True,
        task_indices=[0],
        videos=False,
        verifier_backend=None,
        num_eval=None,
        magma_agent_address=None,
        sim_backend=None,
        shader=None,
        save_dir=None,
        seed=None,
    )

    launch_subset.main(args)

    assert load_calls == {
        "path": "/tmp/config.yaml",
        "accept_no_backend": False,
    }
    assert override_calls["value"]["benchmark"]["logs"] is True
    assert "skip_judge" not in override_calls["value"]["benchmark"]
    assert "skip_judge_verification" not in override_calls["value"]["benchmark"]
    assert runner_state["init"][3] is False
    assert runner_state["load_benchmark"] == (None, ["demo_scenario"])
    assert runner_state["run_args"] is args
