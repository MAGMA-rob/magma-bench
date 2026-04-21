import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "magma-bench-dev" / "src"
sys.path.insert(0, str(SRC_ROOT))


def _load_scenario_class():
    module_names = [
        "magma_bench",
        "magma_bench.builders",
        "magma_bench.builders.task",
        "magma_bench.builders.stage",
        "magma_bench.builders.scenario",
        "magma_bench.executor",
        "magma_bench.evalutations",
        "magma_core",
        "magma_core.base",
        "magma_core.base.data_structures",
        "magma_core.base.tasks",
        "magma_scenarios",
    ]
    previous_modules = {name: sys.modules.get(name) for name in module_names}

    try:
        magma_bench_pkg = types.ModuleType("magma_bench")
        magma_bench_pkg.__path__ = [str(SRC_ROOT / "magma_bench")]
        sys.modules["magma_bench"] = magma_bench_pkg

        builders_pkg = types.ModuleType("magma_bench.builders")
        builders_pkg.__path__ = [str(SRC_ROOT / "magma_bench" / "builders")]
        sys.modules["magma_bench.builders"] = builders_pkg

        task_module = types.ModuleType("magma_bench.builders.task")
        task_module.Task = object
        sys.modules["magma_bench.builders.task"] = task_module

        stage_module = types.ModuleType("magma_bench.builders.stage")
        stage_module.Stage = object
        sys.modules["magma_bench.builders.stage"] = stage_module

        executor_module = types.ModuleType("magma_bench.executor")
        executor_module.ToolsEvalExecutor = object
        sys.modules["magma_bench.executor"] = executor_module

        evalutations_module = types.ModuleType("magma_bench.evalutations")
        evalutations_module.evaluate_env_success = lambda *_args, **_kwargs: True
        sys.modules["magma_bench.evalutations"] = evalutations_module

        sys.modules["magma_core"] = types.ModuleType("magma_core")
        sys.modules["magma_core.base"] = types.ModuleType("magma_core.base")
        data_structures_module = types.ModuleType("magma_core.base.data_structures")
        data_structures_module.ActiveStageErrorState = dict
        sys.modules["magma_core.base.data_structures"] = data_structures_module
        tasks_module = types.ModuleType("magma_core.base.tasks")
        tasks_module.BaseBenchmarkTask = object
        sys.modules["magma_core.base.tasks"] = tasks_module

        magma_scenarios_module = types.ModuleType("magma_scenarios")
        magma_scenarios_module.load_preset = lambda _name: None
        sys.modules["magma_scenarios"] = magma_scenarios_module

        scenario_module = importlib.import_module("magma_bench.builders.scenario")
        return scenario_module.Scenario
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


Scenario = _load_scenario_class()


def test_validate_runtime_error_injections_calls_error_argument_validation():
    seen = {}

    class DummyError:
        def get_name(self):
            return "DemoError"

        def validate_arguments(self, arguments):
            seen["arguments"] = arguments

    fake_scenario = SimpleNamespace(
        tasks=[
            SimpleNamespace(
                id="task_3",
                stages=[
                    SimpleNamespace(
                        id="task_30",
                        get_error_state=lambda: {"DemoError": {"required": ["cube_1"]}},
                    )
                ],
            )
        ]
    )
    task_ref = SimpleNamespace(get_active_stage_error=lambda _stage_id, _error_state: [DummyError()])

    Scenario._validate_runtime_error_injections(fake_scenario, task_ref)

    assert seen["arguments"] == {"required": ["cube_1"]}


def test_validate_runtime_error_injections_wraps_argument_validation_failures():
    class DummyError:
        def get_name(self):
            return "DemoError"

        def validate_arguments(self, arguments):
            raise RuntimeError(f"bad arguments: {arguments}")

    fake_scenario = SimpleNamespace(
        tasks=[
            SimpleNamespace(
                id="task_3",
                stages=[
                    SimpleNamespace(
                        id="task_30",
                        get_error_state=lambda: {"DemoError": {"wrong": True}},
                    )
                ],
            )
        ]
    )
    task_ref = SimpleNamespace(get_active_stage_error=lambda _stage_id, _error_state: [DummyError()])

    with pytest.raises(ValueError, match="invalid runtime_error injection"):
        Scenario._validate_runtime_error_injections(fake_scenario, task_ref)
