import importlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "magma-bench-dev" / "src"
sys.path.insert(0, str(SRC_ROOT))


@dataclass
class _StageResult:
    success: bool = False
    conversation: list = field(default_factory=list)
    explanation: str = ""
    executions_result: list = field(default_factory=list)


def _install_runner_stubs():
    benchmark_module = types.ModuleType("magma_scenarios.benchmark")
    benchmark_module.KNOWN_CRITERIA = ["multi-steps", "c-reasoning", "lg-memorization", "recovery"]
    benchmark_module.LENGTH_GROUPS = {
        "2-3": (2, 3),
        "4-5": (4, 5),
        "6-9": (6, 9),
        "10-16": (10, 16),
    }
    sys.modules["magma_scenarios"] = types.ModuleType("magma_scenarios")
    sys.modules["magma_scenarios.benchmark"] = benchmark_module

    magma_bench_pkg = types.ModuleType("magma_bench")
    magma_bench_pkg.__path__ = [str(SRC_ROOT / "magma_bench")]
    sys.modules["magma_bench"] = magma_bench_pkg

    runner_pkg = types.ModuleType("magma_bench.runner")
    runner_pkg.__path__ = [str(SRC_ROOT / "magma_bench" / "runner")]
    sys.modules["magma_bench.runner"] = runner_pkg

    system_module = types.ModuleType("magma_bench.system")
    system_module.System = object
    sys.modules["magma_bench.system"] = system_module

    builders_module = types.ModuleType("magma_bench.builders")
    builders_module.ScenarioConfig = object
    builders_module.Scenario = object
    builders_module.BenchmarkLoader = object
    builders_module.ScenarioBuilder = object
    builders_module.Task = object
    builders_module.Stage = object
    sys.modules["magma_bench.builders"] = builders_module

    sys.modules.pop("magma_bench.results", None)
    importlib.import_module("magma_bench.results")

    executor_module = types.ModuleType("magma_bench.executor")
    executor_module.ToolsEvalExecutor = object
    sys.modules["magma_bench.executor"] = executor_module

    sys.modules["magma_core"] = types.ModuleType("magma_core")
    sys.modules["magma_core.configs"] = types.ModuleType("magma_core.configs")
    config_module = types.ModuleType("magma_core.configs.config")
    config_module.MAGMAConfig = object
    sys.modules["magma_core.configs.config"] = config_module

    sys.modules["magma_core.utils"] = types.ModuleType("magma_core.utils")
    data_utils_module = types.ModuleType("magma_core.utils.data_utils")
    data_utils_module.apply_att_modif = lambda *_args, **_kwargs: None
    sys.modules["magma_core.utils.data_utils"] = data_utils_module

    text_utils_module = types.ModuleType("magma_core.utils.text_utils")
    text_utils_module.build_model_return_from_executor = (
        lambda call_action, success, reason: {
            "infos": reason[0] if reason else "ok",
            "success": success[0] if success else True,
            "previous_tool_call": call_action,
        }
    )
    text_utils_module.build_fake_execution_fail = (
        lambda _call_action, status_dict, _error_flag: status_dict
    )
    sys.modules["magma_core.utils.text_utils"] = text_utils_module

    global_utils_module = types.ModuleType("magma_core.utils.global_utils")
    global_utils_module.load_module_from_name = lambda *_args, **_kwargs: object
    sys.modules["magma_core.utils.global_utils"] = global_utils_module

    workers_module = types.ModuleType("magma_core.workers")
    workers_module.LMWorker = object
    sys.modules["magma_core.workers"] = workers_module

    tqdm_module = types.ModuleType("tqdm")
    tqdm_module.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules["tqdm"] = tqdm_module


_install_runner_stubs()
sys.modules.pop("magma_bench.runner.runner", None)
runner_module = importlib.import_module("magma_bench.runner.runner")
BenchmarkRunner = runner_module.BenchmarkRunner


class FakeSystem:
    def __init__(self, responses):
        self._responses = list(responses)

    def compute_answer(self, _instruction, _task_attributes):
        if not self._responses:
            raise AssertionError("Unexpected extra model call.")
        return self._responses.pop(0)


class FakeAnswerStage:
    def __init__(self, max_agents_step):
        self.max_agents_step = max_agents_step
        self.expected_behavior = "answer"

    def get_error_state(self):
        return {}

    def has_flag_recovery(self):
        return False

    def has_flag_failure(self):
        return False

    def should_act(self):
        return True

    def is_answer(self):
        return True

    def has_flag_answer_to_user(self):
        return False

    def get_error_flag(self):
        raise AssertionError("Error injection flow should not be used in this test.")


class FakeTextOnlyStage:
    def __init__(self, max_agents_step, is_answer=False):
        self.max_agents_step = max_agents_step
        self._is_answer = is_answer
        self.expected_behavior = "answer" if is_answer else "acknowledge"

    def get_error_state(self):
        return {}

    def has_flag_recovery(self):
        return False

    def has_flag_failure(self):
        return False

    def should_act(self):
        return False

    def is_answer(self):
        return self._is_answer

    def has_flag_answer_to_user(self):
        return False

    def get_error_flag(self):
        raise AssertionError("Error injection flow should not be used in this test.")


class FakeScenario:
    def __init__(self):
        self.sent_actions = []
        self.evaluate_calls = []
        self._turn = 0

    def send_action(self, action, error_state):
        self.sent_actions.append((action, error_state))

    def execute_env_step(self):
        self._turn += 1
        if self._turn == 1:
            return (
                {
                    0: {
                        "planning_error": [False],
                        "success": [True],
                        "reason": ["tool ok"],
                        "att_modif": [],
                    }
                },
                {"obs": "after_tool"},
            )
        return {}, {"obs": f"turn_{self._turn}"}

    def evaluate_stage(self, stage, model_response, obs, do_judge):
        self.evaluate_calls.append((stage, model_response, obs, do_judge))
        return True, "judge ok"


class FakeFailingScenario(FakeScenario):
    def evaluate_stage(self, stage, model_response, obs, do_judge):
        self.evaluate_calls.append((stage, model_response, obs, do_judge))
        return False, "judge failed"


class FakeRuntimeErrorScenario(FakeScenario):
    def execute_env_step(self):
        self._turn += 1
        if self._turn == 1:
            return (
                {
                    0: {
                        "planning_error": [False],
                        "success": [False],
                        "reason": ["runtime error"],
                        "att_modif": [],
                        "runtime_error_triggered": True,
                    }
                },
                {"obs": "after_runtime_error"},
            )
        return {}, {"obs": f"turn_{self._turn}"}


def _make_runner(responses):
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.system = FakeSystem(responses)
    runner._skip_backends = False
    runner.tool_executor = SimpleNamespace(
        ask_for_retry=lambda *_args, **_kwargs: None,
        verif_complementary_bench=lambda *_args, **_kwargs: {
            "verdict": True,
            "explanation": "",
        },
    )
    return runner


def test_answer_stage_with_tools_is_judged_only_on_final_text_answer():
    runner = _make_runner(
        [
            {
                "say": "Je vais verifier.",
                "action": {"name": "lookup_stock", "arguments": {"item": "box_a"}},
            },
            {
                "say": "Il reste 3 boites.",
                "action": {},
            },
        ]
    )
    scenario = FakeScenario()
    stage = FakeAnswerStage(max_agents_step=2)

    result = runner._run_stage(
        scenario,
        stage,
        task_attributes={},
        instruction={"author": "user", "content": "Combien reste-t-il de boites ?"},
    )

    assert result.success is True
    assert result.explanation == "judge ok"
    assert len(scenario.evaluate_calls) == 1
    assert scenario.evaluate_calls[0][1]["say"] == "Il reste 3 boites."
    assert scenario.evaluate_calls[0][1]["action"] == {}
    assert [message["author"] for message in result.conversation] == [
        "user",
        "MODEL",
        "SYSTEM",
        "MODEL",
    ]


def test_answer_stage_with_tools_does_not_call_judge_before_final_answer():
    runner = _make_runner(
        [
            {
                "say": "Je vais verifier.",
                "action": {"name": "lookup_stock", "arguments": {"item": "box_a"}},
            }
        ]
    )
    scenario = FakeScenario()
    stage = FakeAnswerStage(max_agents_step=1)

    result = runner._run_stage(
        scenario,
        stage,
        task_attributes={},
        instruction={"author": "user", "content": "Combien reste-t-il de boites ?"},
    )

    assert result.success is False
    assert result.explanation == "Waiting for a final answer after tool execution."
    assert scenario.evaluate_calls == []
    assert [message["author"] for message in result.conversation] == [
        "user",
        "MODEL",
        "SYSTEM",
    ]


def test_runtime_error_tool_failure_does_not_consume_agent_step_budget():
    runner = _make_runner(
        [
            {
                "say": "Je tente l'action.",
                "action": {"name": "lookup_stock", "arguments": {"item": "box_a"}},
            },
            {
                "say": "L'outil a echoue a cause de l'erreur injectee.",
                "action": {},
            },
        ]
    )
    scenario = FakeRuntimeErrorScenario()
    stage = FakeAnswerStage(max_agents_step=1)

    result = runner._run_stage(
        scenario,
        stage,
        task_attributes={},
        instruction={"author": "user", "content": "Essaie puis explique."},
    )

    assert result.success is True
    assert result.explanation == "judge ok"
    assert len(scenario.evaluate_calls) == 1
    assert scenario.evaluate_calls[0][1]["say"] == "L'outil a echoue a cause de l'erreur injectee."
    assert [message["author"] for message in result.conversation] == [
        "user",
        "MODEL",
        "SYSTEM",
        "MODEL",
    ]


def test_text_only_stage_stops_immediately_after_failed_verification():
    runner = _make_runner(
        [
            {
                "say": "Wrong acknowledgement.",
                "action": {},
            }
        ]
    )
    scenario = FakeFailingScenario()
    stage = FakeTextOnlyStage(max_agents_step=3)

    result = runner._run_stage(
        scenario,
        stage,
        task_attributes={},
        instruction={"author": "user", "content": "Just acknowledge."},
    )

    assert result.success is False
    assert result.explanation == "judge failed"
    assert len(scenario.evaluate_calls) == 1
    assert [message["author"] for message in result.conversation] == [
        "user",
        "MODEL",
    ]


def test_tool_enabled_answer_stage_stops_immediately_after_failed_final_answer():
    runner = _make_runner(
        [
            {
                "say": "Je vais verifier.",
                "action": {"name": "lookup_stock", "arguments": {"item": "box_a"}},
            },
            {
                "say": "Il reste 8 boites.",
                "action": {},
            },
        ]
    )
    scenario = FakeFailingScenario()
    stage = FakeAnswerStage(max_agents_step=4)

    result = runner._run_stage(
        scenario,
        stage,
        task_attributes={},
        instruction={"author": "user", "content": "Combien reste-t-il de boites ?"},
    )

    assert result.success is False
    assert result.explanation == "judge failed"
    assert len(scenario.evaluate_calls) == 1
    assert scenario.evaluate_calls[0][1]["action"] == {}
    assert [message["author"] for message in result.conversation] == [
        "user",
        "MODEL",
        "SYSTEM",
        "MODEL",
    ]


def test_run_filters_selected_tasks_and_skips_metrics(monkeypatch):
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner._benchmark_configs = [SimpleNamespace(scenario_id="scenario_1")]
    runner.output_path = "/tmp/out"
    runner.tool_executor = SimpleNamespace()
    runner.per_task_log = False
    runner.system = SimpleNamespace()

    state = {"stop_called": False, "compute_called": False, "selected_task_ids": None, "closed": False}

    class FakeResultManager:
        def stop(self):
            state["stop_called"] = True

        def compute_global_metrics(self):
            state["compute_called"] = True
            return {}

    class FakeScenario:
        def __init__(self):
            self.name = "scenario_name"
            self.id = "scenario_1"
            self.tasks = [
                SimpleNamespace(id="task_1"),
                SimpleNamespace(id="task_8"),
                SimpleNamespace(id="task_3"),
            ]
            self.nb_tasks = len(self.tasks)

        def close(self):
            state["closed"] = True

    runner.result_manager = FakeResultManager()
    fake_scenario = FakeScenario()

    def fake_run_scenario(scenario):
        state["selected_task_ids"] = [task.id for task in scenario.tasks]

    runner._run_scenario = fake_run_scenario
    monkeypatch.setattr(
        runner_module,
        "ScenarioBuilder",
        SimpleNamespace(load=lambda *_args, **_kwargs: fake_scenario),
    )

    runner.run(SimpleNamespace(no_metrics=True, task_indices=[3, 1], videos=False, seed=42))

    assert state["selected_task_ids"] == ["task_3", "task_1"]
    assert fake_scenario.nb_tasks == 2
    assert state["stop_called"] is True
    assert state["compute_called"] is False
    assert state["closed"] is True


def test_run_scenario_uses_real_task_id_in_video_name(monkeypatch):
    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.num_try = 1
    runner.per_task_log = False
    runner._skip_metrics = True
    runner.system = SimpleNamespace(init_task=lambda *_args: None, reset_step=lambda: None)
    runner.tool_executor = SimpleNamespace(
        set_randomizer_index=lambda *_args: None,
        get_try_randomization_info=lambda: {},
        randomized=False,
    )

    class FakeScenarioResult:
        def __init__(self, *_args, **_kwargs):
            pass

        def record_stage(self, *_args, **_kwargs):
            return None

        def record_skip(self, *_args, **_kwargs):
            return None

        def export_task_log(self, *_args, **_kwargs):
            return None

    class FakeStage:
        id = "stage_0"
        instruction = {"author": "user", "content": "Do something", "timestamp": 0}
        default_instruction = None
        keys_evaluator = []

        def should_reset_env(self):
            return False

    class FakeTask:
        id = "task_42"
        stages = [FakeStage()]

    video_names = []

    class FakeScenario:
        name = "demo"
        id = "scenario_1"
        evaluated_criteria = []
        nb_tasks = 1

        def __init__(self):
            self.tasks = [FakeTask()]

        def get_init_elements(self):
            return {"attributes": {}, "tools": []}

        def get_task(self, idx):
            return self.tasks[idx]

        def env_reset(self):
            return None

        def save_video(self, name):
            video_names.append(name)

    monkeypatch.setattr(runner_module, "ScenarioResult", FakeScenarioResult)
    runner._run_stage = lambda *_args, **_kwargs: _StageResult(
        success=True,
        conversation=[{"author": "status", "content": {"infos": "ok"}}],
        explanation="ok",
        executions_result=[True],
    )

    runner._run_scenario(FakeScenario())

    assert video_names == ["demo-try-0-task-task_42"]


def test_runner_init_can_skip_backends_for_magma_single(monkeypatch):
    state = {
        "lmworker_calls": [],
        "system_kwargs": None,
        "executor_worker": "unset",
        "judge_calls": [],
    }

    class FakeSystemClass:
        def __init__(self, **kwargs):
            state["system_kwargs"] = kwargs
            self.system_name = "demo_system"

    class FakeResultManager:
        def __init__(self, *_args, **_kwargs):
            return None

    class FakeToolsEvalExecutor:
        def __init__(self, _planner_endpoint, worker, nb_env=1, randomize_variation=0):
            state["executor_worker"] = worker

        def verif_complementary_bench(self, model_say, complementary_verif):
            state["judge_calls"].append((model_say, complementary_verif))
            return {
                "verdict": True,
                "explanation": f"complementary_verif={complementary_verif}",
            }

    monkeypatch.setattr(runner_module, "load_module_from_name", lambda *_args, **_kwargs: FakeSystemClass)
    monkeypatch.setattr(runner_module, "ResultManager", FakeResultManager)
    monkeypatch.setattr(runner_module, "ToolsEvalExecutor", FakeToolsEvalExecutor)
    monkeypatch.setattr(
        runner_module,
        "LMWorker",
        lambda backend: state["lmworker_calls"].append(backend) or object(),
    )

    config = SimpleNamespace(
        benchmark={
            "save_dir": "/tmp/out",
            "logs": True,
            "num_eval": 1,
        },
        backends={},
        magma_agent_address="http://agent",
        magma_planner_address="http://planner",
    )

    runner = BenchmarkRunner("MagmaSingle", config, class_specific_args={}, skip_backends=True)
    out = runner.tool_executor.verif_complementary_bench(
        "final answer",
        {
            "logs": [{"action": "demo"}],
            "judge": "must be checked by judge",
        },
    )

    assert state["lmworker_calls"] == []
    assert state["executor_worker"] is None
    assert state["system_kwargs"] == {"agent_url": "http://agent"}
    assert state["judge_calls"] == [("final answer", {"logs": [{"action": "demo"}]})]
    assert out["verdict"] is True
    assert "complementary_verif={'logs': [{'action': 'demo'}]}" in out["explanation"]


def test_runner_init_rejects_skip_backends_for_non_magma_single():
    config = SimpleNamespace(
        benchmark={
            "save_dir": "/tmp/out",
            "logs": True,
            "num_eval": 1,
        },
        backends={},
        magma_agent_address="http://agent",
        magma_planner_address="http://planner",
    )

    with pytest.raises(ValueError, match="skip_backends=True is only supported with MagmaSingle"):
        BenchmarkRunner("StandaloneAgentSystem", config, class_specific_args={}, skip_backends=True)
