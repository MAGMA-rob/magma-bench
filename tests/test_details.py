import sys
import types
import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "magma-bench-dev" / "src"
sys.path.insert(0, str(SRC_ROOT))

benchmark_module = types.ModuleType("magma_scenarios.benchmark")
benchmark_module.KNOWN_CRITERIA = ["multi-steps", "c-reasoning", "lg-memorization", "recovery"]


def _count_task_horizon(stage_list):
    total = 0
    for stage in stage_list:
        expected_behavior = stage.get("expected_behavior", "act")
        max_step = stage.get("max_step", 1 if expected_behavior in {"acknowledge", "answer"} else None)
        total += max_step
        if stage.get("answer_to_user") or stage.get("flag_answer_to_user"):
            total += 1
        raw_injections = stage.get("injection", stage.get("injections"))
        if raw_injections is None:
            continue
        if isinstance(raw_injections, dict):
            raw_injections = [raw_injections]
        if any(injection.get("mode") in {"force_recovery", "force_failure"} for injection in raw_injections):
            total += 1
    return total


def _get_length_bucket(task_horizon):
    if 2 <= task_horizon <= 3:
        return "2-3"
    if 4 <= task_horizon <= 5:
        return "4-5"
    if 6 <= task_horizon <= 9:
        return "6-9"
    if 10 <= task_horizon <= 16:
        return "10-16"
    return "other"


def _task_has_recovery_criterion(stage_list):
    for stage in stage_list:
        raw_injections = stage.get("injection", stage.get("injections"))
        if raw_injections is None:
            continue
        if isinstance(raw_injections, dict):
            raw_injections = [raw_injections]
        if any(injection.get("mode") in {"force_recovery", "force_failure"} for injection in raw_injections):
            return True
    return False


benchmark_module.count_task_horizon = _count_task_horizon
benchmark_module.get_length_bucket = _get_length_bucket
benchmark_module.task_has_recovery_criterion = _task_has_recovery_criterion
sys.modules["magma_scenarios"] = types.ModuleType("magma_scenarios")
sys.modules["magma_scenarios.benchmark"] = benchmark_module

data_structures_module = types.ModuleType("magma_core.base.data_structures")
data_structures_module.ActiveStageErrorState = dict
goals_module = types.ModuleType("magma_core.base.goals")


class _BaseGoal:
    pass


class _Goal(_BaseGoal):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


goals_module.BaseGoal = _BaseGoal
goals_module.And = _Goal
goals_module.At = _Goal
goals_module.AtLeastCountAt = _Goal
goals_module.NotAt = _Goal
goals_module.On = _Goal
goals_module.Or = _Goal

text_utils_module = types.ModuleType("magma_core.utils.text_utils")
text_utils_module.star_extractor = lambda entity_names, pattern: [
    name for name in entity_names if pattern.replace("*", "") in name
]

sys.modules["magma_core"] = types.ModuleType("magma_core")
base_module = types.ModuleType("magma_core.base")
base_module.__path__ = []
sys.modules["magma_core.base"] = base_module
sys.modules["magma_core.base.data_structures"] = data_structures_module
sys.modules["magma_core.base.goals"] = goals_module
sys.modules["magma_core.utils"] = types.ModuleType("magma_core.utils")
sys.modules["magma_core.utils.text_utils"] = text_utils_module

magma_bench_pkg = types.ModuleType("magma_bench")
magma_bench_pkg.__path__ = [str(SRC_ROOT / "magma_bench")]
sys.modules["magma_bench"] = magma_bench_pkg

builders_pkg = types.ModuleType("magma_bench.builders")
builders_pkg.__path__ = [str(SRC_ROOT / "magma_bench" / "builders")]
sys.modules["magma_bench.builders"] = builders_pkg

stage_module = importlib.import_module("magma_bench.builders.stage")
task_module = importlib.import_module("magma_bench.builders.task")

Stage = stage_module.Stage
Task = task_module.Task


def test_acknowledge_stage_parses_without_max_step():
    stage = Stage(
        {
            "instruction": "Remember this rule.",
            "expected_behavior": "acknowledge",
        },
        id="stage0",
    )

    assert stage.is_acknowledge()
    assert stage.is_text_only()
    assert not stage.should_act()
    assert stage.max_agents_step == 1
    assert stage.keys_evaluator == []
    assert stage.get_evaluation_elements() == ([], {})


@pytest.mark.parametrize(
    "extra_field",
    [
        {"complementary_verif": {"judge": "Say yes"}},
        {"keys_evaluator": ["c-reasoning"]},
        {"action_goal": [{"predicate": "in", "args": ["obj", "area1"]}]},
        {"injection": {"mode": "force_recovery"}},
        {"should_reset_env": True},
    ],
)
def test_acknowledge_stage_rejects_forbidden_fields(extra_field):
    with pytest.raises(TypeError):
        Stage(
            {
                "instruction": "Remember this rule.",
                "expected_behavior": "acknowledge",
                **extra_field,
            },
            id="stage0",
        )


def test_answer_stage_accepts_complementary_verif_and_keys_evaluator():
    stage = Stage(
        {
            "instruction": "Where should object A go?",
            "expected_behavior": "answer",
            "complementary_verif": {"judge": "The model answers area1."},
            "keys_evaluator": ["lg-memorization"],
        },
        id="stage0",
    )

    assert stage.is_answer()
    assert stage.is_text_only()
    assert not stage.should_act()
    assert stage.max_agents_step == 1
    assert stage.keys_evaluator == ["lg-memorization"]
    assert stage.get_evaluation_elements() == ([], {"judge": "The model answers area1."})


def test_answer_stage_can_opt_in_to_tools():
    stage = Stage(
        {
            "instruction": "Check the inventory before answering.",
            "expected_behavior": "answer",
            "complementary_verif": {"judge": "The model answers with the fetched quantity."},
            "allows_tools": True,
            "max_step": 2,
        },
        id="stage0",
    )

    assert stage.is_answer()
    assert stage.should_act()
    assert not stage.is_text_only()
    assert stage.max_agents_step == 2


def test_answer_stage_rejects_non_boolean_allows_tools():
    with pytest.raises(TypeError):
        Stage(
            {
                "instruction": "Where should object A go?",
                "expected_behavior": "answer",
                "complementary_verif": {"judge": "The model answers area1."},
                "allows_tools": "yes",
            },
            id="stage0",
        )


@pytest.mark.parametrize(
    ("expected_behavior", "extra_field"),
    [
        ("answer", {"action_goal": [{"predicate": "in", "args": ["obj", "area1"]}]}),
        ("answer", {"injection": {"mode": "force_recovery"}}),
        ("answer", {"default_instruction": {"author": "status", "content": "done"}}),
        ("answer", {"answer_to_user": True}),
        ("acknowledge", {"allows_tools": True}),
        ("act", {"allows_tools": True}),
    ],
)
def test_non_supported_stage_fields_raise(expected_behavior, extra_field):
    payload = {
        "instruction": "Where should object A go?",
        **extra_field,
    }
    if expected_behavior == "answer":
        payload.update(
            {
                "expected_behavior": "answer",
                "complementary_verif": {"judge": "The model answers area1."},
            }
        )
    elif expected_behavior == "acknowledge":
        payload["expected_behavior"] = "acknowledge"
    else:
        payload.update(
            {
                "max_step": 1,
                "action_goal": [],
            }
        )

    with pytest.raises(TypeError):
        Stage(payload, id="stage0")


def test_act_stage_behavior_is_still_supported():
    stage = Stage(
        {
            "instruction": "Launch a cycle.",
            "max_step": 2,
            "action_goal": [{"predicate": "in", "args": ["ref_obj_*", "area1"]}],
            "keys_evaluator": ["c-reasoning"],
        },
        id="stage0",
    )

    assert stage.should_act()
    assert not stage.is_text_only()
    assert stage.max_agents_step == 2
    assert stage.keys_evaluator == ["c-reasoning"]


@pytest.mark.parametrize(
    ("answer_flag_key", "injection_mode"),
    [
        ("answer_to_user", "force_failure"),
        ("flag_answer_to_user", "force_failure"),
        ("answer_to_user", "force_recovery"),
        ("flag_answer_to_user", "force_recovery"),
    ],
)
def test_act_stage_rejects_answer_to_user_with_forced_error_injection(answer_flag_key, injection_mode):
    with pytest.raises(TypeError):
        Stage(
            {
                "instruction": "Launch a cycle.",
                "max_step": 2,
                "action_goal": [{"predicate": "in", "args": ["ref_obj_*", "area1"]}],
                answer_flag_key: True,
                "injection": {"mode": injection_mode},
            },
            id="stage0",
        )


def test_multi_steps_propagates_only_on_null_instruction_chain():
    task = Task(
        {
            "id": "task_0",
            "criteria": ["multi-steps", "c-reasoning", "lg-memorization"],
            "stages": [
                {
                    "instruction": "Do the first part.",
                    "max_step": 1,
                    "keys_evaluator": ["multi-steps", "c-reasoning"],
                },
                {
                    "instruction": None,
                    "default_instruction": {"author": "status", "content": "step1 done"},
                    "max_step": 1,
                },
                {
                    "instruction": None,
                    "default_instruction": {"author": "status", "content": "step2 done"},
                    "max_step": 1,
                    "keys_evaluator": ["lg-memorization"],
                },
                {
                    "instruction": "Start another branch.",
                    "max_step": 1,
                    "keys_evaluator": ["c-reasoning"],
                },
                {
                    "instruction": None,
                    "default_instruction": {"author": "status", "content": "after stop"},
                    "max_step": 1,
                },
            ],
        }
    )

    assert task.stages[0].keys_evaluator == ["multi-steps", "c-reasoning"]
    assert task.stages[1].keys_evaluator == ["multi-steps"]
    assert task.stages[2].keys_evaluator == ["lg-memorization", "multi-steps"]
    assert task.stages[3].keys_evaluator == ["c-reasoning"]
    assert task.stages[4].keys_evaluator == []
