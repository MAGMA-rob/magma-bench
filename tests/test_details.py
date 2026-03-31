import sys
import types
import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "magma-bench-dev" / "src"
sys.path.insert(0, str(SRC_ROOT))

benchmark_module = types.ModuleType("magma_scenarios.benchmark")
benchmark_module.KNOWN_CRITERIA = ["multi-steps", "c-reasoning", "lg-memorization"]
sys.modules["magma_scenarios"] = types.ModuleType("magma_scenarios")
sys.modules["magma_scenarios.benchmark"] = benchmark_module

data_structures_module = types.ModuleType("magma_core.base.data_structures")
data_structures_module.ActiveStageErrorState = dict
sys.modules["magma_core"] = types.ModuleType("magma_core")
sys.modules["magma_core.base"] = types.ModuleType("magma_core.base")
sys.modules["magma_core.base.data_structures"] = data_structures_module

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


@pytest.mark.parametrize(
    "extra_field",
    [
        {"action_goal": [{"predicate": "in", "args": ["obj", "area1"]}]},
        {"injection": {"mode": "force_recovery"}},
        {"default_instruction": {"author": "status", "content": "done"}},
        {"answer_to_user": True},
    ],
)
def test_answer_stage_rejects_forbidden_fields(extra_field):
    with pytest.raises(TypeError):
        Stage(
            {
                "instruction": "Where should object A go?",
                "expected_behavior": "answer",
                "complementary_verif": {"judge": "The model answers area1."},
                **extra_field,
            },
            id="stage0",
        )


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
