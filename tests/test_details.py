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


class _Log:
    def __init__(self, function_name, content=None, action=None):
        self.function = function_name
        self.content = content
        self.action = action


class _BaseGoal:
    def __init__(self, *args, **kwargs):
        self.base_args = args
        self.base_kwargs = kwargs


class _Goal(_BaseGoal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs


class _AtGoal(_BaseGoal):
    def __init__(self, obj_name, location, thresh=0.1):
        super().__init__(obj_name, location, thresh=thresh)
        self.args = (obj_name, location)
        self.kwargs = {} if thresh == 0.1 else {"thresh": thresh}


class _AtLeastCountAtGoal(_BaseGoal):
    def __init__(self, objects, location, minimum, thresh=0.1):
        super().__init__(objects, location, minimum, thresh=thresh)
        self.args = (objects, location, minimum)
        self.kwargs = {} if thresh == 0.1 else {"thresh": thresh}


class _NotAtGoal(_BaseGoal):
    def __init__(self, obj_name, locations, strict, thresh=0.1):
        super().__init__(obj_name, locations, strict=strict, thresh=thresh)
        self.args = (obj_name, locations)
        self.kwargs = {"strict": strict}
        if thresh != 0.1:
            self.kwargs["thresh"] = thresh


class _OnGoal(_BaseGoal):
    def __init__(self, top_object, bottom_object, thresh=0.05):
        super().__init__(top_object, bottom_object, thresh=thresh)
        self.args = (top_object, bottom_object)
        self.kwargs = {} if thresh == 0.05 else {"thresh": thresh}


goals_module.BaseGoal = _BaseGoal
goals_module.And = _Goal
goals_module.At = _AtGoal
goals_module.AtLeastCountAt = _AtLeastCountAtGoal
goals_module.NotAt = _NotAtGoal
goals_module.On = _OnGoal
goals_module.Or = _Goal

text_utils_module = types.ModuleType("magma_core.utils.text_utils")
text_utils_module.star_extractor = lambda entity_names, pattern: [
    name for name in entity_names if pattern.replace("*", "") in name
]

sys.modules["magma_core"] = types.ModuleType("magma_core")
base_module = types.ModuleType("magma_core.base")
base_module.__path__ = []
sys.modules["magma_core.base"] = base_module
data_structures_module.Log = _Log
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


def _make_log(function_name, content=None, action=None):
    return _Log(function_name=function_name, content=content, action=action)


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
        {"action_goal": [{"predicate": "in", "arguments": {"obj_name": "obj", "location": "area1"}}]},
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


def test_stage_accepts_log_rules_in_complementary_verif():
    stage = Stage(
        {
            "instruction": "Press only after loading the mug.",
            "max_step": 2,
            "complementary_verif": {
                "log_rules": [
                    {
                        "rule_name": "count",
                        "arguments": {
                            "event": {"function_name": "press_button"},
                            "count": 1,
                        },
                    }
                ]
            },
        },
        id="stage0",
    )

    predicates, comp_eval = stage.get_evaluation_elements()
    assert predicates == []
    assert "log_rules" in comp_eval
    assert len(comp_eval["log_rules"]) == 1
    assert comp_eval["log_rules"][0].rule_name == "count"


def test_stage_rejects_unknown_complementary_verif_keys():
    with pytest.raises(TypeError, match="complementary_verif only supports keys"):
        Stage(
            {
                "instruction": "Launch a cycle.",
                "max_step": 1,
                "complementary_verif": {"unknown": True},
            },
            id="stage0",
        )


def test_act_stage_can_use_only_log_rules():
    stage = Stage(
        {
            "instruction": "Wait for the right sequence.",
            "max_step": 1,
            "complementary_verif": {
                "log_rules": [
                    {
                        "rule_name": "requires_before",
                        "arguments": {
                            "event": {"function_name": "press_button"},
                            "required_before": [{"function_name": "place_mug"}],
                        },
                    }
                ]
            },
        },
        id="stage0",
    )

    assert stage.should_act()
    assert stage.get_evaluation_elements()[1]["log_rules"][0].rule_name == "requires_before"


def test_stage_accepts_not_in_log_rule():
    stage = Stage(
        {
            "instruction": "Never press the forbidden button.",
            "max_step": 1,
            "complementary_verif": {
                "log_rules": [
                    {
                        "rule_name": "not_in",
                        "arguments": {
                            "event": {
                                "function_name": "press_button",
                                "content": ["forbidden_button"],
                            }
                        },
                    }
                ]
            },
        },
        id="stage0",
    )

    rule = stage.get_evaluation_elements()[1]["log_rules"][0]
    assert rule.rule_name == "not_in"


def test_not_in_log_rule_passes_when_event_is_absent():
    stage = Stage(
        {
            "instruction": "Avoid the forbidden button.",
            "max_step": 1,
            "complementary_verif": {
                "log_rules": [
                    {
                        "rule_name": "not_in",
                        "arguments": {
                            "event": {
                                "function_name": "press_button",
                                "content": ["forbidden_button"],
                            }
                        },
                    }
                ]
            },
        },
        id="stage0",
    )

    rule = stage.get_evaluation_elements()[1]["log_rules"][0]
    verdict, reason = rule.verify(
        [
            _make_log("open_drawer", content=["left_drawer"]),
            _make_log("press_button", content=["allowed_button"]),
        ]
    )

    assert verdict is True
    assert reason == ""


def test_not_in_log_rule_fails_when_event_is_present():
    stage = Stage(
        {
            "instruction": "Avoid the forbidden button.",
            "max_step": 1,
            "complementary_verif": {
                "log_rules": [
                    {
                        "rule_name": "not_in",
                        "arguments": {
                            "event": {
                                "function_name": "press_button",
                                "content": ["forbidden_button"],
                            }
                        },
                    }
                ]
            },
        },
        id="stage0",
    )

    rule = stage.get_evaluation_elements()[1]["log_rules"][0]
    verdict, reason = rule.verify(
        [
            _make_log("open_drawer", content=["left_drawer"]),
            _make_log("press_button", content=["forbidden_button"]),
        ]
    )

    assert verdict is False
    assert "not_in" in reason
    assert "press_button" in reason


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
        ("answer", {"action_goal": [{"predicate": "in", "arguments": {"obj_name": "obj", "location": "area1"}}]}),
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
            "action_goal": [{"predicate": "in", "arguments": {"obj_name": "ref_obj_*", "location": "area1"}}],
            "keys_evaluator": ["c-reasoning"],
        },
        id="stage0",
    )

    assert stage.should_act()
    assert not stage.is_text_only()
    assert stage.max_agents_step == 2
    assert stage.keys_evaluator == ["c-reasoning"]


def test_act_stage_accepts_named_predicate_args_with_threshold():
    stage = Stage(
        {
            "instruction": "Place the object precisely.",
            "max_step": 1,
            "action_goal": [
                {
                    "predicate": "in",
                    "arguments": {
                        "obj_name": "ref_obj_*",
                        "location": "area1",
                        "thresh": 0.03,
                    },
                }
            ],
        },
        id="stage0",
    )

    predicates, _ = stage.get_evaluation_elements(
        {"extra": {"ref_obj_1": None, "area1": None}}
    )

    assert len(predicates) == 1
    assert predicates[0].args == ("ref_obj_1", "area1")
    assert predicates[0].kwargs == {"thresh": 0.03}


def test_act_stage_rejects_legacy_args_field():
    with pytest.raises(TypeError, match="only supports keys"):
        Stage(
            {
                "instruction": "Place the object precisely.",
                "max_step": 1,
                "action_goal": [
                    {
                        "predicate": "in",
                        "args": ["ref_obj_*", "area1"],
                    }
                ],
            },
            id="stage0",
        )


def test_act_stage_requires_arguments_field():
    with pytest.raises(TypeError, match="must define an 'arguments' dict"):
        Stage(
            {
                "instruction": "Place the object precisely.",
                "max_step": 1,
                "action_goal": [
                    {
                        "predicate": "in",
                    }
                ],
            },
            id="stage0",
        )


def test_act_stage_keeps_star_and_list_expansion():
    stage = Stage(
        {
            "instruction": "Place the objects precisely.",
            "max_step": 1,
            "action_goal": [
                {
                    "predicate": "in",
                    "arguments": {
                        "obj_name": ["ref_obj_*", "extra_obj"],
                        "location": ["area1", "area2"],
                    },
                }
            ],
        },
        id="stage0",
    )

    predicates, _ = stage.get_evaluation_elements(
        {"extra": {"ref_obj_1": None, "ref_obj_2": None, "extra_obj": None, "area1": None, "area2": None}}
    )

    assert len(predicates) == 1
    expanded_goals = predicates[0].args[0]
    assert len(expanded_goals) == 6
    assert sorted(goal.args for goal in expanded_goals) == [
        ("extra_obj", "area1"),
        ("extra_obj", "area2"),
        ("ref_obj_1", "area1"),
        ("ref_obj_1", "area2"),
        ("ref_obj_2", "area1"),
        ("ref_obj_2", "area2"),
    ]


def test_act_stage_rejects_unknown_named_predicate_argument():
    with pytest.raises(TypeError, match="does not support argument"):
        Stage(
            {
                "instruction": "Place the object precisely.",
                "max_step": 1,
                "action_goal": [
                    {
                        "predicate": "in",
                        "arguments": {
                            "obj_name": "ref_obj_*",
                            "location": "area1",
                            "thresh": 0.03,
                            "unknown": True,
                        },
                    }
                ],
            },
            id="stage0",
        )


def test_act_stage_rejects_legacy_argument_names():
    with pytest.raises(TypeError, match="does not support argument 'object'"):
        Stage(
            {
                "instruction": "Place the object precisely.",
                "max_step": 1,
                "action_goal": [
                    {
                        "predicate": "in",
                        "arguments": {
                            "object": "ref_obj_*",
                            "target": "area1",
                        },
                    }
                ],
            },
            id="stage0",
        )


def test_not_in_requires_explicit_strict_argument():
    with pytest.raises(ValueError, match="missing required arguments \\['strict'\\]"):
        Stage(
            {
                "instruction": "Keep the object out of the area.",
                "max_step": 1,
                "action_goal": [
                    {
                        "predicate": "not_in",
                        "arguments": {
                            "obj_name": "ref_obj_*",
                            "locations": "area1",
                        },
                    }
                ],
            },
            id="stage0",
        )


def test_mug_and_capsule_goal_predicate_is_supported():
    stage = Stage(
        {
            "instruction": "Serve a black coffee.",
            "max_step": 1,
            "action_goal": [
                {
                    "predicate": "MugAndCapsuleGoal",
                    "arguments": {
                        "capsule": "black",
                    },
                }
            ],
        },
        id="stage0",
    )

    predicates, _ = stage.get_evaluation_elements({"extra": {"coffee_maker": None, "mug": None, "black": None}})

    assert len(predicates) == 1
    assert predicates[0].capsule == "black"


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
                "action_goal": [{"predicate": "in", "arguments": {"obj_name": "ref_obj_*", "location": "area1"}}],
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
