import sys
import types
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "magma-bench-dev" / "src"))
sys.path.insert(0, str(ROOT / "magma-scenarios-dev" / "src"))

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

from magma_bench.results import ResultManager, ScenarioResult, StageResult
from magma_bench.results.metrics import compute_benchmark_metrics


def _make_task(task_id, horizon, bucket):
    return SimpleNamespace(
        id=task_id,
        criteria=["c-reasoning", "lg-memorization", "multi-steps", "recovery"],
        task_horizon=horizon,
        length_bucket=bucket,
    )


def _make_stage(stage_id, horizon, keys=None, recovery=False):
    return SimpleNamespace(
        id=stage_id,
        stage_horizon=horizon,
        keys_evaluator=keys or [],
        has_flag_recovery=lambda: recovery,
        has_flag_failure=lambda: False,
    )


def _make_stage_result(success, executions=None):
    result = StageResult()
    result.success = success
    result.executions_result = [True] if executions is None else executions
    result.conversation = [
        {"author": "user", "content": "instruction"},
        {
            "author": "model",
            "content": {
                "say": "I am doing it." if success else "I failed.",
                "action": {},
            },
        },
        {
            "author": "status",
            "content": (
                {
                    "infos": "Tool succeeded.",
                    "previous_tool_call": {"name": "pick", "arguments": {"item": "box"}},
                }
                if success
                else {
                    "error": "Tool failed.",
                    "previous_tool_call": {"name": "pick", "arguments": {"item": "box"}},
                }
            ),
        },
    ]
    result.explanation = "ok" if success else "failed"
    return result


def test_goal_completion_uses_task_horizon_and_length_buckets():
    scenario_result = ScenarioResult("scenario_a")

    short_task = _make_task("task_short", 4, "4-5")
    long_task = _make_task("task_long", 16, "10-16")

    scenario_result.record_stage(short_task, _make_stage("s0", 2), _make_stage_result(True))
    scenario_result.record_stage(short_task, _make_stage("s1", 2), _make_stage_result(False))

    scenario_result.record_stage(long_task, _make_stage("l0", 6), _make_stage_result(True))
    scenario_result.record_stage(long_task, _make_stage("l1", 10), _make_stage_result(False))

    metrics = scenario_result.compute_result()

    assert metrics.goal_completion_macro == 43.75
    assert metrics.goal_completion_micro == 40.0
    assert metrics.length_breakdown["4-5"].to_dict()["primary_metrics"]["goal_completion_macro"] == 50.0
    assert metrics.length_breakdown["10-16"].to_dict()["primary_metrics"]["goal_completion_macro"] == 37.5


def test_skip_counts_as_eligible_but_not_reached():
    scenario_result = ScenarioResult("scenario_skip")
    task = _make_task("task_skip", 3, "2-3")

    scenario_result.record_skip(task, _make_stage("s0", 1, keys=["lg-memorization"]), "unresolved_instruction")
    metrics = scenario_result.compute_result()

    lg_memory = metrics.secondary_metrics["lg-memorization"].to_dict()
    assert lg_memory["eligible"] == 1
    assert lg_memory["reached"] == 0
    assert lg_memory["success"] == 0
    assert lg_memory["reach_rate"] == 0.0
    assert lg_memory["success_when_reached"] == "non-measured"


def test_secondary_metrics_continue_after_first_failure():
    scenario_result = ScenarioResult("scenario_secondary")
    task = _make_task("task_secondary", 3, "2-3")

    scenario_result.record_stage(
        task,
        _make_stage("s0", 1, keys=["c-reasoning"]),
        _make_stage_result(False),
    )
    scenario_result.record_stage(
        task,
        _make_stage("s1", 2, keys=["lg-memorization"]),
        _make_stage_result(True),
    )

    metrics = scenario_result.compute_result()

    assert metrics.goal_completion_micro == 0.0
    constraint = metrics.secondary_metrics["c-reasoning"].to_dict()
    memory = metrics.secondary_metrics["lg-memorization"].to_dict()
    assert constraint["success"] == 0
    assert memory["reached"] == 1
    assert memory["success"] == 1
    assert memory["success_when_reached"] == 100.0


def test_recovery_is_measured_automatically():
    scenario_result = ScenarioResult("scenario_recovery")
    task = _make_task("task_recovery", 2, "2-3")

    scenario_result.record_stage(
        task,
        _make_stage("s0", 2, recovery=True),
        _make_stage_result(True),
    )

    metrics = scenario_result.compute_result()
    recovery = metrics.secondary_metrics["recovery"].to_dict()
    assert recovery["eligible"] == 1
    assert recovery["reached"] == 1
    assert recovery["success"] == 1
    assert recovery["success_when_reached"] == 100.0


def test_benchmark_metrics_use_raw_counts_not_mean_of_means():
    scenario_a = ScenarioResult("scenario_a")
    scenario_b = ScenarioResult("scenario_b")

    task_a = _make_task("task_a", 4, "4-5")
    task_b = _make_task("task_b", 16, "10-16")

    scenario_a.record_stage(task_a, _make_stage("a0", 2), _make_stage_result(True))
    scenario_a.record_stage(task_a, _make_stage("a1", 2), _make_stage_result(False))

    scenario_b.record_stage(task_b, _make_stage("b0", 16), _make_stage_result(True))

    scenario_a.compute_result()
    scenario_b.compute_result()

    benchmark = compute_benchmark_metrics(
        {
            "scenario_a": [scenario_a],
            "scenario_b": [scenario_b],
        }
    )

    assert benchmark.summary.goal_completion_macro == 75.0
    assert benchmark.summary.goal_completion_micro == 90.0


def test_task_log_exports_eagerly_and_final_export_does_not_rewrite_it(tmp_path):
    manager = ResultManager(str(tmp_path), per_task_log=True)
    try:
        scenario_result = ScenarioResult(
            "scenario_export",
            ["c-reasoning"],
            try_number=2,
            randomization_info={
                "enabled": True,
                "variation_index": 1,
                "attributes": {"objects": ["alpha", "beta"]},
                "tools": [
                    {
                        "name": "press",
                        "description": "Press one object.",
                        "parameter_names": ["target"],
                    }
                ],
            },
        )
        task = _make_task("task_export", 4, "4-5")

        scenario_result.record_stage(
            task,
            _make_stage("s0", 2, keys=["c-reasoning"]),
            _make_stage_result(True),
        )
        scenario_result.record_stage(
            task,
            _make_stage("s1", 2),
            _make_stage_result(False),
        )

        try_path = Path(
            manager.get_try_output_path(scenario_result.scenario_id, scenario_result.try_number)
        )
        log_path = try_path / "logs" / "task_export.json"
        try_score_path = try_path / "try_score.json"

        scenario_result.export_task_log(task, str(try_path))

        payload = json.loads(log_path.read_text())
        assert set(payload.keys()) == {"header", "stage"}
        assert payload["header"] == {
            "task_success": False,
            "goal_completion": 50.0,
            "secondary_metrics": {
                "c-reasoning": "1 / 1",
                "lg-memorization": "0 / 0",
                "multi-steps": "0 / 0",
                "recovery": "0 / 0",
            },
        }
        assert payload["stage"] == [
            {
                "success": True,
                "verification_log": "ok",
                "conversation": [
                    {"author": "user", "content": "instruction"},
                    {"author": "model", "content": {"say": "I am doing it.", "action": {}}},
                    {"author": "status", "content": "Tool succeeded."},
                ],
            },
            {
                "success": False,
                "verification_log": "failed",
                "conversation": [
                    {"author": "user", "content": "instruction"},
                    {"author": "model", "content": {"say": "I failed.", "action": {}}},
                    {"author": "status", "content": "Tool failed."},
                ],
            },
        ]
        assert not try_score_path.exists()

        log_path.write_text('{"sentinel": true}')

        scenario_result.compute_result()
        scenario_result.export(str(try_path))

        assert json.loads(log_path.read_text()) == {"sentinel": True}
        try_score = json.loads(try_score_path.read_text())
        assert try_score["counts"]["task_count"] == 1
        assert try_score["counts"]["completed_prefix_horizon"] == 2
        assert try_score["randomization"] == {
            "enabled": True,
            "variation_index": 1,
            "attributes": {"objects": ["alpha", "beta"]},
            "tools": [
                {
                    "name": "press",
                    "description": "Press one object.",
                    "parameter_names": ["target"],
                }
            ],
        }
    finally:
        manager.stop()


def test_result_manager_exports_scenario_score_to_try_number_without_logs(tmp_path):
    manager = ResultManager(str(tmp_path), per_task_log=False)
    scenario_result = ScenarioResult("scenario_manager", ["c-reasoning"], try_number=3)
    task = _make_task("task_manager", 2, "2-3")

    scenario_result.record_stage(
        task,
        _make_stage("s0", 2, keys=["c-reasoning"]),
        _make_stage_result(True),
    )

    manager.push_scenario_result(scenario_result)
    manager.stop()

    try_path = tmp_path / "scenario_manager" / "try-3"
    assert (try_path / "try_score.json").exists()
    assert not (try_path / "logs").exists()
