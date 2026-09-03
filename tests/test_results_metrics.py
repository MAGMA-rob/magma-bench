import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "magma-bench-dev" / "src"))

from magma_bench.results.metrics import compute_metrics
from magma_bench.results.models import (
    EpisodeOutcome,
    EpisodeResult,
    EpisodeTerminal,
    TraceEvent,
)


CONDITIONS = (
    "clean",
    "mission_update",
    "interruption",
    "execution_error",
    "combined",
)


def _build_fixture():
    bindings = []
    results = {}
    clean_success = {
        ("skeleton_a", "semantic_0"): True,
        ("skeleton_a", "semantic_1"): False,
        ("skeleton_b", "semantic_0"): True,
        ("skeleton_b", "semantic_1"): True,
    }
    update_success = {
        ("skeleton_a", "semantic_0"): True,
        ("skeleton_a", "semantic_1"): True,
        ("skeleton_b", "semantic_0"): False,
        ("skeleton_b", "semantic_1"): True,
    }

    for skeleton_id in ("skeleton_a", "skeleton_b"):
        for semantic_id in ("semantic_0", "semantic_1"):
            control_id = f"{skeleton_id}.{semantic_id}.clean"
            for condition in CONDITIONS:
                episode_id = f"{skeleton_id}.{semantic_id}.{condition}"
                lag = 2 if skeleton_id == "skeleton_a" else 12
                intervention_lags = (
                    []
                    if condition == "clean"
                    else ([lag] if condition == "mission_update" else [None])
                )
                episode = SimpleNamespace(
                    episode_id=episode_id,
                    skeleton_id=skeleton_id,
                    track=(
                        "in_domain"
                        if skeleton_id == "skeleton_a"
                        else "compositional"
                    ),
                    condition=condition,
                    semantic=SimpleNamespace(semantic_id=semantic_id),
                    metadata=SimpleNamespace(
                        control_episode_id=control_id,
                        length_bucket=(
                            "short" if skeleton_id == "skeleton_a" else "long"
                        ),
                        intervention_lags=intervention_lags,
                    ),
                )
                if condition == "clean":
                    success = clean_success[(skeleton_id, semantic_id)]
                elif condition == "mission_update":
                    success = update_success[(skeleton_id, semantic_id)]
                else:
                    success = True
                bindings.append(("scenario_a", episode))
                results[episode_id] = EpisodeResult(
                    episode_id=episode_id,
                    success=success,
                    terminal=EpisodeTerminal(
                        status="success" if success else "stage_failure"
                    ),
                    trace=[],
                )
    return bindings, results


def test_metrics_compute_paired_rates_lengths_and_lags():
    bindings, results = _build_fixture()

    metrics = compute_metrics(bindings, results)

    assert metrics.counts.scenario_count == 1
    assert metrics.counts.skeleton_count == 2
    assert metrics.counts.semantic_unit_count == 4
    assert metrics.counts.episode_count == 24
    assert metrics.clean_success_rate.value == 0.75
    assert metrics.success_rate_by_condition["mission_update"].value == 0.75
    assert (
        metrics.conditional_robustness_by_condition["mission_update"].value
        == pytest.approx(2 / 3)
    )
    assert metrics.success_rate_by_length["short"].clean_success_rate.value == 0.5
    assert metrics.success_rate_by_length["long"].clean_success_rate.value == 1.0
    assert metrics.mission_update_retention_by_lag["0-3"].value == 1.0
    assert metrics.mission_update_retention_by_lag["10-19"].value == 0.5
    assert metrics.mission_update_retention_by_lag["4-9"].value is None
    assert metrics.mission_update_retention_by_lag["4-9"].denominator == 0


def test_metrics_reject_missing_pairs_and_ambiguous_update_lag():
    bindings, results = _build_fixture()
    results.pop("skeleton_a.semantic_0.clean")
    with pytest.raises(ValueError, match="result set mismatch"):
        compute_metrics(bindings, results)

    bindings, results = _build_fixture()
    _, update = next(
        binding
        for binding in bindings
        if binding[1].episode_id == "skeleton_a.semantic_0.mission_update"
    )
    update.metadata.intervention_lags = [2, 8]
    with pytest.raises(ValueError, match="exactly one non-null"):
        compute_metrics(bindings, results)


def test_metrics_allow_conditions_on_only_eligible_skeletons():
    bindings, results = _build_fixture()
    bindings = [
        binding
        for binding in bindings
        if not (
            binding[1].skeleton_id == "skeleton_b"
            and binding[1].condition == "interruption"
        )
    ]
    results = {
        episode_id: result
        for episode_id, result in results.items()
        if not episode_id.startswith("skeleton_b.")
        or not episode_id.endswith(".interruption")
    }

    metrics = compute_metrics(bindings, results)

    assert metrics.success_rate_by_condition["interruption"].denominator == 2
    assert metrics.success_rate_by_condition["interruption"].value == 1.0


def test_episode_models_are_strict_and_preserve_last_action_reference():
    action = TraceEvent(
        index=1,
        stage_index=0,
        kind="agent_answer",
        payload={
            "valid": True,
            "say": "",
            "action": {"robot": {"name": "pick", "arguments": {}}},
        },
    )
    result = EpisodeResult(
        episode_id="episode_1",
        success=False,
        terminal=EpisodeTerminal(
            status="stage_failure",
            reason="pick failed",
            last_action_event_index=action.index,
        ),
        trace=[
            TraceEvent(
                index=0,
                stage_index=0,
                kind="instruction",
                payload={"role": "user", "content": "Pick it"},
            ),
            action,
        ],
    )

    decoded = EpisodeResult.model_validate_json(result.model_dump_json())
    assert decoded.terminal.last_action_event_index == 1
    assert decoded.trace[1].payload["action"]["robot"]["name"] == "pick"
    with pytest.raises(Exception):
        EpisodeResult.model_validate({**result.model_dump(), "unused": True})


def _load_result_manager():
    previous_artifacts = sys.modules.get("magma_bench.artifacts")
    previous_data = sys.modules.get("magma_bench.data_structures")

    class BenchmarkManifest(BaseModel):
        benchmark_version: str
        scenarios: list[str] = []
        episodes: list = []

    artifacts = types.ModuleType("magma_bench.artifacts")
    artifacts.BenchmarkManifest = BenchmarkManifest
    artifacts.load_json_model = lambda path, _model: BenchmarkManifest.model_validate_json(
        path.read_text(encoding="utf-8")
    )
    data_structures = types.ModuleType("magma_bench.data_structures")
    data_structures.Episode = object
    data_structures.Scenario = object
    sys.modules["magma_bench.artifacts"] = artifacts
    sys.modules["magma_bench.data_structures"] = data_structures
    sys.modules.pop("magma_bench.results.manager", None)
    try:
        return importlib.import_module("magma_bench.results.manager").ResultManager
    finally:
        if previous_artifacts is None:
            sys.modules.pop("magma_bench.artifacts", None)
        else:
            sys.modules["magma_bench.artifacts"] = previous_artifacts
        if previous_data is None:
            sys.modules.pop("magma_bench.data_structures", None)
        else:
            sys.modules["magma_bench.data_structures"] = previous_data


ResultManager = _load_result_manager()


def _manager_fixture(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    (benchmark_root / "benchmark.json").write_text(
        json.dumps({"benchmark_version": "test-v1"}),
        encoding="utf-8",
    )
    bindings, _ = _build_fixture()
    episodes = tuple(episode for _, episode in bindings)
    scenario = SimpleNamespace(
        scenario_id="scenario_a",
        episodes=episodes,
    )
    agent = {
        "agent": "test-agent",
        "remote_agent": "history_reactive",
        "prediction_mode": "tool_select",
        "agent_url": "http://ignored",
    }
    return benchmark_root, scenario, agent


def _outcome(episode_id, success=True, status=None):
    return EpisodeOutcome(
        episode_id=episode_id,
        success=success,
        terminal=EpisodeTerminal(
            status=status or ("success" if success else "stage_failure")
        ),
        trace=[],
    )


def test_result_manager_saves_scenario_and_resumes(tmp_path):
    benchmark_root, scenario, agent = _manager_fixture(tmp_path)
    results_path = tmp_path / "results"
    manager = ResultManager(results_path, benchmark_root, agent, [scenario])
    assert manager.start_scenario(scenario) is True
    for episode in scenario.episodes:
        manager.record_episode(_outcome(episode.episode_id))
    scenario_result = manager.finish_scenario(scenario)
    benchmark_result = manager.finish_benchmark()

    assert scenario_result.metrics.clean_success_rate.value == 1.0
    assert set(scenario_result.metrics_by_track) == {
        "in_domain",
        "compositional",
    }
    assert benchmark_result.metrics.counts.episode_count == 24
    assert benchmark_result.status == "complete"
    assert benchmark_result.completed_scenario_count == 1
    assert benchmark_result.partial_scenario_ids == []
    assert benchmark_result.metrics_by_track["in_domain"].counts.skeleton_count == 1
    assert (results_path / "result.json").is_file()
    assert (results_path / "scenarios" / "scenario_a" / "result.json").is_file()

    resumed = ResultManager(results_path, benchmark_root, agent, [scenario])
    assert resumed.start_scenario(scenario) is False
    resumed.finish_benchmark()


def test_result_manager_saves_ordered_model_logs_with_scenario(tmp_path):
    benchmark_root, full_scenario, agent = _manager_fixture(tmp_path)
    episode = full_scenario.episodes[0]
    scenario = SimpleNamespace(
        scenario_id=full_scenario.scenario_id,
        episodes=(episode,),
    )
    results_path = tmp_path / "results"
    manager = ResultManager(
        results_path,
        benchmark_root,
        agent,
        [scenario],
        model_logs=True,
    )
    assert manager.start_scenario(scenario) is True
    manager.start_episode_model_logs(episode.episode_id)
    manager.record_model_diagnostics(
        episode.episode_id,
        2,
        [
            {
                "component": "tsm",
                "input": {
                    "instruction": "Move le cube",
                    "permanent_rules": [],
                    "rules": [],
                    "goals": ["[g0] Move the cube"],
                },
                "raw_output": "ADD_GOAL(...)",
                "error": None,
            },
            {
                "component": "dispatcher",
                "input": {
                    "permanent_rules": [],
                    "rules": [],
                    "goals": ["[g0] Move the cube"],
                    "attributes": {"known_robots": ["panda"]},
                    "history": [
                        '{\"name\": \"pick\"}',
                        "The cube was picked",
                    ],
                },
                "raw_output": {"tools": []},
                "error": None,
            },
        ],
    )
    manager.record_episode(_outcome(episode.episode_id))
    manager.finish_scenario(scenario)

    model_log_path = (
        results_path
        / "scenarios"
        / "scenario_a"
        / "model_logs"
        / episode.skeleton_id
        / episode.semantic.semantic_id
        / episode.condition
    )
    assert [path.name for path in sorted(model_log_path.iterdir())] == [
        "0000_tsm.txt",
        "0001_dispatcher.txt",
    ]
    tsm_log = (model_log_path / "0000_tsm.txt").read_text(encoding="utf-8")
    assert "STAGE_INDEX: 2" in tsm_log
    assert "Move le cube" in tsm_log
    assert "ADD_GOAL(...)" in tsm_log
    assert "PARSED OUTPUT" not in tsm_log
    assert "\nSTATE\n" not in tsm_log
    dispatcher_log = (
        model_log_path / "0001_dispatcher.txt"
    ).read_text(encoding="utf-8")
    assert "TOOLS\n" not in dispatcher_log
    assert '{"name": "pick"}\nThe cube was picked' in dispatcher_log


def test_result_manager_replays_partial_and_rejects_incompatible_agent(tmp_path):
    benchmark_root, scenario, agent = _manager_fixture(tmp_path)
    results_path = tmp_path / "results"
    manager = ResultManager(results_path, benchmark_root, agent, [scenario])
    assert manager.start_scenario(scenario) is True
    video_directory = manager.video_directory("scenario_a")
    (video_directory / "preserved.mp4").write_bytes(b"existing-video")
    manager.record_episode(_outcome(scenario.episodes[0].episode_id))

    resumed = ResultManager(results_path, benchmark_root, agent, [scenario])
    assert resumed.start_scenario(scenario) is True
    partial_episodes = results_path / "scenarios" / ".partial" / "scenario_a" / "episodes"
    completed_episode_id = scenario.episodes[0].episode_id
    assert (partial_episodes / f"{completed_episode_id}.json").is_file()
    assert completed_episode_id not in resumed.pending_episode_ids("scenario_a")
    assert len(resumed.pending_episode_ids("scenario_a")) == 23
    assert (resumed.video_directory("scenario_a") / "preserved.mp4").is_file()

    incompatible = dict(agent)
    incompatible["agent"] = "another-agent"
    with pytest.raises(ValueError, match="incompatible"):
        ResultManager(results_path, benchmark_root, incompatible, [scenario])


def test_result_manager_keeps_infrastructure_failure_partial(tmp_path):
    benchmark_root, scenario, agent = _manager_fixture(tmp_path)
    manager = ResultManager(tmp_path / "results", benchmark_root, agent, [scenario])
    manager.start_scenario(scenario)
    for index, episode in enumerate(scenario.episodes):
        manager.record_episode(
            _outcome(
                episode.episode_id,
                success=index != 0,
                status="infrastructure_failure" if index == 0 else "success",
            )
        )
    assert manager.finish_scenario(scenario) is None
    assert (
        tmp_path / "results" / "scenarios" / ".partial" / "scenario_a"
    ).is_dir()
    benchmark_result = manager.finish_benchmark()
    assert benchmark_result.status == "partial"
    assert benchmark_result.completed_scenario_count == 0
    assert benchmark_result.partial_scenario_ids == ["scenario_a"]
    assert benchmark_result.metrics.counts.episode_count == 0
    assert [
        failure.episode_id
        for failure in benchmark_result.infrastructure_failures
    ] == [scenario.episodes[0].episode_id]

    resumed = ResultManager(
        tmp_path / "results",
        benchmark_root,
        agent,
        [scenario],
    )
    assert resumed.start_scenario(scenario) is True
    failed_episode_id = scenario.episodes[0].episode_id
    assert resumed.pending_episode_ids("scenario_a") == {failed_episode_id}
    resumed.record_episode(_outcome(failed_episode_id))
    resumed.finish_scenario(scenario)


def test_result_manager_rejects_corrupt_partial_episode(tmp_path):
    benchmark_root, scenario, agent = _manager_fixture(tmp_path)
    results_path = tmp_path / "results"
    manager = ResultManager(results_path, benchmark_root, agent, [scenario])
    manager.start_scenario(scenario)
    episode_path = (
        results_path
        / "scenarios"
        / ".partial"
        / "scenario_a"
        / "episodes"
        / f"{scenario.episodes[0].episode_id}.json"
    )
    episode_path.write_text("not-json", encoding="utf-8")

    resumed = ResultManager(results_path, benchmark_root, agent, [scenario])
    with pytest.raises(ValueError):
        resumed.start_scenario(scenario)


def test_result_manager_rejects_runtime_identity_changes(tmp_path):
    benchmark_root, scenario, agent = _manager_fixture(tmp_path)
    results_path = tmp_path / "results"
    ResultManager(
        results_path,
        benchmark_root,
        agent,
        [scenario],
        judge_mode="skipped",
        sim_backend="cpu",
        seed=17,
    )

    with pytest.raises(ValueError, match="incompatible"):
        ResultManager(
            results_path,
            benchmark_root,
            agent,
            [scenario],
            judge_mode="backend",
            sim_backend="cpu",
            seed=17,
        )
