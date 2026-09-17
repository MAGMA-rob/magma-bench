from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Literal, Optional, Sequence, Set

from packaging.version import InvalidVersion, Version

from magma_bench.artifacts import BenchmarkManifest, load_json_model
from magma_bench.data_structures import Episode, Scenario

from .metrics import EpisodeBinding, compute_metrics
from .models import (
    AgentIdentity,
    BenchmarkResult,
    EpisodeOutcome,
    EpisodeResult,
    ExecutionIdentity,
    InfrastructureFailure,
    MetricsPayload,
    RunManifest,
    RunResumeIdentity,
    ScenarioResult,
)


def _compute_metrics_by_track(
    bindings: Sequence[EpisodeBinding],
    results: Dict[str, EpisodeResult],
) -> Dict[str, MetricsPayload]:
    metrics_by_track = {}
    tracks = sorted({episode.track for _, episode in bindings})
    for track in tracks:
        track_bindings = [
            binding for binding in bindings if binding[1].track == track
        ]
        track_results = {
            episode.episode_id: results[episode.episode_id]
            for _, episode in track_bindings
        }
        metrics_by_track[track] = compute_metrics(
            track_bindings,
            track_results,
        )
    return metrics_by_track


def benchmark_fingerprint(
    root: Path,
    manifest: BenchmarkManifest,
    *,
    include_benchmark_version: bool = False,
) -> str:
    """Hash indexed artifacts independently from the declared release version."""

    digest = hashlib.sha256()
    relative_paths = {Path("benchmark.json")}
    relative_paths.update(
        Path(scenario_id) / "scenario.json"
        for scenario_id in manifest.scenarios
    )
    for entry in manifest.episodes:
        episode_path = Path(entry.path)
        relative_paths.add(episode_path)
        relative_paths.add(episode_path.parents[1] / "semantic.json")
        relative_paths.add(episode_path.parents[2] / "skeleton.json")

    for relative_path in sorted(relative_paths, key=str):
        path = root / relative_path
        if not path.is_file():
            raise ValueError(f"Missing indexed benchmark artifact: {path}")
        relative = str(path.relative_to(root)).replace(os.sep, "/")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        payload = path.read_bytes()
        if relative_path == Path("benchmark.json") and not include_benchmark_version:
            benchmark_payload = json.loads(payload)
            benchmark_payload.pop("benchmark_version", None)
            payload = json.dumps(
                benchmark_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _resume_version(version: str) -> str:
    """Ignore only the serial number that distinguishes beta releases."""

    try:
        parsed = Version(version)
    except InvalidVersion:
        return version
    if parsed.pre is None or parsed.pre[0] != "b":
        return str(parsed)
    release = ".".join(str(component) for component in parsed.release)
    epoch = f"{parsed.epoch}!" if parsed.epoch else ""
    post = f".post{parsed.post}" if parsed.post is not None else ""
    dev = f".dev{parsed.dev}" if parsed.dev is not None else ""
    local = f"+{parsed.local}" if parsed.local is not None else ""
    return f"{epoch}{release}b{post}{dev}{local}"


def _resume_identity_differences(
    stored: RunResumeIdentity,
    current: RunResumeIdentity,
) -> List[str]:
    stored_values = stored.model_dump(mode="json")
    current_values = current.model_dump(mode="json")
    differences = []
    missing = object()

    def compare(stored_value: Any, current_value: Any, path: str) -> None:
        if isinstance(stored_value, dict) and isinstance(current_value, dict):
            for key in sorted(stored_value.keys() | current_value.keys()):
                child_path = f"{path}.{key}" if path else key
                compare(
                    stored_value.get(key, missing),
                    current_value.get(key, missing),
                    child_path,
                )
            return

        values_match = stored_value == current_value
        if path in {"benchmark_version", "agent.agent_version"}:
            values_match = _resume_version(stored_value) == _resume_version(
                current_value
            )
        if not values_match:
            stored_display = (
                "<missing>" if stored_value is missing else repr(stored_value)
            )
            current_display = (
                "<missing>" if current_value is missing else repr(current_value)
            )
            differences.append(
                f"{path}: stored={stored_display}, current={current_display}"
            )

    compare(stored_values, current_values, "")
    return differences


def _write_model_atomic(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = model.model_dump_json(indent=2)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


class ResultManager:
    """Persist episode traces and derive resumable scenario/global metrics."""

    def __init__(
        self,
        results_path: Path,
        benchmark_root: Path,
        agent_card: Dict,
        scenarios: Sequence[Scenario],
        judge_mode: Literal["backend", "skipped"] = "backend",
        sim_backend: str = "auto",
        seed: int = 0,
        model_logs: bool = False,
    ) -> None:
        self.results_path = results_path.resolve()
        self.benchmark_root = benchmark_root.resolve()
        self.scenarios = tuple(scenarios)
        self.model_logs = model_logs
        self._model_log_directories: Dict[str, Path] = {}
        self._model_log_counters: Dict[str, int] = {}
        self.scenario_by_id = {
            scenario.scenario_id: scenario for scenario in self.scenarios
        }
        if len(self.scenario_by_id) != len(self.scenarios):
            raise ValueError("Scenario IDs must be unique")

        self.episode_bindings: Dict[str, EpisodeBinding] = {}
        for scenario in self.scenarios:
            for episode in scenario.episodes:
                if episode.episode_id in self.episode_bindings:
                    raise ValueError(
                        f"Duplicated episode ID {episode.episode_id!r}"
                    )
                if Path(episode.episode_id).name != episode.episode_id:
                    raise ValueError(
                        f"Episode ID cannot be used as a filename: {episode.episode_id!r}"
                    )
                self.episode_bindings[episode.episode_id] = (
                    scenario.scenario_id,
                    episode,
                )

        manifest_model = load_json_model(
            self.benchmark_root / "benchmark.json",
            BenchmarkManifest,
        )
        if not isinstance(manifest_model, BenchmarkManifest):
            raise TypeError("Unexpected benchmark manifest model")
        identity = AgentIdentity.model_validate(agent_card)
        self.run_manifest = RunManifest(
            benchmark_version=manifest_model.benchmark_version,
            benchmark_fingerprint=benchmark_fingerprint(
                self.benchmark_root,
                manifest_model,
            ),
            agent=identity,
            created_at=datetime.now(timezone.utc).isoformat(),
            scenario_ids=[scenario.scenario_id for scenario in self.scenarios],
            execution=ExecutionIdentity(
                judge_mode=judge_mode,
                sim_backend=sim_backend,
                seed=seed,
            ),
        )
        self._legacy_benchmark_fingerprint = benchmark_fingerprint(
            self.benchmark_root,
            manifest_model,
            include_benchmark_version=True,
        )
        self.active_scenario_id: Optional[str] = None
        self._pending_episode_ids: Set[str] = set()
        self._initialize_run()

    @property
    def scenarios_path(self) -> Path:
        return self.results_path / "scenarios"

    def _initialize_run(self) -> None:
        run_path = self.results_path / "run.json"
        if run_path.exists():
            existing = RunManifest.model_validate_json(
                run_path.read_text(encoding="utf-8")
            )
            current_manifest = self.run_manifest
            if existing.benchmark_fingerprint == self._legacy_benchmark_fingerprint:
                current_manifest = current_manifest.model_copy(
                    update={
                        "benchmark_fingerprint": existing.benchmark_fingerprint,
                    }
                )
            differences = _resume_identity_differences(
                RunResumeIdentity.from_manifest(existing),
                RunResumeIdentity.from_manifest(current_manifest),
            )
            if differences:
                details = "\n- ".join(differences)
                raise ValueError(
                    "Existing results cannot be resumed because these keys differ:\n"
                    f"- {details}"
                )
        else:
            if self.results_path.exists() and any(self.results_path.iterdir()):
                raise ValueError(
                    f"Results directory {self.results_path} is non-empty but has no run.json"
                )
            self.results_path.mkdir(parents=True, exist_ok=True)
            _write_model_atomic(run_path, self.run_manifest)
        self.scenarios_path.mkdir(parents=True, exist_ok=True)
        (self.scenarios_path / ".partial").mkdir(parents=True, exist_ok=True)

    def _completed_scenario_path(self, scenario_id: str) -> Path:
        return self.scenarios_path / scenario_id

    def _partial_scenario_path(self, scenario_id: str) -> Path:
        return self.scenarios_path / ".partial" / scenario_id

    def start_scenario(self, scenario: Scenario) -> bool:
        if self.active_scenario_id is not None:
            raise RuntimeError(
                f"Scenario {self.active_scenario_id!r} is already active"
            )
        if scenario.scenario_id not in self.scenario_by_id:
            raise ValueError(
                f"Scenario {scenario.scenario_id!r} is not registered in this run"
            )

        completed_path = self._completed_scenario_path(scenario.scenario_id)
        if completed_path.exists():
            self._load_completed_scenario(scenario)
            return False

        partial_path = self._partial_scenario_path(scenario.scenario_id)
        episodes_path = partial_path / "episodes"
        episodes_path.mkdir(parents=True, exist_ok=True)
        expected_ids = {episode.episode_id for episode in scenario.episodes}
        actual_files = set(episodes_path.iterdir())
        expected_files = {
            episodes_path / f"{episode_id}.json" for episode_id in expected_ids
        }
        unexpected = actual_files - expected_files
        if unexpected:
            raise ValueError(
                f"Unknown episode result files for {scenario.scenario_id!r}: "
                f"{sorted(path.name for path in unexpected)}"
            )

        pending_ids = expected_ids.copy()
        for path in sorted(actual_files, key=lambda item: item.name):
            result = EpisodeResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            expected_episode_id = path.stem
            if result.episode_id != expected_episode_id:
                raise ValueError(
                    f"Episode result identity mismatch in {path.name!r}"
                )
            if result.terminal.status != "infrastructure_failure":
                pending_ids.remove(expected_episode_id)

        self.active_scenario_id = scenario.scenario_id
        self._pending_episode_ids = pending_ids
        return True

    def pending_episode_ids(self, scenario_id: str) -> Set[str]:
        """Return the episodes that still need execution for the active scenario."""

        if self.active_scenario_id != scenario_id:
            raise RuntimeError(f"Scenario {scenario_id!r} is not active")
        return set(self._pending_episode_ids)

    def video_directory(self, scenario_id: str) -> Path:
        """Return the video directory that follows one partial scenario."""

        if self.active_scenario_id != scenario_id:
            raise RuntimeError(f"Scenario {scenario_id!r} is not active")
        path = self._partial_scenario_path(scenario_id) / "videos"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def start_episode_model_logs(self, episode_id: str) -> None:
        """Prepare an isolated model-log directory for one episode attempt."""

        if not self.model_logs:
            return
        if self.active_scenario_id is None:
            raise RuntimeError("No scenario is active")
        binding = self.episode_bindings.get(episode_id)
        if binding is None:
            raise KeyError(f"Unknown episode {episode_id!r}")
        scenario_id, episode = binding
        if scenario_id != self.active_scenario_id:
            raise ValueError(
                f"Episode {episode_id!r} belongs to {scenario_id!r}, "
                f"not active scenario {self.active_scenario_id!r}"
            )
        path = (
            self._partial_scenario_path(scenario_id)
            / "model_logs"
            / episode.skeleton_id
            / episode.semantic.semantic_id
            / episode.condition
        )
        path.mkdir(parents=True, exist_ok=True)
        for existing in path.iterdir():
            if (
                existing.is_file()
                and (
                    existing.name == "final.md"
                    or (
                        existing.suffix in {".json", ".md"}
                        and existing.name[:4].isdigit()
                        and existing.name[4:5] == "_"
                    )
                )
            ):
                existing.unlink()
        self._model_log_directories[episode_id] = path
        self._model_log_counters[episode_id] = 0

    def record_model_diagnostics(
        self,
        episode_id: str,
        stage_index: int,
        diagnostics: List[Dict[str, Any]],
    ) -> None:
        """Persist ordered model inputs and outputs for one benchmark turn."""

        if not self.model_logs:
            return
        directory = self._model_log_directories.get(episode_id)
        if directory is None:
            raise RuntimeError(
                f"Model logs were not initialized for episode {episode_id!r}"
            )
        counter = self._model_log_counters[episode_id]
        for diagnostic in diagnostics:
            body = [f"# Stage {stage_index}"]
            for index, step in enumerate(diagnostic["internal_steps"]):
                component = step.get("component", "unknown")
                body.extend(("", f"## Step {index}", "", f"Component: `{component}`"))
                if "full_prompt" in step:
                    body.extend(("", "### Prompt", "", str(step["full_prompt"])))
                if "output_raw" in step:
                    body.extend(("", "### Response", "", str(step["output_raw"])))
            if diagnostic["error"] is not None:
                body.extend(("", "## Error", "", json.dumps(
                    diagnostic["error"], ensure_ascii=False, indent=2,
                )))
            _write_text_atomic(
                directory / f"{counter:04d}_exchange.md", "\n".join(body) + "\n",
            )
            counter += 1
        self._model_log_counters[episode_id] = counter

    def record_episode(self, outcome: EpisodeOutcome) -> None:
        if self.active_scenario_id is None:
            raise RuntimeError("No scenario is active")
        binding = self.episode_bindings.get(outcome.episode_id)
        if binding is None:
            raise KeyError(f"Unknown episode {outcome.episode_id!r}")
        scenario_id, _ = binding
        if scenario_id != self.active_scenario_id:
            raise ValueError(
                f"Episode {outcome.episode_id!r} belongs to {scenario_id!r}, "
                f"not active scenario {self.active_scenario_id!r}"
            )
        if outcome.episode_id not in self._pending_episode_ids:
            raise ValueError(
                f"Episode {outcome.episode_id!r} is already complete"
            )

        result = EpisodeResult.model_validate(outcome.model_dump(mode="python"))
        path = (
            self._partial_scenario_path(scenario_id)
            / "episodes"
            / f"{outcome.episode_id}.json"
        )
        if path.exists():
            previous = EpisodeResult.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            if previous.episode_id != outcome.episode_id:
                raise ValueError(
                    f"Episode result identity mismatch in {path.name!r}"
                )
            if previous.terminal.status != "infrastructure_failure":
                raise ValueError(
                    f"Episode {outcome.episode_id!r} was completed more than once"
                )

        if self.model_logs:
            directory = self._model_log_directories.get(outcome.episode_id)
            if directory is None:
                raise RuntimeError(
                    f"Model logs were not initialized for episode {outcome.episode_id!r}"
                )
            diagnostic_events = [
                event for event in outcome.trace
                if event.kind == "failure_diagnostics"
            ]
            stage_index = (
                diagnostic_events[-1].stage_index
                if diagnostic_events
                else (outcome.trace[-1].stage_index if outcome.trace else None)
            )
            body = ["# Result", "", f"- Status: `{outcome.terminal.status}`"]
            if stage_index is not None:
                body.append(f"- Stage: `{stage_index}`")
            if outcome.terminal.reason:
                body.append(f"- Reason: {outcome.terminal.reason}")

            for event in diagnostic_events:
                diagnostic = event.payload
                source = diagnostic.get("source", "unknown")
                if source == "stage_verification":
                    env_score = diagnostic.get("env_score")
                    log_score = diagnostic.get("log_score")
                    failed_sources = []
                    if isinstance(env_score, (int, float)) and env_score < 0:
                        failed_sources.append("environment")
                    if isinstance(log_score, (int, float)) and log_score < 0:
                        failed_sources.append("logs")
                    failure_source = " and ".join(failed_sources) or "combined verification"
                    body.extend((
                        "", "## Stage verification", "",
                        f"- Failure source: `{failure_source}`",
                        f"- Environment score: `{env_score}`",
                        f"- Log score: `{log_score}`",
                        f"- Combined score: `{diagnostic.get('combined_score')}`",
                    ))
                    if diagnostic.get("stage_goal_description"):
                        body.extend((
                            "", "### Stage goal", "",
                            str(diagnostic["stage_goal_description"]),
                        ))
                    for title, key in (
                        ("Tool results", "tool_results"),
                        ("Goal scores", "goal_scores"),
                        ("Stage logs", "stage_logs"),
                        ("Full logs", "full_logs"),
                    ):
                        if diagnostic.get(key):
                            body.extend((
                                "", f"### {title}", "", "```json",
                                json.dumps(diagnostic[key], ensure_ascii=False, indent=2),
                                "```",
                            ))
                elif source == "judge":
                    body.extend(("", "## Judge"))
                    for title, key in (
                        ("Question", "question"),
                        ("Verification prompt", "verification_prompt"),
                        ("Model answer", "model_answer"),
                        ("Verdict", "judge_verdict"),
                        ("Reason", "judge_reason"),
                        ("Raw response", "judge_response"),
                    ):
                        if key in diagnostic and diagnostic[key] is not None:
                            body.extend(("", f"### {title}", "", str(diagnostic[key])))
                else:
                    body.extend((
                        "", f"## Diagnostic: {source}", "", "```json",
                        json.dumps(diagnostic, ensure_ascii=False, indent=2), "```",
                    ))
            _write_text_atomic(directory / "final.md", "\n".join(body) + "\n")

        _write_model_atomic(path, result)
        self._pending_episode_ids.remove(outcome.episode_id)
        self._model_log_directories.pop(outcome.episode_id, None)
        self._model_log_counters.pop(outcome.episode_id, None)

    def finish_scenario(self, scenario: Scenario) -> Optional[ScenarioResult]:
        if self.active_scenario_id != scenario.scenario_id:
            raise RuntimeError(
                f"Scenario {scenario.scenario_id!r} is not the active scenario"
            )
        if self._pending_episode_ids:
            raise RuntimeError(
                "Scenario still has pending episodes: "
                f"{sorted(self._pending_episode_ids)}"
            )
        partial_path = self._partial_scenario_path(scenario.scenario_id)
        results = self._load_episode_results(partial_path, scenario)
        infrastructure_failures = [
            result.episode_id
            for result in results.values()
            if result.terminal.status == "infrastructure_failure"
        ]
        if infrastructure_failures:
            self.active_scenario_id = None
            self._pending_episode_ids.clear()
            return None

        bindings = [
            (scenario.scenario_id, episode) for episode in scenario.episodes
        ]
        scenario_result = ScenarioResult(
            scenario_id=scenario.scenario_id,
            metrics=compute_metrics(bindings, results),
            metrics_by_track=_compute_metrics_by_track(bindings, results),
        )
        _write_model_atomic(partial_path / "result.json", scenario_result)
        completed_path = self._completed_scenario_path(scenario.scenario_id)
        if completed_path.exists():
            raise FileExistsError(
                f"Completed scenario path already exists: {completed_path}"
            )
        os.replace(partial_path, completed_path)
        self.active_scenario_id = None
        self._pending_episode_ids.clear()
        return scenario_result

    def finish_benchmark(self) -> BenchmarkResult:
        if self.active_scenario_id is not None:
            raise RuntimeError(
                f"Scenario {self.active_scenario_id!r} is still active"
            )
        all_bindings: List[EpisodeBinding] = []
        all_results: Dict[str, EpisodeResult] = {}
        partial_scenario_ids: List[str] = []
        infrastructure_failures: List[InfrastructureFailure] = []
        for scenario in self.scenarios:
            completed_path = self._completed_scenario_path(scenario.scenario_id)
            if not completed_path.exists():
                partial_scenario_ids.append(scenario.scenario_id)
                episodes_path = (
                    self._partial_scenario_path(scenario.scenario_id) / "episodes"
                )
                for episode in scenario.episodes:
                    episode_path = episodes_path / f"{episode.episode_id}.json"
                    if not episode_path.is_file():
                        continue
                    result = EpisodeResult.model_validate_json(
                        episode_path.read_text(encoding="utf-8")
                    )
                    if result.episode_id != episode.episode_id:
                        raise ValueError(
                            "Episode result identity mismatch in "
                            f"{episode.episode_id!r}"
                        )
                    if result.terminal.status == "infrastructure_failure":
                        infrastructure_failures.append(
                            InfrastructureFailure(
                                scenario_id=scenario.scenario_id,
                                episode_id=episode.episode_id,
                                reason=result.terminal.reason,
                            )
                        )
                continue
            results = self._load_completed_scenario(scenario)
            all_bindings.extend(
                (scenario.scenario_id, episode) for episode in scenario.episodes
            )
            all_results.update(results)

        benchmark_result = BenchmarkResult(
            status="partial" if partial_scenario_ids else "complete",
            expected_scenario_count=len(self.scenarios),
            completed_scenario_count=(
                len(self.scenarios) - len(partial_scenario_ids)
            ),
            partial_scenario_ids=partial_scenario_ids,
            infrastructure_failures=infrastructure_failures,
            metrics=compute_metrics(all_bindings, all_results),
            metrics_by_track=_compute_metrics_by_track(
                all_bindings,
                all_results,
            ),
        )
        _write_model_atomic(self.results_path / "result.json", benchmark_result)
        return benchmark_result

    def _load_completed_scenario(
        self,
        scenario: Scenario,
    ) -> Dict[str, EpisodeResult]:
        completed_path = self._completed_scenario_path(scenario.scenario_id)
        result_path = completed_path / "result.json"
        if not result_path.is_file():
            raise ValueError(
                f"Completed scenario {scenario.scenario_id!r} has no valid result.json"
            )
        stored = ScenarioResult.model_validate_json(
            result_path.read_text(encoding="utf-8")
        )
        if stored.scenario_id != scenario.scenario_id:
            raise ValueError(
                f"Stored scenario result does not match {scenario.scenario_id!r}"
            )
        results = self._load_episode_results(completed_path, scenario)
        expected_metrics = compute_metrics(
            [(scenario.scenario_id, episode) for episode in scenario.episodes],
            results,
        )
        if stored.metrics != expected_metrics:
            raise ValueError(
                f"Stored metrics are stale or corrupted for {scenario.scenario_id!r}"
            )
        expected_metrics_by_track = _compute_metrics_by_track(
            [(scenario.scenario_id, episode) for episode in scenario.episodes],
            results,
        )
        if stored.metrics_by_track != expected_metrics_by_track:
            raise ValueError(
                "Stored track metrics are stale or corrupted for "
                f"{scenario.scenario_id!r}"
            )
        return results

    def _load_episode_results(
        self,
        scenario_path: Path,
        scenario: Scenario,
    ) -> Dict[str, EpisodeResult]:
        episodes_path = scenario_path / "episodes"
        actual_files = set(episodes_path.glob("*.json")) if episodes_path.is_dir() else set()
        expected_files = {
            episodes_path / f"{episode.episode_id}.json"
            for episode in scenario.episodes
        }
        if actual_files != expected_files:
            raise ValueError(
                f"Episode results mismatch for {scenario.scenario_id!r}: "
                f"missing={sorted(path.name for path in expected_files - actual_files)}, "
                f"unexpected={sorted(path.name for path in actual_files - expected_files)}"
            )
        results = {
            episode.episode_id: EpisodeResult.model_validate_json(
                (episodes_path / f"{episode.episode_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            for episode in scenario.episodes
        }
        for episode_id, result in results.items():
            if result.episode_id != episode_id:
                raise ValueError(
                    f"Episode result identity mismatch in {episode_id!r}"
                )
        return results
