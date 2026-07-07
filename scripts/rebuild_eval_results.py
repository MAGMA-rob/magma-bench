#!/usr/bin/env python3
"""Rebuild MAGMA-BENCH aggregate result files from saved eval artifacts.

The normal runner computes scenario/benchmark summaries from in-memory
ScenarioResult objects.  This script rebuilds the same JSON schema after a
crash by aggregating saved try_score.json files, with a fallback that rebuilds
a compatible try score from compact per-task logs plus benchmark task configs.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


KNOWN_CRITERIA = ["multi-steps", "c-reasoning", "lg-memorization", "recovery"]
LENGTH_GROUPS = {
    "2-3": (2, 3),
    "4-5": (4, 5),
    "6-9": (6, 9),
    "10-16": (10, 16),
}

# Edit these lists to define the aggregate subsets generated automatically.
SCENARIO_GROUPS = {
    "recovery": ["R_C", "R_CS"],
    "ood": ["Packaging_v1", "US_C", "COMP_Laundry"],
    "id": ["ID_CS", "ID_MC", "ID_WS","R_C","R_CS"],
}

TRY_DIR_RE = re.compile(r"^try-(\d+)$")
SECONDARY_RE = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")


def safe_percentage(num: float, den: float) -> object:
    if den == 0:
        return "non-measured"
    return num * 100.0 / den


def percent_to_sum(percent: object, count: int) -> float:
    if count == 0 or not isinstance(percent, (int, float)):
        return 0.0
    return float(percent) * count / 100.0


def parse_percent(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return default


def ordered_metric_items(items: Mapping[str, Any]) -> Iterable[Tuple[str, Any]]:
    yielded = set()
    for key in LENGTH_GROUPS:
        if key in items:
            yielded.add(key)
            yield key, items[key]
    for key in sorted(items):
        if key not in yielded:
            yield key, items[key]


@dataclass
class SecondaryCounter:
    eligible: int = 0
    reached: int = 0
    success: int = 0

    def merge_counts(self, eligible: int, reached: int, success: int) -> None:
        self.eligible += eligible
        self.reached += reached
        self.success += success

    def merge_payload(self, payload: Mapping[str, Any]) -> None:
        self.merge_counts(
            int(payload.get("eligible", 0)),
            int(payload.get("reached", 0)),
            int(payload.get("success", 0)),
        )

    def to_payload(self) -> Dict[str, object]:
        return {
            "eligible": self.eligible,
            "reached": self.reached,
            "success": self.success,
            "reach_rate": safe_percentage(self.reached, self.eligible),
            "success_when_reached": safe_percentage(self.success, self.reached),
        }


@dataclass
class LengthAccumulator:
    task_count: int = 0
    success_count: int = 0
    goal_completion_sum: float = 0.0
    completed_prefix_horizon: int = 0
    total_task_horizon: int = 0

    def register_task(
        self,
        task_success: bool,
        goal_completion_ratio: float,
        completed_prefix_horizon: int,
        task_horizon: int,
    ) -> None:
        self.task_count += 1
        self.success_count += int(task_success)
        self.goal_completion_sum += goal_completion_ratio
        self.completed_prefix_horizon += completed_prefix_horizon
        self.total_task_horizon += task_horizon

    def merge_payload(self, payload: Mapping[str, Any]) -> None:
        counts = payload.get("counts", {})
        primary = payload.get("primary_metrics", {})
        task_count = int(counts.get("task_count", 0))
        self.task_count += task_count
        self.success_count += int(counts.get("success_count", 0))
        self.total_task_horizon += int(counts.get("total_task_horizon", 0))
        self.completed_prefix_horizon += int(counts.get("completed_prefix_horizon", 0))
        self.goal_completion_sum += percent_to_sum(
            primary.get("goal_completion_macro"), task_count
        )

    def to_payload(self) -> Dict[str, Any]:
        return {
            "counts": {
                "task_count": self.task_count,
                "success_count": self.success_count,
                "total_task_horizon": self.total_task_horizon,
                "completed_prefix_horizon": self.completed_prefix_horizon,
            },
            "primary_metrics": {
                "success_rate": safe_percentage(self.success_count, self.task_count),
                "goal_completion_macro": safe_percentage(
                    self.goal_completion_sum, self.task_count
                ),
                "goal_completion_micro": safe_percentage(
                    self.completed_prefix_horizon, self.total_task_horizon
                ),
            },
        }


@dataclass
class MetricAccumulator:
    task_count: int = 0
    success_count: int = 0
    goal_completion_sum: float = 0.0
    completed_prefix_horizon: int = 0
    total_task_horizon: int = 0
    execution_score_sum: float = 0.0
    reached_stage_count: int = 0
    skipped_stage_count: int = 0
    secondary: Dict[str, SecondaryCounter] = field(
        default_factory=lambda: {criterion: SecondaryCounter() for criterion in KNOWN_CRITERIA}
    )
    length_breakdown: Dict[str, LengthAccumulator] = field(
        default_factory=lambda: {bucket: LengthAccumulator() for bucket in LENGTH_GROUPS}
    )

    def register_task(
        self,
        task: "TaskMetadata",
        task_success: bool,
        goal_completion_ratio: float,
        completed_prefix_horizon: int,
    ) -> None:
        self.task_count += 1
        self.success_count += int(task_success)
        self.goal_completion_sum += goal_completion_ratio
        self.completed_prefix_horizon += completed_prefix_horizon
        self.total_task_horizon += task.task_horizon

        bucket = self.length_breakdown.setdefault(task.length_bucket, LengthAccumulator())
        bucket.register_task(
            task_success,
            goal_completion_ratio,
            completed_prefix_horizon,
            task.task_horizon,
        )

    def merge_payload(self, payload: Mapping[str, Any]) -> None:
        counts = payload.get("counts", {})
        primary = payload.get("primary_metrics", {})

        task_count = int(counts.get("task_count", 0))
        reached_stage_count = int(counts.get("reached_stage_count", 0))

        self.task_count += task_count
        self.success_count += int(counts.get("success_count", 0))
        self.total_task_horizon += int(counts.get("total_task_horizon", 0))
        self.completed_prefix_horizon += int(counts.get("completed_prefix_horizon", 0))
        self.reached_stage_count += reached_stage_count
        self.skipped_stage_count += int(counts.get("skipped_stage_count", 0))
        self.goal_completion_sum += percent_to_sum(
            primary.get("goal_completion_macro"), task_count
        )

        tool_accuracy = payload.get("auxiliary_metrics", {}).get("tool_accuracy")
        if isinstance(tool_accuracy, (int, float)):
            self.execution_score_sum += float(tool_accuracy) * reached_stage_count / 100.0

        for bucket, bucket_payload in ordered_metric_items(payload.get("length_breakdown", {})):
            self.length_breakdown.setdefault(bucket, LengthAccumulator()).merge_payload(bucket_payload)

        for criterion, counter_payload in payload.get("secondary_metrics", {}).items():
            self.secondary.setdefault(criterion, SecondaryCounter()).merge_payload(counter_payload)

    def to_payload(self, extra_counts: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        counts = {
            "task_count": self.task_count,
            "success_count": self.success_count,
            "total_task_horizon": self.total_task_horizon,
            "completed_prefix_horizon": self.completed_prefix_horizon,
            "reached_stage_count": self.reached_stage_count,
            "skipped_stage_count": self.skipped_stage_count,
        }
        if extra_counts:
            counts.update(dict(extra_counts))

        return {
            "counts": counts,
            "primary_metrics": {
                "success_rate": safe_percentage(self.success_count, self.task_count),
                "goal_completion_macro": safe_percentage(
                    self.goal_completion_sum, self.task_count
                ),
                "goal_completion_micro": safe_percentage(
                    self.completed_prefix_horizon, self.total_task_horizon
                ),
            },
            "length_breakdown": {
                bucket: metrics.to_payload()
                for bucket, metrics in ordered_metric_items(self.length_breakdown)
            },
            "secondary_metrics": {
                criterion: self.secondary[criterion].to_payload()
                for criterion in self._ordered_secondary_keys()
            },
            "auxiliary_metrics": {
                "tool_accuracy": safe_percentage(
                    self.execution_score_sum, self.reached_stage_count
                ),
            },
        }

    def _ordered_secondary_keys(self) -> List[str]:
        keys = []
        for criterion in KNOWN_CRITERIA:
            if criterion in self.secondary:
                keys.append(criterion)
        keys.extend(sorted(set(self.secondary) - set(keys)))
        return keys


@dataclass
class StageMetadata:
    keys_evaluator: List[str]
    is_recovery: bool
    stage_horizon: int


@dataclass
class TaskMetadata:
    task_id: str
    criteria: List[str]
    task_horizon: int
    length_bucket: str
    stages: List[StageMetadata]


@dataclass
class ScenarioMetadata:
    scenario_id: str
    scenario_name: str
    tasks: Dict[str, TaskMetadata]


@dataclass
class RebuildReport:
    run_dir: Path
    scenarios: int = 0
    try_scores: int = 0
    rebuilt_tries: int = 0
    skipped_tries: int = 0
    selected_scenarios: List[str] = field(default_factory=list)
    written_files: List[Path] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    aggregate_payload: Dict[str, Any] = field(default_factory=dict)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


@dataclass
class ScenarioSelection:
    include: List[str] = field(default_factory=list)
    exclude: List[str] = field(default_factory=list)
    exclude_recovery: bool = False

    @property
    def active(self) -> bool:
        return bool(self.include or self.exclude or self.exclude_recovery)

    def allows(self, scenario_id: str, metadata: Optional["ScenarioMetadata"]) -> bool:
        labels = [scenario_id]
        if metadata is not None:
            labels.append(metadata.scenario_name)

        if self.include and not any(matches_any(pattern, labels) for pattern in self.include):
            return False

        if self.exclude and any(matches_any(pattern, labels) for pattern in self.exclude):
            return False

        if self.exclude_recovery and is_recovery_scenario(scenario_id, metadata):
            return False

        return True


class BenchmarkCatalog:
    def __init__(self, benchmark_root: Path) -> None:
        self.benchmark_root = benchmark_root
        self.scenarios = self._load_scenarios()

    def get(self, scenario_id: str) -> Optional[ScenarioMetadata]:
        return self.scenarios.get(scenario_id)

    def _load_scenarios(self) -> Dict[str, ScenarioMetadata]:
        metadata_path = self.benchmark_root / "bench_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Cannot find benchmark metadata at {metadata_path}")

        bench_metadata = read_json(metadata_path)
        scenarios: Dict[str, ScenarioMetadata] = {}
        for scenario_entry in bench_metadata:
            folder_name = scenario_entry["folder_name"]
            scenario_root = self.benchmark_root / folder_name
            meta_info = read_json(scenario_root / "meta_info.json")
            scenario_id = meta_info["scenario_id"]
            scenarios[scenario_id] = ScenarioMetadata(
                scenario_id=scenario_id,
                scenario_name=meta_info["scenario_name"],
                tasks=self._load_tasks(scenario_root / "tasks"),
            )
        return scenarios

    def _load_tasks(self, tasks_root: Path) -> Dict[str, TaskMetadata]:
        tasks: Dict[str, TaskMetadata] = {}
        for task_path in sorted(tasks_root.glob("task_*.json")):
            task_id = task_path.stem
            task_payload = read_json(task_path)
            tasks[task_id] = build_task_metadata(task_id, task_payload)
        return tasks


def build_task_metadata(task_id: str, task_payload: Mapping[str, Any]) -> TaskMetadata:
    raw_stages = list(task_payload.get("stages", []))
    task_criteria = list(dict.fromkeys(task_payload.get("criteria", [])))
    if task_has_recovery_criterion(raw_stages) and "recovery" not in task_criteria:
        task_criteria.append("recovery")

    stage_metadatas: List[StageMetadata] = []
    propagate_multi_steps = False
    for raw_stage in raw_stages:
        keys_evaluator = list(dict.fromkeys(raw_stage.get("keys_evaluator", [])))
        if raw_stage.get("instruction", None) is not None:
            propagate_multi_steps = "multi-steps" in keys_evaluator
        elif propagate_multi_steps and "multi-steps" not in keys_evaluator:
            keys_evaluator.append("multi-steps")

        stage_metadatas.append(
            StageMetadata(
                keys_evaluator=keys_evaluator,
                is_recovery=stage_has_recovery(raw_stage),
                stage_horizon=count_stage_horizon(raw_stage),
            )
        )

    task_horizon = sum(stage.stage_horizon for stage in stage_metadatas)
    return TaskMetadata(
        task_id=task_id,
        criteria=task_criteria,
        task_horizon=task_horizon,
        length_bucket=get_length_bucket(task_horizon),
        stages=stage_metadatas,
    )


def stage_has_recovery(stage: Mapping[str, Any]) -> bool:
    return any(
        injection.get("mode") in {"force_recovery", "force_failure"}
        for injection in iter_injections(stage)
    )


def task_has_recovery_criterion(stages: Iterable[Mapping[str, Any]]) -> bool:
    return any(stage_has_recovery(stage) for stage in stages)


def iter_injections(stage: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    raw_injections = stage.get("injection", stage.get("injections", None))
    if raw_injections is None:
        return []
    if isinstance(raw_injections, Mapping):
        return [raw_injections]
    if isinstance(raw_injections, list):
        return [injection for injection in raw_injections if isinstance(injection, Mapping)]
    return []


def count_stage_horizon(stage: Mapping[str, Any]) -> int:
    expected_behavior = stage.get("expected_behavior", "act")
    default_max_step = 1 if expected_behavior in {"acknowledge", "answer"} else None
    max_step = stage.get("max_step", default_max_step)
    if not isinstance(max_step, int) or max_step <= 0:
        raise ValueError(f"Invalid max_step={max_step!r} in benchmark task stage")

    horizon = max_step
    if stage.get("answer_to_user", None) or stage.get("flag_answer_to_user", None):
        horizon += 1
    if stage_has_recovery(stage):
        horizon += 1
    return horizon


def get_length_bucket(task_horizon: int) -> str:
    for bucket_name, (lower, upper) in LENGTH_GROUPS.items():
        if lower <= task_horizon <= upper:
            return bucket_name
    return "other"


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def rebuild_run(
    run_dir: Path,
    catalog: BenchmarkCatalog,
    *,
    dry_run: bool,
    write_rebuilt_try_scores: bool,
    overwrite_try_scores: bool,
    selection: ScenarioSelection,
    output_name: str,
    write_scenario_results: bool,
    subset_metadata: Optional[Mapping[str, Any]] = None,
) -> RebuildReport:
    report = RebuildReport(run_dir=run_dir)
    if not run_dir.exists():
        report.warn(f"{run_dir} does not exist")
        return report

    benchmark_acc = MetricAccumulator()
    scenario_payloads: Dict[str, Dict[str, Any]] = {}

    for scenario_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        scenario_id = scenario_dir.name
        scenario_meta = catalog.get(scenario_id)
        if not selection.allows(scenario_id, scenario_meta):
            continue

        try_dirs = sorted(
            (path for path in scenario_dir.iterdir() if path.is_dir() and TRY_DIR_RE.match(path.name)),
            key=lambda path: int(TRY_DIR_RE.match(path.name).group(1)),
        )
        if not try_dirs:
            continue

        scenario_acc = MetricAccumulator()
        included_tries = 0

        for try_dir in try_dirs:
            try_score_path = try_dir / "try_score.json"
            if try_score_path.exists():
                payload = read_json(try_score_path)
                report.try_scores += 1
            else:
                payload = rebuild_try_from_logs(scenario_id, try_dir, catalog, report)
                if payload is None:
                    report.skipped_tries += 1
                    continue
                report.rebuilt_tries += 1
                if write_rebuilt_try_scores and (overwrite_try_scores or not try_score_path.exists()):
                    if not dry_run:
                        write_json(try_score_path, payload)
                    report.written_files.append(try_score_path)

            scenario_acc.merge_payload(payload)
            benchmark_acc.merge_payload(payload)
            included_tries += 1

        if included_tries == 0:
            continue

        scenario_payload = scenario_acc.to_payload(extra_counts={"num_try": included_tries})
        scenario_payloads[scenario_id] = scenario_acc.to_payload()
        report.selected_scenarios.append(scenario_id)

        if write_scenario_results:
            scenario_result_path = scenario_dir / "result.json"
            if not dry_run:
                write_json(scenario_result_path, scenario_payload)
            report.written_files.append(scenario_result_path)
        report.scenarios += 1

    benchmark_payload = benchmark_acc.to_payload(
        extra_counts={"scenario_count": len(scenario_payloads)}
    )
    benchmark_payload["scenarios"] = scenario_payloads

    if subset_metadata is not None:
        benchmark_payload = {
            "subset": {
                **dict(subset_metadata),
                "selected_scenarios": list(report.selected_scenarios),
            },
            **benchmark_payload,
        }

    report.aggregate_payload = benchmark_payload

    result_path = run_dir / output_name
    if not dry_run:
        write_json(result_path, benchmark_payload)
    report.written_files.append(result_path)
    return report


def matches_any(pattern: str, labels: Iterable[str]) -> bool:
    normalized_pattern = pattern.strip().lower()
    for label in labels:
        normalized_label = label.lower()
        if fnmatch.fnmatchcase(normalized_label, normalized_pattern):
            return True
        if normalized_label == normalized_pattern:
            return True
    return False


def is_recovery_scenario(
    scenario_id: str,
    metadata: Optional[ScenarioMetadata],
) -> bool:
    scenario_name = metadata.scenario_name if metadata is not None else ""
    normalized_id = scenario_id.lower()
    normalized_name = scenario_name.lower()
    return (
        normalized_id == "rl"
        or normalized_id.startswith("r_")
        or "recovery" in normalized_id
        or normalized_name.startswith("recovery")
        or normalized_name.startswith("recovery_")
        or "recovery_benchmark" in normalized_name
    )


def rebuild_try_from_logs(
    scenario_id: str,
    try_dir: Path,
    catalog: BenchmarkCatalog,
    report: RebuildReport,
) -> Optional[Dict[str, Any]]:
    scenario_meta = catalog.get(scenario_id)
    if scenario_meta is None:
        report.warn(f"{try_dir}: no benchmark config found for scenario {scenario_id!r}")
        return None

    log_paths = sorted((try_dir / "logs").glob("task_*.json"), key=task_log_sort_key)
    if not log_paths:
        report.warn(f"{try_dir}: no try_score.json and no task logs")
        return None

    acc = MetricAccumulator()
    for log_path in log_paths:
        task_id = log_path.stem
        task = scenario_meta.tasks.get(task_id)
        if task is None:
            report.warn(f"{log_path}: task not found in benchmark config, skipped")
            continue
        register_task_log(acc, task, read_json(log_path), log_path, report)

    if acc.task_count == 0:
        report.warn(f"{try_dir}: no usable task logs")
        return None
    return acc.to_payload()


def task_log_sort_key(path: Path) -> Tuple[int, str]:
    suffix = path.stem.rsplit("_", 1)[-1]
    return (int(suffix), path.name) if suffix.isdigit() else (sys.maxsize, path.name)


def register_task_log(
    acc: MetricAccumulator,
    task: TaskMetadata,
    task_log: Mapping[str, Any],
    log_path: Path,
    report: RebuildReport,
) -> None:
    header = task_log.get("header", {})
    stage_payloads = list(task_log.get("stage", []))
    task_success = bool(header.get("task_success", False))
    goal_completion_percent = parse_percent(header.get("goal_completion"))
    goal_completion_ratio = goal_completion_percent / 100.0
    completed_prefix_horizon = int(round(goal_completion_ratio * task.task_horizon))

    acc.register_task(
        task=task,
        task_success=task_success,
        goal_completion_ratio=goal_completion_ratio,
        completed_prefix_horizon=completed_prefix_horizon,
    )

    eligible_counts = {criterion: 0 for criterion in KNOWN_CRITERIA}
    for index, stage_payload in enumerate(stage_payloads):
        stage_meta = task.stages[index] if index < len(task.stages) else None
        if stage_meta is None:
            report.warn(f"{log_path}: extra logged stage at index {index} ignored for eligibility")
        else:
            for criterion in KNOWN_CRITERIA:
                if criterion in stage_meta.keys_evaluator or (
                    criterion == "recovery" and stage_meta.is_recovery
                ):
                    eligible_counts[criterion] += 1

        if stage_was_reached(stage_payload):
            acc.reached_stage_count += 1
            acc.execution_score_sum += estimate_execution_score(stage_payload)
        else:
            acc.skipped_stage_count += 1

    compact_secondary = header.get("secondary_metrics", {})
    for criterion in KNOWN_CRITERIA:
        success, reached = parse_compact_secondary(compact_secondary.get(criterion))
        if success is None or reached is None:
            success, reached = infer_secondary_from_stages(task, stage_payloads, criterion)
        acc.secondary[criterion].merge_counts(
            eligible=eligible_counts[criterion],
            reached=reached,
            success=success,
        )


def parse_compact_secondary(value: object) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(value, str):
        match = SECONDARY_RE.match(value)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None, None


def infer_secondary_from_stages(
    task: TaskMetadata,
    stage_payloads: List[Mapping[str, Any]],
    criterion: str,
) -> Tuple[int, int]:
    success = 0
    reached = 0
    for index, stage_payload in enumerate(stage_payloads):
        if index >= len(task.stages):
            continue
        stage_meta = task.stages[index]
        eligible = criterion in stage_meta.keys_evaluator or (
            criterion == "recovery" and stage_meta.is_recovery
        )
        if not eligible or not stage_was_reached(stage_payload):
            continue
        reached += 1
        success += int(bool(stage_payload.get("success", False)))
    return success, reached


def stage_was_reached(stage_payload: Mapping[str, Any]) -> bool:
    return bool(stage_payload.get("conversation"))


def estimate_execution_score(stage_payload: Mapping[str, Any]) -> float:
    conversation = stage_payload.get("conversation", [])
    outcomes: List[bool] = []
    for message in conversation:
        author = str(message.get("author", "")).upper()
        if author != "SYSTEM":
            continue
        for text in iter_text_values(message.get("content")):
            outcomes.extend([True] * len(re.findall(r"\bsucceed\b", text)))
            outcomes.extend([False] * len(re.findall(r"\bfails\b", text)))

    if outcomes:
        return sum(1 for outcome in outcomes if outcome) / len(outcomes)

    model_actions = []
    for message in conversation:
        author = str(message.get("author", "")).upper()
        if author != "MODEL":
            continue
        content = message.get("content")
        if isinstance(content, Mapping) and "action" in content:
            model_actions.append(content.get("action"))

    if not model_actions:
        return 0.0
    if any(bool(action) for action in model_actions):
        return 0.0
    return 1.0


def iter_text_values(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from iter_text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_text_values(child)


def discover_benchmark_root(eval_paths: List[Path], explicit_root: Optional[str]) -> Path:
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    candidates: List[Path] = []
    script_root = Path(__file__).resolve().parents[2]
    cwd = Path.cwd().resolve()
    for base in [cwd, script_root, *[path.resolve() for path in eval_paths]]:
        candidates.extend(
            [
                base / "magma-scenarios-dev" / "src" / "magma_scenarios" / "benchmark",
                base.parent / "magma-scenarios-dev" / "src" / "magma_scenarios" / "benchmark",
                base.parents[1] / "magma-scenarios-dev" / "src" / "magma_scenarios" / "benchmark"
                if len(base.parents) > 1
                else base,
            ]
        )

    for candidate in candidates:
        if (candidate / "bench_metadata.json").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate magma_scenarios benchmark metadata. "
        "Pass --benchmark-root /path/to/magma_scenarios/benchmark."
    )


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild MAGMA-BENCH result.json files from eval outputs."
    )
    parser.add_argument(
        "eval_run",
        nargs="+",
        help="Evaluation run folder(s), e.g. eval/model-name/05-07_09-54",
    )
    parser.add_argument(
        "--benchmark-root",
        help="Path to magma_scenarios/benchmark. Auto-detected in this workspace by default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report what would be written without touching files.",
    )
    parser.add_argument(
        "--no-write-rebuilt-try-scores",
        action="store_true",
        help="Do not write missing try_score.json files rebuilt from task logs.",
    )
    parser.add_argument(
        "--overwrite-try-scores",
        action="store_true",
        help="Allow overwriting try_score.json files when a rebuilt payload is written.",
    )
    parser.add_argument(
        "--include-scenarios",
        action="append",
        default=[],
        help=(
            "Comma-separated scenario ids/names/globs to include, e.g. "
            "ID_CS,Packaging_v1 or 'ID_*'. Can be repeated."
        ),
    )
    parser.add_argument(
        "--exclude-scenarios",
        action="append",
        default=[],
        help=(
            "Comma-separated scenario ids/names/globs to exclude, e.g. "
            "R_*,RL,Packaging_recovery. Can be repeated."
        ),
    )
    parser.add_argument(
        "--exclude-recovery",
        action="store_true",
        help="Exclude recovery scenario families such as R_*, RL and Packaging_recovery.",
    )
    parser.add_argument(
        "--output-name",
        help=(
            "Aggregate result filename to write in each eval run. Defaults to "
            "result.json without filters and result_filtered.json with filters."
        ),
    )
    parser.add_argument(
        "--write-scenario-results",
        action="store_true",
        help=(
            "Also rewrite per-scenario result.json files. By default this only happens "
            "for unfiltered generic rebuilds."
        ),
    )
    parser.add_argument(
        "--no-groups",
        action="store_true",
        help=(
            "Do not write predefined aggregate subsets. By default the script also "
            "writes result_recovery.json, result_ood.json and result_id.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    eval_paths = [Path(path).expanduser().resolve() for path in args.eval_run]
    benchmark_root = discover_benchmark_root(eval_paths, args.benchmark_root)
    catalog = BenchmarkCatalog(benchmark_root)
    selection = ScenarioSelection(
        include=split_patterns(args.include_scenarios),
        exclude=split_patterns(args.exclude_scenarios),
        exclude_recovery=args.exclude_recovery,
    )
    output_name = args.output_name or ("result_filtered.json" if selection.active else "result.json")
    write_scenario_results = args.write_scenario_results or (
        not selection.active and output_name == "result.json"
    )

    exit_code = 0
    for eval_path in eval_paths:
        report = rebuild_run(
            eval_path,
            catalog,
            dry_run=args.dry_run,
            write_rebuilt_try_scores=not args.no_write_rebuilt_try_scores,
            overwrite_try_scores=args.overwrite_try_scores,
            selection=selection,
            output_name=output_name,
            write_scenario_results=write_scenario_results,
        )
        print_report(report, dry_run=args.dry_run)
        if report.skipped_tries:
            exit_code = 1

        if args.no_groups:
            continue

        for group_name, scenario_ids in SCENARIO_GROUPS.items():
            group_report = rebuild_run(
                eval_path,
                catalog,
                dry_run=args.dry_run,
                write_rebuilt_try_scores=not args.no_write_rebuilt_try_scores,
                overwrite_try_scores=args.overwrite_try_scores,
                selection=ScenarioSelection(include=scenario_ids),
                output_name=f"result_{group_name}.json",
                write_scenario_results=False,
                subset_metadata={
                    "name": group_name,
                    "configured_scenarios": scenario_ids,
                },
            )
            print_report(group_report, dry_run=args.dry_run)
            if group_report.skipped_tries:
                exit_code = 1
    return exit_code


def split_patterns(raw_values: List[str]) -> List[str]:
    patterns: List[str] = []
    for raw_value in raw_values:
        for part in raw_value.split(","):
            pattern = part.strip()
            if pattern:
                patterns.append(pattern)
    return patterns


def print_report(report: RebuildReport, *, dry_run: bool) -> None:
    mode = "DRY-RUN" if dry_run else "DONE"
    print(f"[{mode}] {report.run_dir}")
    print(
        "  "
        f"scenarios={report.scenarios} "
        f"try_scores={report.try_scores} "
        f"rebuilt_from_logs={report.rebuilt_tries} "
        f"skipped_tries={report.skipped_tries}"
    )
    if report.selected_scenarios:
        print(f"  selected: {', '.join(report.selected_scenarios)}")
    print_metric_summary(report.aggregate_payload)
    if report.written_files:
        verb = "would write" if dry_run else "wrote"
        for path in report.written_files:
            print(f"  {verb}: {path}")
    for warning in report.warnings:
        print(f"  warning: {warning}", file=sys.stderr)


def print_metric_summary(payload: Mapping[str, Any]) -> None:
    if not payload:
        return
    counts = payload.get("counts", {})
    primary = payload.get("primary_metrics", {})
    print(
        "  primary: "
        f"tasks={counts.get('task_count', 0)} "
        f"SR={format_metric(primary.get('success_rate'))} "
        f"GC_macro={format_metric(primary.get('goal_completion_macro'))} "
        f"GC_micro={format_metric(primary.get('goal_completion_micro'))}"
    )

    length_parts = []
    for bucket, bucket_payload in ordered_metric_items(payload.get("length_breakdown", {})):
        bucket_counts = bucket_payload.get("counts", {})
        bucket_primary = bucket_payload.get("primary_metrics", {})
        if int(bucket_counts.get("task_count", 0)) == 0:
            continue
        length_parts.append(
            f"{bucket}:n={bucket_counts.get('task_count')} "
            f"GCm={format_metric(bucket_primary.get('goal_completion_macro'))}"
        )
    if length_parts:
        print(f"  length: {' | '.join(length_parts)}")

    secondary_parts = []
    for criterion, criterion_payload in payload.get("secondary_metrics", {}).items():
        secondary_parts.append(
            f"{criterion}={criterion_payload.get('success', 0)}/"
            f"{criterion_payload.get('reached', 0)} "
            f"({format_metric(criterion_payload.get('success_when_reached'))})"
        )
    if secondary_parts:
        print(f"  secondary: {' | '.join(secondary_parts)}")


def format_metric(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}%"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
