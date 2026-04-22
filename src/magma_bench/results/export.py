import json
import os
from typing import Any, Dict, Optional

from .metrics import BenchmarkMetrics, ScenarioMetrics, TaskMetrics, compute_task_metrics
from .metrics_scenario import ScenarioResult, TaskRecord

# Formatting for the export

def _compact_secondary_metrics(task_metrics: TaskMetrics) -> Dict[str, str]:
    return {
        criterion: f"{counter.success} / {counter.reached}"
        for criterion, counter in task_metrics.secondary.items()
    }


def _build_task_log(task_record: TaskRecord, task_metrics: TaskMetrics) -> Dict[str, Any]:
    """Build the compact per-task log payload kept in `logs/<task_id>.json`."""
    return {
        "header": {
            "task_success": task_metrics.primary.task_success,
            "goal_completion": task_metrics.primary.goal_completion_ratio * 100,
            "secondary_metrics": _compact_secondary_metrics(task_metrics),
        },
        "stage": [
            {
                "success": record.success,
                "verification_log": _normalize_stage_reason(record),
                "conversation": [_export_conversation_message(message) for message in record.conversation],
            }
            for record in task_record.stage_records
        ],
    }


def _normalize_stage_reason(task_record) -> str:
    reason = str(task_record.explanation or "").strip()
    if reason:
        return reason
    return "Stage succeeded." if task_record.success else "Stage failed."


def _export_conversation_message(message: Dict[str, Any]) -> Dict[str, Any]:
    exported_message = dict(message)
    if exported_message.get("author") != "SYSTEM":
        return exported_message

    content = exported_message.get("content")
    if not isinstance(content, dict):
        return exported_message

    exported_message["content"] = content.get("infos") or content.get("error") or ""
    return exported_message


def export_task_log(task_record: TaskRecord, criteria: list[str], folder_path: str) -> None:
    """Write the detailed JSON log for a single completed task."""
    per_task_path = os.path.join(folder_path, "logs")
    os.makedirs(per_task_path, exist_ok=True)

    task_metrics = compute_task_metrics(task_record, criteria)
    with open(os.path.join(per_task_path, f"{task_record.task_id}.json"), "w+") as file:
        json.dump(_build_task_log(task_record, task_metrics), file, indent=2)


def metrics_to_payload(metrics: ScenarioMetrics, extra_counts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert aggregated metrics into the explicit JSON schema used on disk."""
    counts = {
        "task_count": metrics.task_count,
        "success_count": metrics.success_count,
        "total_task_horizon": metrics.total_task_horizon,
        "completed_prefix_horizon": metrics.completed_prefix_horizon,
        "reached_stage_count": metrics.reached_stage_count,
        "skipped_stage_count": metrics.skipped_stage_count,
    }
    if extra_counts:
        counts.update(extra_counts)

    return {
        "counts": counts,
        "primary_metrics": {
            "success_rate": metrics.success_rate,
            "goal_completion_macro": metrics.goal_completion_macro,
            "goal_completion_micro": metrics.goal_completion_micro,
        },
        "length_breakdown": {
            bucket: bucket_metrics.to_dict()
            for bucket, bucket_metrics in metrics.length_breakdown.items()
        },
        "secondary_metrics": {
            criterion: counter.to_dict()
            for criterion, counter in metrics.secondary_metrics.items()
        },
        "auxiliary_metrics": {
            "tool_accuracy": metrics.tool_accuracy,
        },
    }


def export_scenario_try_score(scenario_result: ScenarioResult, folder_path: str) -> None:
    if scenario_result.metrics is None:
        raise RuntimeError("Scenario metrics must be computed before export.")

    payload = metrics_to_payload(scenario_result.metrics)
    if scenario_result.randomization_info is not None:
        payload["randomization"] = scenario_result.randomization_info

    try_score_path = os.path.join(folder_path, "try_score.json")
    with open(try_score_path, "w+") as file:
        json.dump(payload, file, indent=2)


def export_scenario_summary(metrics: ScenarioMetrics, output_path: str, num_try: int) -> None:
    with open(output_path, "w+") as file:
        json.dump(metrics_to_payload(metrics, extra_counts={"num_try": num_try}), file, indent=2)


def build_benchmark_payload(benchmark_metrics: BenchmarkMetrics) -> Dict[str, Any]:
    payload = metrics_to_payload(
        benchmark_metrics.summary,
        extra_counts={"scenario_count": benchmark_metrics.scenario_count},
    )
    payload["scenarios"] = {
        scenario_id: metrics_to_payload(metrics)
        for scenario_id, metrics in benchmark_metrics.per_scenario.items()
    }
    return payload
