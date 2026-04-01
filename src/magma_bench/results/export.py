import json
import os
from typing import Any, Dict, Optional

from .metrics import BenchmarkMetrics, ScenarioMetrics, TaskMetrics
from .metrics_scenario import ScenarioResult, TaskRecord

# Formatting for the export

def _build_task_log(task_record: TaskRecord, task_metrics: TaskMetrics) -> Dict[str, Any]:
    """Build the detailed per-task log payload kept in `logs/<task_id>.json`."""
    return {
        "header": {
            "counts": {
                "task_horizon": task_record.task_horizon,
                "length_bucket": task_record.length_bucket,
                "stage_count": len(task_record.stage_records),
                "reached_stage_count": task_metrics.reached_stage_count,
                "skipped_stage_count": task_metrics.skipped_stage_count,
                "completed_prefix_horizon": task_metrics.primary.completed_prefix_horizon,
            },
            "primary_metrics": {
                "task_success": task_metrics.primary.task_success,
                "goal_completion": task_metrics.primary.goal_completion_ratio * 100,
            },
            "secondary_metrics": {
                criterion: counter.to_dict()
                for criterion, counter in task_metrics.secondary.items()
            },
            "auxiliary_metrics": {
                "tool_accuracy": task_metrics.tool_accuracy,
            },
        },
        "stage": [record.to_log_dict() for record in task_record.stage_records],
    }


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


def export_scenario_try(scenario_result: ScenarioResult, folder_path: str, detailled_log: bool) -> None:
    if scenario_result.metrics is None:
        raise RuntimeError("Scenario metrics must be computed before export.")

    if detailled_log:
        # Detailed logs stay task-oriented so it is easy to inspect one rollout
        # after a benchmark run and understand where the prefix broke.
        per_task_path = os.path.join(folder_path, "logs")
        os.makedirs(per_task_path, exist_ok=True)

        for task_record in scenario_result.iter_task_records():
            task_metrics = scenario_result.metrics.task_metrics[task_record.task_id]
            with open(os.path.join(per_task_path, f"{task_record.task_id}.json"), "w+") as file:
                json.dump(_build_task_log(task_record, task_metrics), file, indent=2)

    try_score_path = os.path.join(folder_path, "try_score.json")
    with open(try_score_path, "w+") as file:
        json.dump(metrics_to_payload(scenario_result.metrics), file, indent=2)


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
