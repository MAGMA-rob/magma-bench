from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from magma_scenarios.benchmark import KNOWN_CRITERIA, LENGTH_GROUPS

from .metrics_scenario import ScenarioResult, TaskRecord


def _safe_percentage(num: float, den: float):
    if den == 0:
        return "non-measured"
    return num * 100.0 / den


@dataclass
class ReachSuccessCounter:
    eligible: int = 0
    reached: int = 0
    success: int = 0

    def register(self, reached: bool, success: bool) -> None:
        # `eligible` counts every opportunity for a criterion.
        # `reached` counts only stages actually executed by the runner.
        # `success` is conditional on reach.
        self.eligible += 1
        if reached:
            self.reached += 1
            if success:
                self.success += 1

    def merge(self, other: "ReachSuccessCounter") -> None:
        self.eligible += other.eligible
        self.reached += other.reached
        self.success += other.success

    def to_dict(self) -> Dict[str, object]:
        return {
            "eligible": self.eligible,
            "reached": self.reached,
            "success": self.success,
            "reach_rate": _safe_percentage(self.reached, self.eligible),
            "success_when_reached": _safe_percentage(self.success, self.reached),
        }


@dataclass
class PrimaryTaskMetrics:
    task_success: bool
    completed_prefix_horizon: int
    goal_completion_ratio: float


@dataclass
class TaskMetrics:
    task_id: str
    primary: PrimaryTaskMetrics
    secondary: Dict[str, ReachSuccessCounter]
    execution_score_sum: float
    reached_stage_count: int
    skipped_stage_count: int

    @property
    def tool_accuracy(self):
        return _safe_percentage(self.execution_score_sum, self.reached_stage_count)


@dataclass
class LengthBucketMetrics:
    task_count: int = 0
    success_count: int = 0
    goal_completion_sum: float = 0.0
    completed_prefix_horizon: int = 0
    total_task_horizon: int = 0

    def register(self, task_horizon: int, task_metrics: TaskMetrics) -> None:
        self.task_count += 1
        self.success_count += int(task_metrics.primary.task_success)
        self.goal_completion_sum += task_metrics.primary.goal_completion_ratio
        self.completed_prefix_horizon += task_metrics.primary.completed_prefix_horizon
        self.total_task_horizon += task_horizon

    def merge(self, other: "LengthBucketMetrics") -> None:
        self.task_count += other.task_count
        self.success_count += other.success_count
        self.goal_completion_sum += other.goal_completion_sum
        self.completed_prefix_horizon += other.completed_prefix_horizon
        self.total_task_horizon += other.total_task_horizon

    def to_dict(self) -> Dict[str, object]:
        return {
            "counts": {
                "task_count": self.task_count,
                "success_count": self.success_count,
                "total_task_horizon": self.total_task_horizon,
                "completed_prefix_horizon": self.completed_prefix_horizon,
            },
            "primary_metrics": {
                "success_rate": _safe_percentage(self.success_count, self.task_count),
                "goal_completion_macro": _safe_percentage(self.goal_completion_sum, self.task_count),
                "goal_completion_micro": _safe_percentage(
                    self.completed_prefix_horizon, self.total_task_horizon
                ),
            },
        }


@dataclass
class ScenarioMetrics:
    scenario_id: str
    criteria: List[str]
    task_count: int = 0
    success_count: int = 0
    goal_completion_sum: float = 0.0
    completed_prefix_horizon: int = 0
    total_task_horizon: int = 0
    execution_score_sum: float = 0.0
    reached_stage_count: int = 0
    skipped_stage_count: int = 0
    secondary_metrics: Dict[str, ReachSuccessCounter] = field(default_factory=dict)
    length_breakdown: Dict[str, LengthBucketMetrics] = field(default_factory=dict)
    task_metrics: Dict[str, TaskMetrics] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.secondary_metrics:
            self.secondary_metrics = {criterion: ReachSuccessCounter() for criterion in self.criteria}
        if not self.length_breakdown:
            self.length_breakdown = {bucket: LengthBucketMetrics() for bucket in LENGTH_GROUPS}

    def register_task_metrics(self, task_record: TaskRecord, task_metrics: TaskMetrics) -> None:
        # Macro metrics aggregate one value per task, while micro metrics keep
        # the raw horizon counts so that longer tasks weigh more.
        self.task_count += 1
        self.success_count += int(task_metrics.primary.task_success)
        self.goal_completion_sum += task_metrics.primary.goal_completion_ratio
        self.completed_prefix_horizon += task_metrics.primary.completed_prefix_horizon
        self.total_task_horizon += task_record.task_horizon
        self.execution_score_sum += task_metrics.execution_score_sum
        self.reached_stage_count += task_metrics.reached_stage_count
        self.skipped_stage_count += task_metrics.skipped_stage_count
        self.task_metrics[task_record.task_id] = task_metrics

        if task_record.length_bucket not in self.length_breakdown:
            self.length_breakdown[task_record.length_bucket] = LengthBucketMetrics()
        self.length_breakdown[task_record.length_bucket].register(task_record.task_horizon, task_metrics)

        for criterion, counter in task_metrics.secondary.items():
            self.secondary_metrics[criterion].merge(counter)

    def merge(self, other: "ScenarioMetrics") -> None:
        self.task_count += other.task_count
        self.success_count += other.success_count
        self.goal_completion_sum += other.goal_completion_sum
        self.completed_prefix_horizon += other.completed_prefix_horizon
        self.total_task_horizon += other.total_task_horizon
        self.execution_score_sum += other.execution_score_sum
        self.reached_stage_count += other.reached_stage_count
        self.skipped_stage_count += other.skipped_stage_count

        for criterion, counter in other.secondary_metrics.items():
            if criterion not in self.secondary_metrics:
                self.secondary_metrics[criterion] = ReachSuccessCounter()
            self.secondary_metrics[criterion].merge(counter)

        for bucket, bucket_metrics in other.length_breakdown.items():
            if bucket not in self.length_breakdown:
                self.length_breakdown[bucket] = LengthBucketMetrics()
            self.length_breakdown[bucket].merge(bucket_metrics)

    @property
    def success_rate(self):
        return _safe_percentage(self.success_count, self.task_count)

    @property
    def goal_completion_macro(self):
        return _safe_percentage(self.goal_completion_sum, self.task_count)

    @property
    def goal_completion_micro(self):
        return _safe_percentage(self.completed_prefix_horizon, self.total_task_horizon)

    @property
    def tool_accuracy(self):
        return _safe_percentage(self.execution_score_sum, self.reached_stage_count)


@dataclass
class BenchmarkMetrics:
    summary: ScenarioMetrics
    per_scenario: Dict[str, ScenarioMetrics]

    @property
    def scenario_count(self) -> int:
        return len(self.per_scenario)


def compute_task_metrics(task_record: TaskRecord, criteria: List[str]) -> TaskMetrics:
    """Compute primary, secondary and auxiliary metrics for one task trace."""
    secondary = {criterion: ReachSuccessCounter() for criterion in criteria}
    completed_prefix_horizon = 0
    execution_score_sum = 0.0
    reached_stage_count = 0
    skipped_stage_count = 0

    for record in task_record.stage_records:
        # The GC prefix is already decided during recording. We only need to
        # sum the horizon contribution of the stages marked as part of it.
        if record.counted_in_prefix:
            completed_prefix_horizon += record.stage_horizon

        if record.reached:
            reached_stage_count += 1
            execution_score_sum += record.execution_score
        if record.skipped:
            skipped_stage_count += 1

        for criterion in criteria:
            eligible = criterion in record.keys_evaluator or (criterion == "recovery" and record.is_recovery)
            if eligible:
                secondary[criterion].register(record.reached, record.success)

    # A task is strictly successful only if every recorded stage was reached and
    # successful. Skips therefore make the task fail for SR while remaining
    # useful for secondary `reach` diagnostics.
    task_success = bool(task_record.stage_records) and all(
        record.reached and record.success for record in task_record.stage_records
    )
    goal_completion_ratio = (
        completed_prefix_horizon / task_record.task_horizon if task_record.task_horizon > 0 else 0.0
    )

    return TaskMetrics(
        task_id=task_record.task_id,
        primary=PrimaryTaskMetrics(
            task_success=task_success,
            completed_prefix_horizon=completed_prefix_horizon,
            goal_completion_ratio=goal_completion_ratio,
        ),
        secondary=secondary,
        execution_score_sum=execution_score_sum,
        reached_stage_count=reached_stage_count,
        skipped_stage_count=skipped_stage_count,
    )


def compute_scenario_metrics(scenario_result: ScenarioResult) -> ScenarioMetrics:
    """Aggregate every task trace of one scenario try into scenario-level metrics."""
    metrics = ScenarioMetrics(scenario_id=scenario_result.scenario_id, criteria=scenario_result.criteria_to_evaluate)
    for task_record in scenario_result.iter_task_records():
        task_metrics = compute_task_metrics(task_record, metrics.criteria)
        metrics.register_task_metrics(task_record, task_metrics)
    return metrics


def aggregate_scenario_metrics(scenario_id: str, scenario_results: Iterable[ScenarioResult]) -> ScenarioMetrics:
    """Merge several tries of the same scenario from raw counters."""
    aggregated = ScenarioMetrics(scenario_id=scenario_id, criteria=list(KNOWN_CRITERIA))
    for scenario_result in scenario_results:
        metrics = scenario_result.metrics if scenario_result.metrics is not None else scenario_result.compute_result()
        aggregated.merge(metrics)
    return aggregated


def compute_benchmark_metrics(
    computed_bench_results: Dict[str, List[ScenarioResult]]
) -> BenchmarkMetrics:
    """Build the final benchmark-level summary from every scenario try."""
    per_scenario: Dict[str, ScenarioMetrics] = {}
    summary = ScenarioMetrics(scenario_id="benchmark", criteria=list(KNOWN_CRITERIA))

    for scenario_id, scenario_results in computed_bench_results.items():
        per_scenario_metrics = aggregate_scenario_metrics(scenario_id, scenario_results)
        per_scenario[scenario_id] = per_scenario_metrics
        summary.merge(per_scenario_metrics)

    return BenchmarkMetrics(summary=summary, per_scenario=per_scenario)
