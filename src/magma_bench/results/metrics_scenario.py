from typing import Any, Dict, List, Optional

from magma_scenarios.benchmark import KNOWN_CRITERIA
from .metrics_stage import StageRecord, StageResult


class TaskRecord:
    """Ordered trace of one task rollout for a single try."""

    def __init__(self, task_id: str, criteria: List[str], task_horizon: int, length_bucket: str) -> None:
        self.task_id = task_id
        self.criteria = list(dict.fromkeys(criteria))
        self.task_horizon = task_horizon
        self.length_bucket = length_bucket
        self.stage_records: List[StageRecord] = []
        self._prefix_open = True

    @staticmethod
    def _compute_execution_score(executions_result: List[bool]) -> float:
        if not executions_result:
            return 0.0
        success_count = sum(1 for val in executions_result if val)
        return success_count / len(executions_result)

    def record_stage(self, stage_ref: Any, stage_result: StageResult) -> None:
        # The prefix used by SR/GC only advances while every previously reached
        # stage succeeded. After the first executed failure, we still record the
        # following stages, but they stop contributing to the primary prefix.
        success = bool(stage_result.success)
        counted_in_prefix = self._prefix_open and success
        if self._prefix_open and not success:
            self._prefix_open = False

        self.stage_records.append(
            StageRecord(
                task_id=self.task_id,
                stage_id=stage_ref.id,
                success=success,
                reached=True,
                skipped=False,
                skip_reason=None,
                conversation=stage_result.conversation,
                explanation=stage_result.explanation,
                execution_score=self._compute_execution_score(stage_result.executions_result),
                keys_evaluator=list(stage_ref.keys_evaluator),
                is_recovery=stage_ref.has_flag_recovery() or stage_ref.has_flag_failure(),
                stage_horizon=stage_ref.stage_horizon,
                counted_in_prefix=counted_in_prefix,
            )
        )

    def record_skip(self, stage_ref: Any, reason: str) -> None:
        # A skipped stage is part of the trace and can still be "eligible" for
        # secondary metrics, but it is never counted as reached nor in the GC prefix.
        self.stage_records.append(
            StageRecord(
                task_id=self.task_id,
                stage_id=stage_ref.id,
                success=False,
                reached=False,
                skipped=True,
                skip_reason=reason,
                conversation=[],
                explanation=reason,
                execution_score=0.0,
                keys_evaluator=list(stage_ref.keys_evaluator),
                is_recovery=stage_ref.has_flag_recovery() or stage_ref.has_flag_failure(),
                stage_horizon=stage_ref.stage_horizon,
                counted_in_prefix=False,
            )
        )


class ScenarioResult:
    """Recorded outcome of one full scenario try before aggregation."""

    def __init__(self, scenario_id: str, criteria_to_evaluate: Optional[List[str]] = None) -> None:
        effective_criteria: List[str] = []
        for criterion in (criteria_to_evaluate or []) + KNOWN_CRITERIA:
            if criterion in KNOWN_CRITERIA and criterion not in effective_criteria:
                effective_criteria.append(criterion)

        self.scenario_id = scenario_id
        self.criteria_to_evaluate = effective_criteria
        self._task_records: Dict[str, TaskRecord] = {}
        self.metrics = None

    def _get_task_record(self, task_ref: Any) -> TaskRecord:
        if task_ref.id not in self._task_records:
            self._task_records[task_ref.id] = TaskRecord(
                task_id=task_ref.id,
                criteria=task_ref.criteria,
                task_horizon=task_ref.task_horizon,
                length_bucket=task_ref.length_bucket,
            )
        return self._task_records[task_ref.id]

    def record_stage(self, task_ref: Any, stage_ref: Any, stage_result: StageResult) -> None:
        self._get_task_record(task_ref).record_stage(stage_ref, stage_result)

    def record_skip(self, task_ref: Any, stage_ref: Any, reason: str) -> None:
        self._get_task_record(task_ref).record_skip(stage_ref, reason)

    def iter_task_records(self) -> List[TaskRecord]:
        return list(self._task_records.values())

    def compute_result(self):
        self.metrics = compute_scenario_metrics(self)
        return self.metrics

    def export(self, folder_path: str, detailled_log: bool) -> None:
        if self.metrics is None:
            self.compute_result()

        export_scenario_try(self, folder_path, detailled_log)


# Late imports keep the public methods above straightforward while still
# avoiding import cycles between the recording, aggregation and export layers.
from .export import export_scenario_try
from .metrics import compute_scenario_metrics
