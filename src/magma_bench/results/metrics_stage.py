from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StageResult:
    success: bool = False
    conversation: List[Dict[str, Any]] = field(default_factory=list)
    explanation: str = ""
    executions_result: List[bool] = field(default_factory=list)


@dataclass
class StageRecord:
    """
    Represent the execution of one stage
    """
    task_id: str
    stage_id: str
    success: bool
    reached: bool
    skipped: bool
    skip_reason: Optional[str]
    conversation: List[Dict[str, Any]]
    explanation: str
    execution_score: float
    keys_evaluator: List[str]
    is_recovery: bool
    stage_horizon: int
    counted_in_prefix: bool

    def to_log_dict(self) -> Dict[str, Any]:
        """Return the JSON-friendly representation written in detailed logs."""
        status = "skipped" if self.skipped else ("success" if self.success else "failed")
        return {
            "stage_id": self.stage_id,
            "status": status,
            "success": self.success,
            "reached": self.reached,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "counted_in_prefix": self.counted_in_prefix,
            "stage_horizon": self.stage_horizon,
            "criteria": self.keys_evaluator,
            "is_recovery": self.is_recovery,
            "execution_score": self.execution_score,
            "conversation": self.conversation,
            "explanation": self.explanation,
        }
