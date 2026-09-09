from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..artifacts.models import Track


RESULT_SCHEMA_VERSION = "2.0"
NON_CLEAN_CONDITIONS = (
    "mission_update",
    "interruption",
    "execution_error",
    "combined",
)
LENGTH_BUCKETS = ("short", "mid", "long", "very-long", "extreme")
LAG_BUCKETS = ("0-3", "4-9", "10-19", "20+")


class ResultModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class TraceEvent(ResultModel):
    """One agent-visible event in an episode trace."""

    index: int = Field(ge=0)
    stage_index: int = Field(ge=0)
    kind: Literal[
        "instruction",
        "agent_answer",
        "tool_feedback",
        "failure_diagnostics",
    ]
    payload: Dict[str, Any]


class EpisodeTerminal(ResultModel):
    status: Literal[
        "success",
        "invalid_agent_answer",
        "protocol_failure",
        "stage_failure",
        "budget_exceeded",
        "infrastructure_failure",
    ]
    reason: Optional[str] = None
    last_action_event_index: Optional[int] = Field(default=None, ge=0)


class EpisodeOutcome(ResultModel):
    episode_id: str
    success: bool
    terminal: EpisodeTerminal
    trace: List[TraceEvent]

    @model_validator(mode="after")
    def validate_outcome(self) -> "EpisodeOutcome":
        if self.success != (self.terminal.status == "success"):
            raise ValueError("success must agree with terminal status")
        if [event.index for event in self.trace] != list(range(len(self.trace))):
            raise ValueError("Trace event indices must be contiguous and ordered")
        last_action = self.terminal.last_action_event_index
        if last_action is not None:
            if last_action >= len(self.trace):
                raise ValueError("last_action_event_index is outside the trace")
            event = self.trace[last_action]
            if event.kind != "agent_answer" or not event.payload.get("action"):
                raise ValueError(
                    "last_action_event_index must reference an agent action"
                )
        return self


class EpisodeResult(EpisodeOutcome):
    schema_version: Literal["2.0"] = RESULT_SCHEMA_VERSION


class Rate(ResultModel):
    value: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_rate(self) -> "Rate":
        if self.numerator > self.denominator:
            raise ValueError("Rate numerator cannot exceed denominator")
        if self.denominator == 0:
            if self.numerator != 0 or self.value is not None:
                raise ValueError("An unmeasured rate must be 0/0 with value null")
            return self
        expected = self.numerator / self.denominator
        if self.value is None or abs(self.value - expected) > 1e-12:
            raise ValueError("Rate value does not match numerator/denominator")
        return self


class LengthMetrics(ResultModel):
    clean_success_rate: Rate
    success_rate_by_condition: Dict[str, Rate]


class MetricCounts(ResultModel):
    scenario_count: int = Field(ge=0)
    skeleton_count: int = Field(ge=0)
    semantic_unit_count: int = Field(ge=0)
    episode_count: int = Field(ge=0)


class MetricsPayload(ResultModel):
    counts: MetricCounts
    clean_success_rate: Rate
    success_rate_by_condition: Dict[str, Rate]
    conditional_robustness_by_condition: Dict[str, Rate]
    success_rate_by_length: Dict[str, LengthMetrics]
    mission_update_retention_by_lag: Dict[str, Rate]


class ScenarioResult(ResultModel):
    schema_version: Literal["2.0"] = RESULT_SCHEMA_VERSION
    scenario_id: str
    metrics: MetricsPayload
    metrics_by_track: Dict[Track, MetricsPayload]


class InfrastructureFailure(ResultModel):
    scenario_id: str
    episode_id: str
    reason: Optional[str] = None


class BenchmarkResult(ResultModel):
    schema_version: Literal["2.0"] = RESULT_SCHEMA_VERSION
    status: Literal["complete", "partial"] = "complete"
    expected_scenario_count: int = Field(default=0, ge=0)
    completed_scenario_count: int = Field(default=0, ge=0)
    partial_scenario_ids: List[str] = Field(default_factory=list)
    infrastructure_failures: List[InfrastructureFailure] = Field(
        default_factory=list
    )
    metrics: MetricsPayload
    metrics_by_track: Dict[Track, MetricsPayload]


class AgentIdentity(ResultModel):
    agent: str
    agent_id: str
    agent_version: str
    protocol_version: str
    extra_keys: Dict[str, Any]


class ExecutionIdentity(ResultModel):
    """Runtime choices that must remain stable when a run is resumed."""

    judge_mode: Literal["backend", "skipped"] = "backend"
    sim_backend: str = "auto"
    seed: int = 0


class RunManifest(ResultModel):
    schema_version: Literal["2.0"] = RESULT_SCHEMA_VERSION
    benchmark_version: str
    benchmark_fingerprint: str
    agent: AgentIdentity
    created_at: str
    scenario_ids: List[str]
    execution: ExecutionIdentity = Field(default_factory=ExecutionIdentity)

    @model_validator(mode="after")
    def validate_scenarios(self) -> "RunManifest":
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("Run manifest scenario IDs must be unique")
        return self
