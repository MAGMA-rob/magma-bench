from .models import (
    AgentIdentity,
    BenchmarkResult,
    EpisodeOutcome,
    EpisodeResult,
    EpisodeTerminal,
    ExecutionIdentity,
    InfrastructureFailure,
    MetricsPayload,
    Rate,
    RunManifest,
    ScenarioResult,
    TraceEvent,
)

__all__ = [
    "AgentIdentity",
    "BenchmarkResult",
    "EpisodeOutcome",
    "EpisodeResult",
    "EpisodeTerminal",
    "ExecutionIdentity",
    "InfrastructureFailure",
    "MetricsPayload",
    "Rate",
    "RunManifest",
    "ScenarioResult",
    "TraceEvent",
    "ResultManager",
    "compute_metrics",
    "validate_metric_inputs",
]


def __getattr__(name: str):
    if name == "ResultManager":
        from .manager import ResultManager

        return ResultManager
    if name in {"compute_metrics", "validate_metric_inputs"}:
        from .metrics import compute_metrics, validate_metric_inputs

        return {
            "compute_metrics": compute_metrics,
            "validate_metric_inputs": validate_metric_inputs,
        }[name]
    raise AttributeError(name)
