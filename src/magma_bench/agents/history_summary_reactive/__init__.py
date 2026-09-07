from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"HistorySummaryReactiveBenchmarkAgent", "AGENT_CLASS"}:
        from .agent import HistorySummaryReactiveBenchmarkAgent

        if name == "AGENT_CLASS":
            return HistorySummaryReactiveBenchmarkAgent
        return HistorySummaryReactiveBenchmarkAgent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HistorySummaryReactiveBenchmarkAgent",
    "AGENT_CLASS",
]
