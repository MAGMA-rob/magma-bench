from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"HistoryReactiveBenchmarkAgent", "AGENT_CLASS"}:
        from .agent import HistoryReactiveBenchmarkAgent

        if name == "AGENT_CLASS":
            return HistoryReactiveBenchmarkAgent
        return HistoryReactiveBenchmarkAgent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HistoryReactiveBenchmarkAgent",
    "AGENT_CLASS",
]
