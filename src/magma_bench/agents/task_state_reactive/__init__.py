from typing import Any


def __getattr__(name: str) -> Any:
    if name in {"TaskStateReactiveBenchmarkAgent", "AGENT_CLASS"}:
        from .agent import TaskStateReactiveBenchmarkAgent

        if name == "AGENT_CLASS":
            return TaskStateReactiveBenchmarkAgent
        return TaskStateReactiveBenchmarkAgent

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TaskStateReactiveBenchmarkAgent",
    "AGENT_CLASS",
]
