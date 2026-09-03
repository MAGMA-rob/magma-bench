from dataclasses import dataclass
from importlib import import_module
from typing import Type

from magma_bench.agents.base import BenchmarkAgent


@dataclass(frozen=True)
class BenchmarkAgentMode:
    """
    Registry entry for a benchmark agent packet.

    The packet is imported only when selected. Each packet must expose
    ``AGENT_CLASS`` from its ``__init__.py``.
    """

    name: str
    packet: str

    def load_agent_class(self) -> Type[BenchmarkAgent]:
        module = import_module(self.packet)

        try:
            agent_class = module.AGENT_CLASS
        except AttributeError as err:
            raise ValueError(
                f"Benchmark agent packet {self.packet!r} must expose AGENT_CLASS"
            ) from err

        if not isinstance(agent_class, type):
            raise TypeError(f"{self.packet}.AGENT_CLASS must be a class")

        if not issubclass(agent_class, BenchmarkAgent):
            raise TypeError(
                f"{self.packet}.AGENT_CLASS must inherit from BenchmarkAgent"
            )

        return agent_class


AGENT_MODES = {
    "history_reactive": BenchmarkAgentMode(
        name="history_reactive",
        packet="magma_bench.agents.history_reactive",
    ),
    "history_summary_reactive": BenchmarkAgentMode(
        name="history_summary_reactive",
        packet="magma_bench.agents.history_summary_reactive",
    ),
    "task_state_reactive": BenchmarkAgentMode(
        name="task_state_reactive",
        packet="magma_bench.agents.task_state_reactive",
    ),
}


def get_agent_mode(mode_name: str) -> BenchmarkAgentMode:
    try:
        return AGENT_MODES[mode_name]
    except KeyError as err:
        available_modes = ", ".join(sorted(AGENT_MODES))
        raise ValueError(
            f"Unknown benchmark agent mode {mode_name!r}. Available modes: {available_modes}"
        ) from err


def available_agent_modes() -> list[str]:
    return sorted(AGENT_MODES)
