import copy
from dataclasses import InitVar, dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple
from enum import Enum

from magma_core.simulation.agents import AgentAnswer
from magma_core.simulation.data_structures import Instruction
from magma_core.utils.data_utils import apply_att_modif

from magma_bench.results.models import TraceEvent


@dataclass
class EpisodeSituation:
    """Episode observation with opaque memory owned by the remote runtime."""

    tools: List[Dict]
    attributes: Dict
    memory: Dict[str, Any]
    current_instruction: Instruction
    history: InitVar[Optional[List[Any]]] = None

    def __post_init__(self, history: Optional[List[Any]]) -> None:
        self.memory = copy.deepcopy(self.memory)
        if history is not None:
            self.memory["history"] = copy.deepcopy(history)
        else:
            self.memory.setdefault("history", [])

    def modify_attributes(
        self,
        modif_list: List[Tuple[Literal["ADD", "REMOVE"], Tuple[str, str]]],
    ) -> None:
        apply_att_modif(self.attributes, modif_list)

    def set_current_instruction(self, instruction: Instruction) -> None:
        self.current_instruction = instruction

    def snapshot(self) -> "EpisodeSituation":
        """Return an independent copy suitable for background processing."""

        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the generic situation without flattening agent state."""

        return {
            "tools": self.tools,
            "attributes": self.attributes,
            "memory": self.memory,
            "current_instruction": self.current_instruction.to_spec(),
        }


@dataclass(frozen=True)
class BenchmarkAgentResult:
    """One normalized answer and the episode state produced alongside it."""

    answer: AgentAnswer
    situation: EpisodeSituation
    model_diagnostics: List[Dict[str, Any]] = field(default_factory=list)


class RunningState(Enum):
    RUNNING = "running"
    WAITING_MODEL_ANSWER = "waiting_model_answer"


@dataclass
class EpisodeData:
    """
    Episode Runtime Data used to carry conversations of the episode.
    """

    episode_id: str
    state: RunningState
    situation: EpisodeSituation
    env_idx: int
    last_agent_answer: Optional[AgentAnswer] = None
    trace: List[TraceEvent] = field(default_factory=list)
    next_trace_index: int = 0
    last_action_event_index: Optional[int] = None

    def record_trace(
        self,
        stage_index: int,
        kind: Literal[
            "instruction",
            "agent_answer",
            "tool_feedback",
            "failure_diagnostics",
        ],
        payload: Dict[str, Any],
    ) -> TraceEvent:
        event = TraceEvent(
            index=self.next_trace_index,
            stage_index=stage_index,
            kind=kind,
            payload=copy.deepcopy(payload),
        )
        self.trace.append(event)
        self.next_trace_index += 1
        return event
