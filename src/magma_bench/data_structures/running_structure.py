from dataclasses import dataclass, field
from typing import Any, List, Dict, Tuple, Literal
from enum import Enum

from magma_core.utils.data_utils import apply_att_modif
from magma_core.base.data_structures import Instruction


@dataclass
class EpisodeSituation:
    """Agent-visible state for one running benchmark episode."""

    tools: List[Dict]
    attributes: Dict
    current_instruction: Instruction
    history: List[Any] = field(default_factory=list)


    def modify_attributes(
        self,
        modif_list: List[Tuple[Literal["ADD", "REMOVE"], Tuple[str, str]]],
    ) -> None:
        apply_att_modif(self.attributes, modif_list)

    def add_to_history(self, model_answer: Dict) -> None:
        self.history.append(model_answer)

    def set_current_instruction(self, instruction: Instruction) -> None:
        self.current_instruction = instruction
        self.history.append(instruction)

    def to_dict(self) -> Dict:
        return {
            "tools": self.tools,
            "attributes": self.attributes,
            "history": self.history,
            "current_instruction": self.current_instruction.to_spec(),
        }


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
