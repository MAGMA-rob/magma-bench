from typing import List, Dict, Tuple, Literal
from enum import Enum

from magma_core.utils.data_utils import apply_att_modif
from magma_core.base.data_structures import Instruction

class EpisodeSituation:

    tools : List[Dict[str,Dict]]
    attributes : Dict


    history : List

    current_instruction : Instruction


    def modify_attributes(self, modif_list : List[Tuple[Literal["ADD","REMOVE"],Tuple[str,str]]]):
        apply_att_modif(self.attributes, modif_list)


    def add_to_history(
            self,
            model_answer : Dict
    ):
        self.history.append(model_answer)

    def set_current_instruction(
            self,
            instruction : Instruction
    ):
        self.current_instruction = instruction
        self.history.append(instruction)

    def to_dict(self):
        return {
            "tools"
        }
    
class RunningState(Enum):
    RUNNING = "running"
    WAITING_MODEL_ANSWER = "waiting_model_answer"


class EpisodeData:
    """
    Episode Runtime Data used to carry conversations of the episode.
    """

    state : RunningState
    situation : EpisodeSituation

    env_idx : int # index of the env