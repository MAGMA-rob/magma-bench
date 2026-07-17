from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from magma_core.base.data_structures import EnvToolContext
from magma_core.base.agents import ValidAgentAnswer

@dataclass
class EvalEpisodeContext:
    episode_id: str
    tool_context : EnvToolContext

    last_env_state : Dict = field(default_factory=dict)
    last_agent_answer: Optional[ValidAgentAnswer] = None
    nb_of_retry : int = 0

    def get_answer(self) -> ValidAgentAnswer:
        if self.last_agent_answer is None:
            raise RuntimeError("Trying to access the last agent answer of an episode which is None")
        return self.last_agent_answer

    def set_tool_context(self, tool_infos : EnvToolContext):
        self.tool_context = tool_infos