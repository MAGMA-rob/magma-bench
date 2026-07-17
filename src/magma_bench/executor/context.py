from dataclasses import dataclass
from typing import Optional

from magma_core.base.data_structures import EnvToolContext, SavedEnvData
from magma_core.base.agents import ValidAgentAnswer
from magma_core.base.randomizer import RuntimeRandomizer
from magma_core.base.tasks import BaseTask

from magma_bench.data_structures import Episode

@dataclass
class EvalEpisodeContext:
    """Mutable execution state isolated to one active benchmark episode."""

    episode: Episode
    task_ref: BaseTask
    randomizer: RuntimeRandomizer
    saved_data: SavedEnvData
    registration_id: int
    tool_context: Optional[EnvToolContext] = None
    last_agent_answer: Optional[ValidAgentAnswer] = None
    planner_retry_count: int = 0
    judge_retry_count: int = 0

    def get_answer(self) -> ValidAgentAnswer:
        if self.last_agent_answer is None:
            raise RuntimeError("Trying to access the last agent answer of an episode which is None")
        return self.last_agent_answer

    def get_tool_context(self) -> EnvToolContext:
        if self.tool_context is None:
            raise RuntimeError(
                f"Episode {self.episode.episode_id!r} has no active tool context"
            )
        return self.tool_context

    def set_tool_context(self, tool_infos: EnvToolContext) -> None:
        self.tool_context = tool_infos
