from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from magma_core.base.data_structures import EnvToolContext, SavedEnvData, ValidExecutionReq
from magma_core.base.randomizer import RuntimeRandomizer
from magma_core.base.skills.skill_manager import SkillAPIProvider
from magma_core.base.tasks import BaseTask

from magma_bench.data_structures import Episode

@dataclass
class EvalEpisodeContext:
    """Mutable execution state isolated to one active benchmark episode."""

    episode: Episode
    task_ref: BaseTask
    randomizer: RuntimeRandomizer
    saved_data: SavedEnvData
    tool_context: Optional[EnvToolContext] = None
    last_agent_answer: Optional[ValidExecutionReq] = None
    retry_count: int = 0
    judge_pending: bool = False

    def get_answer(self) -> ValidExecutionReq:
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


@dataclass(frozen=True)
class EvalSkillAPIProvider(SkillAPIProvider):
    """Public primitive API and translator bound to one episode runtime."""

    tools_with_real_names: Tuple[Tuple[Dict, str], ...]
    randomizer: RuntimeRandomizer

    def get_tools_with_real_names(self) -> List[Tuple[Dict, str]]:
        return deepcopy(list(self.tools_with_real_names))

    def get_skill_vocabulary_translator(self) -> RuntimeRandomizer:
        return self.randomizer
