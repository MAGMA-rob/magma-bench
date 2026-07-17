from typing import Dict, List
from enum import Enum

from magma_bench.loader import EpisodeGroup
from magma_bench.data_structures import EpisodeData
from magma_bench.executor import ToolsEvalExecutor

from magma_core.base.agents import AgentAnswer
from magma_core.base.skills import SkillManager

class GroupRunner:
    """
    Manage running process of the EpisodeGroup
    """

    group : EpisodeGroup
    skill_manager : SkillManager

    episode_data_per_env : Dict[int,EpisodeData]

    def __init__(
        self,
        group : EpisodeGroup,
        executor_ref : ToolsEvalExecutor
    ) -> None:
        self.group = group
        self.episode_data_per_env = {}
        for _ in range(executor_ref.nb_env):
            episode_spec = self.group.get_episode()
            if episode_spec is None:
                break
            episode_data = executor_ref.register(episode_spec)
            self.episode_data_per_env[episode_data.env_idx] = episode_data


    def tick(
        self,
        status_from_env : List[Dict],
        answers_from_agent : List[AgentAnswer]
    ):
        # 1 Apply answers from agents
        if len(answers_from_agent)>0:
            self._answers_tick(answers_from_agent)


        # 2 apply status_from_env












    def _answers_tick(self, answers_list : List[AgentAnswer]):
        for answer in answers_list:
            if not answer.is_valid():
                self.episode_data_per_env[answer.source_node_id].apply_answer()