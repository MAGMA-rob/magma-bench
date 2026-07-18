from typing import Dict, List
from dataclasses import dataclass

from magma_bench.loader import EpisodeGroup
from magma_bench.data_structures import EpisodeData, Scenario, EpisodeSituation
from magma_bench.executor import ToolsEvalExecutor

from magma_core.base.agents import AgentAnswer
from magma_core.base.data_structures import ValidExecutionReq, ToolStatus

@dataclass(frozen=True)
class BenchmarkTick():

    to_agents : Dict[int, EpisodeSituation]
    to_executor : Dict[int, ValidExecutionReq]

    def has_inputs_for_agents(self) -> bool:
        return len(self.to_agents) > 0
    
    def has_call_for_executor(self) -> bool:
        return len(self.to_executor) > 0

class GroupRunner:
    """
    Manage running process of the EpisodeGroup
    """

    scenario: Scenario
    group : EpisodeGroup
    executor_ref: ToolsEvalExecutor

    episode_data_per_env : Dict[int,EpisodeData]

    def __init__(
        self,
        scenario: Scenario,
        group: EpisodeGroup,
        executor_ref: ToolsEvalExecutor,
    ) -> None:
        self.scenario = scenario
        self.group = group
        self.executor_ref = executor_ref
        self.episode_data_per_env = {}
        executor_ref.initialize_group(scenario, group)
        for _ in range(executor_ref.nb_env):
            episode = self.group.get_episode()
            if episode is None:
                break
            episode_data = executor_ref.register(episode)
            self.episode_data_per_env[episode_data.env_idx] = episode_data


    def tick(
        self,
        status_from_env : Dict[int, ToolStatus],
        answers_from_agent : List[AgentAnswer]
    ) -> BenchmarkTick:
        # 1 Apply answers from agents
        if len(answers_from_agent)>0:
            self._answers_tick(answers_from_agent)


        # 2 apply status_from_env


    ...









    def _answers_tick(self, answers_list : List[AgentAnswer]):
        for answer in answers_list:
            if not answer.is_valid():
                self.episode_data_per_env[answer.source_node_id].apply_answer()
