from dataclasses import dataclass
from typing import Dict, List

from magma_bench.loader import EpisodeGroup
from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeData,
    EpisodeSituation,
    RunningState,
    Scenario,
)
from magma_bench.executor import ToolsEvalExecutor

from magma_core.base.data_structures import ToolStatus, ValidExecutionReq

@dataclass(frozen=True)
class BenchmarkTick:

    to_agents: Dict[int, EpisodeSituation]
    to_executor: Dict[int, ValidExecutionReq]

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
        status_from_env: Dict[int, ToolStatus],
        results_from_agent: List[BenchmarkAgentResult],
    ) -> BenchmarkTick:
        self.to_release = []
        to_executor = self._answers_tick(results_from_agent)

        # Tool-status transitions and terminal result registration are handled
        # by the next runner implementation step. Agent-state updates are
        # intentionally independent from that scheduling logic.
        if status_from_env:
            raise NotImplementedError(
                "Tool-status transitions are not implemented in GroupRunner yet"
            )

        to_agents = {}


        for env_idx in self.to_release:
            self.executor_ref.release_idx(env_idx)
            episode_data = self.episode_data_per_env.pop(env_idx)
            new_episode = self.group.get_episode()
            if new_episode is not None:
                self.executor_ref.register(new_episode)

        return BenchmarkTick(
            to_agents=to_agents,
            to_executor=to_executor,
        )

    def _answers_tick(
        self,
        results: List[BenchmarkAgentResult],
    ) -> Dict[int, ValidExecutionReq]:
        to_executor = {}
        for result in results:
            answer = result.answer
            env_idx = answer.source_node_id
            episode_data = self.episode_data_per_env.get(env_idx)
            if episode_data is None:
                raise KeyError(
                    f"No running episode is assigned to environment {env_idx}"
                )
            if episode_data.state != RunningState.WAITING_MODEL_ANSWER:
                raise RuntimeError(
                    f"Environment {env_idx} did not wait for an agent answer"
                )

            episode_data.situation = result.situation
            if not answer.is_valid():
                self.to_release.append(env_idx)
                continue
            
            episode_data.state = RunningState.RUNNING

            to_executor[env_idx] = ValidExecutionReq(
                source_node_id=env_idx,
                agent_step_id=0,
                calls=answer.get_action(),
                say=answer.get_say()
            )
            
        return to_executor
