from types import SimpleNamespace

import pytest


pytest.importorskip("sapien")

from magma_core.base.agents import BadAgentAnswer
from magma_core.base.data_structures import (
    RobotToolStatus,
    StageSuccess,
    ToolErrorFlag,
    ToolStatus,
)

from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeData,
    EpisodeSituation,
    RunningState,
)
from magma_bench.runner import group_runner as group_runner_module
from magma_bench.runner.group_runner import GroupRunner


class FakeInstruction:
    def get_role(self):
        return "user"

    def get_content(self):
        return "Do the task"

    def to_spec(self):
        return {"type": "fake", "arguments": {}}


class FakeEpisodeGroup:
    def __init__(self):
        self.episodes = [SimpleNamespace(episode_id="episode_1")]

    def get_episode(self):
        return self.episodes.pop(0) if self.episodes else None


class FakeExecutor:
    nb_env = 1

    def __init__(self):
        self.released = []

    def initialize_group(self, _scenario, _group):
        return None

    def register(self, episode):
        return EpisodeData(
            episode_id=episode.episode_id,
            state=RunningState.WAITING_MODEL_ANSWER,
            situation=EpisodeSituation(
                tools=[],
                attributes={"known_robots": ["robot"]},
                memory={},
                current_instruction=FakeInstruction(),
            ),
            env_idx=0,
        )

    def get_skill_api_provider(self, _env_idx):
        return object()

    def get_skill_execution_context(self, _env_idx, attributes):
        return SimpleNamespace(stage_id=0, attributes=attributes)

    def release_idx(self, env_idx):
        self.released.append(env_idx)


class FakeSkillManager:
    def __init__(self, _skill_types):
        pass

    def build_api(self, _provider):
        pass

    def initialize_robot_statuses(self, _attributes):
        pass

    def get_api(self):
        return []


def test_group_runner_emits_one_traced_outcome_for_invalid_answer(monkeypatch):
    monkeypatch.setattr(group_runner_module, "SkillManager", FakeSkillManager)
    outcomes = []
    executor = FakeExecutor()
    runner = GroupRunner(
        scenario=SimpleNamespace(skill_types=()),
        group=FakeEpisodeGroup(),
        executor_ref=executor,
        on_episode_finished=outcomes.append,
    )

    initial = runner.tick({}, [])
    assert list(initial.to_agents) == [0]
    episode_data = runner.episode_data_per_env[0]
    answer = BadAgentAnswer(
        source_node_id=0,
        agent_step_id=0,
        say="",
        raw_action="not-json",
        reason="invalid JSON",
    )
    runner.tick(
        {},
        [BenchmarkAgentResult(answer=answer, situation=episode_data.situation)],
    )

    assert runner.is_done()
    assert executor.released == [0]
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome.success is False
    assert outcome.terminal.status == "invalid_agent_answer"
    assert [event.kind for event in outcome.trace] == [
        "instruction",
        "agent_answer",
    ]
    assert outcome.trace[1].payload["raw_action"] == "not-json"


def test_group_runner_classifies_planner_failure_as_infrastructure():
    status = ToolStatus(
        robots_status=[
            RobotToolStatus(
                robot_name="robot",
                mess="planner unavailable",
                result=False,
                error_flag=ToolErrorFlag.PLANNER_ERROR,
            )
        ],
        stage_id=0,
        attributes={},
        error_descriptions=[""],
        stage_success=StageSuccess.FAILED,
    )

    assert GroupRunner._failure_terminal_status(status) == "infrastructure_failure"
