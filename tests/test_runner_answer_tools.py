from concurrent.futures import Future
import json
import threading
from types import SimpleNamespace

import pytest


pytest.importorskip("sapien")

from magma_core.base.agents import BadAgentAnswer, ValidAgentAnswer
from magma_core.base.data_structures import (
    Call,
    RobotToolStatus,
    StageInput,
    StageSuccess,
    ToolErrorFlag,
    ToolStatus,
    ValidExecutionReq,
)
from magma_core.base.skills import (
    DeferredInputTransition,
    SkillDeferredInputResult,
    SkillExecutionResult,
    SkillStateRef,
    SkillStatusEvent,
    SkillStatusResult,
)
from magma_core.base.skills.structure import Tick

from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeData,
    EpisodeSituation,
    RunningState,
)
from magma_bench.runner import group_runner as group_runner_module
from magma_bench.runner.group_runner import GroupRunner
from magma_bench.executor.eval_executor import ToolsEvalExecutor


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
        self.initialize_kwargs = {}

    def initialize_group(self, _scenario, _group, **kwargs):
        self.initialize_kwargs = kwargs
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
        self.states_reset = False
        self.suspended_answer = None
        self.registrations = {}

    def build_api(self, _provider):
        pass

    def initialize_robot_statuses(self, _attributes):
        return SimpleNamespace(value=1)

    def get_api(self):
        return []

    def reset_states(self):
        self.states_reset = True

    def get_suspended_answer(self, _state_ref):
        return self.suspended_answer

    def register(self, registrations):
        self.registrations = registrations

    def tick(self, _statuses):
        return Tick({})


def test_group_runner_emits_one_traced_outcome_for_invalid_answer(monkeypatch):
    monkeypatch.setattr(group_runner_module, "SkillManager", FakeSkillManager)
    outcomes = []
    executor = FakeExecutor()
    runner = GroupRunner(
        scenario=SimpleNamespace(skill_types=()),
        group=FakeEpisodeGroup(),
        executor_ref=executor,
        on_episode_finished=outcomes.append,
        sim_backend="gpu",
        seed=17,
    )
    assert executor.initialize_kwargs == {"sim_backend": "gpu", "seed": 17}

    initial = runner.tick({}, [])
    assert list(initial.to_agents) == [0]
    episode_data = runner.episode_data_per_env[0]
    skill_manager = runner.skill_managers[0]
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
    assert skill_manager.states_reset is True
    assert runner.skill_state_refs == {}
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


def test_empty_answer_resumes_suspended_work(monkeypatch):
    monkeypatch.setattr(group_runner_module, "SkillManager", FakeSkillManager)
    runner = GroupRunner(
        scenario=SimpleNamespace(skill_types=()),
        group=FakeEpisodeGroup(),
        executor_ref=FakeExecutor(),
        on_episode_finished=lambda _outcome: None,
    )
    runner.tick({}, [])
    manager = runner.skill_managers[0]
    manager.suspended_answer = ValidAgentAnswer(0, 3, "", [])
    empty_answer = ValidAgentAnswer(0, 4, "", [])

    runner.tick(
        {},
        [
            BenchmarkAgentResult(
                answer=empty_answer,
                situation=runner.episode_data_per_env[0].situation,
            )
        ],
    )

    assert manager.registrations[0].answer is empty_answer


def test_empty_answer_without_suspended_work_is_protocol_failure(monkeypatch):
    monkeypatch.setattr(group_runner_module, "SkillManager", FakeSkillManager)
    outcomes = []
    runner = GroupRunner(
        scenario=SimpleNamespace(skill_types=()),
        group=FakeEpisodeGroup(),
        executor_ref=FakeExecutor(),
        on_episode_finished=outcomes.append,
    )
    runner.tick({}, [])
    empty_answer = ValidAgentAnswer(0, 0, "", [])

    runner.tick(
        {},
        [
            BenchmarkAgentResult(
                answer=empty_answer,
                situation=runner.episode_data_per_env[0].situation,
            )
        ],
    )

    assert outcomes[0].terminal.status == "protocol_failure"


def test_group_runner_consumes_typed_skill_results(monkeypatch):
    monkeypatch.setattr(group_runner_module, "SkillManager", FakeSkillManager)
    runner = GroupRunner(
        scenario=SimpleNamespace(skill_types=()),
        group=FakeEpisodeGroup(),
        executor_ref=FakeExecutor(),
        on_episode_finished=lambda _outcome: None,
    )
    runner.tick({}, [])
    episode_data = runner.episode_data_per_env[0]
    executed_answer = ValidAgentAnswer(
        0,
        1,
        "",
        [Call("pick", {}, "robot")],
    )
    request = ValidExecutionReq(0, 1, [], "")
    to_executor = {}

    runner._consume_skill_tick(
        Tick({0: SkillExecutionResult(0, request, executed_answer)}),
        to_executor,
    )

    assert to_executor == {0: request}
    assert episode_data.last_agent_answer is executed_answer

    status = ToolStatus(
        robots_status=[
            RobotToolStatus("robot", "done", True, ToolErrorFlag.NONE)
        ],
        stage_id=0,
        attributes={"known_robots": ["robot"]},
        error_descriptions=[""],
        stage_success=StageSuccess.ONGOING,
    )
    status_ref = SkillStateRef(2)
    runner._consume_skill_tick(
        Tick(
            {
                0: SkillStatusResult(
                    0,
                    SkillStatusEvent(status, status_ref, executed_answer),
                )
            }
        ),
        {},
    )
    assert runner.skill_state_refs[0] == status_ref
    assert episode_data.trace[-1].kind == "tool_feedback"

    deferred_ref = SkillStateRef(3)
    deferred_instruction = FakeInstruction()
    transition = DeferredInputTransition(
        stage_input=StageInput(deferred_instruction, False),
        attributes={"known_robots": ["robot"], "updated": True},
        tool_calls=1,
        forgiven_tool_calls=0,
        state_ref=deferred_ref,
    )
    runner._consume_skill_tick(
        Tick({0: SkillDeferredInputResult(0, transition)}),
        {},
    )
    assert runner.skill_state_refs[0] == deferred_ref
    assert episode_data.situation.current_instruction is deferred_instruction
    assert episode_data.situation.attributes["updated"] is True


class ImmediateJudgeWorker:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.submissions = 0

    def submit(self, payload, callback):
        self.submissions += 1
        future = Future()
        response = next(self.responses)
        if isinstance(response, Exception):
            future.set_exception(response)
        else:
            future.set_result((payload.id, response))
        callback(future)


def _judge_executor(responses):
    worker = ImmediateJudgeWorker(responses)
    executor = ToolsEvalExecutor.__new__(ToolsEvalExecutor)
    executor.worker = worker
    executor.lock = threading.Lock()
    executor._judge_done = {}
    executor._fatal_error = None
    answer = ValidExecutionReq(0, 0, [], "answer")
    stage_input = SimpleNamespace(
        instruction=SimpleNamespace(get_content=lambda: "question")
    )
    context = SimpleNamespace(
        episode=SimpleNamespace(episode_id="episode_1"),
        saved_data=SimpleNamespace(
            stage_id=0,
            attributes={},
            tool_calls=0,
            forgiven_tool_calls=0,
        ),
        task_ref=SimpleNamespace(
            get_stage_rule=lambda _stage_id: "rule",
            get_stage_input=lambda _stage_id: stage_input,
        ),
        judge_pending=False,
        judge_attempt_count=0,
        get_answer=lambda: answer,
    )
    executor._envs = {0: context}
    return executor, context, worker


def test_judge_valid_negative_verdict_does_not_retry():
    response = json.dumps({"verdict": False, "reason": "incorrect"})
    executor, context, worker = _judge_executor([response])

    executor._submit_judge(0)

    assert worker.submissions == 1
    assert context.judge_attempt_count == 1
    assert executor._judge_done[0].stage_success == StageSuccess.FAILED


@pytest.mark.parametrize(
    "response",
    ["not-json", RuntimeError("judge unavailable")],
)
def test_judge_stops_after_six_invalid_attempts(response):
    executor, context, worker = _judge_executor([response] * 6)

    executor._submit_judge(0)

    assert worker.submissions == 6
    assert context.judge_pending is False
    assert str(executor._fatal_error) == (
        "Impossible to obtain a valid judge answer for episode episode_1 "
        "after 6 attempts."
    )


def test_skip_judge_queues_a_success_without_worker():
    executor, context, _ = _judge_executor([])
    executor.worker = None
    executor.skip_judge = True
    executor.env = SimpleNamespace(
        unwrapped=SimpleNamespace(get_obs=lambda: {})
    )
    context.task_ref.is_stage_text_only = lambda _stage_id: True

    executor.compute_actions({0: context.get_answer()})

    assert context.judge_pending is True
    assert executor._judge_done[0].stage_success == StageSuccess.FINISH
