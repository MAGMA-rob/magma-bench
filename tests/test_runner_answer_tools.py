from concurrent.futures import Future
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest


pytest.importorskip("sapien")

from magma_core.simulation.agents import BadAgentAnswer, ValidAgentAnswer
from magma_core.domain import Call, ValidExecutionReq
from magma_core.simulation.data_structures import (
    RobotToolStatus,
    StageInput,
    StageSuccess,
    ToolErrorFlag,
    ToolStatus,
    UserInstruction,
)
from magma_core.simulation.stage import AskingBaseStage
from magma_core.simulation.skills import (
    DeferredInputTransition,
    SkillDeferredInputResult,
    SkillExecutionResult,
    SkillStateRef,
    SkillStatusEvent,
    SkillStatusResult,
)
from magma_core.simulation.skills.structure import Tick

from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeData,
    EpisodeSituation,
    RunningState,
)
from magma_bench.artifacts import (
    DeclarativeStageSpec,
    InstructionSpec,
    StageInputSpec,
    StagePresentationSpec,
    deserialize_stage,
    serialize_stage,
)
from magma_bench.runner import group_runner as group_runner_module
from magma_bench.runner.group_runner import GroupRunner
from magma_bench.executor.eval_executor import ToolsEvalExecutor
from magma_bench.executor.context import PlannerRetryEvent
from magma_bench.video import EpisodeVideoRecorder, VideoConfig


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


class NonPhysicalLoopSkillManager(FakeSkillManager):
    def __init__(self, skill_types):
        super().__init__(skill_types)
        self.next_state_ref = 2

    def tick(self, _statuses):
        if not self.registrations:
            return Tick({})
        env_idx, registration = self.registrations.popitem()
        call = registration.answer.get_action()[0]
        if call.name == "physical_tool":
            return Tick(
                {
                    env_idx: SkillExecutionResult(
                        env_idx,
                        ValidExecutionReq(
                            env_idx,
                            registration.answer.agent_step_id,
                            [call],
                            "",
                        ),
                        registration.answer,
                    )
                }
            )
        status = ToolStatus(
            robots_status=[
                RobotToolStatus(
                    call.target_robot_name,
                    "The control call did not launch a physical tool.",
                    False,
                    ToolErrorFlag.BAD_CALL,
                )
            ],
            stage_id=0,
            attributes={"known_robots": ["robot"]},
            error_descriptions=[""],
            stage_success=StageSuccess.ONGOING,
        )
        state_ref = SkillStateRef(self.next_state_ref)
        self.next_state_ref += 1
        return Tick(
            {
                env_idx: SkillStatusResult(
                    env_idx,
                    SkillStatusEvent(status, state_ref, registration.answer),
                )
            }
        )


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


def test_group_runner_stops_after_three_non_physical_agent_turns(monkeypatch):
    monkeypatch.setattr(
        group_runner_module,
        "SkillManager",
        NonPhysicalLoopSkillManager,
    )
    outcomes = []
    runner = GroupRunner(
        scenario=SimpleNamespace(skill_types=()),
        group=FakeEpisodeGroup(),
        executor_ref=FakeExecutor(),
        on_episode_finished=outcomes.append,
    )
    runner.tick({}, [])

    for step in range(3):
        answer = ValidAgentAnswer(
            0,
            step,
            "",
            [Call("control_only", {}, "robot")],
        )
        tick = runner.tick(
            {},
            [
                BenchmarkAgentResult(
                    answer=answer,
                    situation=runner.episode_data_per_env[0].situation,
                )
            ],
        )
        assert tick.to_executor == {}

    assert runner.is_done()
    assert len(outcomes) == 1
    assert outcomes[0].terminal.status == "protocol_failure"
    assert outcomes[0].terminal.reason == (
        "The agent produced 3 consecutive answers without launching a physical "
        "tool call."
    )


def test_physical_tool_call_resets_non_physical_turn_limit(monkeypatch):
    monkeypatch.setattr(
        group_runner_module,
        "SkillManager",
        NonPhysicalLoopSkillManager,
    )
    outcomes = []
    runner = GroupRunner(
        scenario=SimpleNamespace(skill_types=()),
        group=FakeEpisodeGroup(),
        executor_ref=FakeExecutor(),
        on_episode_finished=outcomes.append,
    )
    runner.tick({}, [])

    for step in range(2):
        answer = ValidAgentAnswer(
            0,
            step,
            "",
            [Call("control_only", {}, "robot")],
        )
        runner.tick(
            {},
            [
                BenchmarkAgentResult(
                    answer=answer,
                    situation=runner.episode_data_per_env[0].situation,
                )
            ],
        )

    physical_answer = ValidAgentAnswer(
        0,
        2,
        "",
        [Call("physical_tool", {}, "robot")],
    )
    tick = runner.tick(
        {},
        [
            BenchmarkAgentResult(
                answer=physical_answer,
                situation=runner.episode_data_per_env[0].situation,
            )
        ],
    )

    assert list(tick.to_executor) == [0]
    assert tick.to_executor[0].calls == physical_answer.get_action()
    assert runner.consecutive_non_physical_agent_turns[0] == 0
    assert outcomes == []


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
        tool_context=None,
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


def test_stage_artifacts_preserve_allowed_tools():
    declarative_spec = DeclarativeStageSpec(
        id="answer",
        type="answer",
        presentation=StagePresentationSpec(
            stage_input=StageInputSpec(
                instruction=InstructionSpec.model_validate(
                    UserInstruction("Inspect before answering.").to_spec()
                ),
            ),
            verification_prompt="Verify the answer.",
            stage_goal_description="Answer after inspection.",
        ),
        target_tool_calls=1,
        max_tool_calls=3,
        allow_tools_before_answer=True,
        allowed_tools=["detect"],
    )
    declarative_stage = deserialize_stage(declarative_spec)

    serialized_spec = serialize_stage(
        AskingBaseStage(
            question="What is visible?",
            answer="a cube",
            allow_tools_before_answer=True,
            allowed_tools=["detect"],
        ),
        "serialized_answer",
    )
    serialized_stage = deserialize_stage(serialized_spec)

    assert declarative_stage.get_allowed_tools() == ["detect"]
    assert serialized_stage.get_allowed_tools() == ["detect"]


def test_planner_terminal_retry_keeps_a_runtime_event():
    executor = ToolsEvalExecutor.__new__(ToolsEvalExecutor)
    executor._planner_runtime_events = []
    executor.trajectory_converter = SimpleNamespace(agents_name=["robot"])
    executor.env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            get_state_dict=lambda: {},
            set_state_dict=lambda _state: None,
        )
    )
    context = SimpleNamespace(
        episode=SimpleNamespace(episode_id="episode_0"),
        planner_retry_count=ToolsEvalExecutor.MAX_PLANNER_RETRIES,
        saved_data=SimpleNamespace(
            stage_id=2,
            env_state={},
            attributes={},
            tool_calls=3,
            forgiven_tool_calls=0,
        ),
        randomizer=SimpleNamespace(traduce_end=lambda statuses: statuses),
    )
    executor._envs = {0: context}

    failed = executor._handle_tool_retry({
        0: ("planner blocked by collision", ()),
    })
    events = executor.drain_planner_runtime_events()

    assert failed[0].stage_success == StageSuccess.FAILED
    assert events == [
        PlannerRetryEvent(
            env_idx=0,
            episode_id="episode_0",
            attempt=ToolsEvalExecutor.MAX_PLANNER_RETRIES,
            max_attempts=ToolsEvalExecutor.MAX_PLANNER_RETRIES,
            message="planner blocked by collision",
        )
    ]


class FakeRenderedEnvironment:
    def __init__(self, frames):
        self.frames = frames
        self.render_calls = 0

    def render_rgb_array(self):
        self.render_calls += 1
        return self.frames


class FakeVideoWriter:
    def __init__(self, path, written_frames):
        self.path = Path(path)
        self.written_frames = written_frames
        self.seeded = False

    def send(self, frame):
        if frame is None:
            self.seeded = True
            return
        assert self.seeded
        self.written_frames.append(frame.copy())

    def close(self):
        self.path.write_bytes(b"fake-video")


def _video_recorder(monkeypatch, tmp_path, frames, fps=2, hold_seconds=0.5):
    writers = []

    def write_frames(path, _size, **_kwargs):
        written_frames = []
        writers.append(written_frames)
        return FakeVideoWriter(path, written_frames)

    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        SimpleNamespace(write_frames=write_frames),
    )
    recorder = EpisodeVideoRecorder(
        VideoConfig(enabled=True, fps=fps, hold_seconds=hold_seconds)
    )
    video_directory = tmp_path / "videos"
    recorder.start_scenario(video_directory)
    environment = FakeRenderedEnvironment(frames)
    recorder.bind_environment(environment)
    return recorder, environment, writers, video_directory


def test_video_recorder_splits_vector_frames_and_recycles_slot(
    monkeypatch,
    tmp_path,
):
    numpy = pytest.importorskip("numpy")
    frames = numpy.zeros((2, 32, 48, 3), dtype=numpy.uint8)
    frames[0, ..., 0] = 40
    frames[1, ..., 1] = 80
    recorder, environment, writers, video_directory = _video_recorder(
        monkeypatch,
        tmp_path,
        frames,
    )
    recorder.start_episode(0, "episode_0", 0, "USER", "first")
    recorder.start_episode(1, "episode_1", 1, "SYSTEM", "second")
    recorder.update_instruction(0, 0, "USER", "first")
    recorder.update_instruction(1, 1, "SYSTEM", "second")

    recorder.flush_holds()

    assert environment.render_calls == 1
    assert len(writers) == 2
    assert tuple(writers[0][0][10, 10]) == (40, 0, 0)
    assert tuple(writers[1][0][10, 10]) == (0, 80, 0)
    assert writers[0][0].shape == (272, 48, 3)

    recorder.finish_episode(0, "success")
    recorder.start_episode(0, "episode_2", 2, "USER", "recycled")
    recorder.update_instruction(0, 2, "USER", "recycled")
    recorder.flush_holds([0])
    recorder.finish_episode(0, "stage_failure")
    recorder.finish_episode(1, "success")

    assert (video_directory / "episode_0.mp4").is_file()
    assert (video_directory / "episode_1.mp4").is_file()
    assert (video_directory / "episode_2.mp4").is_file()
    manifest = json.loads(
        (video_directory / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["episodes"]["episode_2"]["terminal_status"] == "stage_failure"


def test_video_recorder_holds_context_and_draws_planner_banner(
    monkeypatch,
    tmp_path,
):
    numpy = pytest.importorskip("numpy")
    frames = numpy.zeros((1, 32, 64, 3), dtype=numpy.uint8)
    recorder, _, writers, _ = _video_recorder(
        monkeypatch,
        tmp_path,
        frames,
        fps=4,
        hold_seconds=1.0,
    )
    recorder.start_episode(0, "episode", 3, "USER", "instruction")
    recorder.update_instruction(0, 3, "USER", "instruction")
    recorder.flush_holds()
    assert len(writers[0]) == 4

    say = ValidAgentAnswer(0, 1, "Waiting for confirmation", [])
    recorder.update_answer(0, say, hold=True)
    recorder.flush_holds()
    assert len(writers[0]) == 8
    assert recorder._sessions[0].activity_label == "Current answer"

    cancel = ValidAgentAnswer(
        0,
        2,
        "",
        [Call("cancel_current_action", {}, "robot")],
    )
    recorder.update_answer(0, cancel, hold=True)
    previous_activity = recorder._sessions[0].activity
    recorder.update_instruction(
        0,
        3,
        "USER",
        "Interruption: stop and inspect the tray",
    )
    recorder.flush_holds()
    assert len(writers[0]) == 16
    assert recorder._sessions[0].activity == previous_activity

    recorder.set_planner_retry(0, 4, 10, "path unavailable")
    recorder.flush_holds()

    assert len(writers[0]) == 20
    assert tuple(writers[0][-1][5, 5]) == (190, 20, 20)
    recorder.clear_planner_retry(0)
    recorder.capture_physical_step([0])
    assert tuple(writers[0][-1][5, 5]) == (0, 0, 0)
    recorder.finish_episode(0, "infrastructure_failure")


def test_disabled_video_recorder_never_renders():
    recorder = EpisodeVideoRecorder(VideoConfig(enabled=False))
    environment = FakeRenderedEnvironment(None)

    recorder.bind_environment(environment)
    recorder.start_episode(0, "episode", 0, "USER", "instruction")
    recorder.capture_physical_step([0])
    recorder.flush_holds()
    recorder.finish_episode(0, "success")

    assert environment.render_calls == 0


class FailingVideoRecorder:
    config = VideoConfig(enabled=False)

    def start_episode(self, *_args, **_kwargs):
        pass

    def update_instruction(self, *_args, **_kwargs):
        pass

    def update_answer(self, *_args, **_kwargs):
        pass

    def finish_episode(self, _env_idx, _terminal_status):
        raise RuntimeError("video encoding failed")


def test_video_failure_prevents_episode_outcome_persistence(monkeypatch):
    monkeypatch.setattr(group_runner_module, "SkillManager", FakeSkillManager)
    outcomes = []
    runner = GroupRunner(
        scenario=SimpleNamespace(skill_types=()),
        group=FakeEpisodeGroup(),
        executor_ref=FakeExecutor(),
        on_episode_finished=outcomes.append,
        video_recorder=FailingVideoRecorder(),
    )
    runner.tick({}, [])
    answer = BadAgentAnswer(0, 0, "", "invalid", "invalid JSON")

    with pytest.raises(RuntimeError, match="video encoding failed"):
        runner.tick(
            {},
            [
                BenchmarkAgentResult(
                    answer=answer,
                    situation=runner.episode_data_per_env[0].situation,
                )
            ],
        )

    assert outcomes == []
