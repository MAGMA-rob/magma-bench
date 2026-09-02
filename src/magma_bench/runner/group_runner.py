from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional

from magma_core.base.agents import ValidAgentAnswer
from magma_core.base.data_structures import (
    EmptyInstruction,
    StageState,
    StageSuccess,
    StatusReturn,
    ToolStatus,
    ToolErrorFlag,
    ValidExecutionReq,
)
from magma_core.base.skills import (
    SkillDeferredInputResult,
    SkillExecutionResult,
    SkillManager,
    SkillRegistration,
    SkillStateRef,
    SkillStatusResult,
)
from magma_core.base.skills.structure import Tick

from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeData,
    EpisodeSituation,
    RunningState,
    Scenario,
)
from magma_bench.executor import ToolsEvalExecutor
from magma_bench.loader import EpisodeGroup
from magma_bench.results.models import EpisodeOutcome, EpisodeTerminal
from magma_bench.video import EpisodeVideoRecorder, VideoConfig


@dataclass(frozen=True)
class BenchmarkTick:
    to_agents: Dict[int, EpisodeSituation]
    to_executor: Dict[int, ValidExecutionReq]

    def has_inputs_for_agents(self) -> bool:
        return bool(self.to_agents)

    def has_call_for_executor(self) -> bool:
        return bool(self.to_executor)


class GroupRunner:
    """Orchestrate agents, skills and simulator slots for one episode group."""

    def __init__(
        self,
        scenario: Scenario,
        group: EpisodeGroup,
        executor_ref: ToolsEvalExecutor,
        on_episode_finished: Callable[[EpisodeOutcome], None],
        sim_backend: str = "auto",
        seed: int = 0,
        video_recorder: Optional[EpisodeVideoRecorder] = None,
        on_episode_started: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.scenario = scenario
        self.group = group
        self.executor_ref = executor_ref
        self.on_episode_finished = on_episode_finished
        self.on_episode_started = on_episode_started
        self.video_recorder = video_recorder or EpisodeVideoRecorder(
            VideoConfig(enabled=False)
        )
        self.episode_data_per_env: Dict[int, EpisodeData] = {}
        self.skill_managers: Dict[int, SkillManager] = {}
        self.skill_state_refs: Dict[int, SkillStateRef] = {}
        self.pending_agent_inputs: Dict[int, EpisodeSituation] = {}
        self.group_exhausted = False

        executor_ref.initialize_group(
            scenario,
            group,
            sim_backend=sim_backend,
            seed=seed,
        )
        if self.video_recorder.config.enabled:
            self.video_recorder.bind_environment(executor_ref.env)
        for _ in range(executor_ref.nb_env):
            if not self._register_next_episode():
                break

    def tick(
        self,
        status_from_env: Dict[int, ToolStatus],
        results_from_agent: List[BenchmarkAgentResult],
    ) -> BenchmarkTick:
        to_executor: Dict[int, ValidExecutionReq] = {}

        for env_idx, status in status_from_env.items():
            if env_idx not in self.episode_data_per_env:
                raise KeyError(f"No running episode uses environment {env_idx}")
            if self._handle_terminal_status(env_idx, status):
                continue
            skill_tick = self.skill_managers[env_idx].tick({env_idx: status})
            self._consume_skill_tick(skill_tick, to_executor)

        for result in results_from_agent:
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
            answer_payload = {
                "valid": answer.is_valid(),
                **answer.to_dict(),
            }
            stage_index = self.executor_ref.get_skill_execution_context(
                env_idx,
                episode_data.situation.attributes,
            ).stage_id
            answer_event = episode_data.record_trace(
                stage_index,
                "agent_answer",
                answer_payload,
            )
            if answer.is_valid() and answer_payload.get("action"):
                episode_data.last_action_event_index = answer_event.index
            if not answer.is_valid():
                self._finish_episode(
                    env_idx,
                    success=False,
                    terminal_status="invalid_agent_answer",
                    reason=answer.to_string(),
                )
                continue
            if not isinstance(answer, ValidAgentAnswer):
                raise TypeError(
                    "A valid benchmark answer must be a ValidAgentAnswer"
                )

            internal_answer = answer
            if answer.get_say() == "" and not answer.get_action():
                manager = self.skill_managers[env_idx]
                if manager.get_suspended_answer(
                    self.skill_state_refs[env_idx]
                ) is None:
                    self._finish_episode(
                        env_idx,
                        success=False,
                        terminal_status="protocol_failure",
                        reason="The agent returned neither a message nor a tool call.",
                    )
                    continue

            episode_data.state = RunningState.RUNNING
            context = self.executor_ref.get_skill_execution_context(
                env_idx,
                episode_data.situation.attributes,
            )
            manager = self.skill_managers[env_idx]
            manager.register(
                {
                    env_idx: SkillRegistration(
                        answer=internal_answer,
                        execution_context=context,
                        state_ref=self.skill_state_refs[env_idx],
                    )
                }
            )
            self._consume_skill_tick(manager.tick({}), to_executor)

        to_agents = {
            env_idx: situation.snapshot()
            for env_idx, situation in self.pending_agent_inputs.items()
        }
        self.pending_agent_inputs.clear()
        return BenchmarkTick(to_agents=to_agents, to_executor=to_executor)

    def is_done(self) -> bool:
        return self.group_exhausted and not self.episode_data_per_env

    def _register_next_episode(self) -> bool:
        episode = self.group.get_episode()
        if episode is None:
            self.group_exhausted = True
            return False

        episode_data = self.executor_ref.register(episode)
        env_idx = episode_data.env_idx
        manager = SkillManager(list(self.scenario.skill_types))
        manager.build_api(self.executor_ref.get_skill_api_provider(env_idx))
        state_ref = manager.initialize_robot_statuses(
            episode_data.situation.attributes
        )
        episode_data.situation.tools = manager.get_api()
        self.episode_data_per_env[env_idx] = episode_data
        self.skill_managers[env_idx] = manager
        self.skill_state_refs[env_idx] = state_ref
        if self.on_episode_started is not None:
            self.on_episode_started(episode_data.episode_id)
        stage_index = self.executor_ref.get_skill_execution_context(
            env_idx,
            episode_data.situation.attributes,
        ).stage_id
        instruction = episode_data.situation.current_instruction
        self.video_recorder.start_episode(
            env_idx,
            episode_data.episode_id,
            stage_index,
            instruction.get_role(),
            instruction.get_content(),
        )
        self._queue_agent_input(env_idx, "instruction")
        return True

    def _queue_agent_input(
        self,
        env_idx: int,
        trace_kind: Literal["instruction", "tool_feedback"],
    ) -> None:
        episode_data = self.episode_data_per_env[env_idx]
        episode_data.state = RunningState.WAITING_MODEL_ANSWER
        instruction = episode_data.situation.current_instruction
        stage_index = self.executor_ref.get_skill_execution_context(
            env_idx,
            episode_data.situation.attributes,
        ).stage_id
        episode_data.record_trace(
            stage_index,
            trace_kind,
            {
                "role": instruction.get_role(),
                "content": instruction.get_content(),
            },
        )
        video_content = (
            instruction.to_string()
            if isinstance(instruction, StatusReturn)
            else instruction.get_content()
        )
        self.video_recorder.update_instruction(
            env_idx,
            stage_index,
            instruction.get_role(),
            video_content,
        )
        self.pending_agent_inputs[env_idx] = episode_data.situation

    def _finish_episode(
        self,
        env_idx: int,
        success: bool,
        terminal_status: str,
        reason: Optional[str],
    ) -> None:
        episode_data = self.episode_data_per_env[env_idx]
        outcome = EpisodeOutcome(
            episode_id=episode_data.episode_id,
            success=success,
            terminal=EpisodeTerminal(
                status=terminal_status,
                reason=reason,
                last_action_event_index=episode_data.last_action_event_index,
            ),
            trace=list(episode_data.trace),
        )
        self.video_recorder.finish_episode(env_idx, terminal_status)
        self.on_episode_finished(outcome)
        self.pending_agent_inputs.pop(env_idx, None)
        manager = self.skill_managers.pop(env_idx)
        manager.reset_states()
        self.skill_state_refs.pop(env_idx)
        self.executor_ref.release_idx(env_idx)
        self.episode_data_per_env.pop(env_idx)
        self._register_next_episode()

    def _handle_terminal_status(self, env_idx: int, status: ToolStatus) -> bool:
        if status.stage_success == StageSuccess.FAILED:
            if status.failure_diagnostics is not None:
                self.episode_data_per_env[env_idx].record_trace(
                    status.stage_id,
                    "failure_diagnostics",
                    status.failure_diagnostics,
                )
            terminal_status = self._failure_terminal_status(status)
            self._finish_episode(
                env_idx,
                success=False,
                terminal_status=terminal_status,
                reason=status.failure_reason or "The stage failed.",
            )
            return True
        if status.stage_state == StageState.EXCEEDED:
            self._finish_episode(
                env_idx,
                success=False,
                terminal_status="budget_exceeded",
                reason=status.failure_reason or "The stage tool-call budget was exceeded.",
            )
            return True
        if status.stage_success == StageSuccess.FINISH and status.next_input is None:
            self._finish_episode(
                env_idx,
                success=True,
                terminal_status="success",
                reason=None,
            )
            return True
        return False

    @staticmethod
    def _failure_terminal_status(status: ToolStatus) -> str:
        if any(
            robot_status.error_flag == ToolErrorFlag.PLANNER_ERROR
            for robot_status in status.robots_status
        ):
            return "infrastructure_failure"
        reason = (status.failure_reason or "").lower()
        if (
            "requires an action" in reason
            or "requires a user-facing response" in reason
            or "neither a message nor a tool call" in reason
        ):
            return "protocol_failure"
        return "stage_failure"

    def _consume_skill_tick(
        self,
        skill_tick: Tick,
        to_executor: Dict[int, ValidExecutionReq],
    ) -> None:
        for env_idx, result in skill_tick.results.items():
            if env_idx not in self.episode_data_per_env:
                raise KeyError(f"No running episode uses environment {env_idx}")
            if result.environment_source_node_id != env_idx:
                raise RuntimeError(
                    "Skill result source does not match its environment: "
                    f"{result.environment_source_node_id} != {env_idx}"
                )

            episode_data = self.episode_data_per_env[env_idx]
            if isinstance(result, SkillExecutionResult):
                if env_idx in to_executor:
                    raise RuntimeError(
                        f"Two executor requests were produced for environment {env_idx}"
                    )
                if result.executed_answer is not None:
                    episode_data.last_agent_answer = result.executed_answer
                executed_answer = result.executed_answer or result.request
                self.video_recorder.update_answer(
                    env_idx,
                    executed_answer,
                    hold=bool(executed_answer.get_say()),
                )
                to_executor[env_idx] = result.request
                continue

            if isinstance(result, SkillStatusResult):
                event = result.status_event
                self.skill_state_refs[env_idx] = event.state_ref
                if event.executed_answer is not None:
                    episode_data.last_agent_answer = event.executed_answer
                    self.video_recorder.update_answer(
                        env_idx,
                        event.executed_answer,
                        hold=False,
                    )
                if self._handle_terminal_status(env_idx, event.status):
                    continue
                self._return_status_to_agent(env_idx, event.status)
                continue

            if isinstance(result, SkillDeferredInputResult):
                transition = result.deferred_input
                self.skill_state_refs[env_idx] = transition.state_ref
                episode_data.situation.attributes = dict(transition.attributes)
                episode_data.situation.set_current_instruction(
                    transition.stage_input.instruction
                )
                self._queue_agent_input(env_idx, "instruction")
                continue

            raise TypeError(f"Unsupported skill result {type(result).__name__}")

    def _return_status_to_agent(self, env_idx: int, status: ToolStatus) -> None:
        episode_data = self.episode_data_per_env[env_idx]
        episode_data.situation.attributes = dict(status.attributes)
        previous_answer = episode_data.last_agent_answer
        if previous_answer is None:
            raise RuntimeError(
                f"Environment {env_idx} has no preceding agent answer for its status"
            )

        next_input = status.next_input
        if (
            status.stage_success == StageSuccess.FINISH
            and next_input is not None
            and not isinstance(next_input.instruction, EmptyInstruction)
        ):
            instruction = next_input.instruction
            trace_kind = "instruction"
        else:
            status_payload = status.build_status_return(previous_answer)
            if status_payload is None:
                status_payload = {
                    "infos": "The action has completed.",
                    "previous_tool_call": previous_answer.to_dict().get("action", {}),
                }
            elif "infos" not in status_payload and "error" not in status_payload:
                status_payload["error"] = status_payload.pop(
                    "failure_reason",
                    "The action failed.",
                )
            instruction = StatusReturn(status_payload)
            trace_kind = "tool_feedback"

        episode_data.situation.set_current_instruction(instruction)
        self._queue_agent_input(env_idx, trace_kind)
