# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""Vectorized benchmark executor with one task runtime per active episode."""

from collections import OrderedDict
from concurrent.futures import Future
import copy
import json
import logging
import threading
from typing import Dict, List, Optional, Tuple, Union

import torch

from magma_core.base.data_structures import (
    EnvToolContext,
    RobotToolStatus,
    SavedEnvData,
    StageSuccess,
    ToolErrorFlag,
    ToolStatus,
    ValidExecutionReq,
)
from magma_core.base.data_structures.env import build_text_only_action_failure_status
from magma_core.base.envs import DefaultEnv
from magma_core.base.executor import ToolsBaseExecutor
from magma_core.base.skills import SkillExecutionContext
from magma_core.base.skills.skill_manager import SkillAPIProvider
from magma_core.protocol.payload.user_sim_payload import JudgePayload
from magma_core.serialization import decode_value
from magma_core.utils.global_utils import (
    apply_env_state_updates,
    batch_set_value,
    extract_env_state_val,
    merge_robot_articulations,
    restore_disallowed_actor_states,
)
from magma_core.workers import LMWorker

from magma_bench.artifacts import deserialize_runtime_randomizer
from magma_bench.data_structures import (
    Episode,
    EpisodeData,
    EpisodeSituation,
    RunningState,
    Scenario,
)
from magma_bench.loader import EpisodeGroup

from .context import (
    EvalEpisodeContext,
    EvalSkillAPIProvider,
    PlannerRetryEvent,
    PlannerRetryResolvedEvent,
    PlannerRuntimeEvent,
)
from .episode_runtime import build_episode_task


LOGGER = logging.getLogger(__name__)


class ToolsEvalExecutor(ToolsBaseExecutor):
    """Execute independent benchmark episodes in shared simulator slots."""

    MAX_JUDGE_ATTEMPTS = 6

    _envs : Dict[int, EvalEpisodeContext]


    def __init__(
            self,
            planner_endpoint: str,
            worker: Optional[LMWorker],
            nb_env: int = 1,
            skip_judge: bool = False,
        ) -> None:
        super().__init__(
            nb_env,
            planner_endpoint=planner_endpoint,
            ollama_worker=worker,
        )
        self._envs = {}
        self._free_idx = list(range(nb_env))
        self._scenario: Optional[Scenario] = None
        self._group: Optional[EpisodeGroup] = None
        self._group_base_state: Optional[Dict] = None
        self._judge_done: Dict[int, ToolStatus] = {}
        self._immediate_done: Dict[int, ToolStatus] = {}
        self._fatal_error: Optional[RuntimeError] = None
        self._planner_runtime_events: List[PlannerRuntimeEvent] = []
        self.skip_judge = skip_judge
        self.lock = threading.Lock()

    def initialize_group(
            self,
            scenario: Scenario,
            group: EpisodeGroup,
            obs_mode: str = "state_dict",
            sim_backend: str = "auto",
            seed: int = 0,
        ) -> DefaultEnv:
        """Initialize the simulator configuration shared by an episode group."""

        if self._envs:
            raise RuntimeError(
                "Cannot initialize a new episode group while slots are active"
            )

        self._scenario = scenario
        self._group = group
        self._free_idx = list(range(self.nb_env))
        with self.lock:
            self._judge_done.clear()
            self._immediate_done.clear()
            self._fatal_error = None
            self._planner_runtime_events.clear()

        self.env = self._create_envs(
            scenario.environment_id,
            group.env_options,
            group.planner_options,
            obs_mode,
            sim_backend,
        )
        reset_options = copy.deepcopy(group.env_options)
        reset_options["reconfigure"] = False
        self.env.reset(seed=seed, options=reset_options)
        action = self.step()
        self.env.step(action)
        self._group_base_state = copy.deepcopy(self.env.unwrapped.get_state_dict())
        return self.env

    def register(self, episode: Episode) -> EpisodeData:
        """Create and bind one episode runtime to the next free environment."""

        if self._scenario is None or self._group is None:
            raise RuntimeError("initialize_group must be called before register")
        if self._group_base_state is None:
            raise RuntimeError("The group base state has not been initialized")
        if not self._free_idx:
            raise RuntimeError(
                "Trying to register an episode while no free env idx is left"
            )

        env_idx = self._free_idx.pop(0)
        try:
            task_ref = build_episode_task(self._scenario, episode)
            randomizer = deserialize_runtime_randomizer(episode.semantic)

            # set the env state to the group default one
            env_state = copy.deepcopy(self.env.unwrapped.get_state_dict())
            base_slot_state = extract_env_state_val(
                self._group_base_state,
                env_idx,
            )
            batch_set_value(
                env_state,
                torch.tensor([env_idx]),
                base_slot_state,
            )

            initialized_state = task_ref.initialize_task(
                self.trajectory_converter.agents,
                self.trajectory_converter.agents_name,
                env_state=env_state,
                nb_env=self.nb_env,
                build_first_stage=True,
                env_ids=[env_idx],
            )
            self.env.unwrapped.set_state_dict(initialized_state)

            saved_data = SavedEnvData(
                env_state=extract_env_state_val(initialized_state, env_idx),
                logs=[],
                stage_id=0,
                stage_log_length=0,
                attributes=copy.deepcopy(task_ref.get_init_attributes()),
            )
            context = EvalEpisodeContext(
                episode=episode,
                task_ref=task_ref,
                randomizer=randomizer,
                saved_data=saved_data,
            )
            self._envs[env_idx] = context

            initial_situation = randomizer.get_randomized_situation(
                task_ref.get_init_situation()
            )
            public_history = json.loads(
                randomizer.traduce_attributes_to_llm(
                    json.dumps(initial_situation.history)
                )
            )
            public_attributes = decode_value(
                copy.deepcopy(episode.semantic.visible_attributes)
            )
            if not isinstance(public_attributes, dict):
                raise TypeError("Decoded semantic attributes must be a dictionary")
            public_attributes["known_robots"] = list(task_ref.get_agent_names())
            situation = EpisodeSituation(
                tools=copy.deepcopy(randomizer.get_tools()),
                attributes=public_attributes,
                memory=initial_situation.memory,
                current_instruction=task_ref.get_stage_input(0).instruction,
                history=public_history,
            )
            return EpisodeData(
                episode_id=episode.episode_id,
                state=RunningState.WAITING_MODEL_ANSWER,
                situation=situation,
                env_idx=env_idx,
            )
        except Exception:
            self._free_idx.append(env_idx)
            self._free_idx.sort()
            raise

    def release_idx(self, env_idx: int) -> None:
        """Release a slot only after all executor work has completed."""

        context = self._envs.get(env_idx)
        if context is None:
            raise RuntimeError("Asking to release an env_idx that is not used")
        
        if context.tool_context is not None:
            raise RuntimeError(
                f"Cannot release environment {env_idx}: a tool is still active"
            )
        if context.judge_pending:
            raise RuntimeError(
                f"Cannot release environment {env_idx}: a judge is still active"
            )

        self._envs.pop(env_idx)
        with self.lock:
            self._judge_done.pop(env_idx, None)
            self._immediate_done.pop(env_idx, None)
        self._free_idx.append(env_idx)
        self._free_idx.sort()

    def compute_actions(
            self,
            tools_call: Dict[int, ValidExecutionReq],
        ) -> None:
        """Transform agent answers into actions for their assigned slots."""

        obs = self.env.unwrapped.get_obs()
        for env_idx, answer in tools_call.items():
            context = self._envs.get(env_idx)
            if context is None:
                raise KeyError(f"No active episode in environment {env_idx}")
            if context.tool_context is not None:
                raise RuntimeError(
                    f"Episode {context.episode.episode_id!r} already executes a tool"
                )

            context.last_agent_answer = answer
            stage_id = context.saved_data.stage_id
            is_text_only = context.task_ref.is_stage_text_only(stage_id)
            if answer.get_say() != "":
                if not is_text_only:
                    self._store_local_failure(
                        env_idx,
                        "The agent returned a message while this stage requires an action.",
                    )
                    continue
                if self.skip_judge:
                    context.judge_pending = True
                    self._store_judge_verdict(
                        env_idx,
                        verdict=True,
                        failure_reason="",
                        judge_response=None,
                    )
                else:
                    self._submit_judge(env_idx)
                continue

            calls = answer.get_tool_calls()
            if not calls:
                self._store_local_failure(
                    env_idx,
                    "The agent returned neither a message nor a tool call.",
                )
                continue

            stage = context.task_ref.stages[stage_id]
            if is_text_only and not stage.allows_tools_before_answer():
                canonical = build_text_only_action_failure_status(
                    robot_names=[call.target_robot_name for call in calls],
                    stage_id=stage_id,
                    attributes=copy.deepcopy(context.saved_data.attributes),
                    tool_calls=context.saved_data.tool_calls,
                    forgiven_tool_calls=context.saved_data.forgiven_tool_calls,
                )
                with self.lock:
                    self._immediate_done[env_idx] = self._translate_status_to_public(
                        env_idx,
                        canonical,
                    )
                continue

            saved_data = context.saved_data
            tool_context = self._compute_tool(
                task_ref=context.task_ref,
                randomizer=context.randomizer,
                calls=calls,
                env_id=env_idx,
                obs=obs,
                error_state=saved_data.active_stage_error_state,
                stage_id=saved_data.stage_id,
                current_node_step=0,
                original_log_length=saved_data.stage_log_length,
                source_node_id=env_idx,
                attributes=saved_data.attributes,
                logs=saved_data.logs,
                previous_tool_calls=saved_data.tool_calls,
                previous_forgiven_tool_calls=saved_data.forgiven_tool_calls,
                composite_progress=saved_data.composite_progress,
                node_id=env_idx,
            )
            context.set_tool_context(tool_context)
            if not tool_context.is_full_error():
                self.trajectory_converter.transform_poses_in_actions(
                    tool_context,
                    env_idx,
                )

    def step(self) -> Union[torch.Tensor, OrderedDict]:
        """Return the next batched low-level action."""

        active_tools = {
            env_idx: context.tool_context
            for env_idx, context in self._envs.items()
            if context.tool_context is not None
        }
        return self.trajectory_converter.step(active_tools)

    def has_active_tools(self) -> bool:
        """Return whether a physical simulator step is currently required."""

        return any(
            context.tool_context is not None
            for context in self._envs.values()
        )

    def get_active_tool_env_ids(self) -> List[int]:
        """Return slots whose physical tool contributes video frames."""

        return [
            env_idx
            for env_idx, context in self._envs.items()
            if context.tool_context is not None
        ]

    def drain_planner_runtime_events(self) -> List[PlannerRuntimeEvent]:
        """Consume planner retry events produced on the main execution thread."""

        events = self._planner_runtime_events
        self._planner_runtime_events = []
        return events

    def get_skill_api_provider(self, env_idx: int) -> SkillAPIProvider:
        """Build the skill API provider for the episode occupying a slot."""

        context = self._envs.get(env_idx)
        if context is None:
            raise KeyError(f"No active episode in environment {env_idx}")
        tools = copy.deepcopy(context.randomizer.get_tools())
        if not isinstance(tools, list):
            raise TypeError("Decoded semantic tools must be a list")
        pairs = []
        for tool in tools:
            public_name = tool.get("name")
            mapping = context.randomizer.tools_equivalence.get(public_name)
            if not isinstance(mapping, dict) or not isinstance(mapping.get("name"), str):
                raise ValueError(
                    f"Missing canonical mapping for public tool {public_name!r}"
                )
            pairs.append((tool, mapping["name"]))
        return EvalSkillAPIProvider(tuple(pairs), context.randomizer)

    def get_skill_execution_context(
        self,
        env_idx: int,
        public_attributes: Dict,
    ) -> SkillExecutionContext:
        """Expose the current counters and public state to one skill manager."""

        context = self._envs.get(env_idx)
        if context is None:
            raise KeyError(f"No active episode in environment {env_idx}")
        return SkillExecutionContext(
            stage_id=context.saved_data.stage_id,
            attributes=copy.deepcopy(public_attributes),
            tool_calls=context.saved_data.tool_calls,
            forgiven_tool_calls=context.saved_data.forgiven_tool_calls,
            flag_answer_to_user=context.task_ref.get_stage_input(
                context.saved_data.stage_id
            ).flag_answer_to_user,
        )
    

    def verif_ended_tool(self, obs: Dict) -> Dict[int, ToolStatus]:
        """Verify completed tools and advance each episode independently."""

        with self.lock:
            fatal_error = self._fatal_error
        if fatal_error is not None:
            raise fatal_error

        out = self._consume_judge_results()
        with self.lock:
            out.update(self._immediate_done)
            self._immediate_done = {}
        completed: List[
            Tuple[int, EvalEpisodeContext, EnvToolContext, ToolStatus]
        ] = []
        retry_failures: Dict[int, str] = {}

        env_state = copy.deepcopy(self.env.unwrapped.get_state_dict())
        state_changed = False

        for env_idx, context in self._envs.items():
            tool_context = context.tool_context
            if tool_context is None:
                continue
            tool_context.compute_tool_results(obs)
            if not tool_context.all_finished():
                continue

            tool_status = tool_context.build_tool_status()
            if tool_status.should_restore_source_env_state():
                context.tool_context = None
                retry_messages = [
                    robot_status.mess
                    for robot_status in tool_status.robots_status
                    if robot_status.error_flag == ToolErrorFlag.PLANNER_ERROR
                    and robot_status.mess
                ]
                retry_failures[env_idx] = "\n".join(retry_messages)
                continue
            
            # Apply possible state changement (move_to, object state changement)
            state_changed |= apply_env_state_updates(
                env_state,
                env_idx,
                tool_context.get_state_updates(),
            )
            allowed_actors = tool_context.get_allowed_moving_actors()
            if allowed_actors is not None:
                restored_actors = restore_disallowed_actor_states(
                    env_state,
                    context.saved_data.env_state,
                    env_idx,
                    allowed_actors,
                )
                if restored_actors is None:
                    LOGGER.warning(
                        "Actor movement protection disabled due to an incompatible "
                        "state layout episode=%s env=%d allowed=%s",
                        context.episode.episode_id,
                        env_idx,
                        sorted(allowed_actors),
                    )
                elif restored_actors:
                    state_changed = True
                    LOGGER.info(
                        "Restored disallowed actors episode=%s env=%d actors=%s",
                        context.episode.episode_id,
                        env_idx,
                        restored_actors,
                    )
            completed.append((env_idx, context, tool_context, tool_status))

        if state_changed:
            self.env.unwrapped.set_state_dict(env_state)
            obs = self.env.unwrapped.get_obs()

        # MAIN VERIFICATION LOOP
        # Verify Goals and Logs on completed env (tool finished)
        for env_idx, context, tool_context, tool_status in completed:
            task_ref = context.task_ref
            saved_data = context.saved_data
            stage_id = saved_data.stage_id
            advanced_stage = False
            is_last_stage = stage_id == task_ref.get_nb_total_stage() - 1
            next_input = (
                None
                if is_last_stage
                else task_ref.get_stage_input(stage_id + 1)
            )

            if task_ref.is_stage_text_only(stage_id):
                tool_status.stage_state = task_ref.get_stage_state(
                    stage_id,
                    tool_status.tool_calls,
                    tool_status.forgiven_tool_calls,
                    tool_status.stage_success,
                    tool_context.logs,
                )
            elif tool_context.is_full_error():
                #All tools have fail (error in processing, bad robot name, full syntax errors)
                if tool_context.has_terminal_tool_failure():
                    tool_status.stage_success = StageSuccess.FAILED
                tool_status.stage_state = task_ref.get_stage_state(
                    stage_id,
                    tool_status.tool_calls,
                    tool_status.forgiven_tool_calls,
                    tool_status.stage_success,
                    tool_context.logs,
                )
            else:
                #valid execution
                env_score = task_ref.verif_stage_env_completion(
                    stage_id,
                    obs,
                    [env_idx],
                )[0].item()
                full_logs, stage_logs = tool_context._get_logs()
                log_score = task_ref.verif_stage_log_completion(
                    stage_id,
                    full_log=full_logs,
                    stage_log=stage_logs,
                    composite_progress=tool_context.composite_progress,
                )
                stage_score = task_ref.combine_stage_verif_scores(
                    stage_id,
                    env_score,
                    log_score,
                )
                tool_status.reward = stage_score
                if stage_score < 0:
                    tool_status.stage_success = StageSuccess.FAILED
                    tool_status.failure_reason = (
                        "Stage verification failed: "
                        f"env_score={env_score}, log_score={log_score}, "
                        f"combined_score={stage_score}."
                    )
                    goal_scores = []
                    for goal in task_ref.stages[stage_id].get_goals():
                        goal_score = goal.verify(obs)[env_idx].item()
                        goal_scores.append({
                            "name": goal.name,
                            "metadata": goal.metadata,
                            "spec": goal.to_spec(),
                            "score": goal_score,
                        })
                    tool_status.failure_diagnostics = {
                        "source": "stage_verification",
                        "stage_id": stage_id,
                        "stage_goal_description": (
                            task_ref.stages[stage_id].get_stage_goal_description()
                        ),
                        "env_score": env_score,
                        "log_score": log_score,
                        "combined_score": stage_score,
                        "tool_results": [
                            {
                                "robot_name": robot_status.robot_name,
                                "success": robot_status.result,
                                "message": robot_status.mess,
                                "error_flag": robot_status.error_flag.name,
                            }
                            for robot_status in tool_status.robots_status
                        ],
                        "goal_scores": goal_scores,
                        "stage_logs": [
                            {
                                "stage_id": log.stage_id,
                                "function": log.function,
                                "content": repr(log.content),
                                "action": log.action,
                            }
                            for log in stage_logs
                        ],
                        "full_logs": [
                            {
                                "stage_id": log.stage_id,
                                "function": log.function,
                                "content": repr(log.content),
                                "action": log.action,
                            }
                            for log in full_logs
                        ],
                    }
                elif stage_score > 0:
                    tool_status.stage_success = StageSuccess.FINISH
                    tool_status.next_input = next_input
                    if next_input is not None:
                        env_state = self._pass_to_the_next_stage(
                            task_ref,
                            stage_id,
                            [env_idx],
                            env_state,
                        )
                        state_changed = True
                        saved_data.stage_id += 1
                        saved_data.stage_log_length = len(tool_context.logs)
                        saved_data.active_stage_error_state = {}
                        saved_data.composite_progress = {}
                        advanced_stage = True
                elif task_ref.should_reset_same_stage(stage_id):
                    batch_set_value(
                        env_state,
                        torch.tensor([env_idx]),
                        task_ref._default_env_state,
                    )
                    state_changed = True
                    saved_data.stage_log_length = len(tool_context.logs)

                tool_status.stage_state = task_ref.get_stage_state(
                    stage_id,
                    tool_status.tool_calls,
                    tool_status.forgiven_tool_calls,
                    tool_status.stage_success,
                    full_logs,
                )

            saved_data.logs = tool_context.logs
            saved_data.attributes = copy.deepcopy(tool_context.attributes)
            if advanced_stage:
                saved_data.tool_calls = 0
                saved_data.forgiven_tool_calls = 0
                saved_data.active_stage_error_state = {}
                saved_data.composite_progress = {}
            else:
                saved_data.tool_calls = tool_context.tool_calls
                saved_data.forgiven_tool_calls = tool_context.forgiven_tool_calls
                saved_data.active_stage_error_state = tool_context.error_state
                saved_data.composite_progress = tool_context.composite_progress
            if context.planner_retry_count > 0:
                self._planner_runtime_events.append(
                    PlannerRetryResolvedEvent(env_idx=env_idx)
                )
            context.planner_retry_count = 0
            context.tool_context = None

            translated = self._translate_status_to_public(env_idx, tool_status)
            out[env_idx] = translated

        if state_changed:
            self.env.unwrapped.set_state_dict(env_state)
        for env_idx, context, _, _ in completed:
            context.saved_data.env_state = extract_env_state_val(
                env_state,
                env_idx,
            )

        if retry_failures:
            out.update(self._handle_tool_retry(retry_failures))

        return out

    def _handle_tool_retry(
        self,
        retry_failures: Dict[int, str],
    ) -> Dict[int, ToolStatus]:
        """Restore committed states and replay answers after planner failures."""

        env_state = copy.deepcopy(self.env.unwrapped.get_state_dict())
        answers: Dict[int, ValidExecutionReq] = {}
        failed: Dict[int, ToolStatus] = {}
        robot_names = self.trajectory_converter.agents_name

        for env_idx, planner_message in retry_failures.items():
            context = self._envs[env_idx]
            if context.planner_retry_count >= 10:
                self._planner_runtime_events.append(
                    PlannerRetryEvent(
                        env_idx=env_idx,
                        attempt=10,
                        max_attempts=10,
                        message=planner_message,
                    )
                )
                canonical = ToolStatus(
                    robots_status=[
                        RobotToolStatus(
                            "",
                            "Planner failed after 10 retries.",
                            False,
                            ToolErrorFlag.PLANNER_ERROR,
                        )
                    ],
                    stage_id=context.saved_data.stage_id,
                    attributes=copy.deepcopy(context.saved_data.attributes),
                    error_descriptions=[""],
                    failure_reason="Planner error exceeded 10 retries.",
                    stage_success=StageSuccess.FAILED,
                    tool_calls=context.saved_data.tool_calls,
                    forgiven_tool_calls=context.saved_data.forgiven_tool_calls,
                )
                failed[env_idx] = self._translate_status_to_public(
                    env_idx,
                    canonical,
                )
                continue
            context.planner_retry_count += 1
            self._planner_runtime_events.append(
                PlannerRetryEvent(
                    env_idx=env_idx,
                    attempt=context.planner_retry_count,
                    max_attempts=10,
                    message=planner_message,
                )
            )
            restored_state = copy.deepcopy(context.saved_data.env_state)
            current_state = extract_env_state_val(env_state, env_idx)

            if context.planner_retry_count % 2:
                source_articulations = current_state.get("articulations")
            else:
                source_articulations = context.task_ref._default_env_state.get(
                    "articulations"
                )
            if source_articulations is not None:
                restored_state["articulations"], _ = merge_robot_articulations(
                    restored_state.get("articulations", {}),
                    source_articulations,
                    robot_names,
                )

            batch_set_value(
                env_state,
                torch.tensor([env_idx]),
                restored_state,
                strict=False,
            )
            answers[env_idx] = context.get_answer()

        self.env.unwrapped.set_state_dict(env_state)
        if answers:
            self.compute_actions(answers)
        return failed

    def _store_local_failure(self, env_idx: int, reason: str) -> None:
        """Queue an episode-local protocol failure for the main runner."""

        context = self._envs[env_idx]
        status = ToolStatus(
            robots_status=[],
            stage_id=context.saved_data.stage_id,
            attributes=copy.deepcopy(context.saved_data.attributes),
            error_descriptions=[],
            failure_reason=reason,
            stage_success=StageSuccess.FAILED,
            tool_calls=context.saved_data.tool_calls,
            forgiven_tool_calls=context.saved_data.forgiven_tool_calls,
        )
        with self.lock:
            self._immediate_done[env_idx] = self._translate_status_to_public(
                env_idx,
                status,
            )

    def _translate_status_to_public(
        self,
        env_idx: int,
        status: ToolStatus,
    ) -> ToolStatus:
        """Translate runtime results without translating persisted stage input twice.

        Compiled benchmark stage presentations are already expressed in the
        episode's public vocabulary.  ``RuntimeRandomizer.traduce_end`` normally
        receives canonical task stages and therefore translates ``next_input``.
        Temporarily detaching it preserves the benchmark presentation while the
        status messages and canonical attributes still follow the normal core
        translation path.
        """

        context = self._envs[env_idx]
        next_input = status.next_input
        status.next_input = None
        try:
            translated = context.randomizer.traduce_end(
                {env_idx: status}
            )[env_idx]
        finally:
            status.next_input = next_input
        translated.next_input = next_input
        return translated

    def _submit_judge(self, env_idx: int, retry: bool = False) -> None:
        """Submit one text-only answer using its environment as identifier."""

        if self.worker is None:
            raise RuntimeError("Text-only verification requires a judge worker")
        context = self._envs[env_idx]
        if context.judge_pending and not retry:
            raise RuntimeError(
                f"Episode {context.episode.episode_id!r} already waits for a judge"
            )
        stage_id = context.saved_data.stage_id
        rule = context.task_ref.get_stage_rule(stage_id)
        if not rule:
            raise RuntimeError(
                f"Stage {stage_id} of {context.episode.episode_id!r} "
                "has no verification prompt"
            )

        context.judge_pending = True
        context.judge_attempt_count += 1
        payload = JudgePayload(
            rule=rule,
            model_answer=context.get_answer().get_say(),
            id=env_idx,
            question=context.task_ref.get_stage_input(
                stage_id
            ).instruction.get_content(),
        )
        try:
            self.worker.submit(
                payload,
                callback=lambda future, idx=env_idx: self._on_judge_correction(
                    idx,
                    future,
                ),
            )
        except Exception:
            self._handle_judge_failure(env_idx)

    def _on_judge_correction(self, env_idx: int, future: Future) -> None:
        """Store a judge result for the episode occupying its environment."""

        context = self._envs.get(env_idx)
        if context is None:
            return

        try:
            response_env_idx, judge_response = future.result()
            if response_env_idx != env_idx:
                raise ValueError(
                    f"Judge response environment {response_env_idx} does not match "
                    f"submitted environment {env_idx}"
                )
            judge_dict = json.loads(judge_response)
            verdict = judge_dict["verdict"]
            if not isinstance(verdict, bool):
                raise TypeError("Judge verdict must be a boolean")
        except Exception:
            self._handle_judge_failure(env_idx)
            return

        failure_reason = str(
            judge_dict.get("reason", judge_dict.get("explanation", ""))
        ).strip()
        self._store_judge_verdict(
            env_idx,
            verdict=verdict,
            failure_reason=failure_reason,
            judge_response=judge_response,
        )

    def _handle_judge_failure(self, env_idx: int) -> None:
        """Retry a failed judge request, then expose a fatal runner error."""

        context = self._envs.get(env_idx)
        if context is None:
            return
        if context.judge_attempt_count < self.MAX_JUDGE_ATTEMPTS:
            self._submit_judge(env_idx, retry=True)
            return

        context.judge_pending = False
        error = RuntimeError(
            "Impossible to obtain a valid judge answer for episode "
            f"{context.episode.episode_id} after {self.MAX_JUDGE_ATTEMPTS} attempts."
        )
        with self.lock:
            self._fatal_error = error

    def _store_judge_verdict(
        self,
        env_idx: int,
        verdict: bool,
        failure_reason: str,
        judge_response: Optional[str],
    ) -> None:
        """Queue a valid judge verdict for main-thread stage processing."""

        context = self._envs.get(env_idx)
        if context is None:
            return

        stage_id = context.saved_data.stage_id
        status = ToolStatus(
            robots_status=[
                RobotToolStatus("", "", verdict, ToolErrorFlag.NONE)
            ],
            error_descriptions=[""],
            failure_reason=failure_reason,
            failure_diagnostics=(
                None
                if verdict
                else {
                    "source": "judge",
                    "stage_id": stage_id,
                    "question": context.task_ref.get_stage_input(
                        stage_id
                    ).instruction.get_content(),
                    "verification_prompt": context.task_ref.get_stage_rule(stage_id),
                    "model_answer": context.get_answer().get_say(),
                    "judge_response": judge_response,
                    "judge_verdict": verdict,
                    "judge_reason": failure_reason,
                }
            ),
            stage_id=stage_id,
            stage_success=(
                StageSuccess.FINISH if verdict else StageSuccess.FAILED
            ),
            attributes=copy.deepcopy(context.saved_data.attributes),
            tool_calls=context.saved_data.tool_calls,
            forgiven_tool_calls=context.saved_data.forgiven_tool_calls,
        )
        with self.lock:
            self._judge_done[env_idx] = status

    def _consume_judge_results(self) -> Dict[int, ToolStatus]:
        """Apply main-thread stage transitions for completed judge requests."""

        with self.lock:
            completed = self._judge_done
            self._judge_done = {}

        out: Dict[int, ToolStatus] = {}
        if not completed:
            return out

        env_state = copy.deepcopy(self.env.unwrapped.get_state_dict())
        state_changed = False
        for env_idx, status in completed.items():
            context = self._envs.get(env_idx)
            if context is None:
                continue

            stage_id = context.saved_data.stage_id
            task_ref = context.task_ref
            has_next_stage = stage_id < task_ref.get_nb_total_stage() - 1
            if status.stage_success == StageSuccess.FINISH and has_next_stage:
                status.next_input = task_ref.get_stage_input(stage_id + 1)
                env_state = self._pass_to_the_next_stage(
                    task_ref,
                    stage_id,
                    [env_idx],
                    env_state,
                )
                state_changed = True
                context.saved_data.stage_id += 1
                context.saved_data.stage_log_length = len(
                    context.saved_data.logs
                )
                context.saved_data.active_stage_error_state = {}
                context.saved_data.composite_progress = {}
                context.saved_data.tool_calls = 0
                context.saved_data.forgiven_tool_calls = 0

            status.stage_state = task_ref.get_stage_state(
                stage_id,
                status.tool_calls,
                status.forgiven_tool_calls,
                status.stage_success,
                context.saved_data.logs,
            )
            context.judge_attempt_count = 0
            context.judge_pending = False
            out[env_idx] = self._translate_status_to_public(env_idx, status)

        if state_changed:
            self.env.unwrapped.set_state_dict(env_state)
            for env_idx in completed:
                context = self._envs.get(env_idx)
                if context is not None:
                    context.saved_data.env_state = extract_env_state_val(
                        env_state,
                        env_idx,
                    )
        return out
