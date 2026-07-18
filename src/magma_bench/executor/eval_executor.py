# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

"""Vectorized benchmark executor with one task runtime per active episode."""

from collections import OrderedDict
from concurrent.futures import Future
import copy
import json
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
    ValidExecutionReq
)
from magma_core.base.envs import DefaultEnv
from magma_core.base.executor import ToolsBaseExecutor
from magma_core.protocol.payload.user_sim_payload import JudgePayload
from magma_core.serialization import decode_value
from magma_core.utils.global_utils import (
    apply_env_state_updates,
    batch_set_value,
    extract_env_state_val,
    merge_robot_articulations,
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

from .context import EvalEpisodeContext
from .episode_runtime import build_episode_task


class ToolsEvalExecutor(ToolsBaseExecutor):
    """Execute independent benchmark episodes in shared simulator slots."""

    _envs : Dict[int, EvalEpisodeContext]


    def __init__(
            self,
            planner_endpoint: str,
            worker: Optional[LMWorker],
            nb_env: int = 1,
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
        self.lock = threading.Lock()

    def initialize_group(
            self,
            scenario: Scenario,
            group: EpisodeGroup,
            obs_mode: str = "state_dict",
            sim_backend: str = "auto",
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

        self.env = self._create_envs(
            scenario.environment_id,
            group.env_options,
            group.planner_options,
            obs_mode,
            sim_backend,
        )
        self._group_base_state = copy.deepcopy(self.env.get_state_dict())
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
            env_state = copy.deepcopy(self.env.get_state_dict())
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
            self.env.set_state_dict(initialized_state)

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

            situation = EpisodeSituation(
                tools=decode_value(copy.deepcopy(episode.semantic.tools)),
                attributes=decode_value(
                    copy.deepcopy(episode.semantic.visible_attributes)
                ),
                current_instruction=task_ref.get_stage_input(0).instruction,
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
        self._free_idx.append(env_idx)
        self._free_idx.sort()

    def compute_actions(
            self,
            tools_call: Dict[int, ValidExecutionReq],
        ) -> None:
        """Transform agent answers into actions for their assigned slots."""

        obs = self.env.get_obs()
        for env_idx, answer in tools_call.items():
            context = self._envs.get(env_idx)
            if context is None:
                raise KeyError(f"No active episode in environment {env_idx}")
            if context.tool_context is not None:
                raise RuntimeError(
                    f"Episode {context.episode.episode_id!r} already executes a tool"
                )

            context.last_agent_answer = answer
            if answer.get_say() != "":
                self._submit_judge(env_idx)
                continue

            calls = answer.get_tool_calls()
            if not calls:
                raise RuntimeError(
                    "An empty answer reached the evaluation executor"
                )

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
    

    def verif_ended_tool(self, obs: Dict) -> Dict[int, ToolStatus]:
        """Verify completed tools and advance each episode independently."""

        out = self._consume_judge_results()
        completed: List[
            Tuple[int, EvalEpisodeContext, EnvToolContext, ToolStatus]
        ] = []
        retry_env_ids: List[int] = []

        env_state = copy.deepcopy(self.env.get_state_dict())
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
                retry_env_ids.append(env_idx)
                continue
            
            # Apply possible state changement (move_to, object state changement)
            state_changed |= apply_env_state_updates(
                env_state,
                env_idx,
                tool_context.get_state_updates(),
            )
            completed.append((env_idx, context, tool_context, tool_status))

        if state_changed:
            self.env.set_state_dict(env_state)
            obs = self.env.get_obs()

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
                # catastrophic failure
                # faut stopper la tache carrément

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
            saved_data.tool_calls = tool_context.tool_calls
            saved_data.forgiven_tool_calls = tool_context.forgiven_tool_calls
            if advanced_stage:
                saved_data.active_stage_error_state = {}
                saved_data.composite_progress = {}
            else:
                saved_data.active_stage_error_state = tool_context.error_state
                saved_data.composite_progress = tool_context.composite_progress
            context.retry_count = 0
            context.tool_context = None

            translated = context.randomizer.traduce_end(
                {env_idx: tool_status}
            )[env_idx]
            out[env_idx] = translated

        if state_changed:
            self.env.set_state_dict(env_state)
        for env_idx, context, _, _ in completed:
            context.saved_data.env_state = extract_env_state_val(
                env_state,
                env_idx,
            )

        if retry_env_ids:
            self._handle_tool_retry(retry_env_ids)

        return out

    def _handle_tool_retry(self, env_ids: List[int]) -> None:
        """Restore committed states and replay answers after planner failures."""

        env_state = copy.deepcopy(self.env.get_state_dict())
        answers: Dict[int, ValidExecutionReq] = {}
        robot_names = self.trajectory_converter.agents_name

        for env_idx in env_ids:
            context = self._envs[env_idx]
            if context.retry_count >= 10:
                raise RuntimeError(
                    f"Planner error exceeded 10 retries for "
                    f"{context.episode.episode_id}"
                )
            context.retry_count += 1
            restored_state = copy.deepcopy(context.saved_data.env_state)
            current_state = extract_env_state_val(env_state, env_idx)

            if context.retry_count % 2:
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

        self.env.set_state_dict(env_state)
        self.compute_actions(answers)

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
        payload = JudgePayload(
            rule=rule,
            model_answer=context.get_answer().get_say(),
            id=env_idx,
            question=context.task_ref.get_stage_input(
                stage_id
            ).instruction.get_content(),
        )
        try:
            self.worker.submit(payload, callback=self._on_judge_correction)
        except Exception:
            context.judge_pending = False
            raise

    def _on_judge_correction(self, future: Future) -> None:
        """Store a judge result for the episode occupying its environment."""

        env_idx, judge_response = future.result()
        context = self._envs.get(env_idx)
        if context is None:
            return

        try:
            judge_dict = json.loads(judge_response)
            verdict = judge_dict["verdict"]
            if not isinstance(verdict, bool):
                raise TypeError("Judge verdict must be a boolean")
        except (json.JSONDecodeError, KeyError, TypeError):
            if context.retry_count >= 5:
                verdict = False
                failure_reason = "Judge failed to return a valid verdict after 6 attempts"
            else:
                context.retry_count += 1
                self._submit_judge(env_idx, retry=True)
                return
        else:
            failure_reason = str(
                judge_dict.get("reason", judge_dict.get("explanation", ""))
            ).strip()

        stage_id = context.saved_data.stage_id
        status = ToolStatus(
            robots_status=[
                RobotToolStatus("", "", verdict, ToolErrorFlag.NONE)
            ],
            error_descriptions=[""],
            failure_reason=failure_reason,
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

        env_state = copy.deepcopy(self.env.get_state_dict())
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

            status.stage_state = task_ref.get_stage_state(
                stage_id,
                status.tool_calls,
                status.forgiven_tool_calls,
                status.stage_success,
                context.saved_data.logs,
            )
            context.retry_count = 0
            context.judge_pending = False
            out[env_idx] = context.randomizer.traduce_end(
                {env_idx: status}
            )[env_idx]

        if state_changed:
            self.env.set_state_dict(env_state)
            for env_idx in completed:
                context = self._envs.get(env_idx)
                if context is not None:
                    context.saved_data.env_state = extract_env_state_val(
                        env_state,
                        env_idx,
                    )
        return out
