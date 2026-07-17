# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Union, Optional, Tuple, Any
from collections import OrderedDict
from concurrent.futures import Future
from magma_core.base.tasks.base_task import BaseTask
import copy, threading
import torch, json
from mani_skill.utils.wrappers.record import RecordEpisode

from magma_bench.data_structures import EpisodeData
from magma_bench.artifacts import EpisodeSpec

from magma_core.base.envs import DefaultEnv
from magma_core.base.executor import ToolsBaseExecutor
from magma_core.base.agents import ValidAgentAnswer
from magma_core.base.data_structures import (
    ToolStatus, RobotToolStatus, Log, 
    ActiveStageErrorState, ToolErrorFlag,
    StageSuccess
)
from magma_core.workers import LMWorker
from magma_core.protocol.payload.user_sim_payload import JudgePayload
from magma_core.utils.global_utils import (
    batch_set_value,
    extract_env_state_val,
    merge_robot_articulations,
)

from .context import EvalEpisodeContext

class ToolsEvalExecutor(ToolsBaseExecutor):
    """
    Tools executor is the main class responsible for transforming LLM calls into actions using specific defined class.
    """

    # Only for Evaluation Mode
    _envs : Dict[int, EvalEpisodeContext]
    _free_idx : List[int]

    # to allows to define specific variation for randomization
    _task_env_options_override: Dict[str,Any]

    _judge_done : Dict[int,ToolStatus]

    def __init__(
            self,
            planner_endpoint : str,
            worker : LMWorker,
            nb_env : int = 1,
        ):
        # We do not use a global randomizer, each episode will carry its own little randomizer
        super().__init__(
            nb_env,
            planner_endpoint=planner_endpoint,
            ollama_worker=worker,
            nb_randomization=0
        )
        self._free_idx = [e for e in range(nb_env)]
        self._envs = {}
        self._judge_done = {}
        self._task_env_options_override: Dict[str, Any] = {}

        self.lock = threading.Lock()
   
    ################ public function

    def register(self, episode_spec : EpisodeSpec) -> EpisodeData:
        if len(self._free_idx) == 0:
            raise RuntimeError("Trying to register an episode while no free env idx left")
        
        idx = self._free_idx.pop(0)

        self._envs[idx] = EvalEpisodeContext(
            episode_spec.episode_id,
            validation_steps=...
        )

        return EpisodeData(

        )

    def release_idx(self, idx : int):
        if not idx in self._envs:
            raise RuntimeError("Asking to release an env_idx that is not used")
        self._envs.pop(idx)

    def set_task_env_options(self, env_options: Optional[Dict[str, Any]]) -> None:
        """
        Allow to partially override the original task env option
        """
        if env_options is None:
            self._task_env_options_override = {}
            return
        if not isinstance(env_options, dict):
            raise TypeError(f"Task env_options must be a dict. Got {type(env_options)}.")
        self._task_env_options_override = copy.deepcopy(env_options)

    def get_env_options(self) -> Dict:
        env_options = copy.deepcopy(super().get_env_options())
        env_options.update(copy.deepcopy(self._task_env_options_override))
        return env_options

    def initialize(
            self,
            task_ref: BaseTask,
            build_first_stage: bool = True,
            obs_mode: str = "state_dict",
            sim_backend: str = "auto",
            video_path : str = "none"
        ) -> DefaultEnv:
        self.set_task_env_options(None)
        self.env = super().initialize(
            task_ref, build_first_stage,
            obs_mode, sim_backend
        )
        if video_path != "none":
            self.env = RecordEpisode(
                    self.env,
                    output_dir=video_path,
                    save_trajectory=False,
                    save_video=True,
                    source_type="MAGMA-BENCHMARK",
                    source_desc=f"Videos of the execution on {task_ref.name} benchmark",
                    video_fps=30,
                    save_on_reset=False
            ) #type: ignore
        return self.env

    def _recursive_verif_log(self, log_ref : List, log_model : List[Log]) -> Tuple[bool,str]:
        if len(log_ref) == 0 or isinstance(log_ref[0], Dict):
            return self.task_ref.compare_log(log_ref, log_model)
        elif isinstance(log_ref[0], List):
            for possible_log_ref in log_ref:
                out = self._recursive_verif_log(possible_log_ref, log_model)
                if out[0] == True:
                    return out
            return False, "none of the proposition have completed" 
        else:
            raise RuntimeError(f"Unknow log ref structure. Need either a list of dict, with dict with 'action', 'content' or 'function' field. OR a list of list of dict (to show multiple valible possibility)")
    
    


    def _verif_log_rules(self, log_rules: List[Any], logs: List[Log]) -> Tuple[bool, str]:
        # TO KEEP WAITING FOR UPDATE
        # if log_ref or log_rules:
        #     full_log, stage_log = self._get_logs(0)
        #     if log_ref == ['empty']:
        #         if len(full_log) != 0:
        #             log_verdict, log_reason = False, f"Log should be empty but got {len(full_log)} element"
        #     elif log_ref:
        #         # print(log_ref)
        #         # print("VS")
        #         # print("FULL : ", [l.to_string() for l in full_log])
        #         exact_stage_log = full_log[-len(log_ref):]
        #         # print("STAGE : ", [l.to_string() for l in exact_stage_log])
        #         log_verdict, log_reason = self._recursive_verif_log(log_ref, exact_stage_log)
        #         # print(log_verdict)
        #         # print(log_reason)
        #         # print("=========")

        #     if log_rules:
        #         # Benchmark log rules intentionally run on the full log so one
        #         # rule can span multiple benchmark stages when needed.
        #         log_rules_verdict, log_rules_reason = self._verif_log_rules(log_rules, full_log)
        verdict = True
        reasons = []

        for rule in log_rules:
            rule_verdict, reason = rule.verify(logs)
            if not rule_verdict:
                verdict = False
            reason = reason.strip()
            if reason:
                reasons.append(reason)

        return verdict, " - ".join(reasons)

    def check_env_state(self, obs : Dict):
        """
        Used for Checking if env has passed the stage in eval mode.

        Return a list of 0 (running), 1 (finish), -1 (error) representing the status of all envs.
        """
        out = []
        stage_id = self._eval_envs[0].current_task_stage
        env_ids = list(range(self.nb_env))
        env_verif = self.task_ref.verif_stage_env_completion(stage_id, obs, env_ids=env_ids).cpu().tolist()
        for i in range(self.nb_env):
            full_log, stage_log = self._get_logs(i)
            log_verif = self.task_ref.verif_stage_log_completion(stage_id=stage_id ,full_log=full_log, stage_log=stage_log)
            out.append(self.task_ref.combine_stage_verif_scores(stage_id, env_verif[i], log_verif))

        if all([score == 1 for score in out]):
            if stage_id != self.task_ref.get_nb_total_stage()-1:
                new_id = stage_id+1
                st = self.env.get_state_dict().copy()
                self._pass_to_the_next_stage(stage_id, env_ids, st)

                situation = self.get_init_situation(new_id)
                print(f"NEXT STAGE : {new_id} with instruction {situation.instruction.get_content()}")

                for i in range(self.nb_env):
                    # Specific modifications
                    self._eval_envs[i].current_task_stage = new_id
                    self._eval_envs[i].stage_log_start_idx = len(self._eval_envs[i].logs)
                    # A new stage must resample its own active error profile.
                    self._eval_envs[i].error_state = {}

                self.env.set_state_dict(st)
            else:
                print("FINISHED TASK")

        return out

    def _initialize_stage_error_state(
            self,
            stage_id: int,
            error_state: ActiveStageErrorState,
            obs: Dict,
            env_id: int,
            attributes: Dict[str, Any],
            agent_id: int = 0
        ) -> ActiveStageErrorState:
        return error_state
    
    def compute_actions(
            self,
            tools_call : Dict[int, ValidAgentAnswer]
        ):
        """
        Take a batch of tool calls keyed by environment id.

        Transforms this into a batched sequence of steps per environment.
        """

        # print("------- ACTION COMPUTE -------")
 
        # step to have obs
        obs = self.env.get_obs()

        for env_id, answer in tools_call.items():
            if not answer:
                continue

            if answer.get_say() != "":
                judge_verif = ... # TODO: get this from verification episode context
                payload = JudgePayload(rule=judge_verif, model_answer=answer.get_say(), id=env_id)
                self.worker.submit(payload, callback=self._on_judge_correction)
                continue

            calls = answer.get_action()
            if len(calls) == 0:
                raise RuntimeError("Impossible fail: An empty answer has reached the Eval Executor")
            
            episode_context = self._envs[env_id].tool_context


            tool_infos = self._compute_tool(
                calls=calls,
                env_id=env_id,
                obs=obs,
                error_state=episode_context.error_state,
                stage_id=episode_context.current_task_stage,
                current_node_step=0,
                original_log_length=len(episode_context.logs),
                source_node_id=0,
                attributes=episode_context.attributes,
                logs=episode_context.logs,
                previous_tool_calls=episode_context.tool_calls,
                previous_forgiven_tool_calls=episode_context.forgiven_tool_calls,
            )
            self._envs[env_id].set_tool_context(tool_infos)
            if tool_infos.is_full_error():
                # print(f"[FunctionExecutor] No poses returned for {func_name} : {tool_execution.reason}")
                continue

            _ = self.trajectory_converter.transform_poses_in_actions(
                    tool_infos,
                    env_id
                )

    def step(self) -> Union[torch.Tensor, OrderedDict]:
        """
        Execute the actions in the environment.

        Returns:
            Dict: If SingleAGent env, return a batched tensor of action. For MultiAGent, return an OrderedDict
        """
        return self.trajectory_converter.step({k: v.tool_context for k,v in self._envs.items()})

    def verif_ended_tool(self, obs : Dict) -> Dict[int,ToolStatus]:
        with self.lock:
            # Fetch judge verification
            out = self._judge_done.copy()
            self._judge_done.clear()

        retry_env_idx = []

        for env_idx, episode_context in self._envs.items():
            env_infos = episode_context.tool_context
            env_infos.compute_tool_results(obs)

            if env_infos.all_finished():
                # gerer la verification ici



                tool_status = env_infos.build_tool_status()
                
                if tool_status.should_restore_source_env_state():
                    # That's mean there is a planner error and so we need to try again
                    retry_env_idx.append(env_idx)
                    continue

                out[env_idx] = tool_status

        if len(retry_env_idx)>0:
            self._handle_tool_retry(retry_env_idx)

        return out

    ################ private function

    def _handle_tool_retry(
            self,
            desired_env_ids: List[int],
        ):
        """
        Ask the environment to re-try the last action for the desired_env_ids.
        Allows to handle when there is a planning error.

        It resets the env to the state before the last tool call, and retry it with a small randomization on the robot joints pose.
        If retry counter is even, it resets the robot to its base pose.
        It automatically re-computes the latent actions
        """
        full_state_dict = self.env.get_state_dict().copy()
        calls = {}
        robot_names = self.trajectory_converter.agents_name
        for ids in desired_env_ids:
            episode_context = self._envs[ids]
            if episode_context.nb_of_retry > 10:
                raise RuntimeError(f"Planner Error for more than 10 retries")
            
            episode_context.nb_of_retry+=1
            precedent_state_dict = episode_context.last_env_state.copy()
            current_state = extract_env_state_val(full_state_dict, ids)

            if episode_context.nb_of_retry % 2 != 0:
                # We keep current robot pose
                source_articulations = current_state.get('articulations', None)
            else:
                # We reset to base joint pose
                source_articulations = self.task_ref._default_env_state.get('articulations', None)


            if source_articulations is not None:
                precedent_state_dict['articulations'], _ = merge_robot_articulations(
                    precedent_state_dict.get('articulations', {}),
                    source_articulations,
                    robot_names,
                )

            batch_set_value(full_state_dict, torch.tensor([ids]), precedent_state_dict, strict=False)
            calls[ids] = episode_context.get_answer()
        self.env.set_state_dict(full_state_dict)
        action = self.step()
        _ , _, _, _,_ = self.env.step(action)
        
        self.compute_actions(calls)


    def _on_judge_correction(self, future : Future):
        i, judge_str = future.result()
        if not i in self._envs:
            raise RuntimeError("Impossible fail: Get an env id that is not in self._envs")
        episode_context = self._envs[i]
        try:
            judge_dict = json.loads(judge_str)
        except:
            judge_dict = {}

        if not "verdict" in judge_dict:
            if episode_context.nb_of_retry > 5:
                verdict = True
                failure_reason = "Skipped due to 5 judge failure"
            else:
                episode_context.nb_of_retry += 1
                payload = JudgePayload(
                    rule=judge_verif, #TODO: Get from verification episode context
                    model_answer=episode_context.get_answer().get_say(),
                    id=i
                )
                self.worker.submit(payload, callback=self._on_judge_correction)
                return
        else:
            verdict = judge_dict.get("verdict")
            failure_reason = str(
                judge_dict.get("reason", judge_dict.get("explanation", ""))
            ).strip()
        
        with self.lock:
            self._judge_done[i] = ToolStatus(
                robots_status=[RobotToolStatus("","",False,ToolErrorFlag.NONE)],
                error_descriptions=[""],
                failure_reason=failure_reason,
                stage_id= episode_context.tool_context.current_task_stage,
                # A text-only judgement is terminal for the current answer:
                # either the reply satisfies the rule, or the stage failed.
                # Because we enter here only if there is no action that follow
                stage_success = StageSuccess.FINISH if verdict else StageSuccess.FAILED,
                attributes= episode_context.tool_context.attributes,
            )