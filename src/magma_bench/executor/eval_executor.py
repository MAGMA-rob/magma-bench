# SPDX-License-Identifier: BSD-2-Clause
# Copyright (c) 2026, Loan Bernat

from typing import Dict, List, Union, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from magma_core.base.tasks.base_task import BaseTask
import torch, json
from mani_skill.utils.wrappers.record import RecordEpisode

from magma_core.base.envs import DefaultEnv
from magma_core.base.executor import ToolsBaseExecutor
from magma_core.base.data_structures import ToolInfos, Log
from magma_core.workers import LMWorker
from magma_core.protocol.payload.user_sim_payload import JudgePayload
from magma_core.utils.global_utils import extract_env_state_val, batch_set_value

@dataclass
class SavedEnvState:
    """Dataclass to store saved information"""
    env_state: Dict
    tool_call : Dict

class ToolsEvalExecutor(ToolsBaseExecutor):
    """
    Tools executor is the main class responsible for transforming LLM calls into actions using specific defined class.
    """

    # Only for Evaluation Mode
    _eval_envs : Dict[int,ToolInfos]
    _precedent_env_state : Dict[int, SavedEnvState]

    def __init__(
            self,
            planner_endpoint : str,
            worker : LMWorker,
            nb_env : int = 1,
            randomize_variation : int = 0,
        ):
        super().__init__(
            nb_env,
            planner_endpoint=planner_endpoint,
            ollama_worker=worker,
            nb_randomization=randomize_variation
        )

        self._eval_envs : Dict[int,ToolInfos] = {}
        self._precedent_env_state = {}
   
    ################ public function

    def set_randomizer_index(self, id : int):
        if self.randomized:
            self.randomizer.set_variation_index(id)

    def initialize(self, task_ref: BaseTask, build_first_stage: bool = True, obs_mode: str = "state_dict", video_path : str = "none") -> DefaultEnv:
        self.env = super().initialize(task_ref, build_first_stage, obs_mode)
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
            )
        return self.env

    def log_reset(self):
        """
        Benchmark only.

        Reset all logs for all env.
        """
        for tool_info in self._eval_envs.values():
            tool_info.logs = []

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
    
    def ask_for_retry(self, desired_env_ids: List[int], reset_joint_to_env_default : bool, auto_compute : bool = True):
        """
        Ask the environment to re-try the last action for the desired_env_ids.
        Allows to handle when there is a planning error.

        It resets the env to the state before the last tool call, and retry it with a small randomization on the robot joints pose.
        If reset_joint_to_env_default is True, it just reset the robot to its base pose.
        If auto_compute is True, automatically re-compute the latent actions
        """
        full_state_dict = self.env.get_state_dict().copy()
        calls = {}
        for ids in desired_env_ids:
            if ids >= self.nb_env:
                raise RuntimeError(f"Asking reset for an unknow ids = {ids}. Max nb of env {self.nb_env}")
            st = self._precedent_env_state[ids].env_state.copy()
            if not reset_joint_to_env_default:
                st['articulations'] = extract_env_state_val(full_state_dict, ids)['articulations']
            else:
                st['articulations'] = self.task_ref._default_env_state['articulations']
            batch_set_value(full_state_dict, ids, st)
            calls[ids] = self._precedent_env_state[ids].tool_call.copy()
        self.env.set_state_dict(full_state_dict)
        action = self.step()
        _ , _, _, _,_ = self.env.step(action)
        if auto_compute:
            self.compute_actions(calls)


    def verif_complementary_bench(
            self,
            model_say : str,
            log_ref : Optional[List],
            judge_verif : Optional[str]
        ) -> Dict:
        """
        Benchmark only.

        Verify some complementary elements such as logs, model answer...

        Return a dict with a bool 'verdict' and a string 'explanation'.
        """
        log_verdict, log_reason = True, ""
        if log_ref:
            full_log, _ = self._get_logs(0)
            if log_ref == ['empty']:
                if len(full_log) != 0:
                    log_verdict, log_reason = False, f"Log should be empty but got {len(full_log)} element"
            else:
            # print(log_ref)
            # print("VS")
            # print("FULL : ", [l.to_string() for l in full_log])
                stage_log = full_log[-len(log_ref):]
                # print("STAGE : ", [l.to_string() for l in stage_log])
                log_verdict, log_reason = self._recursive_verif_log(log_ref, stage_log)
                # print(log_verdict)
                # print(log_reason)
                # print("=========")            

        if judge_verif:
            if self.randomized:
                judge_verif = self.randomizer.traduce_attributes_to_llm(judge_verif)
            payload = JudgePayload(rule=judge_verif, model_answer=model_say, id=0)
            future = self.worker.submit(payload, callback=None)
            i, judge_str = future.result()
            try:
                judge_dict = json.loads(judge_str)
            except:
                judge_dict = {}
            if not "verdict" in judge_dict:
                payload = JudgePayload(rule=judge_verif, model_answer=model_say, id=0)
                future = self.worker.submit(payload, callback=None)
                i, judge_str = future.result()
                judge_dict = json.loads(judge_str)
        else:
            judge_dict = {"verdict": True, "explanation":""}

        return {
            "verdict" : judge_dict.get("verdict",True) and log_verdict,
            "explanation" : judge_dict.get("explanation","") + " - " + log_reason
        }

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

                self.env.set_state_dict(st)
            else:
                print("FINISHED TASK")

        return out
    
    def get_current_task_attributes(self):
        """
        Specific function to get the task attributes of all env (the latest in case of multiple stage between envs)
        """
        stage = 0
        for tool_infos in self._eval_envs.values():
            if tool_infos.current_task_stage > stage: stage = tool_infos.current_task_stage
        
        return self.get_task_attributes(stage)
    
    def compute_actions(self, tools_call : Dict):
        """
        Take a batch of tools_calls. It's a dict where each key is a node_id associate with a dict.
        In case of GENERATION mode, the value dict contains 'tool' (the action dict) and 'src_id' (the parent node_id).
        In case of EVALUATION mode, the value dict contains directly the action dict.
        Transforms this into a batched sequence of steps per environment.
        """

        # print("------- ACTION COMPUTE -------")
 
        # step to have obs
        action = self.step()
        obs , _, _, _,_ = self.env.step(action)
        state_dict = self.env.get_state_dict().copy()

        for env_id, func in tools_call.items():
            func_name = func.get("name", None)
            params = func.get("arguments", {})

            self._precedent_env_state[env_id] = SavedEnvState(
                env_state=extract_env_state_val(state_dict,env_id),
                tool_call=func
            )

            if env_id in self._eval_envs:
                stage_id = self._eval_envs[env_id].current_task_stage
                logs = self._eval_envs[env_id].logs
                stage_log_length = self._eval_envs[env_id].stage_log_start_idx
            else:
                stage_id = 0
                stage_log_length = 0
                logs = []
            
            # FAUT QUE JARRIVE A DETERMINER ICI SI CEST UN DEBUT DE STAGE OU NON
            if func_name:
                tool_infos = self._compute_single_tool(
                    func_name,
                    params,
                    env_id,
                    obs,
                    current_node_step=0,
                    stage_id=stage_id,
                    node_id=env_id,
                    logs=logs,
                    source_node_id=0,
                    original_log_length=stage_log_length
                )
            else:
                tool_infos = self._compute_multiple_tool(
                    actions = func,
                    env_id=env_id,
                    obs=obs,
                    stage_id=stage_id,
                    current_node_step=0,
                    node_id=env_id,
                    logs=logs,
                    source_node_id=0,
                    original_log_length=stage_log_length
                )
            self._eval_envs[env_id] = tool_infos
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
        return self.trajectory_converter.step(self._eval_envs)

    def verif_ended_tool(self, obs : Dict) -> Dict:
        out = {}
        for env_id, env_infos in self._eval_envs.items():
            traj_to_compute = env_infos.compute_tool_results(obs)

            if traj_to_compute:
                res = self.trajectory_converter.transform_poses_in_actions(
                    env_infos,
                    env_id
                )

            if env_infos.all_finished():

                results, mess, att_modif = env_infos.build_return()
                
                # Detect planning error
                planning_error = []
                for i, r in enumerate(results):
                    if not r and env_infos.tool_robots[i].get_result() is not None:
                        planning_error.append(True)
                    else:
                        planning_error.append(False)

                out[env_infos.node_id] = {'success':results, "reason":mess, "att_modif" : att_modif, "planning_error":planning_error}
                env_infos.tool_robots = []

        return self.randomizer.traduce_end_eval(out) if self.randomized else out

    ################ private function

    def _get_logs(self, env_id: int) -> Tuple[List[Log], List[Log]]:
        """
        Return the log corresponding to the given env_id
        """
        return self._eval_envs[env_id]._get_logs()

    def _get_node_infos(self, node_id):
        """Allows to retrieve the env information (env_state and logs) from a specific nodes"""

        def _extract_env_state_val(st, env_id) -> Dict:
            out = {}
            for key, value in st.items():
                if isinstance(value, Dict):
                    out[key] = _extract_env_state_val(value, env_id)
                else:
                    out[key] = value[env_id].clone()
            return out

        return {
            "logs" : self._eval_envs[node_id].logs,
            "env_state" : _extract_env_state_val(self.env.get_state_dict().copy(),node_id)
        }
   