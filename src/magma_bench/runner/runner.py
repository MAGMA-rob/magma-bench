from pygments.token import String
from dataclasses import dataclass
import magma_bench.system as system_pkg
from magma_bench.system import System
from magma_bench.builders import (
    ScenarioConfig, Scenario, 
    BenchmarkLoader, ScenarioBuilder,
    Task, Stage
)
from magma_bench.results import ResultManager, ScenarioResult, StageResult
from magma_bench.executor import ToolsEvalExecutor

from magma_core.configs.config import MAGMAConfig
from magma_core.utils.data_utils import apply_att_modif
from magma_core.utils.text_utils import build_model_return_from_executor, build_fake_execution_fail
from magma_core.utils.global_utils import load_module_from_name
from magma_core.workers import LMWorker

from typing import Dict, List, Type, Union
from tqdm import tqdm
import os, json, copy
from datetime import datetime

class BenchmarkRunner():
    """
    Main class of the benchmark. Allows to initialize and runs evaluation on multiple benchmark for a given system
    """

    # reference to the system to evaluate
    system : System

    _benchmark_configs : List[ScenarioConfig]

    result_manager : ResultManager
    tool_executor : ToolsEvalExecutor

    #_bench_logs : List = []

    @staticmethod
    def _ensure_empty_or_create_output_dir(output_path: str) -> None:
        if os.path.exists(output_path):
            if not os.path.isdir(output_path):
                raise FileExistsError(f"Benchmark output path exists and is not a directory: {output_path}")
            if any(os.scandir(output_path)):
                raise FileExistsError(f"Benchmark output directory already exists and is not empty: {output_path}")
            return

        os.makedirs(output_path, exist_ok=True)

    def __init__(
            self,
            system_class_name : str,
            magma_config : MAGMAConfig,
            class_specific_args : Dict,
            skip_backends: bool = False,
        ) -> None:
        benchmark_config = magma_config.benchmark
        self._skip_backends = skip_backends
        if self._skip_backends and system_class_name != "MagmaSingle":
            raise ValueError("skip_backends=True is only supported with MagmaSingle.")

        System_class : Type[System] = load_module_from_name(system_pkg, system_class_name)
        if "Magma" in system_class_name:
            class_specific_args["agent_url"] = magma_config.magma_agent_address

        verifier_backend = None
        if not self._skip_backends:
            verifier_backend = magma_config.backends[benchmark_config["backend_verifier"]]
            class_specific_args.setdefault("backend_url",verifier_backend.endpoint) # TO DO: Dedicated option
            class_specific_args.setdefault("backend_header",verifier_backend.headers) # TO DO: Dedicated option

        self.system = System_class(**class_specific_args)

        self.benchmarks = []
        self.output_path = os.path.join(benchmark_config["save_dir"],self.system.system_name, datetime.now().strftime("%m-%d_%H-%M"))
        self._ensure_empty_or_create_output_dir(self.output_path)

        self.per_task_log = benchmark_config["logs"]
        self.result_manager = ResultManager(self.output_path, benchmark_config["logs"], benchmark_config["num_eval"]==1)
        worker = None if self._skip_backends else LMWorker(verifier_backend)

        self.num_try = benchmark_config["num_eval"]
        self._skip_metrics = False

        self.tool_executor : ToolsEvalExecutor = ToolsEvalExecutor(
            magma_config.magma_planner_address, 
            worker, nb_env=1, 
            randomize_variation=self.num_try
        )
        if self._skip_backends:
            original_verif = self.tool_executor.verif_complementary_bench

            def _verif_without_judge(model_say: str, complementary_verif):
                if complementary_verif is None:
                    return original_verif(model_say, None)

                effective_complementary_verif = dict(complementary_verif)
                effective_complementary_verif.pop("judge", None)
                return original_verif(model_say, effective_complementary_verif)

            self.tool_executor.verif_complementary_bench = _verif_without_judge
    
    def load_benchmark(self, criteria, scenario) -> bool:
        """Load one benchmark to evaluate the system on"""
        if criteria == None or criteria == ['all']:
            criteria = 'all'
        if scenario == None:
            scenario = 'all'
        self._benchmark_configs = BenchmarkLoader.load(criteria, scenario)

        return True
    
    @dataclass
    class StageCounters:
        is_running_a_tool = False
        planner_error_count = 0
        step_counter = 0
        end_of_action = False # is action terminated ?
        catastrophic = False
        budget = 4 # store the nb of injection error free recovery remaining

    def _make_answer(self, author: str, content: Dict, timestep = 0):
        """Return a json formated answer."""
        return {'author':author, "content": copy.deepcopy(content), "timestamp":timestep}

    def _get_last_status_instruction(self, conversation: List[Dict]) -> Union[Dict, None]:
        """
        Return the latest status message produced during a stage execution.
        """
        for message in reversed(conversation):
            if message.get("author") == "SYSTEM":
                return message
        return None

    def _resolve_stage_instruction(
            self,
            stage: Stage,
            previous_stage_success: bool,
            previous_status_instruction: Union[Dict, None],
        ) -> Union[Dict, None]:
        """
        Resolve which instruction should start the current benchmark stage.

        If the stage has no explicit instruction, we reuse the previous status
        when the previous stage succeeded. If no reusable status exists, we
        fall back to ``default_instruction`` when provided. Otherwise the stage
        is skipped so the runner can continue until it finds a runnable stage.
        """
        if stage.instruction is not None:
            return stage.instruction

        if previous_stage_success and previous_status_instruction is not None:
            return previous_status_instruction

        if stage.default_instruction is not None:
            return stage.default_instruction

        return None

    def _run_stage(self, scenario : Scenario, stage : Stage, task_attributes : Dict, instruction : Dict) -> StageResult:
        """Run the evaluation process for one inner-step"""

        call_action = {} # functions calling stack
        status_dict = {}
        response_dict = {} # action + think + say stack
        error_state = stage.get_error_state()

        # returned values
        stageResult = StageResult()
        instruction.setdefault("timestamp",0)
        stageResult.conversation = [instruction]

        # stage counters
        stageCounters = self.StageCounters()
        should_recover = stage.has_flag_recovery() or stage.has_flag_failure()

        end_of_loop = False
        while not end_of_loop:

            # make an action if no tool is running
            if not stageCounters.is_running_a_tool:
                stageCounters.step_counter+=1
                # get model answer
                response_dict = self.system.compute_answer(instruction, task_attributes)
                model_answer = self._make_answer("MODEL", response_dict)
                stageResult.conversation.append(model_answer)
                        
                # process action
                call_action : Dict = response_dict.get("action",{})
                if call_action:
                    if stage.should_act():
                        scenario.send_action(call_action, error_state) 
                        stageCounters.is_running_a_tool = True
                    else:
                        stageCounters.end_of_action = True
                        stageCounters.catastrophic = True
                        stageResult.executions_result.append(False)
                        response_dict.clear()
                else:
                    stageCounters.end_of_action = True
                    stageResult.executions_result.append(True)

            # execute an env step
            tools_ended, obs = scenario.execute_env_step()
            
            # if all tools call are finished, manage planner error, env updating and go to next step
            if tools_ended:

                # retry in case of a planning error or stop stage if too many planner errors
                if any(tools_ended[0]['planning_error']):
                    stageCounters.planner_error_count+=1
                    # We got a planning error here. We retry the same function
                    self.tool_executor.ask_for_retry([0], stageCounters.planner_error_count % 3 == 0, error_state)
                    if stageCounters.planner_error_count > 10:
                        stageResult.executions_result.append(True)
                        stageResult.success = False
                        stageResult.explanation = "The planning failed 10 times. The call is probably not good."
                        return stageResult
                    tools_ended = {}
                    continue

                # build the status 
                status_dict = build_model_return_from_executor(call_action, tools_ended[0]['success'], tools_ended[0]['reason'])
                stageResult.executions_result.extend(tools_ended[0]['success'])
                if stageCounters.budget > 0 and tools_ended[0].get("runtime_error_triggered", False) and any(not success for success in tools_ended[0]['success']):
                    stageCounters.step_counter -= 1
                    stageCounters.budget -= 1

                # update stage attributes
                if not should_recover and not stage.has_flag_failure():
                    att_modif = tools_ended[0].get('att_modif',[])
                    if len(att_modif)>0:
                        apply_att_modif(task_attributes, att_modif)

                stageCounters.is_running_a_tool = False
                stageCounters.end_of_action = True

            # manage env status and returned messages on stage end
            if stageCounters.end_of_action:
                # Verify if the stage is still valid
                stageCounters.is_running_a_tool = False
                stageCounters.end_of_action = False
                fail_fast_on_verification_failure = not stage.should_act()

                if stage.expected_behavior == "act" and call_action == {}:
                    stageResult.success = False
                    stageResult.explanation = 'Forget to call a tool on a act stage'
                    stageCounters.catastrophic = False
                    return stageResult
                elif stageCounters.catastrophic:
                    stageResult.success = False
                    stageResult.explanation = 'Launched a tool on a text-only stage'
                    stageCounters.catastrophic = False
                    return stageResult
                
                should_evaluate_stage = not (stage.is_answer() and call_action != {})
                if should_evaluate_stage:
                    try:
                        stageResult.success, stageResult.explanation = scenario.evaluate_stage(stage, response_dict, obs, not self._skip_backends)
                    except:
                        stageResult.success, stageResult.explanation = False, "Exception due to no tool call"
                else:
                    # Tool-enabled answer stages must be judged only once the
                    # model eventually provides a final text answer.
                    stageResult.success = False
                    stageResult.explanation = "Waiting for a final answer after tool execution."

                # if a tool where called  
                if call_action != {}:
                    if status_dict == {}:
                        raise RuntimeError("Empty status dict while having executed an action")
                    
                    # update env status with failure
                    if stageResult.success and (should_recover or stage.has_flag_failure()):
                        # If the stage is successfull + marked as recovery but we haven't done it yet.
                        status_dict = build_fake_execution_fail(
                            call_action, status_dict,
                            stage.get_error_flag()
                        )
                        self.tool_executor.ask_for_retry([0],False,auto_compute=False, error_state=error_state)
                        stageCounters.step_counter-=1
                        stageResult.success = False
                        if not should_recover: #only in force_failure, count the number of model errors before informing about the failure case
                            stageCounters.planner_error_count+=1
                            if stageCounters.planner_error_count > 4:
                                stageResult.success = False
                                stageResult.explanation = "The models persist in its error more than 4 times"
                                return stageResult
                        else:
                            stageCounters.planner_error_count = 0
                        should_recover = False
                        
                    # build returned instruction
                    instruction = self._make_answer("SYSTEM", status_dict)
                    stageResult.conversation.append(instruction)

                # if no tools where called, check model text response
                elif stage.has_flag_failure() and not should_recover:
                    d = self.tool_executor.verif_complementary_bench(
                        response_dict['say'],
                        {
                            "judge": f"The model must inform the user that an error happens"
                        }
                        #this error happens : {stage.force_failure}. It is not strict, the model can just inform of the failure state."
                        )
                    stageResult.success, stageResult.explanation = d['verdict'], d['explanation']

                fail_fast_on_verification_failure = (
                    (not stage.should_act())
                    or (stage.is_answer() and call_action == {})
                )
                call_action.clear()
                stageCounters.catastrophic = False

                # check user response in case of stage sucess                
                if stageResult.success:
                    # check if user got a valid answer
                    if stage.has_flag_answer_to_user():
                        # get model answer
                        response_dict = self.system.compute_answer(instruction,task_attributes)
                        answer = self._make_answer("MODEL", response_dict)
                        stageResult.conversation.append(answer)
                        stageResult.success = (response_dict['action'] == {})
                        if not stageResult.success: stageResult.explanation = "Add a call on the final answer."
                    end_of_loop = True
                else:
                    if fail_fast_on_verification_failure or stageCounters.step_counter >= stage.max_agents_step:
                        end_of_loop = True
                

        return stageResult
    
    def _run_scenario(self, scenario : Scenario):
        nb_tasks = scenario.nb_tasks

        for try_index in tqdm(range(self.num_try), desc="Evaluation Tries", position=1, leave=False):
            try_number = try_index + 1
            try_output_path = None
            if self.per_task_log:
                try_output_path = self.result_manager.get_try_output_path(scenario.id, try_number)

            # Select the indexed benchmark variation before collecting any
            # try-level metadata or exposing tools/attributes to the system.
            self.tool_executor.set_randomizer_index(try_index)
            scenario_result = ScenarioResult(
                scenario.id,
                scenario.evaluated_criteria,
                try_number=try_number,
                randomization_info=self.tool_executor.get_try_randomization_info(),
            )
            init_elements = scenario.get_init_elements()
            self.system.init_task(init_elements)
            
            for task_id in tqdm(range(nb_tasks), desc="Task", position=2, leave=False):
                self.system.reset_step()

                task : Task = scenario.get_task(task_id)
                task_attributes = scenario.get_init_elements()["attributes"]

                previous_stage_success = False
                previous_status_instruction = None
                for stage in task.stages:
                    instruction = self._resolve_stage_instruction(
                        stage,
                        previous_stage_success,
                        previous_status_instruction,
                    )
                    if instruction is None:
                        # No explicit instruction, no fallback and no reusable status: 
                        # skip this chained stage and move on until we reach a runnable one.
                        scenario_result.record_skip(task, stage, "unresolved_instruction")
                        previous_stage_success = False
                        previous_status_instruction = None
                        continue
                    
                    if self.tool_executor.randomized:
                        instruction = json.loads(self.tool_executor.randomizer.traduce_attributes_to_llm(json.dumps(instruction)))

                    stage_result = self._run_stage(scenario,stage,task_attributes,instruction)

                    # saves results and stage info
                    scenario_result.record_stage(task, stage, stage_result)

                    previous_stage_success = stage_result.success
                    previous_status_instruction = self._get_last_status_instruction(stage_result.conversation)

                    if stage.should_reset_env():
                        scenario.env_reset()
                    
                    scenario.save_video(f"{scenario.name}-try-{try_index}-task-{task.id}")

                if self.per_task_log and try_output_path is not None:
                    scenario_result.export_task_log(task, try_output_path)

            if not self._skip_metrics:
                self.result_manager.push_scenario_result(scenario_result) 
    
    def run(self, args):
        """Run the evaluation process on the pre-loaded benchmarks"""
        if not self._benchmark_configs:
            raise ValueError("There is no benchmark loaded. Please use .load(path) before run.")

        self._skip_metrics = bool(getattr(args, "no_metrics", False))

        selected_task_indices = getattr(args, "task_indices", None)
        if selected_task_indices is not None and len(self._benchmark_configs) != 1:
            raise ValueError("task_indices can only be used when executing exactly one scenario.")
        
        for bench_config in tqdm(self._benchmark_configs, desc="Scenarios", position=0, leave=True):
            scenario = ScenarioBuilder.load(bench_config, self.output_path, self.tool_executor, args)
            if selected_task_indices is not None:
                if len(set(selected_task_indices)) != len(selected_task_indices):
                    raise ValueError(f"task_indices must not contain duplicates. Got {selected_task_indices}.")
                task_by_index = {}
                for task in scenario.tasks:
                    suffix = str(task.id).rsplit("_", 1)[-1]
                    try:
                        task_index = int(suffix)
                    except ValueError:
                        continue
                    task_by_index[task_index] = task

                invalid_indices = [
                    idx for idx in selected_task_indices
                    if not isinstance(idx, int) or idx not in task_by_index
                ]
                if invalid_indices:
                    available_indices = sorted(task_by_index.keys())
                    raise ValueError(
                        f"task_indices {invalid_indices} do not match any task id in scenario "
                        f"{scenario.name}. Available task ids are {available_indices}."
                    )
                scenario.tasks = [task_by_index[idx] for idx in selected_task_indices]
                scenario.nb_tasks = len(scenario.tasks)
            self._run_scenario(scenario)
            scenario.close()

        self.result_manager.stop()
        if self._skip_metrics:
            return
        data = self.result_manager.compute_global_metrics()
        data['system_info'] = self.system.get_system_card()

        path = os.path.join(self.output_path, "result.json")
        with open(path,"w+") as f:
            json.dump(data,f,indent=2)
                    
