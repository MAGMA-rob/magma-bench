import magma_bench.system as system_pkg
from magma_bench.system import System
from magma_bench.builders import (
    ScenarioConfig, Scenario, 
    BenchmarkLoader, ScenarioBuilder,
    Task, Stage
)
from magma_bench.results import ResultManager, ScenarioResult
from magma_bench.executor import ToolsEvalExecutor

from magma_core.configs.config import MAGMAConfig
from magma_core.utils.data_utils import apply_att_modif
from magma_core.utils.text_utils import build_model_return_from_executor, build_fake_execution_fail
from magma_core.utils.global_utils import load_module_from_name
from magma_core.workers import LMWorker

from typing import Dict, List, Type, Union
from tqdm import tqdm
import os, json
from datetime import datetime

class BenchmarkRunner():
    """
    Main class of the benchmark. ALlows to initialize and runs evaluation on multiple benchmark for a given system
    """

    # reference to the system to evaluate
    system : System

    _benchmark_configs : List[ScenarioConfig]

    result_manager : ResultManager
    tool_executor : ToolsEvalExecutor

    def __init__(
            self,
            system_class_name : str,
            magma_config : MAGMAConfig,
            class_specific_args : Dict,
        ) -> None:
        System_class : Type[System] = load_module_from_name(system_pkg, system_class_name)
        if "Magma" in system_class_name:
            class_specific_args["agent_url"] = magma_config.magma_agent_address
        verifier_backend = magma_config.backends[magma_config.benchmark["backend_verifier"]]
        class_specific_args.setdefault("backend_url",verifier_backend.endpoint) # TO DO: Dedicated option
        class_specific_args.setdefault("backend_header",verifier_backend.headers) # TO DO: Dedicated option
        self.system = System_class(**class_specific_args)

        self.benchmarks = []
        self.output_path = os.path.join(magma_config.benchmark["save_dir"],self.system.system_name, datetime.now().strftime("%m-%d_%H-%M"))

        self.result_manager = ResultManager(self.output_path, magma_config.benchmark["logs"], magma_config.benchmark["num_eval"]==1)
        worker = LMWorker(magma_config.backends[magma_config.benchmark["backend_verifier"]])
        self.tool_executor : ToolsEvalExecutor = ToolsEvalExecutor(magma_config.magma_planner_address, worker, nb_env=1)

        self.num_try = magma_config.benchmark["num_eval"]
    
    def load_benchmark(self, criteria, scenario) -> bool:
        """Load one benchmark to evaluate the system on"""
        if criteria == None or criteria == ['all']:
            criteria = 'all'
        if scenario == None:
            scenario = 'all'
        self._benchmark_configs = BenchmarkLoader.load(criteria, scenario)

        return True
    
    def _run_stage(self, scenario : Scenario, stage : Stage, task_attributes : Dict, instruction : Dict):
        """Run the evaluation process for one inner-step"""
        should_act = stage.should_act()
        instruction.setdefault("timestamp",0)

        running = False
        success = False
        END_OF_LOOP = False
        end_of_action = False
        step_cpt = 0
        should_recover = stage.has_flag_recovery() or stage.has_flag_failure()

        call_ac = {}
        status_dict = {}
        response_dict = {}
        conversation = [instruction]
        exe_result = []
        reason = ""
        catastrophic = False

        err=0

        while not END_OF_LOOP:

            if not running:
                step_cpt+=1

                # get model answer
                response_dict = self.system.compute_answer(instruction,task_attributes)
                model_answer = {'author':"model", "content":response_dict, "timestamp":0}
                conversation.append(model_answer)
                
                # process action
                call_ac : Dict = response_dict.get("action",{})
                if call_ac:
                    if should_act:
                        scenario.send_action(call_ac) 
                        running = True
                    else:
                        end_of_action = True
                        catastrophic = True
                        exe_result.append(False)
                        response_dict = {}
                else:
                    end_of_action = True
                    exe_result.append(True)

            # execute step
            tools_ended = scenario.execute_env_step()
            
            # check if all steps of the actions are finished
            if tools_ended:
                if any(tools_ended[0]['planning_error']):
                    err+=1
                    # We got a planning error here. We retry the same function
                    self.tool_executor.ask_for_retry([0], err % 3 == 0)
                    if err > 10:
                        exe_result.append(True)
                        return False, "The planning failed 10 times. The call is probably not good.", conversation, exe_result
                    tools_ended = {}
                    continue

                # Build the status 
                status_dict = build_model_return_from_executor(call_ac, tools_ended[0]['success'], tools_ended[0]['reason'])
                exe_result.extend(tools_ended[0]['success'])

                if not should_recover and not stage.has_flag_failure():
                    att_modif = tools_ended[0].get('att_modif',[])
                    if len(att_modif)>0:
                        apply_att_modif(task_attributes, att_modif)

                running = False
                end_of_action = True

            # check for success
            if end_of_action:
                # Verify if the stage is valid
                if catastrophic:
                    success = False
                    reason = 'Launched a tool on a text-only stage'
                else:
                    try:
                        success, reason = scenario.evaluate_stage(stage,response_dict)
                    except:
                        success, reason = False, "Exception due to no tool call"
                        
                    if call_ac != {}:
                        if status_dict == {}:
                            raise RuntimeError("Empty status dict while having executed an action")
                        
                        if success and (should_recover or stage.has_flag_failure()):
                            # If the stage is successfull + marked as recovery but we haven't done it yet.
                            status_dict = build_fake_execution_fail(
                                call_ac, status_dict,
                                stage.force_recovery if stage.has_flag_recovery() else stage.force_failure)
                            self.tool_executor.ask_for_retry([0],False,auto_compute=False)
                            step_cpt-=1
                            success = False
                            if not should_recover: #only in force_failure, count the number of model errors before informing about the failure case
                                err+=1
                                if err > 4:
                                    return False, "The models persist in its error more than 4 times", conversation, exe_result
                            else:
                                err = 0
                            should_recover = False
                            
                        
                        instruction = {'author':"status", "content":str(status_dict), "timestamp":0}
                        conversation.append(instruction)
                    elif stage.has_flag_failure() and not should_recover:
                        d = self.tool_executor.verif_complementary_bench(
                            response_dict['say'], None, 
                            f"The model must inform the user that an error happens" 
                            #this error happens : {stage.force_failure}. It is not strict, the model can just inform of the failure state."
                            )     
                        success, reason = d['verdict'], d['explanation']

                    call_ac = {}
                    catastrophic=False
                if not success:
                    if step_cpt >= stage.max_agents_step:
                        break
                
                running=False
                end_of_action = False

                if success:
                    if stage.has_flag_answer_to_user():
                        response_dict = self.system.compute_answer(instruction,task_attributes)
                        conversation.append({'author':"model", "content":response_dict, "timestamp":0})
                        success = (response_dict['action'] == {})

                    END_OF_LOOP = True

        return success, reason, conversation, exe_result
    
    def _run_scenario(self, scenario : Scenario, task_decomp : Dict[str,List[str]]):
        nb_tasks = scenario.nb_tasks
        init_elements = scenario.get_init_elements()

        self.system.init_task(init_elements)

        for i in tqdm(range(self.num_try), desc="Evaluation Tries", position=1, leave=False):
            scenario_result = ScenarioResult(scenario.id, scenario.evaluated_criteria, task_decomp)
            for j in tqdm(range(nb_tasks), desc="Task", position=2, leave=False):
                self.system.reset_step()
                task : Task = scenario.get_task(j)
                task_attributes = scenario.get_init_elements()["attributes"]
                conv = []
                suc = False
                for stage in task.stages:
                    instruction = stage.instruction
                    if instruction is None:
                        if suc and conv[-1]["author"] == "status": instruction = conv[-1]
                        else: instruction = stage.get_default_instruction()
                    suc, r, conv, exe = self._run_stage(scenario,stage,task_attributes,instruction)
                    scenario_result.add_result(
                        task,conv,suc,
                        stage.has_flag_failure() or stage.has_flag_recovery(),
                        exe,r,stage.keys_evaluator
                    )

                    if stage.should_reset_env():
                        scenario.env_reset()

                scenario.save_video(f"{scenario.name}-{i}-{j}")

            self.result_manager.push_scenario_result(scenario_result)

            
    
    def run(self, args):
        """Run the evaluation process on the pre-loaded benchmarks"""
        if not self._benchmark_configs:
            raise ValueError("There is no benchmark loaded. Please use .load(path) before run.")
        
        for bench_config in tqdm(self._benchmark_configs, desc="Scenarios", position=0, leave=True):
            scenario = ScenarioBuilder.load(bench_config, self.output_path, self.tool_executor, args)
            self._run_scenario(scenario, bench_config.task_decomp)
            scenario.close()

        self.result_manager.stop()
        data = self.result_manager.compute_global_metrics()
        data['system_info'] = self.system.get_system_card()

        path = os.path.join(self.output_path, "result.json")
        with open(path,"w+") as f:
            json.dump(data,f,indent=2)
                    