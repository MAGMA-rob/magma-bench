from .scenario import ScenarioConfig
import magma_scenarios.benchmark as bench_pkg
from magma_core.utils.data_utils import verify_key
from magma_scenarios.benchmark import KNOWN_CRITERIA

import json, os
import importlib.resources
from typing import List, Optional, Any, Dict, Tuple, Union

GROUP = {"CGC 2-3":[2,3],"CGC 4-5":[4,5],"CGC 6-9":[6,9],"CGC 10-16":[10,16]}

class BenchmarkLoader():
    """
    Load one or multiple scenario according to selected criteria to form the benchmark
    """

    @staticmethod
    def _scenario_valid(scenario : Union[str, List[str]], criteria : Union[str, List[str]], scenario_dict : Dict) -> bool:
        if scenario == "all" or scenario_dict['scenario_name'] in scenario:      
            return criteria == "all" or any(x in scenario_dict['criteria'] for x in criteria)
        
        return False

    @staticmethod
    def load(
        criteria : Union[str, List[str]] = "all",
        scenario : Union[str, List[str]] = "all",
    ) -> List[ScenarioConfig]:
        
        out = []
        
        with importlib.resources.files(bench_pkg).joinpath("bench_metadata.json").open('r') as file:
            bench_config = json.load(file)
        for scenario_dict in bench_config:
            if BenchmarkLoader._scenario_valid(scenario,criteria,scenario_dict):
                task_decomp = {k:[] for k in GROUP}

                print(f"[LOADER] Loading {scenario_dict['scenario_name']} for criteria {criteria}")
                
                # if scenario valid, we load its config
                with importlib.resources.files(bench_pkg).joinpath(scenario_dict['folder_name'],"meta_info.json").open('r') as file:
                    scenario_config = json.load(file)
                
                criteria_tasks = verify_key(scenario_config, "criteria_tasks")
                s_name = verify_key(scenario_config, "scenario_name")
                s_task = verify_key(scenario_config, "associate_task")
                s_id = verify_key(scenario_config, "scenario_id")
                s_mplib = verify_key(scenario_config, "options_mplib")
                s_tasks_horizon = verify_key(scenario_config, "tasks_horizon")

                task_folder = importlib.resources.files(bench_pkg).joinpath(scenario_dict['folder_name'], "tasks")
                tasks = []
                seen = set()

                for c, task_list in criteria_tasks.items():
                    if criteria == "all" or c in criteria:
                        for name in task_list:
                            if name not in seen:
                                seen.add(name)
                                tasks.append(task_folder / name)

                for horizon, tasks_list in s_tasks_horizon.items():
                    for k, bound in GROUP.items():
                        if int(horizon) >= bound[0] and int(horizon) <= bound[1]:
                            task_decomp[k].extend(tasks_list)
                            break

                if isinstance(criteria, List):
                    e_criteria = criteria
                else:
                    e_criteria = KNOWN_CRITERIA.copy()
                out.append(ScenarioConfig(
                    scenario_name=s_name,
                    task_decomp=task_decomp,
                    scenario_id=s_id,
                    task_cls_name=s_task,
                    mplib_options=s_mplib,
                    tasks_path=tasks,
                    evaluated_criteria=e_criteria
                ))
                        
        return out
