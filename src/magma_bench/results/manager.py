from .result import ScenarioResult, TaskResult
from magma_scenarios.benchmark import KNOWN_CRITERIA

from typing import List, Dict, Tuple
import os, json
import threading, queue
import numpy as np

class ResultManager():
    """This class handle the export and save of the multiple log, result for the benchmark"""

    _task_log : bool
    _output_path : str

    _computed_bench_results : Dict[str, List[ScenarioResult]]
    _waiting_bench_results : queue.Queue

    def __init__(
            self,
            output_path : str,
            per_task_log : bool = True,
            unique_try : bool = False,
    ) -> None:
        
        self._task_log = per_task_log
        self._output_path = output_path
        self._computed_bench_results = {}
        self._waiting_bench_results = queue.Queue()
        self._unique_try = unique_try

        self.thread = threading.Thread(target=self._compute_result, daemon=True)
        self.thread.start()
    
    def _compute_result(self):
        while True:
            scenario_result : ScenarioResult = self._waiting_bench_results.get()
            if scenario_result == None:
                break

            scenario_result.compute_result()

            if not scenario_result.scenario_id in self._computed_bench_results:
                self._computed_bench_results[scenario_result.scenario_id] = []
            self._computed_bench_results[scenario_result.scenario_id].append(scenario_result)

            self._export(scenario_result, len(self._computed_bench_results[scenario_result.scenario_id]))


    def stop(self):
        self._waiting_bench_results.put(None)   # unblocks the queue
        self.thread.join()
            
    def _export(self, scenario_result : ScenarioResult, num_try : int):
        """Start the export process"""
        base_path = os.path.join(self._output_path, scenario_result.scenario_id)
        os.makedirs(base_path, exist_ok=True)
        out_path = os.path.join(base_path, f"try-{num_try}")
        os.makedirs(out_path)
        scenario_result.export(out_path, self._task_log)

    def push_scenario_result(self, scenario_result : ScenarioResult):            
        self._waiting_bench_results.put(scenario_result)

    def compute_global_metrics(self) -> Dict:

        # def build_MCP_sr(result_list : List[ScenarioResult], task_ids : List[str]) -> Tuple[np.ndarray, float]:
        #     R = len(result_list)
        #     T = len(result_list[0]._tasks_result)
            
        #     MCP = np.zeros((R, T), dtype=float)

        #     for r, scenario_result in enumerate(result_list):
        #         task_result_dict = scenario_result._tasks_result
        #         if len(task_result_dict) != T:
        #             raise ValueError("Inconsistent task number between same scenario.")
        #         for t_idx, task_id in enumerate(task_ids):
        #             MCP[r, t_idx] = task_result_dict[task_id].nb_completed_stage *100 / len(task_result_dict[task_id]._stage_result) 

        #     return MCP # TO DO        
        def mean_metrics(scenario_results : List[ScenarioResult]):
            
            criteria_result = {}
            total_stage = {}
            per_section_cgc = {}
            cgc = 0
            exe = 0
            sr = 0

            for result in scenario_results:

                # per section cgc
                for k, v in result._per_section_cgc.items():
                    if not k in per_section_cgc:
                        per_section_cgc[k] = []
                    per_section_cgc[k].extend(v)
                # criteria
                for k in result.criteria_score.keys():
                    nb_of_stage = result.per_criterion_nb[k]
                    nb_of_success = result.criteria_score[k]

                    if not k in criteria_result:
                        if nb_of_stage == 0:
                            criteria_result[k] = 'non-measured'
                            total_stage[k] = 0
                        else:
                            criteria_result[k] = nb_of_success
                            total_stage[k] = nb_of_stage
                    
                    else:
                        if type(criteria_result[k]) != type(nb_of_success):
                            raise RuntimeError(f"Got a scneario result where {k} is marked as {type(nb_of_success)} but precdent was {type(criteria_result[k])}")
                        if isinstance(criteria_result[k], float):
                            criteria_result[k] += nb_of_success
                            criteria_result[k] += nb_of_stage

                # generic metric
                cgc+=result.cgc
                exe+=result.exe_score
                sr+=result.success_rate

            N = len(scenario_results)
            criteria = {}
            for k,v in criteria_result.items():
                if isinstance(v,int):
                    criteria[k] = v*100/total_stage[k]
                    criteria[f"{k}_"] = f"{v} / {total_stage[k]}"
                else:
                    criteria[k] = 'non-measured'

            per_cgc = {}
            for k, l in per_section_cgc.items():
                if len(l) > 0:
                    per_cgc[k] = sum(l) / len(l)
                else:
                    per_cgc[k] = "non-measured"

            return sr/N, cgc/N, exe/N, criteria, per_cgc
        
        overall : Dict[str, float] = {
            "cumulative-goal-completion" : 0,
            "cumulative-success_rate" : 0,
            "multi-steps solving" : 0,
            "tool-accuracy" : 0,
            "constrained reasoning" : 0,
            "long-horizon memorization": 0,
            "total_tasks" : 0
        }

        score_list = []

        for s_id, r_l in self._computed_bench_results.items():
            task_ids = [t for t in r_l[0]._tasks_result]

            # criteria
            sr, cgc, exe, crit, per_cgc = mean_metrics(r_l)
            
            
            per_id_metrics = {
                "scenario_success_rate" : sr,
                "scenario_cgc" : cgc,
                "detailled_cgc" : per_cgc,
                "scenario_criteria" : crit,
                "scenario_exe" : exe,
                "scenario_per_criterion_nb_task": r_l[0].per_criterion_nb,
                "scenario_nb_total_evaluated_task" : len(task_ids) * len(r_l),
                # "per_task_score" : per_task_score
            }

            score_list.append(per_id_metrics)

            path = os.path.join(self._output_path, s_id, "result.json")
            with open(path,"w+") as f:
                json.dump(per_id_metrics,f,indent=2)

        N = len(score_list)
        if N ==0:
            raise ValueError("Score list empty.")
        
        total_task_per_criterion = {c : 0 for c in KNOWN_CRITERIA + ["recovery"]}
        score_per_criterion = {c : 0 for c in KNOWN_CRITERIA + ["recovery"]}
        
        for score in score_list:
            overall["cumulative-goal-completion"] += score["scenario_cgc"]
            overall["cumulative-success_rate"] += score["scenario_success_rate"]
            overall["tool-accuracy"] += score["scenario_exe"]
            overall["total_tasks"] += score["scenario_nb_total_evaluated_task"]

            for criterion in score["scenario_criteria"]:
                if score["scenario_criteria"][criterion] == "non-measured" or criterion.endswith("_"): continue
                score_per_criterion[criterion] += (score["scenario_criteria"][criterion] * score["scenario_per_criterion_nb_task"][criterion])
                total_task_per_criterion[criterion] += score["scenario_per_criterion_nb_task"][criterion]

        def att_criterion_score(display_name, criterion_name):
            if total_task_per_criterion[criterion_name] > 0:
                overall[display_name] = score_per_criterion[criterion_name] / total_task_per_criterion[criterion_name]
            else:
                overall[display_name] = "non-measured" #type: ignore
        
        att_criterion_score("constrained reasoning","c-reasoning")
        att_criterion_score("multi-steps solving", "multi-steps")
        att_criterion_score("long-horizon memorization", "lg-memorization")
        att_criterion_score("recovery-capacity", "recovery")

        overall["cumulative-goal-completion"] /= N
        overall["cumulative-success_rate"] /= N
        overall["tool-accuracy"] /= N

        return overall