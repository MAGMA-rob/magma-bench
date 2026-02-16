from typing import List, Dict, Optional
from dataclasses import dataclass
import os, json

from magma_bench.builders import Task

@dataclass
class StageResult:
    conversation : List[Dict]
    success : bool
    explanation : str
    exe_score : float # Exe score represent the % of good call (good syntax) that has been passed. [0,1]
    was_recovery : bool
    keys_evaluator : List[str]

class CriterionScore:

    success : int #nb of suc
    nb_of_key_stage : int #nb of key stage

    def __init__(self) -> None:
        self.success = 0
        self.nb_of_key_stage = 0

    def count(self, success : bool):
        self.nb_of_key_stage+=1
        if success: self.success+=1

class TaskResult:

    _stage_result : List[StageResult]

    _success : bool # success rate
    nb_completed_stage : int # Number of Completed Stage (<= nb_stage)
    sum_execution_score : float # Sum of execution score of all stages (<=nb_stage)

    _metrics_computed : bool
    _criteria : Dict[str, CriterionScore] # Score per criterion

    def __init__(self, criteria : List[str]) -> None:
        self._stage_result = []
        self._success = False
        self.nb_completed_stage = 0
        self.sum_execution_score = 0.0
        self._metrics_computed = False
        self._criteria = {c : CriterionScore() for c in criteria + ["recovery"]}

    def add_stage_result(
            self, conversation : List[Dict], success : bool, was_recovery : bool,
            execution : List[bool], keys_evaluator : List[str], explanation : str):
        cpt=0
        for val in execution:
            if val: cpt+=1
        
        self._stage_result.append(
            StageResult(
                conversation=conversation,
                success=success,
                exe_score=cpt/len(execution),
                explanation=explanation,
                was_recovery=was_recovery,
                keys_evaluator=keys_evaluator
            )
        )

    def _compute_result(self):
        if self._metrics_computed:
            raise RuntimeError("Trying to compute two time the result")
        self._success = True
        
        for s in self._stage_result:
            if self._success and s.success:
                self.nb_completed_stage += 1
            else:
                self._success = False

            self.sum_execution_score += s.exe_score

            for key in s.keys_evaluator:
                self._criteria[key].count(s.success)

            if s.was_recovery:
                self._criteria["recovery"].count(s.success)

        for criteria, score in self._criteria.items():
            if score.nb_of_key_stage == 0:
                self._criteria[criteria].count(self._stage_result[-1].success)

        self._metrics_computed = True


    def get_result(self):
        """
        Return the result of the Task.
        First boolean of success, then nb of completed task, exe score and finally the nb of stage in the task.
        """
        if not self._metrics_computed:
            self._compute_result()

        return self._success, self.nb_completed_stage, self.sum_execution_score, len(self._stage_result)
    
    def get_logs(self) -> Dict:
        out = []
        for step in self._stage_result:
            out.append({
                "conversation" : step.conversation,
                "success" : step.success,
                "explanation" : step.explanation
            })


        N = len(self._stage_result)

        h = {
            "task_success" : self._success,
            "stage_completed" : f"{self.nb_completed_stage} / {N}",
            "task_completion_percentage" : self.nb_completed_stage * 100 / N,
            "tool_accuracy" : self.sum_execution_score * 100 / N,
            "task_tool_accuracy" : self.sum_execution_score * 100 / N
        }
        for criterion, score in self._criteria.items():
            h[criterion] = f"{score.success} / {score.nb_of_key_stage}"
            h[f"{criterion}_percentage"] = score.success * 100 / score.nb_of_key_stage

        return {
            "header" : h,
            "stage" : out
        }
    
    def get_criterion_score(self, criterion : str) -> Optional[CriterionScore]:
        """Return criterion score 0-100. -1 if the criterion was not evaluated"""
        return self._criteria.get(criterion,None)

class ScenarioResult:

    _tasks_result : Dict[str, TaskResult]
    scenario_id : str

    cgc : float # cumulative goal completion. Mean completion percentage before failure per task. 0-100
    success_rate : float # success rate 0-100
    exe_score : float # The percentage of good syntax answer (good call with argument / not catastrophic call).
    criteria_score : Dict[str, float] # criteria 0-100 or -1 if non-measured
    per_criterion_nb : Dict
    _per_section_cgc : Dict[str,List[float]]

    def __init__(self, id : str, criteria_to_evaluate : List[str], stage_decomposition : Dict[str,List]) -> None:
        self._tasks_result = {}
        self.scenario_id = id
        self.stage_decomp = stage_decomposition
        self.criteria_score = {c : 0 for c in criteria_to_evaluate}

    def add_result(
            self,
            task_ref : Task,
            conversation : List[Dict],
            success : bool,
            was_recovery : bool,
            execution : List[bool],
            explanation : str,
            keys_evaluator : List[str]
    ):
        """Allows to add a Inner Step results inside the Step result corresponding to the ID"""
        
        if not task_ref.id in self._tasks_result:
            self._tasks_result[task_ref.id] = TaskResult(task_ref.criteria)
        
        self._tasks_result[task_ref.id].add_stage_result(conversation,success,was_recovery,execution,keys_evaluator,explanation)

    def export(self, folder_path : str, detailled_log : bool):
        
        if detailled_log:
            per_step_path = os.path.join(folder_path, "logs")
            os.makedirs(per_step_path)

            for id, step in self._tasks_result.items():
                data = step.get_logs()
                with open(os.path.join(per_step_path, id+".json"), "w+") as f:
                    json.dump(data, f, indent=2)
        
        metrics = self.criteria_score.copy()
        metrics.update({
            "cgc" : self.cgc,
            "success_rate" : self.success_rate,
            "nb_of_task" : len(self._tasks_result)
        })
        
        json_p = os.path.join(folder_path,"try_score.json")
        with open(json_p, "w+") as f:
            json.dump(metrics,f,indent=2)

    def compute_result(self):
        cgc = 0.0
        success_rate = 0.0
        exe_score = 0.0
        total_stage = 0
        task_nb = len(self._tasks_result)
        self._per_section_cgc = {k : [] for k in self.stage_decomp}
        self.per_criterion_nb = {c : 0 for c in self.criteria_score}

        for task_id, task_result in self._tasks_result.items():
            task_sr, task_comp, task_exe, nb_stage = task_result.get_result()
            total_stage += nb_stage

            # general metric
            cgc += task_comp
            exe_score += task_exe
            success_rate += int(task_sr)
            # per section cgc
            for k, l in self.stage_decomp.items():
                
                if task_id in l:
                    self._per_section_cgc[k].append(task_comp/nb_stage)
                    break
            
            # criteria metric
            for criterion in self.criteria_score:
                if not criterion in self.criteria_score:
                    self.per_criterion_nb[criterion] = 0
                    self.criteria_score[criterion] = 0
                    # Do this to go fats, its for the recovery which is not an official criterion
                score = task_result.get_criterion_score(criterion)
                if score is not None:
                    self.per_criterion_nb[criterion] += score.nb_of_key_stage
                    self.criteria_score[criterion] += score.success

        self.cgc = cgc * 100 / total_stage
        self.exe_score = exe_score * 100 / total_stage
        self.success_rate = success_rate * 100 / task_nb

    # def get_per_task_metrics(self) -> Dict:
    #     mu_t = np.mean(mcp, axis=0) # mean score per task
    #     std_t = np.std(mcp, axis=0) # var per task

        
        
    #     per_task_score = {
    #         task_ids[i]: {
    #             "mcp" : mu_t[i],
    #             "std" : std_t[i]
    #         } for i in range(len(task_ids))
    #     }