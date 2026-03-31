from typing import Dict, List

from .stage import Stage


class Task:
    """Allows to stock the inner step from a benchmark step"""

    stages: List[Stage]
    id: str
    criteria: List[str]

    def __init__(self, task_data: Dict):
        self.stages = []
        self.id = task_data["id"]
        self.criteria = task_data["criteria"]

        for i, stage in enumerate(task_data["stages"]):
            try:
                self.stages.append(Stage(stage, id=self.id + str(i)))
            except Exception as exc:
                raise ValueError(f"Stage {i} : {exc}")

        self._propagate_multi_steps()
        self.validate()

    def _propagate_multi_steps(self):
        propagate_multi_steps = False
        for stage in self.stages:
            if stage.instruction is not None:
                propagate_multi_steps = "multi-steps" in stage.keys_evaluator
                continue

            if propagate_multi_steps and "multi-steps" not in stage.keys_evaluator:
                stage.keys_evaluator.append("multi-steps")

    def validate(self):
        precedent_flag = True
        unreset_action_or_log = False
        for i, stage in enumerate(self.stages):
            if precedent_flag and stage.instruction is None and stage.default_instruction is None:
                raise TypeError(
                    f"The stage ({i}) is defined without instruction. Meaning that you are planning to override it by the last return status of the precedent stage or with the default_instruction provided."
                    f"However, stage ({i-1}) has set the flag_user_answer to True. Meaning that the system will ask the model to generate an additional answer to explicitly acknoledge the end of the task."
                    f"This is not compatible. Define a custom user instruction at stage {i} or set the flag to false at stage {i-1}"
                )

            for key in stage.keys_evaluator:
                if key not in self.criteria:
                    raise TypeError(
                        f"Stage {i} defines a {key} evaluator criteria that is not present in the overall evaluated criteria of the task. Please fix this."
                    )

            log = stage._comp_eval.get("logs", None)
            if log == ["empty"] and unreset_action_or_log:
                raise TypeError(
                    f"Stage {i} has logs set to ['empty'] meaning that we want to have empty log. However, there is no reset 'should_env_reset' in precedent stages so logs of precdent stages will false the result."
                )

            precedent_flag = stage.answer_user
            if stage.should_reset_env():
                unreset_action_or_log = False
            elif stage.should_act():
                unreset_action_or_log = True

    def __iter__(self):
        return iter(self.stages)

    def evaluate(self, state):
        pass
