from typing import List, Dict, Optional
from magma_scenarios.benchmark import KNOWN_CRITERIA

class Stage:
    """Stage of the benchmark. Stock some predicates to compute success and step datas"""

    id : str

    instruction : Optional[Dict]
    default_instruction : Optional[Dict]

    max_agents_step : int

    expected_behavior : str

    keys_evaluator : List[str]

    answer_user : bool
    force_recovery : Optional[str] # Set to a string value (representing the failed error mess) when we want to mark the stage has recovery
    force_failure : Optional[str] # Set to a string value (representing the failed error mess) when we want to force the failure

    def __init__(self, inputs : Dict, id : str) -> None:
        self.id = id
        # instruction
        instruction = inputs.get("instruction",None)
        if instruction is None:
            self.instruction = None
            self.default_instruction = inputs.get("default_instruction", None)
            if self.default_instruction is None:
                raise TypeError("If you are setting the 'instruction' to null, you need to define 'default_instruction' in case of error at the precedent stage.")
            if not "content" in self.default_instruction:
                raise TypeError(f"You need to define the content of the default_instruction in 'content' field (str) : {self.default_instruction}")
            role = self.default_instruction.get("author",None)
            if not role in ["status","user"]:
                raise TypeError(f"You need to define a valid author (status/user) in the default_instruction in the field 'author'. Got {role}")
            self.default_instruction.setdefault("timestamp",0)
        elif instruction and isinstance(instruction, str):
            timestamp = inputs.get("timestamp",0)
            self.instruction = {"author":"user","content":instruction,"timestamp":timestamp}
            self.default_instruction = inputs.get("default_instruction", None)
            if self.default_instruction is not None:
                raise TypeError("You are defining a default_instruction whereas a real instruction was provided. The default_instruction is not necessary.")
        else:
            raise ValueError("Each stage must either have a None instruction (using precedent return) or a string")
        
        # max try
        max_try = inputs.get("max_step",-1)
        if max_try <= 0:
            raise ValueError(f"Each step must have a number of max agents steps valid. Get {max_try}.")
        self.max_agents_step = max_try

        # [OPTIONAL]
        # predicate and complementary eval
        self._predicates = inputs.get("action_goal",[])
        self._comp_eval = inputs.get("complementary_verif",{})

        # flag answer_to_user
        self.answer_user = inputs.get("answer_to_user",False)

       # expected_behavior
        self.expected_behavior = inputs.get("expected_behavior","act")
        if self.expected_behavior == "acknowledge" and self._predicates != []:
            raise ValueError("If the expected behavior is set to acknowledge, you can not specify predicates.")
        
        # force recovery
        self.force_recovery = inputs.get("force_recovery",None)
        if self.force_recovery is not None and (self.expected_behavior != "act" or (self._predicates == [] and self._comp_eval.get("logs",[]) == [])) :
            raise TypeError("You can not set the stage to force recovery while not setting expected_behavior to act or not giving predicated or logs verif")

        # force failure
        self.force_failure = inputs.get("force_failure",None)
        if self.force_recovery is not None and self.force_failure is not None:
            raise TypeError("A stage can not be marked as force_failure or force_recovery. Please select only one.")
        if self.force_failure is not None and self._predicates == [] and self._comp_eval.get("logs",[]) == []:
            raise TypeError("A stage marked as force_failure must define either an action_goal or a log eval.")

        # reset env after the inner step
        self._should_reset_env = inputs.get("should_reset_env",False) or inputs.get("should_env_reset",False)

        k_eval = inputs.get("keys_evaluator", [])
        self.keys_evaluator = []
        for key in k_eval:
            if not key in KNOWN_CRITERIA:
                raise ValueError(f"Key {key} is not a known criteria.")
            self.keys_evaluator.append(key)

    def get_evaluation_elements(self):
        """Allows to gets the evaluation elements (logs, judge, predicates) the inner steps"""
        return self._predicates, self._comp_eval
    
    def should_reset_env(self) -> bool:
        """Return True if the env should be reset"""
        return self._should_reset_env
    
    def should_act(self) -> bool:
        """Return True if the inner step should execute action"""
        return self.expected_behavior == "act"

    def has_flag_recovery(self) -> bool:
        """
        Return True if the stage is marked as force_recovery.
        """
        return self.force_recovery is not None
    
    def has_flag_failure(self) -> bool:
        """
        Return True if the stage is marked as force_recovery.
        """
        return self.force_failure is not None

    def has_flag_answer_to_user(self) -> bool:
        """
        Return True if the stage is marked with the flag 'answer_to_user'.
        Meaning that we want the model to aknowledge the end of its stage (answer to the user 'I have done this').
        """
        return self.answer_user

    def get_default_instruction(self) -> Dict:
        if self.default_instruction is None:
            raise RuntimeError("Asking the default instruction of a valid stage.")
        return self.default_instruction


class Task:
    """Allows to stock the inner step from a benchmark step"""

    stages : List[Stage]
    id : str

    criteria : List[str]

    def __init__(
            self,
            task_data : Dict, # Dict with list of user queries, and max_steps
        ):
        
        self.stages = []
        self.id = task_data['id']
        self.criteria = task_data['criteria']

        for i, stage in enumerate(task_data['stages']):
            try:
                self.stages.append(
                    Stage(stage, id=self.id+str(i))
                )
            except Exception as e:
                raise ValueError(f"Stage {i} : {e}")
            
        self.validate()
            
    def validate(self):
        precedent_flag = True
        unreset_action_or_log = False        
        for i, stage in enumerate(self.stages):
            # Check litle particularity
            if precedent_flag and stage.instruction is None:
                raise TypeError(f"The stage ({i}) is defined without instruction. Meaning that you are planning to override it by the last return status of the precedent stage or with the default_instruction provided."
                                f"However, stage ({i-1}) has set the flag_user_answer to True. Meaning that the system will ask the model to generate an additional answer to explicitly acknoledge the end of the task."
                                f"This is not compatible. Define a custom user instruction at stage {i} or set the flag to false at stage {i-1}")

            for key in stage.keys_evaluator:
                if not key in self.criteria:
                    raise TypeError(f"Stage {i} defines a {key} evaluator criteria that is not present in the overall evaluated criteria of the task. Please fix this.")

            log = stage._comp_eval.get("logs", None)
            
            if log == ["empty"] and unreset_action_or_log:
                raise TypeError(f"Stage {i} has logs set to ['empty'] meaning that we want to have empty log. However, there is no reset 'should_env_reset' in precedent stages so logs of precdent stages will false the result.")


            precedent_flag = stage.answer_user
            if stage.should_reset_env(): unreset_action_or_log = False
            else: unreset_action_or_log = stage.expected_behavior != "acknowledge"


    def __iter__(self):
        # yields Step instances
        return iter(self.stages)

    def evaluate(self,state):
        pass


