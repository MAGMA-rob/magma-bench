from typing import Any, List, Dict, Optional, Literal
from dataclasses import dataclass

from magma_scenarios.benchmark import KNOWN_CRITERIA
from magma_core.base.data_structures import ActiveStageErrorState

@dataclass
class StageInjection:
    mode: Literal["runtime_error", "force_recovery", "force_failure"]
    error: Optional[str] = None
    arguments: Optional[Dict] = None
    message: Optional[str] = None

    def verify(self):
        """
        Validate a single benchmark injection entry.

        The benchmark format stays intentionally small:
        - `runtime_error` uses `error` + optional `arguments`
        - `force_recovery` / `force_failure` use the optional `message`
        """
        if self.mode not in {"runtime_error", "force_recovery", "force_failure"}:
            raise ValueError(
                "Injection mode must be one of "
                "'runtime_error', 'force_recovery' or 'force_failure'. "
                f"Got {self.mode!r}."
            )

        if self.mode == "runtime_error":
            if not isinstance(self.error, str) or self.error == "":
                raise ValueError(
                    "A runtime_error injection must define a non-empty string in the 'error' field."
                )
            if self.arguments is not None and not isinstance(self.arguments, dict):
                raise TypeError(
                    "A runtime_error injection must define 'arguments' as a dict when provided."
                )
            if self.message is not None:
                raise TypeError(
                    "A runtime_error injection can not define a 'message' field."
                )
            return

        if self.error is not None or self.arguments is not None:
            raise TypeError(
                f"A {self.mode} injection can only define the optional 'message' field."
            )
        if self.message is not None and not isinstance(self.message, str):
            raise TypeError(
                f"A {self.mode} injection message must be a string when provided."
            )

class Stage:
    """Stage of the benchmark. Stock some predicates to compute success and step datas"""

    id : str

    instruction : Optional[Dict]
    default_instruction : Optional[Dict]

    max_agents_step : int
    expected_behavior : str
    keys_evaluator : List[str]

    # flags
    answer_user : bool
    failure_flag : bool # set to True if the injection contains a force_failure error
    recovery_flag : bool # set to True if the injection contains a force_recovery error

    injections : List[StageInjection]

    def __init__(self, inputs : Dict, id : str) -> None:
        self.id = id
        # instruction
        instruction = inputs.get("instruction",None)
        if instruction is None:
            self.instruction = None
            self.default_instruction = inputs.get("default_instruction", None)
            # A null instruction means "reuse the previous status when available". 
            # ``default_instruction`` is optional and is only used as an explicit fallback 
            # when the runner cannot reuse a previous status message (stage failed)
            if self.default_instruction is not None:
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
        # Keep support for the older `flag_answer_to_user` json key.
        self.answer_user = inputs.get("answer_to_user",False) or inputs.get("flag_answer_to_user",False)

       # expected_behavior
        self.expected_behavior = inputs.get("expected_behavior","act")
        if self.expected_behavior == "acknowledge" and self._predicates != []:
            raise ValueError("If the expected behavior is set to acknowledge, you can not specify predicates.")

        # Benchmark injections
        self.injections = self._parse_injections(inputs) #parse the list
        # Verify that we do not have multiple force_* instance in the list
        self._validate_injections()

        # reset env after the inner step
        # Keep support for historical aliases already present in benchmark jsons.
        self._should_reset_env = (
            inputs.get("should_reset_env",False)
            or inputs.get("should_env_reset",False)
            or inputs.get("should_reset",False)
        )

        k_eval = inputs.get("keys_evaluator", [])
        self.keys_evaluator = []
        for key in k_eval:
            if not key in KNOWN_CRITERIA:
                raise ValueError(f"Key {key} is not a known criteria.")
            self.keys_evaluator.append(key)

    def _parse_injections(self, inputs: Dict) -> List[StageInjection]:
        """
        Normalize the json task format to a list of StageInjection objects.

        Supported inputs:
        - `injection`: dict or list[dict]
        """
        raw_injections = inputs.get("injection", inputs.get("injections", None))
        legacy_force_recovery = inputs.get("force_recovery", None)
        legacy_force_failure = inputs.get("force_failure", None)

        if legacy_force_recovery is not None or legacy_force_failure is not None:
            raise TypeError(
                "Legacy fields 'force_recovery' and 'force_failure' are not supported anymore. "
                "Please use the 'injection' field instead."
            )

        normalized: List[StageInjection] = []

        if raw_injections is None:
            return normalized

        if isinstance(raw_injections, dict):
            raw_injections = [raw_injections]
        elif not isinstance(raw_injections, list):
            raise TypeError(
                "Field 'injection' must be either a dict or a list of dict."
            )

        for i, raw_injection in enumerate(raw_injections):
            if not isinstance(raw_injection, dict):
                raise TypeError(
                    f"Injection at index {i} must be a dict. Got {type(raw_injection)}."
                )

            injection = StageInjection(
                mode=raw_injection.get("mode",""),
                error=raw_injection.get("error"),
                arguments=raw_injection.get("arguments"),
                message=raw_injection.get("message"),
            )
            injection.verify()
            normalized.append(injection)

        return normalized

    def _validate_injections(self):
        """
        Validate stage-level injection constraints and mirror legacy flags for
        the existing runner implementation.

        Verify that there is no mix betwen type
        """
        recovery_injections = [
            injection for injection in self.injections if injection.mode == "force_recovery"
        ]
        failure_injections = [
            injection for injection in self.injections if injection.mode == "force_failure"
        ]
        self.failure_flag = len(failure_injections) > 0
        self.recovery_flag = len(recovery_injections) > 0

        if len(recovery_injections) > 1:
            raise TypeError(
                "A stage can not define more than one force_recovery injection."
            )
        if len(failure_injections) > 1:
            raise TypeError(
                "A stage can not define more than one force_failure injection."
            )
        if recovery_injections and failure_injections:
            raise TypeError(
                "A stage can not mix force_recovery and force_failure injections."
            )

        force_injections = recovery_injections + failure_injections
        if force_injections and (
            self.expected_behavior != "act"
            or (self._predicates == [] and self._comp_eval.get("logs",[]) == [])
        ):
            raise TypeError(
                "A stage can not define force_recovery/force_failure injections while "
                "not expecting an action or while missing predicates/log verifications."
            )

        if failure_injections and self._predicates == [] and self._comp_eval.get("logs",[]) == []:
            raise TypeError(
                "A stage marked as force_failure must define either an action_goal or a log eval."
            )

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
        return self.recovery_flag
    
    def has_flag_failure(self) -> bool:
        """
        Return True if the stage is marked as force_recovery.
        """
        return self.failure_flag
    
    def get_error_flag(self) -> str:
        """
        Return a string corresponding either to flag failure or flag recovery depending on the one set.
        Raise an exception otherwise.
        """
        if self.recovery_flag:
            for e in self.injections:
                if e.mode == "force_recovery" and e.message is not None:
                    return e.message
            return "Communication error: Failed to communicate with the robot."
        elif self.failure_flag:
            for e in self.injections:
                if e.mode == "force_failure" and e.message is not None:
                    return e.message
            return "Communication error: Failed to communicate with the robot."
        raise RuntimeError("Asking for an error flag but got nothing")

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
    
    def get_error_state(self) -> ActiveStageErrorState:
        """
        Extract, from the injected error at the stage, the runtime_errors.
        """
        error_state = {}
        for injected_error in self.injections:
            if injected_error.mode == "runtime_error":
                error_state[injected_error.error] = injected_error.arguments
        return error_state


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
            if precedent_flag and stage.instruction is None and stage.default_instruction is None:
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
