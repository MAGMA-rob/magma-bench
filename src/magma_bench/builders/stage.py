from typing import Dict, List, Optional, Tuple

from magma_core.base.data_structures import ActiveStageErrorState
from magma_core.base.goals import BaseGoal
from magma_scenarios.benchmark import KNOWN_CRITERIA
from ..evalutations.predicate import compile_predicates_to_goals, get_obs_entity_signature
from .stage_injection import StageInjection


def _resolve_expected_behavior(inputs: Dict) -> str:
    expected_behavior = inputs.get("expected_behavior", "act")
    if expected_behavior not in {"act", "acknowledge", "answer"}:
        raise ValueError(
            "expected_behavior must be one of 'act', 'acknowledge' or 'answer'. "
            f"Got {expected_behavior!r}."
        )
    return expected_behavior


class Stage:
    """Base benchmark stage plus factory for the concrete stage types."""

    id: str
    instruction: Optional[Dict]
    default_instruction: Optional[Dict]
    max_agents_step: int
    stage_horizon: int
    expected_behavior: str
    keys_evaluator: List[str]
    answer_user: bool
    failure_flag: bool
    recovery_flag: bool
    allow_tool_flag : bool
    injections: List[StageInjection]
    predicates : List[BaseGoal]
    _predicate_specs: List[Dict]
    _predicate_signature: Optional[Tuple[str, ...]]

    _should_reset_env: bool
    _comp_eval: Dict

    def __new__(cls, inputs: Dict, id: str):
        if cls is Stage:
            stage_cls = {
                "act": ActStage,
                "acknowledge": AcknowledgeStage,
                "answer": AnswerStage,
            }[_resolve_expected_behavior(inputs)]
            instance = object.__new__(stage_cls)
            stage_cls.__init__(instance, inputs, id)
            return instance
        return super().__new__(cls)

    def __init__(self, inputs: Dict, id: str) -> None:
        # ``Stage`` is a factory. Concrete subclasses do the real initialization.
        pass

    def _setup_base(self, id: str, expected_behavior: str):
        self.id = id
        self.expected_behavior = expected_behavior
        self.instruction = None
        self.default_instruction = None
        self.max_agents_step = 1
        self.stage_horizon = 0
        self.predicates = []
        self._predicate_specs = []
        self._predicate_signature = None
        self._comp_eval = {}
        self.answer_user = False
        self.failure_flag = False
        self.recovery_flag = False
        self.allow_tool_flag = False
        self.injections = []
        self._should_reset_env = False
        self.keys_evaluator = []

    def _extract_allows_tools(self, inputs: Dict, expected_behavior: str) -> bool:
        if "allows_tools" not in inputs:
            return False

        allows_tools = inputs["allows_tools"]
        if not isinstance(allows_tools, bool):
            raise TypeError(
                f"allows_tools must be a bool. Got {type(allows_tools)}."
            )

        if expected_behavior != "answer":
            raise TypeError(
                "Field 'allows_tools' is only supported for stages with "
                "expected_behavior='answer'."
            )

        return allows_tools

    def _assert_only_allowed_fields(self, inputs: Dict, allowed_fields: set[str], mode_name: str):
        extra_fields = set(inputs) - allowed_fields
        if extra_fields:
            raise TypeError(
                f"A stage with expected_behavior='{mode_name}' only allows "
                f"{sorted(allowed_fields)}. Got unexpected fields {sorted(extra_fields)}."
            )

    def _build_explicit_instruction(self, inputs: Dict) -> Dict:
        instruction = inputs.get("instruction", None)
        if not isinstance(instruction, str) or instruction == "":
            raise ValueError("Stages with an explicit instruction must define a non-empty string in 'instruction'.")
        timestamp = inputs.get("timestamp", 0)
        return {"author": "USER", "content": instruction, "timestamp": timestamp}

    def _init_act_instruction(self, inputs: Dict):
        instruction = inputs.get("instruction", None)
        if instruction is None:
            self.instruction = None
            self.default_instruction = inputs.get("default_instruction", None)
            # A null instruction means "reuse the previous status when available".
            # ``default_instruction`` is optional and is only used as an explicit fallback
            # when the runner cannot reuse a previous status message (stage failed)
            if self.default_instruction is not None:
                if "content" not in self.default_instruction:
                    raise TypeError(
                        "You need to define the content of the default_instruction in "
                        f"'content' field (str) : {self.default_instruction}"
                    )
                role = self.default_instruction.get("author", None)
                if role not in ["SYSTEM", "USER", "status", "system", "user"]:
                    raise TypeError(
                        "You need to define a valid author (SYSTEM/USER) in the "
                        f"default_instruction in the field 'author'. Got {role}"
                    )
                if role in ["status", "system"]:
                    self.default_instruction["author"] = "SYSTEM"
                elif role == "user":
                    self.default_instruction["author"] = "USER"
                self.default_instruction.setdefault("timestamp", 0)
            return

        if isinstance(instruction, str) and instruction:
            self.instruction = self._build_explicit_instruction(inputs)
            self.default_instruction = inputs.get("default_instruction", None)
            if self.default_instruction is not None:
                raise TypeError(
                    "You are defining a default_instruction whereas a real instruction "
                    "was provided. The default_instruction is not necessary."
                )
            return

        raise ValueError("Each stage must either have a None instruction (using precedent return) or a string")

    def _get_positive_max_step(self, inputs: Dict, default: Optional[int] = None) -> int:
        max_try = inputs.get("max_step", default)
        if not isinstance(max_try, int) or max_try <= 0:
            raise ValueError(f"Each step must have a number of max agents steps valid. Get {max_try}.")
        return max_try

    def _extract_should_reset_env(self, inputs: Dict) -> bool:
        # Keep support for historical aliases already present in benchmark jsons.
        return (
            inputs.get("should_reset_env", False)
            or inputs.get("should_env_reset", False)
            or inputs.get("should_reset", False)
        )

    def _parse_keys_evaluator(self, k_eval: List[str]) -> List[str]:
        if not isinstance(k_eval, list):
            raise TypeError(f"keys_evaluator must be a list. Got {type(k_eval)}.")

        out = []
        for key in k_eval:
            if key not in KNOWN_CRITERIA:
                raise ValueError(f"Key {key} is not a known criteria.")
            if key not in out:
                out.append(key)
        return out

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
            raise TypeError("Field 'injection' must be either a dict or a list of dict.")

        for i, raw_injection in enumerate(raw_injections):
            if not isinstance(raw_injection, dict):
                raise TypeError(
                    f"Injection at index {i} must be a dict. Got {type(raw_injection)}."
                )

            injection = StageInjection(
                mode=raw_injection.get("mode", ""),
                error=raw_injection.get("error"),
                arguments=raw_injection.get("arguments") or raw_injection.get("argument"),
                message=raw_injection.get("message"),
            )
            injection.verify()
            normalized.append(injection)

        return normalized

    def _validate_injections(self):
        """
        Validate stage-level injection constraints and mirror legacy flags for
        the existing runner implementation.
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
            raise TypeError("A stage can not define more than one force_recovery injection.")
        if len(failure_injections) > 1:
            raise TypeError("A stage can not define more than one force_failure injection.")
        if recovery_injections and failure_injections:
            raise TypeError("A stage can not mix force_recovery and force_failure injections.")

        force_injections = recovery_injections + failure_injections
        if force_injections and (
            self.expected_behavior != "act"
            or (not self._has_action_goals() and self._comp_eval.get("logs", []) == [])
        ):
            raise TypeError(
                "A stage can not define force_recovery/force_failure injections while "
                "not expecting an action or while missing predicates/log verifications."
            )

        if force_injections and self.answer_user:
            raise TypeError(
                "A stage can not define force_recovery/force_failure injections together "
                "with answer_to_user/flag_answer_to_user."
            )

        if failure_injections and not self._has_action_goals() and self._comp_eval.get("logs", []) == []:
            raise TypeError(
                "A stage marked as force_failure must define either an action_goal or a log eval."
            )

    def _update_stage_horizon(self):
        self.stage_horizon = self.max_agents_step
        if self.answer_user:
            self.stage_horizon += 1
        if self.has_flag_recovery() or self.has_flag_failure():
            self.stage_horizon += 1

    def _has_action_goals(self) -> bool:
        return len(self.predicates) > 0 or len(self._predicate_specs) > 0

    def _resolve_predicates(self, obs: Dict) -> List[BaseGoal]:
        signature = get_obs_entity_signature(obs)
        if self._predicate_signature != signature:
            self.predicates = compile_predicates_to_goals(self._predicate_specs, obs)
            self._predicate_signature = signature
        return self.predicates

    def get_evaluation_elements(self, obs: Optional[Dict] = None) -> Tuple[List[BaseGoal], Dict]:
        """Allows to gets the evaluation elements (logs, judge, predicates) the inner steps"""
        if obs is not None and len(self._predicate_specs) > 0:
            self._resolve_predicates(obs)
        return self.predicates, self._comp_eval

    def should_reset_env(self) -> bool:
        """Return True if the env should be reset"""
        return self._should_reset_env

    def should_act(self) -> bool:
        """Return True if the inner step should execute action"""
        return self.allow_tool_flag

    def is_acknowledge(self) -> bool:
        return self.expected_behavior == "acknowledge"

    def is_answer(self) -> bool:
        return self.expected_behavior == "answer"

    def is_text_only(self) -> bool:
        return not self.should_act()

    def has_flag_recovery(self) -> bool:
        return self.recovery_flag

    def has_flag_failure(self) -> bool:
        return self.failure_flag

    def get_error_flag(self) -> str:
        if self.recovery_flag:
            for injection in self.injections:
                if injection.mode == "force_recovery" and injection.message is not None:
                    return injection.message
            return "Communication error: Failed to communicate with the robot."

        if self.failure_flag:
            for injection in self.injections:
                if injection.mode == "force_failure" and injection.message is not None:
                    return injection.message
            return "Communication error: Failed to communicate with the robot."

        raise RuntimeError("Asking for an error flag but got nothing")

    def has_flag_answer_to_user(self) -> bool:
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


class AcknowledgeStage(Stage):
    def __init__(self, inputs: Dict, id: str) -> None:
        self._setup_base(id, "acknowledge")
        self._extract_allows_tools(inputs, "acknowledge")
        self._assert_only_allowed_fields(
            inputs,
            {"instruction", "expected_behavior", "timestamp"},
            "acknowledge",
        )
        self.instruction = self._build_explicit_instruction(inputs)
        self._update_stage_horizon()


class AnswerStage(Stage):
    def __init__(self, inputs: Dict, id: str) -> None:
        self._setup_base(id, "answer")
        self._assert_only_allowed_fields(
            inputs,
            {
                "instruction",
                "expected_behavior",
                "complementary_verif",
                "max_step",
                "keys_evaluator",
                "timestamp",
                "should_reset_env",
                "should_env_reset",
                "should_reset",
                "allows_tools",
            },
            "answer",
        )
        self.instruction = self._build_explicit_instruction(inputs)
        self.max_agents_step = self._get_positive_max_step(inputs, default=1)
        self._comp_eval = inputs.get("complementary_verif", {})
        if not isinstance(self._comp_eval, dict) or self._comp_eval == {}:
            raise TypeError(
                "A stage with expected_behavior='answer' must define a non-empty "
                "dict in complementary_verif."
            )
        self._should_reset_env = self._extract_should_reset_env(inputs)
        self.keys_evaluator = self._parse_keys_evaluator(inputs.get("keys_evaluator", []))
        self.allow_tool_flag = self._extract_allows_tools(inputs, "answer")
        self._update_stage_horizon()

class ActStage(Stage):
    def _parse_predicate_specs(self, predicates: List[Dict]) -> List[Dict]:
        if not isinstance(predicates, list):
            raise TypeError(f"action_goal must be a list. Got {type(predicates)}.")
        for i, predicate in enumerate(predicates):
            if not isinstance(predicate, dict):
                raise TypeError(f"action_goal[{i}] must be a dict. Got {type(predicate)}.")
        return predicates

    def __init__(self, inputs: Dict, id: str) -> None:
        self._setup_base(id, "act")
        self._extract_allows_tools(inputs, "act")
        self._init_act_instruction(inputs)
        self.max_agents_step = self._get_positive_max_step(inputs)
        self._predicate_specs = self._parse_predicate_specs(inputs.get("action_goal", []))
        self._comp_eval = inputs.get("complementary_verif", {})
        self.answer_user = inputs.get("answer_to_user", False) or inputs.get("flag_answer_to_user", False)
        self.injections = self._parse_injections(inputs)
        self._validate_injections()
        self._should_reset_env = self._extract_should_reset_env(inputs)
        self.keys_evaluator = self._parse_keys_evaluator(inputs.get("keys_evaluator", []))
        self._update_stage_horizon()
        self.allow_tool_flag = True
