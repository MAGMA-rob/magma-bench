"""Conversion between persistent artifact schemas and executable MAGMA types.

``models.py`` deliberately stays simulator-agnostic and JSON serializable.
Functions here are the boundary that invokes MAGMA registries to reconstruct
instructions, goals, errors and stages for validation or execution.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, List, Optional

from magma_core.simulation.data_structures import Instruction, StageInput
from magma_core.simulation.errors import BaseError
from magma_core.simulation.goals import BaseGoal
from magma_core.simulation.randomizer import RandomizationSpec, RuntimeRandomizer
from magma_core.simulation.stage import (
    BaseTaskStage,
    StageErrorParameters,
    StageGlobalParameters,
)
from magma_core.simulation.tasks import BaseTask
from magma_core.simulation.serialization import canonical_type_name, decode_value, encode_value

from magma_bench.evalutations.log_rules import compile_log_rules

from .models import (
    ActiveErrorSpec,
    DeclarativeStageSpec,
    InstructionSpec,
    ObjectSpec,
    SerializedStageSpec,
    SemanticManifest,
    StageInputSpec,
    StagePresentationSpec,
    StageSpec,
)


def deserialize_runtime_randomizer(
    semantic: SemanticManifest,
) -> RuntimeRandomizer:
    """Rebuild an independent runtime translator from a semantic manifest."""

    return RuntimeRandomizer(
        RandomizationSpec(
            tools=decode_value(deepcopy(semantic.tools)),
            tool_equivalence=decode_value(deepcopy(semantic.tool_equivalence)),
            attributes_equivalence=decode_value(
                deepcopy(semantic.attributes_equivalence)
            ),
            reversed_attributes_equivalence=decode_value(
                deepcopy(semantic.reversed_attributes_equivalence)
            ),
            attribute_keys_order=list(semantic.attribute_keys_order),
            attribute_values_order=decode_value(
                deepcopy(semantic.attribute_values_order)
            ),
        )
    )


def serialize_goal(goal: BaseGoal) -> ObjectSpec:
    """Convert a registered runtime goal into its persistent specification."""

    return ObjectSpec.model_validate(goal.to_spec())


def deserialize_goal(spec: ObjectSpec) -> BaseGoal:
    """Reconstruct a goal through the ``BaseGoal`` serialization registry."""

    return BaseGoal.from_spec(spec.model_dump(mode="python"))


def serialize_error(
    error: BaseError,
    runtime_arguments: Optional[dict] = None,
) -> ActiveErrorSpec:
    """Persist an error definition and its episode-specific runtime settings."""

    error_spec = error.to_spec()
    return ActiveErrorSpec(
        error_type=error_spec["type"],
        constructor_arguments=error_spec["arguments"],
        runtime_arguments={} if runtime_arguments is None else runtime_arguments,
    )


def deserialize_error(spec: ActiveErrorSpec) -> BaseError:
    """Reconstruct and validate an error before attaching it to a stage."""

    error = BaseError.from_spec({
        "type": spec.error_type,
        "arguments": spec.constructor_arguments,
    })
    error.validate_arguments(spec.runtime_arguments)
    return error


def stage_presentation(stage: BaseTaskStage) -> StagePresentationSpec:
    """Capture the exact agent/verifier-facing presentation of a core stage."""

    stage_input = stage.get_stage_input()
    return StagePresentationSpec(
        stage_input=StageInputSpec(
            instruction=InstructionSpec.model_validate(
                stage_input.instruction.to_spec()
            ),
            answer_after_completion=stage_input.flag_answer_to_user,
            linked_to_previous=stage_input.linked_to_prev,
        ),
        verification_prompt=stage.get_verification_prompt(),
        stage_goal_description=stage.get_stage_goal_description(),
    )


def _expected_behavior(stage: BaseTaskStage) -> str:
    """Resolve whether the agent must act, answer, or acknowledge the stage."""

    declared_behavior = getattr(stage, "expected_behavior", None)
    if declared_behavior is not None:
        if declared_behavior not in {"act", "answer", "acknowledge"}:
            raise ValueError(
                f"Invalid expected_behavior {declared_behavior!r} on "
                f"{type(stage).__name__}"
            )
        return declared_behavior
    if not stage.is_text_only():
        return "act"
    if stage.get_stage_input().instruction.has_constraint:
        return "acknowledge"
    return "answer"


def apply_stage_presentation(
    stage: BaseTaskStage,
    presentation: StagePresentationSpec,
) -> None:
    """Restore persisted presentation fields on a reconstructed core stage."""

    stage_input = presentation.stage_input
    instruction = Instruction.from_spec(
        stage_input.instruction.model_dump(mode="python")
    )
    stage.stage_input = StageInput(
        instruction=instruction,
        flag_answer_to_user=stage_input.answer_after_completion,
        linked_to_prev=stage_input.linked_to_previous,
    )
    stage.global_parameters.verification_prompt = presentation.verification_prompt
    stage.stage_goal_description = presentation.stage_goal_description


class DeclarativeStage(BaseTaskStage):
    """Executable counterpart of :class:`DeclarativeStageSpec`.

    This adapter allows benchmark authors to assemble a stage from registered
    goals, log rules and errors without creating a dedicated Python stage class.
    It behaves like any other ``BaseTaskStage`` once reconstructed.
    """

    target_tool_calls = 0
    max_tool_calls = 0

    def __init__(self, spec: DeclarativeStageSpec) -> None:
        self.target_tool_calls = spec.target_tool_calls
        self.max_tool_calls = spec.max_tool_calls
        presentation = spec.presentation
        stage_input = presentation.stage_input
        errors = [deserialize_error(error) for error in spec.active_errors]
        super().__init__(
            goals=[deserialize_goal(goal) for goal in spec.goals],
            stage_goal_description=presentation.stage_goal_description,
            stage_input=StageInput(
                instruction=Instruction.from_spec(
                    stage_input.instruction.model_dump(mode="python")
                ),
                flag_answer_to_user=stage_input.answer_after_completion,
                linked_to_prev=stage_input.linked_to_previous,
            ),
            global_parameters=StageGlobalParameters(
                reset_at_end=spec.reset_environment_after,
                additive_stage=spec.additive,
                verification_prompt=presentation.verification_prompt,
                allow_tools_before_answer=spec.allow_tools_before_answer,
                allowed_tools=list(spec.allowed_tools),
            ),
            error_parameters=StageErrorParameters(
                possible_errors=errors,
                min_active_errors=len(errors),
                max_active_errors=len(errors),
            ),
        )
        self.benchmark_log_rules = compile_log_rules([
            rule.model_dump(mode="python") for rule in spec.log_rules
        ])
        self.active_error_arguments_override = {
            error.get_name(): dict(error_spec.runtime_arguments)
            for error, error_spec in zip(errors, spec.active_errors)
        }
        self.benchmark_expected_behavior = spec.type


class DeclarativeActionStage(DeclarativeStage):
    """Declarative action stage with optional log-based completion rules."""

    def verif_log_completion(self, stage_log, full_log) -> int:
        if not self.benchmark_log_rules:
            return 0
        return int(all(rule.verify(stage_log)[0] for rule in self.benchmark_log_rules))


def serialize_stage(stage: BaseTaskStage, stage_id: str) -> SerializedStageSpec:
    """Serialize one registered Python stage without losing its presentation."""

    core_spec = stage.to_spec()
    return SerializedStageSpec(
        id=stage_id,
        stage_type=core_spec["type"],
        arguments=core_spec["arguments"],
        expected_behavior=_expected_behavior(stage),
        presentation=stage_presentation(stage),
        reset_at_end=stage.should_reset_at_end(),
        additive_stage=stage.is_additive_stage(),
        allow_tools_before_answer=stage.allows_tools_before_answer(),
        allowed_tools=list(stage.get_allowed_tools()),
        active_errors=[],
    )


def deserialize_serialized_stage(
    spec: SerializedStageSpec,
    *,
    validate_presentation: bool = False,
) -> BaseTaskStage:
    """Rebuild a registered Python stage from a ``SerializedStageSpec``.

    Optional validation detects constructor-argument drift and validates the
    fully restored stage.
    """

    stage = BaseTaskStage.from_spec({
        "type": spec.stage_type,
        "arguments": spec.arguments,
    })
    if validate_presentation:
        round_trip_spec = stage.to_spec()
        if (
            round_trip_spec["type"] != canonical_type_name(spec.stage_type)
            or round_trip_spec["arguments"] != encode_value(decode_value(spec.arguments))
        ):
            raise ValueError(
                f"Serialized stage {spec.id!r} arguments do not round-trip exactly"
            )
    apply_stage_presentation(stage, spec.presentation)
    stage.global_parameters.reset_at_end = spec.reset_at_end
    stage.global_parameters.additive_stage = spec.additive_stage
    stage.global_parameters.allow_tools_before_answer = (
        spec.allow_tools_before_answer
    )
    stage.global_parameters.allowed_tools = list(spec.allowed_tools)
    active_errors = [deserialize_error(error) for error in spec.active_errors]
    stage.error_parameters.possible_errors = active_errors
    stage.error_parameters.min_active_errors = len(active_errors)
    stage.error_parameters.max_active_errors = len(active_errors)
    stage.active_error_arguments_override = {
        error.get_name(): dict(error_spec.runtime_arguments)
        for error, error_spec in zip(active_errors, spec.active_errors)
    }
    stage.benchmark_expected_behavior = spec.expected_behavior
    if validate_presentation:
        if _expected_behavior(stage) != spec.expected_behavior:
            raise ValueError(
                f"Serialized stage {spec.id!r} expected behavior differs from its "
                "restored presentation"
            )
        stage.validate(["default"])
    return stage


def deserialize_stage(
    spec: StageSpec,
    *,
    validate_presentation: bool = False,
) -> BaseTaskStage:
    """Dispatch reconstruction according to the stage specification ``kind``."""

    if isinstance(spec, DeclarativeStageSpec):
        if spec.type == "act":
            return DeclarativeActionStage(spec)
        return DeclarativeStage(spec)
    return deserialize_serialized_stage(
        spec,
        validate_presentation=validate_presentation,
    )


def serialize_task_stages(task: BaseTask) -> List[SerializedStageSpec]:
    """Serialize every stage of a generated task with stable ordered IDs."""

    serialized: List[SerializedStageSpec] = []
    for index, stage in enumerate(task.stages):
        try:
            serialized.append(serialize_stage(stage, f"stage_{index:03d}"))
        except Exception as error:
            message = (
                f"Cannot serialize stage {index} "
                f"({stage.__class__.__module__}:{stage.__class__.__qualname__}): {error}"
            )
            if isinstance(error, NotImplementedError):
                raise NotImplementedError(message) from error
            if isinstance(error, TypeError):
                raise TypeError(message) from error
            if isinstance(error, ValueError):
                raise ValueError(message) from error
            raise RuntimeError(message) from error
    return serialized


def deserialize_task_stages(
    specs: Iterable[StageSpec],
    *,
    validate_presentation: bool = False,
) -> List[BaseTaskStage]:
    """Reconstruct an ordered sequence containing either stage-spec variant."""

    return [
        deserialize_stage(spec, validate_presentation=validate_presentation)
        for spec in specs
    ]
