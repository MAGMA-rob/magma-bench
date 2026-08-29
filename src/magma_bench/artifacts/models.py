"""Persistent schemas for a compiled MAGMA benchmark.

These Pydantic models describe JSON files, not live simulator objects. The
generator writes them, :mod:`magma_bench.loader` validates and groups them,
and :mod:`magma_bench.artifacts.runtime` reconstructs executable MAGMA objects
from the stage specifications.

The hierarchy on disk is:

``BenchmarkManifest -> ScenarioManifest -> SkeletonManifest
                    -> SemanticManifest -> CompiledEpisodeSpec``
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.6"
Track = Literal["in_domain", "held_out_domain", "compositional"]
Condition = Literal[
    "clean",
    "mission_update",
    "interruption",
    "execution_error",
    "combined",
]
ArtifactSchemaVersion = Literal["1.5", "1.6"]
VariantCondition = Literal[
    "mission_update",
    "interruption",
    "execution_error",
]
CONDITIONS = (
    "clean",
    "mission_update",
    "interruption",
    "execution_error",
    "combined",
)


class StrictModel(BaseModel):
    """Base schema rejecting unknown JSON fields.

    Rejecting extra fields makes format changes explicit: adding a field to a
    generated artifact requires updating the corresponding schema here.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


class ScenarioManifest(StrictModel):
    """Stable metadata shared by every episode of one scenario.

    A scenario identifies a semantic domain and its simulator environment.
    Tracks belong to skeletons because one domain may contain tasks from
    different evaluation tracks.
    """

    schema_version: ArtifactSchemaVersion = SCHEMA_VERSION
    scenario_id: str
    name: str
    definition: str
    environment_id: str
    tools_type: str
    skill_types: List[str]
    description: str = ""


class SkeletonManifest(StrictModel):
    """Generation provenance and initial state of one task skeleton.

    A skeleton is the common task instance from which all experimental
    conditions and semantic variations are derived. ``initialization`` is the
    canonical environment configuration/state needed to start its episodes.
    The actual stage sequence is stored in each :class:`EpisodeSpec`. The
    skeleton track applies to every condition and semantic variation derived
    from it.
    """

    schema_version: ArtifactSchemaVersion = SCHEMA_VERSION
    skeleton_id: str
    scenario_id: str
    track: Track
    conditions: List[Condition] = Field(default_factory=lambda: list(CONDITIONS))
    source_definition: str
    definition_arguments: Dict[str, Any] = Field(default_factory=dict)
    generation_seed: int
    requested_length_bucket: str
    requested_tool_call_range: List[int]
    initialization: Dict[str, Any]

    @model_validator(mode="after")
    def validate_range(self) -> "SkeletonManifest":
        if self.schema_version == "1.6" and "conditions" not in self.model_fields_set:
            raise ValueError("Schema 1.6 skeletons must declare conditions")
        if len(self.requested_tool_call_range) != 2:
            raise ValueError("requested_tool_call_range must contain [minimum, maximum]")
        if self.requested_tool_call_range[0] > self.requested_tool_call_range[1]:
            raise ValueError("requested_tool_call_range has reversed bounds")
        if len(set(self.conditions)) != len(self.conditions):
            raise ValueError("Skeleton conditions must be unique")
        if "clean" not in self.conditions:
            raise ValueError("Skeleton conditions must include clean")
        combined_expected = {
            "mission_update",
            "execution_error",
        }.issubset(self.conditions)
        if ("combined" in self.conditions) != combined_expected:
            raise ValueError(
                "combined must be present if and only if mission_update and "
                "execution_error are present"
            )
        return self


class ObjectSpec(StrictModel):
    """Generic serialized reference to a registered MAGMA object.

    ``type`` is the registry identifier and ``arguments`` are passed to the
    object's reconstruction contract. Goals use this schema directly;
    instructions specialize it through :class:`InstructionSpec`.
    """

    type: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class InstructionSpec(ObjectSpec):
    """Serialized user/system instruction reconstructed by ``Instruction``."""


class StageInputSpec(StrictModel):
    """Instruction and conversation flags presented when a stage starts."""

    instruction: InstructionSpec
    answer_after_completion: bool = False
    linked_to_previous: bool = False


class StagePresentationSpec(StrictModel):
    """Agent-visible and verifier-facing presentation of a stage.

    This information is stored separately from a serialized core stage so that
    benchmark compilation can preserve the exact prompt and expected semantics,
    even when the core class derives part of them dynamically.
    """

    stage_input: StageInputSpec
    verification_prompt: Optional[str] = None
    stage_goal_description: str


class LogRuleSpec(StrictModel):
    """Declarative rule used to verify completion from execution logs."""

    rule_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ActiveErrorSpec(StrictModel):
    """Serialized error injection attached to an action stage.

    Constructor arguments rebuild the error object. Runtime arguments configure
    how that error behaves in this particular stage/episode.
    """

    error_type: str
    constructor_arguments: Dict[str, Any] = Field(default_factory=dict)
    runtime_arguments: Dict[str, Any] = Field(default_factory=dict)


class DeclarativeStageSpec(StrictModel):
    """Self-contained stage authored directly in benchmark JSON.

    It is independent of a concrete scenario stage class. At runtime it becomes
    :class:`magma_bench.artifacts.runtime.DeclarativeStage`, with goals, log
    rules and errors reconstructed from their registry specifications.
    """

    kind: Literal["declarative"] = "declarative"
    id: str
    type: Literal["act", "answer", "acknowledge"]
    presentation: StagePresentationSpec
    target_tool_calls: int = Field(ge=0)
    max_tool_calls: int = Field(ge=0)
    reset_environment_after: bool = False
    additive: bool = False
    allow_tools_before_answer: bool = False
    allowed_tools: List[str] = Field(default_factory=list)
    goals: List[ObjectSpec] = Field(default_factory=list)
    log_rules: List[LogRuleSpec] = Field(default_factory=list)
    active_errors: List[ActiveErrorSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_stage(self) -> "DeclarativeStageSpec":
        if self.target_tool_calls > self.max_tool_calls:
            raise ValueError("target_tool_calls must not exceed max_tool_calls")
        prompt = self.presentation.verification_prompt
        if self.type == "act" and prompt is not None:
            raise ValueError("An act stage cannot define a verification prompt")
        if self.type in {"answer", "acknowledge"} and prompt is None:
            raise ValueError(f"A {self.type} stage must define a verification prompt")
        if self.type in {"answer", "acknowledge"} and self.goals:
            raise ValueError("Text stages cannot define environment goals")
        if self.type != "act" and self.active_errors:
            raise ValueError("Only act stages can define active_errors")
        if self.allowed_tools and not self.allow_tools_before_answer:
            raise ValueError("allowed_tools requires allow_tools_before_answer=true")
        return self


class SerializedStageSpec(StrictModel):
    """Snapshot of an existing registered ``BaseTaskStage``.

    ``stage_type`` and ``arguments`` reconstruct the original core/scenario
    class. ``presentation`` preserves what the agent and verifier must observe.
    Use this form when a stage already has a reusable Python implementation.
    """

    kind: Literal["serialized"] = "serialized"
    id: str
    stage_type: str
    arguments: Dict[str, Any]
    expected_behavior: Literal["act", "answer", "acknowledge"]
    presentation: StagePresentationSpec
    reset_at_end: bool = False
    additive_stage: bool = False
    allow_tools_before_answer: bool = False
    allowed_tools: List[str] = Field(default_factory=list)
    active_errors: List[ActiveErrorSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_runtime_parameters(self) -> "SerializedStageSpec":
        if self.allowed_tools and not self.allow_tools_before_answer:
            raise ValueError("allowed_tools requires allow_tools_before_answer=true")
        return self


# An episode may freely mix manually authored stages and registered Python
# stages. The ``kind`` discriminator tells the runtime how to reconstruct each.
StageSpec = Union[DeclarativeStageSpec, SerializedStageSpec]


class InterventionSpec(StrictModel):
    """Experimental event injected into an episode.

    ``injected_at_stage`` locates when the event appears. ``relevant_at_stage``
    optionally marks the later stage at which remembering that event matters,
    which allows the generator/evaluator to derive an intervention lag.
    """

    type: Literal[
        "mission_update",
        "interruption",
        "execution_error",
        "combined",
    ]
    injected_at_stage: str
    relevant_at_stage: Optional[str] = None


class EpisodeSpec(StrictModel):
    """Complete ordered task execution for one experimental condition.

    An episode is one condition (clean, interruption, error, etc.) of a
    skeleton. It owns its stage sequence because different conditions can add,
    remove or reorder stages; stages therefore cannot live only on the shared
    :class:`SkeletonManifest`.

    This is the authoring-level form. The compiled form is
    :class:`CompiledEpisodeSpec`, which adds a semantic variation and metrics
    metadata.
    """

    schema_version: ArtifactSchemaVersion = SCHEMA_VERSION
    episode_id: str
    scenario_id: str
    skeleton_id: str
    condition: Literal[
        "clean",
        "mission_update",
        "interruption",
        "execution_error",
        "combined",
    ]
    interventions: List[InterventionSpec] = Field(default_factory=list)
    stages: List[StageSpec]

    @model_validator(mode="after")
    def validate_episode(self) -> "EpisodeSpec":
        if not self.stages:
            raise ValueError("An episode must contain at least one stage")
        stage_ids = [stage.id for stage in self.stages]
        duplicates = sorted(
            {stage_id for stage_id in stage_ids if stage_ids.count(stage_id) > 1}
        )
        if duplicates:
            raise ValueError(f"Duplicated stage IDs: {duplicates}")
        known_ids = set(stage_ids)
        for intervention in self.interventions:
            if intervention.injected_at_stage not in known_ids:
                raise ValueError(
                    f"Unknown injected_at_stage {intervention.injected_at_stage!r}"
                )
            if (
                intervention.relevant_at_stage is not None
                and intervention.relevant_at_stage not in known_ids
            ):
                raise ValueError(
                    f"Unknown relevant_at_stage {intervention.relevant_at_stage!r}"
                )
        if self.condition == "clean" and self.interventions:
            raise ValueError("The clean episode cannot contain interventions")
        if self.condition != "clean" and not self.interventions:
            raise ValueError(f"The {self.condition} episode must declare interventions")
        if self.condition not in {"clean", "combined"}:
            mismatched = [
                intervention.type
                for intervention in self.interventions
                if intervention.type != self.condition
            ]
            if mismatched:
                raise ValueError(
                    f"The {self.condition} episode contains mismatched interventions: "
                    f"{mismatched}"
                )
        if self.condition == "combined" and len(self.interventions) < 2:
            raise ValueError("The combined episode must contain at least two interventions")
        return self


class SemanticManifest(StrictModel):
    """Frozen semantic randomization applied to one skeleton.

    It contains the exact agent-visible tools and attributes plus both
    translation directions. Real simulator values remain canonical at the
    low-level boundary; these mappings translate prompts/tool calls between the
    canonical vocabulary and this episode's randomized vocabulary.

    The same manifest is shared by all conditions of a semantic variation,
    ensuring that clean and perturbed episodes remain paired.
    """

    schema_version: ArtifactSchemaVersion = SCHEMA_VERSION
    semantic_id: str
    variation_index: int = Field(ge=0)
    tools: List[Dict[str, Any]]
    visible_attributes: Dict[str, Any]
    tool_equivalence: Dict[str, Dict[str, Any]]
    attributes_equivalence: Dict[str, str]
    reversed_attributes_equivalence: Dict[str, str]
    attribute_keys_order: List[str]
    attribute_values_order: Dict[str, List[str]]


class EpisodeMetadata(StrictModel):
    """Derived, immutable descriptors used for slicing and paired evaluation.

    The oracle counts and length bucket describe difficulty. Intervention lags
    support delayed-rule analyses. ``control_episode_id`` points to the clean
    episode with the same skeleton and semantic variation.
    """

    clean_oracle_tool_calls: int = Field(ge=0)
    episode_oracle_tool_calls: int = Field(ge=0)
    length_bucket: str
    number_of_stages: int = Field(ge=1)
    number_of_interventions: int = Field(ge=0)
    intervention_lags: List[Optional[int]]
    control_episode_id: str


class CompiledEpisodeSpec(EpisodeSpec):
    """Runtime-ready episode produced by the benchmark compiler.

    Its stages already contain randomized, agent-visible wording.
    ``semantic_id`` links back to the :class:`SemanticManifest` required to
    translate tool calls, while ``metadata`` supports grouping and metrics.
    """

    semantic_id: str
    metadata: EpisodeMetadata


class EpisodeIndexEntry(StrictModel):
    """Lightweight pointer from ``benchmark.json`` to one episode JSON file.

    Identity fields and the skeleton track are duplicated intentionally so the
    loader can detect a misplaced, stale or incorrectly generated file before
    evaluation starts.
    """

    episode_id: str
    scenario_id: str
    skeleton_id: str
    track: Track
    semantic_id: str
    condition: str
    length_bucket: str
    path: str


class BenchmarkManifest(StrictModel):
    """Root index of one immutable compiled benchmark build.

    It lists scenario IDs and every compiled episode path. The loader treats
    this file as authoritative and checks that the indexed files and directory
    tree match exactly.
    """
    schema_version: ArtifactSchemaVersion = SCHEMA_VERSION
    benchmark_version: str
    generated_at: str
    semantic_variations: int = Field(ge=1)
    scenarios: List[str]
    episodes: List[EpisodeIndexEntry]


def load_json_model(path: Path, model_type: type[StrictModel]) -> StrictModel:
    """Read one JSON artifact and validate it against its persistent schema."""

    return model_type.model_validate_json(path.read_text(encoding="utf-8"))
