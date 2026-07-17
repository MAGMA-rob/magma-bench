from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0"
CONDITIONS = (
    "clean",
    "mission_update",
    "interruption",
    "noise",
    "execution_error",
    "combined",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScenarioManifest(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    scenario_id: str
    name: str
    track: Literal["in_domain", "held_out_domain", "compositional"]
    definition: str
    environment_id: str
    description: str = ""


class SkeletonManifest(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    skeleton_id: str
    scenario_id: str
    source_definition: str
    definition_arguments: Dict[str, Any] = Field(default_factory=dict)
    generation_seed: int
    requested_length_bucket: str
    requested_tool_call_range: List[int]
    initialization: Dict[str, Any]

    @model_validator(mode="after")
    def validate_range(self) -> "SkeletonManifest":
        if len(self.requested_tool_call_range) != 2:
            raise ValueError("requested_tool_call_range must contain [minimum, maximum]")
        if self.requested_tool_call_range[0] > self.requested_tool_call_range[1]:
            raise ValueError("requested_tool_call_range has reversed bounds")
        return self


class ObjectSpec(StrictModel):
    type: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class InstructionSpec(ObjectSpec):
    pass


class StageInputSpec(StrictModel):
    instruction: InstructionSpec
    answer_after_completion: bool = False
    linked_to_previous: bool = False


class StagePresentationSpec(StrictModel):
    stage_input: StageInputSpec
    verification_prompt: Optional[str] = None
    stage_goal_description: str


class LogRuleSpec(StrictModel):
    rule_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ActiveErrorSpec(StrictModel):
    error_type: str
    constructor_arguments: Dict[str, Any] = Field(default_factory=dict)
    runtime_arguments: Dict[str, Any] = Field(default_factory=dict)


class DeclarativeStageSpec(StrictModel):
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
    kind: Literal["serialized"] = "serialized"
    id: str
    stage_type: str
    arguments: Dict[str, Any]
    expected_behavior: Literal["act", "answer", "acknowledge"]
    presentation: StagePresentationSpec
    active_errors: List[ActiveErrorSpec] = Field(default_factory=list)


StageSpec = Union[DeclarativeStageSpec, SerializedStageSpec]


class InterventionSpec(StrictModel):
    type: Literal[
        "mission_update",
        "interruption",
        "noise",
        "execution_error",
        "combined",
    ]
    injected_at_stage: str
    relevant_at_stage: Optional[str] = None


class EpisodeSpec(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    episode_id: str
    scenario_id: str
    skeleton_id: str
    condition: Literal[
        "clean",
        "mission_update",
        "interruption",
        "noise",
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
    schema_version: Literal["1.0"] = SCHEMA_VERSION
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
    clean_oracle_tool_calls: int = Field(ge=0)
    episode_oracle_tool_calls: int = Field(ge=0)
    length_bucket: str
    number_of_stages: int = Field(ge=1)
    number_of_interventions: int = Field(ge=0)
    intervention_lags: List[Optional[int]]
    control_episode_id: str


class CompiledEpisodeSpec(EpisodeSpec):
    semantic_id: str
    metadata: EpisodeMetadata


class EpisodeIndexEntry(StrictModel):
    episode_id: str
    scenario_id: str
    skeleton_id: str
    semantic_id: str
    condition: str
    length_bucket: str
    path: str


class BenchmarkManifest(StrictModel):
    schema_version: Literal["1.0"] = SCHEMA_VERSION
    benchmark_version: str
    generated_at: str
    semantic_variations: int = Field(ge=1)
    scenarios: List[str]
    episodes: List[EpisodeIndexEntry]


def load_json_model(path: Path, model_type: type[StrictModel]) -> StrictModel:
    return model_type.model_validate_json(path.read_text(encoding="utf-8"))
