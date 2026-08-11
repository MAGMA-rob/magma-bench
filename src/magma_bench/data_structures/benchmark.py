"""Runtime view of a compiled MAGMA benchmark.

Artifact models mirror the normalized JSON files written by the benchmark
compiler. The two dataclasses below are the smaller, direct view consumed by
the evaluation pipeline after loading those files.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Type

from magma_core.base.tools import BaseToolsAPI
from magma_core.base.skills import BaseSkill

from magma_bench.artifacts import (
    EpisodeMetadata,
    InterventionSpec,
    SemanticManifest,
    StageSpec,
)


@dataclass(frozen=True)
class Episode:
    """One executable benchmark condition with all required persisted data."""

    episode_id: str
    skeleton_id: str
    condition: str
    initialization: Dict[str, Any]
    stages: Tuple[StageSpec, ...]
    semantic: SemanticManifest
    interventions: Tuple[InterventionSpec, ...]
    metadata: EpisodeMetadata


@dataclass(frozen=True)
class Scenario:
    """A benchmark domain and the compiled episodes evaluated within it."""

    scenario_id: str
    name: str
    track: str
    environment_id: str
    tools_cls: Type[BaseToolsAPI]
    skill_types: Tuple[Type[BaseSkill], ...]
    episodes: Tuple[Episode, ...]
