"""Decode the normalized benchmark artifact tree into runtime scenarios."""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from magma_core.base.tools import BaseToolsAPI
from magma_core.base.skills import BaseSkill, CancelCurrentActionSkill
from magma_core.serialization import load_spec

from magma_bench.artifacts import (
    CONDITIONS,
    BenchmarkManifest,
    CompiledEpisodeSpec,
    ScenarioManifest,
    SemanticManifest,
    SkeletonManifest,
    load_json_model,
)
from magma_bench.data_structures import Episode, Scenario


ScenarioSelection = Optional[Union[str, Sequence[str]]]


def load_scenarios(
    root: Path,
    scenarios: ScenarioSelection = None,
) -> List[Scenario]:
    """Load selected scenarios from a complete compiled benchmark.

    Selection accepts scenario IDs or display names. ``None`` and ``"all"``
    load every scenario. The benchmark index and selected artifact files are
    validated before runtime dataclasses are returned.
    """

    root = root.resolve()
    manifest_model = load_json_model(root / "benchmark.json", BenchmarkManifest)
    if not isinstance(manifest_model, BenchmarkManifest):
        raise TypeError(f"Unexpected benchmark manifest loaded from {root}")
    manifest = manifest_model

    if len(set(manifest.scenarios)) != len(manifest.scenarios):
        raise ValueError("Duplicated scenario IDs in benchmark manifest")

    indexed_paths: Set[str] = set()
    indexed_episode_ids: Set[str] = set()
    for entry in manifest.episodes:
        if entry.path in indexed_paths:
            raise ValueError(f"Duplicated episode path in benchmark index: {entry.path}")
        indexed_paths.add(entry.path)
        if entry.episode_id in indexed_episode_ids:
            raise ValueError(
                f"Duplicated episode ID in benchmark index: {entry.episode_id}"
            )
        indexed_episode_ids.add(entry.episode_id)

    actual_paths = {
        str(path.relative_to(root))
        for path in root.glob("*/skeleton_*/semantic_*/episodes/*.json")
    }
    if actual_paths != indexed_paths:
        raise ValueError(
            "Compiled episode tree differs from benchmark index: "
            f"missing={sorted(indexed_paths - actual_paths)}, "
            f"unindexed={sorted(actual_paths - indexed_paths)}"
        )

    indexed_scenario_ids = {entry.scenario_id for entry in manifest.episodes}
    if set(manifest.scenarios) != indexed_scenario_ids:
        raise ValueError(
            f"Benchmark scenario index mismatch: manifest={sorted(manifest.scenarios)}, "
            f"episodes={sorted(indexed_scenario_ids)}"
        )

    scenario_manifests: Dict[str, ScenarioManifest] = {}
    for scenario_id in manifest.scenarios:
        scenario_path = root / scenario_id / "scenario.json"
        scenario_model = load_json_model(scenario_path, ScenarioManifest)
        if not isinstance(scenario_model, ScenarioManifest):
            raise TypeError(f"Unexpected scenario model loaded from {scenario_path}")
        if scenario_model.scenario_id != scenario_id:
            raise ValueError(f"Scenario identity mismatch in {scenario_path}")
        scenario_manifests[scenario_id] = scenario_model

    if scenarios is None:
        selected_ids = set(manifest.scenarios)
    else:
        requested = {scenarios} if isinstance(scenarios, str) else set(scenarios)
        if not requested:
            raise ValueError("Scenario selection cannot be empty")
        if "all" in requested:
            if requested != {"all"}:
                raise ValueError("'all' cannot be combined with scenario selectors")
            selected_ids = set(manifest.scenarios)
        else:
            selected_ids = {
                scenario_id
                for scenario_id, scenario in scenario_manifests.items()
                if scenario_id in requested or scenario.name in requested
            }
            matched_selectors = {
                selector
                for selector in requested
                if any(
                    selector in {scenario_id, scenario.name}
                    for scenario_id, scenario in scenario_manifests.items()
                )
            }
            unknown = requested - matched_selectors
            if unknown:
                available = sorted(scenario_manifests)
                raise ValueError(
                    f"Unknown benchmark scenarios: {sorted(unknown)}. "
                    f"Available scenario IDs: {available}"
                )

    selected_entries = [
        entry for entry in manifest.episodes if entry.scenario_id in selected_ids
    ]
    semantics_by_skeleton: Dict[Tuple[str, str], Set[str]] = {}
    conditions_by_semantic: Dict[Tuple[str, str, str], Set[str]] = {}
    for entry in selected_entries:
        skeleton_key = (entry.scenario_id, entry.skeleton_id)
        semantics_by_skeleton.setdefault(skeleton_key, set()).add(entry.semantic_id)
        semantic_key = (*skeleton_key, entry.semantic_id)
        conditions_by_semantic.setdefault(semantic_key, set()).add(entry.condition)

    for skeleton_key, semantic_ids in semantics_by_skeleton.items():
        if len(semantic_ids) != manifest.semantic_variations:
            raise ValueError(
                f"Skeleton {skeleton_key} has {len(semantic_ids)} semantic variations, "
                f"expected {manifest.semantic_variations}"
            )

    expected_conditions = set(CONDITIONS)
    for semantic_key, conditions in conditions_by_semantic.items():
        if conditions != expected_conditions:
            raise ValueError(
                f"Semantic group {semantic_key} conditions mismatch: "
                f"missing={sorted(expected_conditions - conditions)}, "
                f"unexpected={sorted(conditions - expected_conditions)}"
            )

    skeletons: Dict[Tuple[str, str], SkeletonManifest] = {}
    semantics: Dict[Tuple[str, str, str], SemanticManifest] = {}
    episodes_by_scenario: Dict[str, List[Episode]] = {
        scenario_id: [] for scenario_id in selected_ids
    }

    for entry in selected_entries:
        skeleton_key = (entry.scenario_id, entry.skeleton_id)
        skeleton = skeletons.get(skeleton_key)
        if skeleton is None:
            skeleton_path = root / entry.scenario_id / entry.skeleton_id / "skeleton.json"
            skeleton_model = load_json_model(skeleton_path, SkeletonManifest)
            if not isinstance(skeleton_model, SkeletonManifest):
                raise TypeError(f"Unexpected skeleton model loaded from {skeleton_path}")
            if (
                skeleton_model.scenario_id != entry.scenario_id
                or skeleton_model.skeleton_id != entry.skeleton_id
            ):
                raise ValueError(f"Skeleton identity mismatch in {skeleton_path}")
            skeleton = skeleton_model
            skeletons[skeleton_key] = skeleton

        semantic_key = (*skeleton_key, entry.semantic_id)
        semantic = semantics.get(semantic_key)
        if semantic is None:
            semantic_path = (
                root
                / entry.scenario_id
                / entry.skeleton_id
                / entry.semantic_id
                / "semantic.json"
            )
            semantic_model = load_json_model(semantic_path, SemanticManifest)
            if not isinstance(semantic_model, SemanticManifest):
                raise TypeError(f"Unexpected semantic model loaded from {semantic_path}")
            if semantic_model.semantic_id != entry.semantic_id:
                raise ValueError(f"Semantic identity mismatch in {semantic_path}")
            semantic = semantic_model
            semantics[semantic_key] = semantic

        episode_path = root / entry.path
        episode_model = load_json_model(episode_path, CompiledEpisodeSpec)
        if not isinstance(episode_model, CompiledEpisodeSpec):
            raise TypeError(f"Unexpected episode model loaded from {episode_path}")
        expected_identity = (
            entry.episode_id,
            entry.scenario_id,
            entry.skeleton_id,
            entry.semantic_id,
            entry.condition,
            entry.length_bucket,
        )
        actual_identity = (
            episode_model.episode_id,
            episode_model.scenario_id,
            episode_model.skeleton_id,
            episode_model.semantic_id,
            episode_model.condition,
            episode_model.metadata.length_bucket,
        )
        if actual_identity != expected_identity:
            raise ValueError(
                f"Episode index identity mismatch for {entry.path}: "
                f"expected={expected_identity}, got={actual_identity}"
            )
        if (
            episode_model.metadata.number_of_interventions
            != len(episode_model.interventions)
            or len(episode_model.metadata.intervention_lags)
            != len(episode_model.interventions)
        ):
            raise ValueError(
                f"Intervention metadata mismatch for {episode_model.episode_id!r}"
            )
        if episode_model.condition == "mission_update":
            measurable_lags = [
                lag
                for lag in episode_model.metadata.intervention_lags
                if lag is not None
            ]
            if len(measurable_lags) != 1:
                raise ValueError(
                    f"Mission-update episode {episode_model.episode_id!r} must "
                    "define exactly one non-null intervention lag"
                )

        episodes_by_scenario[entry.scenario_id].append(
            Episode(
                episode_id=episode_model.episode_id,
                skeleton_id=episode_model.skeleton_id,
                condition=episode_model.condition,
                # All conditions and semantics of a skeleton share this
                # read-only source initialization. Execution will copy it when
                # constructing a mutable BaseTask.
                initialization=skeleton.initialization,
                stages=tuple(episode_model.stages),
                semantic=semantic,
                interventions=tuple(episode_model.interventions),
                metadata=episode_model.metadata,
            )
        )

    loaded_episodes = {
        episode.episode_id: (scenario_id, episode)
        for scenario_id, episodes in episodes_by_scenario.items()
        for episode in episodes
    }
    for scenario_id, episode in loaded_episodes.values():
        control_binding = loaded_episodes.get(episode.metadata.control_episode_id)
        if control_binding is None:
            raise ValueError(
                f"Episode {episode.episode_id!r} references unknown control "
                f"{episode.metadata.control_episode_id!r}"
            )
        control_scenario_id, control = control_binding
        if (
            control_scenario_id != scenario_id
            or control.condition != "clean"
            or control.skeleton_id != episode.skeleton_id
            or control.semantic.semantic_id != episode.semantic.semantic_id
        ):
            raise ValueError(
                f"Invalid clean pairing for episode {episode.episode_id!r}"
            )

    loaded_scenarios: List[Scenario] = []
    for scenario_id in manifest.scenarios:
        if scenario_id not in selected_ids:
            continue
        scenario = scenario_manifests[scenario_id]
        tools_cls, _ = load_spec(
            {"type": scenario.tools_type, "arguments": {}},
            BaseToolsAPI,
        )
        skill_types = tuple(
            load_spec({"type": type_name, "arguments": {}}, BaseSkill)[0]
            for type_name in scenario.skill_types
        )
        if len(set(skill_types)) != len(skill_types):
            raise ValueError(
                f"Scenario {scenario.scenario_id!r} declares duplicate skill types"
            )
        skill_names = [skill_type.spec.name for skill_type in skill_types]
        if len(set(skill_names)) != len(skill_names):
            raise ValueError(
                f"Scenario {scenario.scenario_id!r} declares duplicate skill names"
            )
        if CancelCurrentActionSkill.spec.name in skill_names:
            raise ValueError(
                f"Scenario {scenario.scenario_id!r} declares reserved skill name "
                f"{CancelCurrentActionSkill.spec.name!r}"
            )
        loaded_scenarios.append(
            Scenario(
                scenario_id=scenario.scenario_id,
                name=scenario.name,
                track=scenario.track,
                environment_id=scenario.environment_id,
                tools_cls=tools_cls,
                skill_types=skill_types,
                episodes=tuple(episodes_by_scenario[scenario_id]),
            )
        )

    return loaded_scenarios
