from pathlib import Path
from typing import List, Tuple

from .models import (
    CONDITIONS,
    BenchmarkManifest,
    CompiledEpisodeSpec,
    ScenarioManifest,
    SemanticManifest,
    SkeletonManifest,
    load_json_model,
)
from .runtime import deserialize_task_stages


def load_compiled_episode(path: Path) -> CompiledEpisodeSpec:
    episode = load_json_model(path, CompiledEpisodeSpec)
    if not isinstance(episode, CompiledEpisodeSpec):
        raise TypeError(f"Unexpected episode model loaded from {path}")
    deserialize_task_stages(episode.stages)
    return episode


def load_compiled_benchmark(
    root: Path,
) -> Tuple[BenchmarkManifest, List[CompiledEpisodeSpec]]:
    root = root.resolve()
    manifest = load_json_model(root / "benchmark.json", BenchmarkManifest)
    if not isinstance(manifest, BenchmarkManifest):
        raise TypeError(f"Unexpected benchmark manifest loaded from {root}")
    episodes: List[CompiledEpisodeSpec] = []
    seen_paths = set()
    loaded_scenarios = {}
    loaded_skeletons = {}
    loaded_semantics = {}
    for entry in manifest.episodes:
        if entry.path in seen_paths:
            raise ValueError(f"Duplicated episode path in benchmark index: {entry.path}")
        seen_paths.add(entry.path)
        scenario = loaded_scenarios.get(entry.scenario_id)
        if scenario is None:
            scenario_path = root / entry.scenario_id / "scenario.json"
            scenario = load_json_model(scenario_path, ScenarioManifest)
            if scenario.scenario_id != entry.scenario_id:
                raise ValueError(f"Scenario identity mismatch in {scenario_path}")
            loaded_scenarios[entry.scenario_id] = scenario

        skeleton_key = (entry.scenario_id, entry.skeleton_id)
        skeleton = loaded_skeletons.get(skeleton_key)
        if skeleton is None:
            skeleton_path = (
                root / entry.scenario_id / entry.skeleton_id / "skeleton.json"
            )
            skeleton = load_json_model(skeleton_path, SkeletonManifest)
            if (
                skeleton.scenario_id != entry.scenario_id
                or skeleton.skeleton_id != entry.skeleton_id
            ):
                raise ValueError(f"Skeleton identity mismatch in {skeleton_path}")
            loaded_skeletons[skeleton_key] = skeleton

        semantic_key = (
            entry.scenario_id,
            entry.skeleton_id,
            entry.semantic_id,
        )
        semantic = loaded_semantics.get(semantic_key)
        if semantic is None:
            semantic_path = (
                root
                / entry.scenario_id
                / entry.skeleton_id
                / entry.semantic_id
                / "semantic.json"
            )
            semantic = load_json_model(semantic_path, SemanticManifest)
            if semantic.semantic_id != entry.semantic_id:
                raise ValueError(f"Semantic identity mismatch in {semantic_path}")
            loaded_semantics[semantic_key] = semantic

        episode_path = root / entry.path
        if not episode_path.is_file():
            raise FileNotFoundError(f"Missing compiled episode {episode_path}")
        episode = load_compiled_episode(episode_path)
        expected_identity = (
            entry.episode_id,
            entry.scenario_id,
            entry.skeleton_id,
            entry.semantic_id,
            entry.condition,
            entry.length_bucket,
        )
        actual_identity = (
            episode.episode_id,
            episode.scenario_id,
            episode.skeleton_id,
            episode.semantic_id,
            episode.condition,
            episode.metadata.length_bucket,
        )
        if actual_identity != expected_identity:
            raise ValueError(
                f"Episode index identity mismatch for {entry.path}: "
                f"expected={expected_identity}, got={actual_identity}"
            )
        episodes.append(episode)
    actual_paths = {
        str(path.relative_to(root))
        for path in root.glob("*/skeleton_*/semantic_*/episodes/*.json")
    }
    if actual_paths != seen_paths:
        raise ValueError(
            "Compiled episode tree differs from benchmark index: "
            f"missing={sorted(seen_paths - actual_paths)}, "
            f"unindexed={sorted(actual_paths - seen_paths)}"
        )
    if set(manifest.scenarios) != set(loaded_scenarios):
        raise ValueError(
            f"Benchmark scenario index mismatch: manifest={sorted(manifest.scenarios)}, "
            f"episodes={sorted(loaded_scenarios)}"
        )
    semantics_by_skeleton = {}
    conditions_by_semantic = {}
    for entry in manifest.episodes:
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
    return manifest, episodes
