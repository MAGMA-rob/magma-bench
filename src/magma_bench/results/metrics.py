from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from .models import (
    LAG_BUCKETS,
    LENGTH_BUCKETS,
    NON_CLEAN_CONDITIONS,
    EpisodeResult,
    LengthMetrics,
    MetricCounts,
    MetricsPayload,
    Rate,
)


EpisodeBinding = Tuple[str, Any]


def _rate(numerator: int, denominator: int) -> Rate:
    return Rate(
        value=None if denominator == 0 else numerator / denominator,
        numerator=numerator,
        denominator=denominator,
    )


def _lag_bucket(lag: int) -> str:
    if lag <= 3:
        return "0-3"
    if lag <= 9:
        return "4-9"
    if lag <= 19:
        return "10-19"
    return "20+"


def validate_metric_inputs(
    bindings: Sequence[EpisodeBinding],
    results: Dict[str, EpisodeResult],
) -> None:
    episodes = {episode.episode_id: episode for _, episode in bindings}
    if len(episodes) != len(bindings):
        raise ValueError("Episode IDs must be globally unique")

    expected_ids = set(episodes)
    actual_ids = set(results)
    if actual_ids != expected_ids:
        raise ValueError(
            "Episode result set mismatch: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )

    for scenario_id, episode in bindings:
        control = episodes.get(episode.metadata.control_episode_id)
        if control is None:
            raise ValueError(
                f"Episode {episode.episode_id!r} references missing control "
                f"{episode.metadata.control_episode_id!r}"
            )
        if control.condition != "clean":
            raise ValueError(
                f"Control {control.episode_id!r} is not a clean episode"
            )
        control_binding = next(
            (
                (bound_scenario, bound_episode)
                for bound_scenario, bound_episode in bindings
                if bound_episode.episode_id == control.episode_id
            ),
            None,
        )
        if control_binding is None or control_binding[0] != scenario_id:
            raise ValueError(
                f"Control {control.episode_id!r} belongs to another scenario"
            )
        if (
            control.skeleton_id != episode.skeleton_id
            or control.semantic.semantic_id != episode.semantic.semantic_id
        ):
            raise ValueError(
                f"Episode {episode.episode_id!r} is not paired with a clean "
                "episode from the same skeleton and semantic variation"
            )

        if episode.condition == "mission_update":
            measurable_lags = [
                lag for lag in episode.metadata.intervention_lags if lag is not None
            ]
            if len(measurable_lags) != 1:
                raise ValueError(
                    f"Mission-update episode {episode.episode_id!r} must define "
                    "exactly one non-null intervention lag"
                )


def compute_metrics(
    bindings: Sequence[EpisodeBinding],
    results: Dict[str, EpisodeResult],
) -> MetricsPayload:
    validate_metric_inputs(bindings, results)

    clean_numerator = 0
    clean_denominator = 0
    condition_counts = {
        condition: [0, 0] for condition in NON_CLEAN_CONDITIONS
    }
    robustness_counts = {
        condition: [0, 0] for condition in NON_CLEAN_CONDITIONS
    }
    length_counts = {
        bucket: {
            "clean": [0, 0],
            **{
                condition: [0, 0]
                for condition in NON_CLEAN_CONDITIONS
            },
        }
        for bucket in LENGTH_BUCKETS
    }
    lag_counts = {bucket: [0, 0] for bucket in LAG_BUCKETS}

    episode_by_id = {episode.episode_id: episode for _, episode in bindings}
    scenario_ids = set()
    skeleton_ids = set()
    semantic_units = set()

    for scenario_id, episode in bindings:
        scenario_ids.add(scenario_id)
        skeleton_ids.add((scenario_id, episode.skeleton_id))
        semantic_units.add(
            (scenario_id, episode.skeleton_id, episode.semantic.semantic_id)
        )
        result = results[episode.episode_id]
        bucket = episode.metadata.length_bucket
        if bucket not in length_counts:
            raise ValueError(
                f"Unknown length bucket {bucket!r} on {episode.episode_id!r}"
            )

        if episode.condition == "clean":
            clean_denominator += 1
            clean_numerator += int(result.success)
            length_counts[bucket]["clean"][1] += 1
            length_counts[bucket]["clean"][0] += int(result.success)
            continue

        if episode.condition not in condition_counts:
            raise ValueError(
                f"Unknown condition {episode.condition!r} on {episode.episode_id!r}"
            )
        condition_counts[episode.condition][1] += 1
        condition_counts[episode.condition][0] += int(result.success)
        length_counts[bucket][episode.condition][1] += 1
        length_counts[bucket][episode.condition][0] += int(result.success)

        control_result = results[episode.metadata.control_episode_id]
        if control_result.success:
            robustness_counts[episode.condition][1] += 1
            robustness_counts[episode.condition][0] += int(result.success)

            if episode.condition == "mission_update":
                lag = next(
                    lag
                    for lag in episode.metadata.intervention_lags
                    if lag is not None
                )
                lag_bucket = _lag_bucket(lag)
                lag_counts[lag_bucket][1] += 1
                lag_counts[lag_bucket][0] += int(result.success)

    return MetricsPayload(
        counts=MetricCounts(
            scenario_count=len(scenario_ids),
            skeleton_count=len(skeleton_ids),
            semantic_unit_count=len(semantic_units),
            episode_count=len(episode_by_id),
        ),
        clean_success_rate=_rate(clean_numerator, clean_denominator),
        success_rate_by_condition={
            condition: _rate(*condition_counts[condition])
            for condition in NON_CLEAN_CONDITIONS
        },
        conditional_robustness_by_condition={
            condition: _rate(*robustness_counts[condition])
            for condition in NON_CLEAN_CONDITIONS
        },
        success_rate_by_length={
            bucket: LengthMetrics(
                clean_success_rate=_rate(*length_counts[bucket]["clean"]),
                success_rate_by_condition={
                    condition: _rate(*length_counts[bucket][condition])
                    for condition in NON_CLEAN_CONDITIONS
                },
            )
            for bucket in LENGTH_BUCKETS
        },
        mission_update_retention_by_lag={
            bucket: _rate(*lag_counts[bucket])
            for bucket in LAG_BUCKETS
        },
    )
