from dataclasses import dataclass, field
import json
from typing import Dict, List, Optional, Set, Tuple

from magma_bench.data_structures import Episode, Scenario
from magma_core.simulation.tasks import InitializationParameters


@dataclass
class EpisodeGroup:
    """Episodes that can share one vectorized simulator configuration."""

    episodes: Tuple[Episode, ...]
    env_options: Dict
    planner_options: Dict
    cur_idx: int = field(default=0, init=False)

    def get_episode(self) -> Optional[Episode]:
        if self.cur_idx >= len(self.episodes):
            return None
        out = self.episodes[self.cur_idx]
        self.cur_idx += 1
        return out


def load_groups_from_config(
    scenario: Scenario,
    episode_ids: Optional[Set[str]] = None,
) -> List[EpisodeGroup]:
    """Group episodes by the options shared by the environment and planner."""

    grouped_episodes: Dict[str, List[Episode]] = {}
    group_parameters: Dict[str, InitializationParameters] = {}
    group_order: List[str] = []

    for episode in scenario.episodes:
        if episode_ids is not None and episode.episode_id not in episode_ids:
            continue
        parameters_spec = episode.initialization.get("parameters")
        if not isinstance(parameters_spec, dict):
            raise ValueError(
                f"Episode {episode.episode_id!r} has no valid initialization parameters"
            )
        parameters = InitializationParameters.from_spec(parameters_spec)
        encoded_group_key = json.dumps(
            {
                "env_options": parameters_spec.get("env_options"),
                "planner_options": parameters_spec.get("planner_options"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if encoded_group_key not in grouped_episodes:
            grouped_episodes[encoded_group_key] = []
            group_parameters[encoded_group_key] = parameters
            group_order.append(encoded_group_key)
        grouped_episodes[encoded_group_key].append(episode)

    return [
        EpisodeGroup(
            episodes=tuple(grouped_episodes[key]),
            env_options=group_parameters[key].env_options,
            planner_options=group_parameters[key].planner_options,
        )
        for key in group_order
    ]
