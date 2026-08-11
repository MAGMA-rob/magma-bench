"""Construction of mutable runtime objects from loaded benchmark episodes."""

from copy import deepcopy

from magma_core.base.tasks import BaseTask

from magma_bench.artifacts import deserialize_task_stages
from magma_bench.data_structures import Episode, Scenario


def build_episode_task(scenario: Scenario, episode: Episode) -> BaseTask:
    """Build and validate one independent task runtime for an episode."""

    task = BaseTask()
    task.name = episode.episode_id
    task.maniskill_env_id = scenario.environment_id
    task.Tools_cls = scenario.tools_cls
    task.apply_initialization_spec(deepcopy(episode.initialization))
    task.stages = deserialize_task_stages(
        episode.stages,
        validate_presentation=False,
    )
    task.validate()
    return task
