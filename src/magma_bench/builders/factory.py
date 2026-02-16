from .scenario import Scenario, ScenarioConfig
from .details import Task
from magma_bench.executor import ToolsEvalExecutor

import json
from typing import Optional
import os

class ScenarioBuilder():

    @staticmethod
    def load(
        config : ScenarioConfig,
        out_path : Optional[str],
        tool_executor_ref : ToolsEvalExecutor,
        args,
        obs_mode : str = "state_dict"
    ) -> Scenario:
        video_path = "none"
        if out_path is not None:
            output_path = os.path.join(out_path, config.scenario_id)
            os.makedirs(output_path)
            if args.videos:
                video_path = os.path.join(output_path, "videos")
            

        tasks = []
        for task_path in config.tasks_path:
            with task_path.open("r") as f:
                task_config = json.load(f)
            task_id = task_path.stem
            task_config['id'] = str(task_id)
            try:
                tasks.append(
                    Task(task_config)
                )
            except Exception as e:
                raise ValueError(f"Scenario {config.scenario_name} : Task {task_id} : {e}")

        return Scenario(
            name=config.scenario_name,
            id=config.scenario_id,
            task_name=config.task_cls_name,
            tool_executor_ref=tool_executor_ref,
            video_output_path=video_path,
            tasks=tasks,
            criteria=config.evaluated_criteria,
            seed=args.seed,
            obs_mode=obs_mode,
        )
    