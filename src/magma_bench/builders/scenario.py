from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from magma_bench.executor import ToolsEvalExecutor
from .task import Task
from .stage import Stage
from magma_bench.evalutations import evaluate_env_success
from magma_core.base.data_structures import ActiveStageErrorState
from magma_core.base.tasks import BaseBenchmarkTask
from magma_scenarios import load_preset

@dataclass
class ScenarioConfig:

    scenario_name : str
    scenario_id : str
    task_cls_name : str
    mplib_options : Dict
    tasks_path : List[Path]
    evaluated_criteria : List[str]
    task_decomp : Dict[str,List[str]]

class Scenario():

    name : str
    id : str
    nb_tasks : int

    tasks : List[Task]
    tool_executor : ToolsEvalExecutor

    saving_video : bool

    evaluated_criteria : List[str]

    def __init__(
            self,
            name : str,
            id : str,
            task_name : str,
            tool_executor_ref : ToolsEvalExecutor,
            video_output_path : str,
            tasks : List[Task],
            criteria : List[str],
            seed : int,
            obs_mode : str = "state_dict",
        ) -> None:
        # Args
        self.name = name
        self.id = id
        self.nb_tasks = len(tasks)
        self.tasks = tasks
        self.evaluated_criteria = criteria
        self.seed = seed
        self.tool_executor = tool_executor_ref

        Cls_Task = load_preset(task_name)
        task_ref = Cls_Task()

        if not isinstance(task_ref, BaseBenchmarkTask):
            raise ValueError(f"Benchmark tasks must inherits from a BaseBenchmarkTask class. It's not the case of {task_ref.__class__.__name__}")

        self.env = self.tool_executor.initialize(
            task_ref=task_ref,
            video_path=video_output_path,
            build_first_stage=False,
            obs_mode=obs_mode
        )
        if video_output_path != "none":
            self.saving_video = True
        else:
            self.saving_video = False

        # Verify that all used errors exist in the BenchmarkTask
        self._validate_runtime_error_injections(task_ref)

    def _validate_runtime_error_injections(self, task_ref : BaseBenchmarkTask):
        """
        Verify that every runtime_error declared in benchmark JSON can be
        resolved against the benchmark task error registry.
        """
        for task in self.tasks:
            for stage in task.stages:
                error_state = stage.get_error_state()
                if not error_state:
                    continue
                try:
                    task_ref.get_active_stage_error(0, error_state)
                except Exception as exc:
                    raise ValueError(
                        f"Task {task.id}, stage {stage.id}: invalid runtime_error injection. {exc}"
                    ) from exc

    def save_video(self, video_name):
        if self.saving_video and self.env is not None:
            print(video_name)
            self.env.flush_video(
                video_name, ignore_empty_transition=False
            )
        else:
            self.saving_video = False

    ############ EVAL

    def evaluate_stage(self, stage : Stage, model_response : Dict) -> Tuple[bool, str]:
        """
        Allows to return for a specific Step if the step is marked as successfull or not
        return a tuple bool, str -> boolean success and reason
        """
        model_say = model_response['say']

        if not stage.should_act() and model_response['action'] != {}:
            return False, "The model try to call a tool whereas it should just acknowledge."

        predicates, complementary_verif = stage.get_evaluation_elements()
        env_state = self.env.unwrapped.get_state_dict()['actors']

        success = True
        reason = ""

        if predicates:
            if not evaluate_env_success(env_state, predicates): 
                success = False
                reason+="Predicate fails. "

        if success and complementary_verif: #no need to check this if predicates fails
            log_ref = complementary_verif.get("logs",None)
            judge = complementary_verif.get("judge", None)

            out_dict = self.tool_executor.verif_complementary_bench(model_say, log_ref, judge)
            if not out_dict['verdict']: success = False
            reason += out_dict["explanation"]

        return success, reason
    

    ############ Getter

    def get_task(self, idx : int) -> Task:
        """
        Allows to get the idx-th tasks in the benchmark
        + reset env and tool_executor
        """
        if not self.env: raise ValueError("Env not initialized")
        self.env.reset(seed=self.seed,options=self.tool_executor.get_env_options(0))
        self.tool_executor.log_reset() #reset log at each new tasks
        self.tool_executor.task_ref.reset_stage() #Allows to reset the attributes properly (in case of modification)
        return self.tasks[idx]
    
    def get_init_elements(self):
        """Return a dict containing different init ellement such as tools and attributes."""
        att = self.tool_executor.get_task_attributes(0)
        tools = self.tool_executor.get_tools()
        return {
            "attributes" : att,
            "tools" : tools
        }
    
    ########## EXECUTION
    
    def send_action(self, action : Dict, error_state : ActiveStageErrorState):
        """Send the tool call to the evaluation env"""
        tool_calls = {0 : action}
        self.tool_executor.compute_actions(tool_calls, error_state)
        return True
    
    def execute_env_step(self) -> Dict:
        """Execute one env step. Return a dict with status when tool is finish"""
        if not self.env: raise ValueError("Env not initialized")
        action = self.tool_executor.step()
        obs, _, _, _, _ = self.env.step(action)

        return self.tool_executor.verif_ended_tool(obs)
    
    def env_reset(self):
        if self.env: self.env.reset(seed=self.seed,options=self.tool_executor.get_env_options(0))
        self.tool_executor.log_reset()

    ########## CLEANER
    
    def close(self):
        """Close the Benchmark and free the memory"""
        try:
            if hasattr(self, "env") and self.env is not None:
                self.env.close()
        except Exception:
            pass

        # 2. Access underlying ManiSkill environment
        try:
            env = self.env.unwrapped
        except Exception:
            env = None

        if env is not None:
            # 3. Close viewer (if created)
            viewer = getattr(env, "viewer", None)
            if viewer is not None:
                try:
                    viewer.close()
                except Exception:
                    pass

            # 4. Close SAPIEN renderer
            renderer = getattr(env, "renderer", None)
            if renderer is not None:
                try:
                    renderer.remove_all_cameras()
                except Exception:
                    pass
                try:
                    renderer.destroy()  # frees GPU memory
                except Exception:
                    pass

            # 5. Close SAPIEN scene and engine
            scene = getattr(env, "scene", None)
            if scene is not None:
                try:
                    scene.clear()       # remove actors
                except Exception:
                    pass

            engine = getattr(env, "engine", None)
            if engine is not None:
                try:
                    engine.destroy()
                except Exception:
                    pass

        # 6. Force Python to release references
        self.env = None
        # gc.collect()
