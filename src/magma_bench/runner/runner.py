from dataclasses import dataclass
from magma_bench.agents import BenchmarkAgent, get_agent_mode
from magma_bench.builders import (
    ScenarioConfig, Scenario, 
    BenchmarkLoader, ScenarioBuilder,
    Task, Stage
)
from magma_bench.results import ResultManager, ScenarioResult, StageResult
from magma_bench.executor import ToolsEvalExecutor
from magma_bench.loader import load_groups_from_config, EpisodeGroup
from .group_runner import GroupRunner

from magma_core.configs.config import MAGMAConfig
from magma_core.base.agents import AgentAnswer, BadAgentAnswer
from magma_core.utils.data_utils import apply_att_modif
from magma_core.utils.text_utils import build_model_return_from_executor, build_fake_execution_fail
from magma_core.workers import LMWorker

from typing import Dict, List, Type, Union, Optional
from tqdm import tqdm
import os, json, copy
from datetime import datetime

class BenchmarkRunner():
    """
    Main class of the benchmark. Allows to initialize and run evaluations for a benchmark agent.
    """

    # reference to the agent to evaluate
    agent : BenchmarkAgent

    _benchmark_configs : List[ScenarioConfig]

    result_manager : ResultManager
    tool_executor : ToolsEvalExecutor

    #_bench_logs : List = []

    @staticmethod
    def _ensure_empty_or_create_output_dir(output_path: str) -> None:
        if os.path.exists(output_path):
            if not os.path.isdir(output_path):
                raise FileExistsError(f"Benchmark output path exists and is not a directory: {output_path}")
            if any(os.scandir(output_path)):
                raise FileExistsError(f"Benchmark output directory already exists and is not empty: {output_path}")
            return

        os.makedirs(output_path, exist_ok=True)

    def __init__(
            self,
            agent_mode_name : str,
            magma_config : MAGMAConfig,
            class_specific_args : Dict,
            skip_backends: bool = False,
        ) -> None:
        benchmark_config = magma_config.benchmark
        self._skip_backends = skip_backends
        if self._skip_backends and agent_mode_name != "task_state_reactive":
            raise ValueError("skip_backends=True is only supported with task_state_reactive.")

        Agent_class : Type[BenchmarkAgent] = get_agent_mode(agent_mode_name).load_agent_class()
        class_specific_args.setdefault("agent_url", magma_config.magma_agent_address)

        verifier_backend = None
        if not self._skip_backends:
            verifier_backend = magma_config.backends[benchmark_config["backend_verifier"]]
            class_specific_args.setdefault("backend_url",verifier_backend.endpoint) # TO DO: Dedicated option
            class_specific_args.setdefault("backend_header",verifier_backend.headers) # TO DO: Dedicated option

        self.agent = Agent_class(**class_specific_args)

        self.benchmarks = []
        self.output_path = os.path.join(benchmark_config["save_dir"],self.agent.agent_name, datetime.now().strftime("%m-%d_%H-%M"))
        self._ensure_empty_or_create_output_dir(self.output_path)

        self.per_task_log = benchmark_config["logs"]
        self.result_manager = ResultManager(self.output_path, benchmark_config["logs"], benchmark_config["num_eval"]==1)
        worker = None if self._skip_backends else LMWorker(verifier_backend)

        self.num_try = benchmark_config["num_eval"]
        self._skip_metrics = False

        self.tool_executor : ToolsEvalExecutor = ToolsEvalExecutor(
            magma_config.magma_planner_address, 
            worker, nb_env=1, 
            randomize_variation=self.num_try
        )
        if self._skip_backends:
            original_verif = self.tool_executor.verif_complementary_bench

            def _verif_without_judge(model_say: str, complementary_verif):
                if complementary_verif is None:
                    return original_verif(model_say, None)

                effective_complementary_verif = dict(complementary_verif)
                effective_complementary_verif.pop("judge", None)
                return original_verif(model_say, effective_complementary_verif)

            self.tool_executor.verif_complementary_bench = _verif_without_judge
    
    def load_benchmark(self, scenario) -> bool:
        """Load one benchmark to evaluate the agent on."""
        if scenario == None or scenario == ['all']:
            scenario = 'all'
        self._benchmark_configs = BenchmarkLoader.load(scenario)

        return True

    def _make_answer(self, author: str, content: Dict, timestep = 0):
        """Return a json formated answer."""
        return {'author':author, "content": copy.deepcopy(content), "timestamp":timestep}

    def _build_planning_retry_exceeded_reason(self, call_action: Dict, status_dict: Dict) -> str:
        action_name = call_action.get("name")
        if action_name is None and call_action:
            action_name = ", ".join(
                f"{robot}:{action.get('name', 'unknown')}"
                for robot, action in call_action.items()
                if isinstance(action, dict)
            )
        if not action_name:
            action_name = "unknown"

        executor_message = status_dict.get("error") or status_dict.get("infos") or ""
        if executor_message:
            return (
                f"Exceeded 10 planning retries while executing '{action_name}'. "
                f"Last executor status: {executor_message}"
            )
        return f"Exceeded 10 planning retries while executing '{action_name}'."

    def _get_last_status_instruction(self, conversation: List[Dict]) -> Union[Dict, None]:
        """
        Return the latest status message produced during a stage execution.
        """
        for message in reversed(conversation):
            if message.get("author") == "SYSTEM":
                return message
        return None

    def _resolve_stage_instruction(
            self,
            stage: Stage,
            previous_stage_success: bool,
            previous_status_instruction: Union[Dict, None],
        ) -> Union[Dict, None]:
        """
        Resolve which instruction should start the current benchmark stage.

        If the stage has no explicit instruction, we reuse the previous status
        when the previous stage succeeded. If no reusable status exists, we
        fall back to ``default_instruction`` when provided. Otherwise the stage
        is skipped so the runner can continue until it finds a runnable stage.
        """
        if stage.instruction is not None:
            return stage.instruction

        if previous_stage_success and previous_status_instruction is not None:
            return previous_status_instruction

        if stage.default_instruction is not None:
            return stage.default_instruction

        return None

    
    
    def _run_group(self, group : GroupRunner):

        group_not_done = True
        while group_not_done:

            group.tick(None)
    
    def run(self, args):
        """Run the evaluation process on the pre-loaded benchmarks"""
        if not self._benchmark_configs:
            raise ValueError("There is no benchmark loaded. Please use .load(path) before run.")

        self._skip_metrics = bool(getattr(args, "no_metrics", False))

        selected_task_indices = getattr(args, "task_indices", None)
        if selected_task_indices is not None and len(self._benchmark_configs) != 1:
            raise ValueError("task_indices can only be used when executing exactly one scenario.")
        
        # TODO: Add selected indice logic to run only certain 
        
        for bench_config in tqdm(self._benchmark_configs, desc="Scenarios", position=0, leave=True):
            group_list = load_groups_from_config(bench_config)
            for group in group_list:
                runner = GroupRunner(group, self.tool_executor)
                # TODO: Manage video saving ??
                self._run_group(runner)

        self.result_manager.stop()
        if self._skip_metrics:
            return
        data = self.result_manager.compute_global_metrics()
        data['agent_info'] = self.agent.get_agent_card()

        path = os.path.join(self.output_path, "result.json")
        with open(path,"w+") as f:
            json.dump(data,f,indent=2)
                    
