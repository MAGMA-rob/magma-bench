from magma_bench.agents import BenchmarkAgent, get_agent_mode
from magma_bench.data_structures import Scenario
from magma_bench.results import ResultManager
from magma_bench.executor import ToolsEvalExecutor
from magma_bench.loader import load_groups_from_config, load_scenarios
from .group_runner import GroupRunner

from magma_core.configs.config import MAGMAConfig
from magma_core.workers import LMWorker

from pathlib import Path
from typing import Dict, List, Type, Union, Optional
from tqdm import tqdm
import os, json
from datetime import datetime

class BenchmarkRunner():
    """
    Main class of the benchmark. Allows to initialize and run evaluations for a benchmark agent.
    """

    # reference to the agent to evaluate
    agent : BenchmarkAgent

    _scenarios : List[Scenario]

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
        self._scenarios = []
        self.output_path = os.path.join(benchmark_config["save_dir"],self.agent.agent_name, datetime.now().strftime("%m-%d_%H-%M"))
        self._ensure_empty_or_create_output_dir(self.output_path)

        self.per_task_log = benchmark_config["logs"]
        self.result_manager = ResultManager(self.output_path, benchmark_config["logs"], benchmark_config["num_eval"]==1)
        worker = None if self._skip_backends else LMWorker(verifier_backend)

        self.num_try = benchmark_config["num_eval"]
        self._skip_metrics = False

        self.tool_executor : ToolsEvalExecutor = ToolsEvalExecutor(
            magma_config.magma_planner_address, 
            worker, nb_env=1
        )
    
    def load_benchmark(
        self,
        benchmark_root: Optional[Union[str, Path]],
        scenarios: Optional[Union[str, List[str]]] = None,
    ) -> bool:
        """Load compiled benchmark artifacts and select scenarios by ID or name."""
        if benchmark_root is None:
            raise ValueError(
                "A compiled benchmark root is required. Pass the directory "
                "produced by magma-bench-build."
            )

        self._scenarios = load_scenarios(Path(benchmark_root), scenarios)
        return True

    def _run_group(self, group : GroupRunner):

        group_not_done = True
        while group_not_done:
            action = self.tool_executor.step() # this will return the action to take for each env
            obs, _, _, _, _ = self.tool_executor.env.step(action)

            tools_ended = self.tool_executor.verif_ended_tool(obs)
            fetched_answer = self.agent.get_pending_answer()

            bench_tick = group.tick(tools_ended, fetched_answer)

            if bench_tick.has_inputs_for_agents():
                self.agent.add_inputs(bench_tick.to_agents)

            if bench_tick.has_call_for_executor():
                self.tool_executor.compute_actions(
                    bench_tick.to_executor
                )

            
    
    def run(self, args):
        """Run the evaluation process on the pre-loaded benchmarks"""
        if not self._scenarios:
            raise ValueError("There is no benchmark loaded. Please use .load(path) before run.")

        self._skip_metrics = bool(getattr(args, "no_metrics", False))

        selected_task_indices = getattr(args, "task_indices", None)
        if selected_task_indices is not None and len(self._scenarios) != 1:
            raise ValueError("task_indices can only be used when executing exactly one scenario.")
        
        # TODO: Add selected indice logic to run only certain 
        
        for scenario in tqdm(self._scenarios, desc="Scenarios", position=0, leave=True):
            group_list = load_groups_from_config(scenario)
            for group in group_list:
                runner = GroupRunner(scenario, group, self.tool_executor)
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
                    
