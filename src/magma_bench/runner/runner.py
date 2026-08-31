import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from tqdm import tqdm

from magma_core.configs.config import MAGMAConfig
from magma_core.workers import LMWorker

from magma_bench.agents import BenchmarkAgent, get_agent_mode
from magma_bench.data_structures import Scenario
from magma_bench.executor import ToolsEvalExecutor
from magma_bench.loader import load_groups_from_config, load_scenarios
from magma_bench.results.manager import ResultManager
from magma_bench.results.models import EpisodeOutcome

from .group_runner import GroupRunner
import magma_scenarios.envs # to load gym envs

class BenchmarkRunner:
    """Load and execute compiled benchmark scenarios against one agent."""

    def __init__(
        self,
        agent_mode_name: str,
        magma_config: MAGMAConfig,
        class_specific_args: Dict,
        skip_judge: bool = False,
    ) -> None:
        benchmark_config = magma_config.benchmark
        self._skip_judge = skip_judge
        if benchmark_config.get("videos", False):
            raise NotImplementedError(
                "Benchmark video recording is not supported by magma_bench."
            )
        if benchmark_config.get("shader", "default") != "default":
            raise NotImplementedError(
                "Non-default benchmark shaders are not supported by magma_bench."
            )
        self._sim_backend = str(benchmark_config.get("sim_backend", "auto"))
        if self._sim_backend not in {"auto", "cpu", "gpu"}:
            raise ValueError(
                "benchmark.sim_backend must be one of: auto, cpu, gpu"
            )
        configured_seed = benchmark_config.get("seed", 42)
        self._seed = 42 if configured_seed is None else int(configured_seed)

        agent_class: Type[BenchmarkAgent] = get_agent_mode(
            agent_mode_name
        ).load_agent_class()
        class_specific_args.setdefault("agent_url", magma_config.magma_agent_address)
        class_specific_args.setdefault(
            "inference_mode",
            bool(benchmark_config.get("deterministic_decoding", True)),
        )

        if skip_judge:
            worker = None
        else:
            verifier_backend = magma_config.backends[
                benchmark_config["backend_verifier"]
            ]
            class_specific_args.setdefault("backend_url", verifier_backend.endpoint)
            class_specific_args.setdefault("backend_header", verifier_backend.headers)
            worker = LMWorker(verifier_backend)

        self.agent = agent_class(**class_specific_args)
        self._scenarios: List[Scenario] = []
        self._benchmark_root: Optional[Path] = None
        self._benchmark_config = benchmark_config
        self.result_manager: Optional[ResultManager] = None
        self.tool_executor = ToolsEvalExecutor(
            magma_config.magma_planner_address,
            worker,
            nb_env=int(benchmark_config.get("nb_env", 1)),
            skip_judge=skip_judge,
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
        self._benchmark_root = Path(benchmark_root).resolve()
        self._scenarios = load_scenarios(self._benchmark_root, scenarios)
        return True

    def _record_episode_outcome(self, outcome: EpisodeOutcome) -> None:
        if self.result_manager is None:
            raise RuntimeError("ResultManager has not been initialized")
        self.result_manager.record_episode(outcome)

    def _build_result_manager(self) -> ResultManager:
        if self._benchmark_root is None:
            raise RuntimeError("Benchmark root is not available")
        configured_path = self._benchmark_config.get("results_path")
        if configured_path is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            results_path = (
                Path(self._benchmark_config.get("save_dir", "eval"))
                / self.agent.agent_name
                / timestamp
            )
        else:
            results_path = Path(configured_path)
        return ResultManager(
            results_path=results_path,
            benchmark_root=self._benchmark_root,
            agent_card=self.agent.get_agent_card(),
            scenarios=self._scenarios,
            judge_mode="skipped" if self._skip_judge else "backend",
            sim_backend=self._sim_backend,
            seed=self._seed,
        )

    def _run_group(self, group: GroupRunner) -> None:
        while not group.is_done():
            if self.tool_executor.has_active_tools():
                action = self.tool_executor.step()
                obs, _, _, _, _ = self.tool_executor.env.step(action)
            else:
                obs = self.tool_executor.env.unwrapped.get_obs()
                time.sleep(0.01)

            tools_ended = self.tool_executor.verif_ended_tool(obs)
            fetched_answers = self.agent.get_pending_results()
            benchmark_tick = group.tick(tools_ended, fetched_answers)

            if benchmark_tick.has_inputs_for_agents():
                self.agent.add_inputs(benchmark_tick.to_agents)
            if benchmark_tick.has_call_for_executor():
                self.tool_executor.compute_actions(benchmark_tick.to_executor)

    def run(self) -> None:
        """Run all loaded scenarios; outcomes are emitted through the callback."""

        if not self._scenarios:
            raise ValueError(
                "There is no benchmark loaded. Please use load_benchmark() before run()."
            )
        self.result_manager = self._build_result_manager()

        try:
            for scenario in tqdm(
                self._scenarios,
                desc="Scenarios",
                position=0,
                leave=True,
            ):
                if not self.result_manager.start_scenario(scenario):
                    continue
                pending_episode_ids = self.result_manager.pending_episode_ids(
                    scenario.scenario_id
                )
                for episode_group in load_groups_from_config(
                    scenario,
                    pending_episode_ids,
                ):
                    group_runner = GroupRunner(
                        scenario,
                        episode_group,
                        self.tool_executor,
                        self._record_episode_outcome,
                        sim_backend=self._sim_backend,
                        seed=self._seed,
                    )
                    self._run_group(group_runner)
                self.result_manager.finish_scenario(scenario)
            self.result_manager.finish_benchmark()
        finally:
            try:
                self.agent.stop()
            finally:
                if hasattr(self.tool_executor, "env"):
                    self.tool_executor.env.close()
