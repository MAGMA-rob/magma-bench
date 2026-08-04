import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

from tqdm import tqdm

from magma_core.configs.config import MAGMAConfig
from magma_core.workers import LMWorker

from magma_bench.agents import BenchmarkAgent, get_agent_mode
from magma_bench.data_structures import EpisodeOutcome, Scenario
from magma_bench.executor import ToolsEvalExecutor
from magma_bench.loader import load_groups_from_config, load_scenarios

from .group_runner import GroupRunner
import magma_scenarios.envs # to load gym envs

class BenchmarkRunner:
    """Load and execute compiled benchmark scenarios against one agent."""

    def __init__(
        self,
        agent_mode_name: str,
        magma_config: MAGMAConfig,
        class_specific_args: Dict,
        skip_backends: bool = False,
        on_episode_finished: Optional[Callable[[EpisodeOutcome], None]] = None,
    ) -> None:
        benchmark_config = magma_config.benchmark
        self._skip_backends = skip_backends
        if skip_backends and agent_mode_name != "task_state_reactive":
            raise ValueError(
                "skip_backends=True is only supported with task_state_reactive."
            )

        agent_class: Type[BenchmarkAgent] = get_agent_mode(
            agent_mode_name
        ).load_agent_class()
        class_specific_args.setdefault("agent_url", magma_config.magma_agent_address)

        verifier_backend = None
        if not skip_backends:
            verifier_backend = magma_config.backends[
                benchmark_config["backend_verifier"]
            ]
            class_specific_args.setdefault("backend_url", verifier_backend.endpoint)
            class_specific_args.setdefault("backend_header", verifier_backend.headers)

        self.agent = agent_class(**class_specific_args)
        self._scenarios: List[Scenario] = []
        self._on_episode_finished = on_episode_finished
        worker = None if skip_backends else LMWorker(verifier_backend)
        self.tool_executor = ToolsEvalExecutor(
            magma_config.magma_planner_address,
            worker,
            nb_env=int(benchmark_config.get("nb_env", 1)),
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

    def _record_episode_outcome(self, outcome: EpisodeOutcome) -> None:
        if self._on_episode_finished is not None:
            self._on_episode_finished(outcome)

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

    def run(self, args) -> None:
        """Run all loaded scenarios; outcomes are emitted through the callback."""

        if not self._scenarios:
            raise ValueError(
                "There is no benchmark loaded. Please use load_benchmark() before run()."
            )
        selected_task_indices = getattr(args, "task_indices", None)
        if selected_task_indices is not None and len(self._scenarios) != 1:
            raise ValueError(
                "task_indices can only be used when executing exactly one scenario."
            )

        try:
            for scenario in tqdm(
                self._scenarios,
                desc="Scenarios",
                position=0,
                leave=True,
            ):
                for episode_group in load_groups_from_config(scenario):
                    group_runner = GroupRunner(
                        scenario,
                        episode_group,
                        self.tool_executor,
                        self._record_episode_outcome,
                    )
                    self._run_group(group_runner)
        finally:
            self.agent.stop()
