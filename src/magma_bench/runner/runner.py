import time
from datetime import datetime
import logging
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from tqdm import tqdm

from magma_core.configs.config import MAGMAConfig
from magma_core.workers import LMWorker

from magma_bench.agents import BenchmarkAgent, get_agent_mode
from magma_bench.data_structures import Scenario
from magma_bench.executor import (
    PlannerRetryEvent,
    PlannerRetryResolvedEvent,
    ToolsEvalExecutor,
)
from magma_bench.loader import load_groups_from_config, load_scenarios
from magma_bench.results.manager import ResultManager
from magma_bench.results.models import EpisodeOutcome
from magma_bench.video import EpisodeVideoRecorder, VideoConfig

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
        self._model_logs = bool(
            benchmark_config.get(
                "model_logs",
                benchmark_config.get("logs", False),
            )
        )
        self.video_recorder = EpisodeVideoRecorder(
            VideoConfig(
                enabled=bool(benchmark_config.get("videos", False)),
                fps=int(benchmark_config.get("video_fps", 20)),
                hold_seconds=float(
                    benchmark_config.get("video_hold_seconds", 1.0)
                ),
            )
        )

        agent_class: Type[BenchmarkAgent] = get_agent_mode(
            agent_mode_name
        ).load_agent_class()
        class_specific_args.setdefault("agent_url", magma_config.magma_agent_address)
        class_specific_args.setdefault(
            "inference_mode",
            bool(benchmark_config.get("deterministic_decoding", True)),
        )
        class_specific_args.setdefault("collect_model_logs", self._model_logs)

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
        self._episode_started_at: Dict[str, float] = {}
        self._progress_logger = logging.getLogger(
            f"magma_bench.progress.{id(self)}"
        )
        self._progress_logger.setLevel(logging.INFO)
        self._progress_logger.propagate = False
        self._progress_log_handler: Optional[logging.Handler] = None
        self.tool_executor = ToolsEvalExecutor(
            magma_config.magma_planner_address,
            worker,
            nb_env=int(benchmark_config.get("nb_env", 1)),
            skip_judge=skip_judge,
            visual_assets=self.video_recorder.config.enabled,
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
        started_at = self._episode_started_at.pop(outcome.episode_id, None)
        elapsed = None if started_at is None else time.monotonic() - started_at
        self._progress_logger.info(
            "EPISODE_COMPLETED episode=%s success=%s status=%s duration_seconds=%s reason=%s",
            outcome.episode_id,
            outcome.success,
            outcome.terminal.status,
            "unknown" if elapsed is None else f"{elapsed:.3f}",
            outcome.terminal.reason or "",
        )

    def _record_episode_started(
        self,
        episode_id: str,
    ) -> None:
        if self.result_manager is None:
            raise RuntimeError("ResultManager has not been initialized")
        self.result_manager.start_episode_model_logs(episode_id)
        self._episode_started_at[episode_id] = time.monotonic()
        self._progress_logger.info(
            "EPISODE_STARTED episode=%s",
            episode_id,
        )

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
            model_logs=self._model_logs,
        )

    def _record_model_diagnostics(
        self,
        episode_id: str,
        stage_index: int,
        diagnostics: List[Dict],
    ) -> None:
        if self.result_manager is None:
            raise RuntimeError("ResultManager has not been initialized")
        self.result_manager.record_model_diagnostics(
            episode_id,
            stage_index,
            diagnostics,
        )

    def _run_group(self, group: GroupRunner) -> None:
        while not group.is_done():
            active_env_ids = self.tool_executor.get_active_tool_env_ids()
            if active_env_ids:
                action = self.tool_executor.step()
                obs, _, _, _, _ = self.tool_executor.env.step(action)
                self.video_recorder.capture_physical_step(active_env_ids)
            else:
                obs = self.tool_executor.env.unwrapped.get_obs()
                time.sleep(0.01)

            tools_ended = self.tool_executor.verif_ended_tool(obs)
            for event in self.tool_executor.drain_planner_runtime_events():
                if isinstance(event, PlannerRetryEvent):
                    if event.tool_failures:
                        tool_failure_messages = []
                        for failure in event.tool_failures:
                            reason = failure.reason.replace("\n", " ")
                            tool_failure_messages.append(
                                f"{failure.tool_name}: {reason}"
                            )
                        tool_failures = " | ".join(tool_failure_messages)
                    else:
                        reason = event.message.replace("\n", " ")
                        tool_failures = f"unknown: {reason}"
                    self._progress_logger.warning(
                        "PLANNER_ERROR episode=%s attempt=%d/%d tools=[%s]",
                        event.episode_id,
                        event.attempt,
                        event.max_attempts,
                        tool_failures,
                    )
                    self.video_recorder.set_planner_retry(
                        event.env_idx,
                        event.attempt,
                        event.max_attempts,
                        event.message,
                    )
                elif isinstance(event, PlannerRetryResolvedEvent):
                    self.video_recorder.clear_planner_retry(event.env_idx)
            self.video_recorder.flush_holds()

            fetched_answers = self.agent.get_pending_results()
            benchmark_tick = group.tick(tools_ended, fetched_answers)
            self.video_recorder.flush_holds()

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
        progress_log_path = self.result_manager.results_path / "progress.log"
        self._progress_log_handler = logging.FileHandler(
            progress_log_path,
            mode="a",
            encoding="utf-8",
        )
        self._progress_log_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        ))
        self._progress_logger.addHandler(self._progress_log_handler)

        try:
            for scenario in tqdm(
                self._scenarios,
                desc="Scenarios",
                position=0,
                leave=True,
            ):
                if not self.result_manager.start_scenario(scenario):
                    continue
                if self.video_recorder.config.enabled:
                    self.video_recorder.start_scenario(
                        self.result_manager.video_directory(scenario.scenario_id)
                    )
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
                        video_recorder=self.video_recorder,
                        on_episode_started=self._record_episode_started,
                        on_model_diagnostics=(
                            self._record_model_diagnostics
                            if self._model_logs
                            else None
                        ),
                    )
                    self._run_group(group_runner)
                self.video_recorder.finish_scenario()
                scenario_result = self.result_manager.finish_scenario(scenario)
                if scenario_result is None:
                    self._progress_logger.warning(
                        "SCENARIO_PARTIAL scenario=%s reason=infrastructure_failure",
                        scenario.scenario_id,
                    )
            benchmark_result = self.result_manager.finish_benchmark()
            if benchmark_result.status == "partial":
                summary = (
                    "Benchmark completed with infrastructure failures: "
                    f"{benchmark_result.completed_scenario_count}/"
                    f"{benchmark_result.expected_scenario_count} scenarios completed; "
                    "partial scenarios were excluded from metrics."
                )
                self._progress_logger.warning(summary)
                tqdm.write(summary)
                for scenario_id in benchmark_result.partial_scenario_ids:
                    message = f"- {scenario_id}"
                    self._progress_logger.warning(
                        "PARTIAL_SCENARIO scenario=%s",
                        scenario_id,
                    )
                    tqdm.write(message)
                    for failure in benchmark_result.infrastructure_failures:
                        if failure.scenario_id != scenario_id:
                            continue
                        reason = (failure.reason or "unknown").replace("\n", " ")
                        detail = f"  - {failure.episode_id}: {reason}"
                        self._progress_logger.warning(
                            "INFRASTRUCTURE_FAILURE scenario=%s episode=%s reason=%s",
                            scenario_id,
                            failure.episode_id,
                            reason,
                        )
                        tqdm.write(detail)
        finally:
            try:
                self.agent.stop()
            finally:
                try:
                    self.video_recorder.close()
                finally:
                    try:
                        if hasattr(self.tool_executor, "env"):
                            self.tool_executor.env.close()
                    finally:
                        if self._progress_log_handler is not None:
                            self._progress_logger.removeHandler(
                                self._progress_log_handler
                            )
                            self._progress_log_handler.close()
                            self._progress_log_handler = None
