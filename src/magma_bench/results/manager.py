from .metrics_scenario import ScenarioResult
from .metrics import compute_benchmark_metrics
from .export import build_benchmark_payload, export_scenario_summary

from typing import List, Dict
import os
import threading, queue

class ResultManager():
    """This class handle the export and save of the multiple log, result for the benchmark"""

    _output_path : str

    _computed_bench_results : Dict[str, List[ScenarioResult]]
    _waiting_bench_results : queue.Queue

    def __init__(
            self,
            output_path : str,
            per_task_log : bool = True,
            unique_try : bool = False,
    ) -> None:
        self._output_path = output_path
        self._computed_bench_results = {}
        self._waiting_bench_results = queue.Queue()
        self._unique_try = unique_try

        self.thread = threading.Thread(target=self._compute_result, daemon=True)
        self.thread.start()
    
    def _compute_result(self):
        while True:
            scenario_result : ScenarioResult = self._waiting_bench_results.get()
            if scenario_result == None:
                break

            scenario_result.compute_result()

            if not scenario_result.scenario_id in self._computed_bench_results:
                self._computed_bench_results[scenario_result.scenario_id] = []
            self._computed_bench_results[scenario_result.scenario_id].append(scenario_result)

            self._export(scenario_result)


    def stop(self):
        self._waiting_bench_results.put(None)   # unblocks the queue
        self.thread.join()
            
    def get_try_output_path(self, scenario_id: str, try_number: int) -> str:
        base_path = os.path.join(self._output_path, scenario_id)
        os.makedirs(base_path, exist_ok=True)
        out_path = os.path.join(base_path, f"try-{try_number}")
        os.makedirs(out_path, exist_ok=True)
        return out_path

    def _export(self, scenario_result : ScenarioResult):
        """Start the export process"""
        out_path = self.get_try_output_path(scenario_result.scenario_id, scenario_result.try_number)
        scenario_result.export(out_path)

    def push_scenario_result(self, scenario_result : ScenarioResult):            
        self._waiting_bench_results.put(scenario_result)

    def compute_global_metrics(self) -> Dict:
        benchmark_metrics = compute_benchmark_metrics(self._computed_bench_results)

        for scenario_id, scenario_metrics in benchmark_metrics.per_scenario.items():
            output_path = os.path.join(self._output_path, scenario_id, "result.json")
            export_scenario_summary(
                scenario_metrics,
                output_path,
                len(self._computed_bench_results[scenario_id]),
            )

        return build_benchmark_payload(benchmark_metrics)
