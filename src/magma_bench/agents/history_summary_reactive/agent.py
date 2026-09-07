from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any, Dict, List, Optional

from magma_core.base.agents import AgentAnswer, BadAgentAnswer
from magma_core.protocol.agent import AgentOutput

from magma_bench.agents.history_reactive.agent import (
    HistoryReactiveBenchmarkAgent,
)
from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeSituation,
)


@dataclass(frozen=True)
class SummaryUpdate:
    called: bool
    input_summary: str
    summary: str
    input_history: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_output(
        cls,
        output: Dict[str, Any],
        expected_input_summary: str,
    ) -> "SummaryUpdate":
        value = output.get("summary_update")
        if not isinstance(value, dict):
            raise ValueError(
                "History summary reactive output is missing summary_update."
            )
        called = value.get("called")
        if not isinstance(called, bool):
            raise TypeError("summary_update.called must be a boolean.")
        input_summary = value.get("input_summary")
        if not isinstance(input_summary, str):
            raise TypeError("summary_update.input_summary must be a string.")
        if input_summary != expected_input_summary:
            raise ValueError(
                "summary_update.input_summary does not match the summary sent "
                "by the benchmark."
            )
        summary = value.get("summary")
        if not isinstance(summary, str):
            raise TypeError("summary_update.summary must be a string.")
        if not called and summary != input_summary:
            raise ValueError(
                "A skipped summary update cannot change the summary."
            )
        input_history = value.get("input_history")
        if input_history is not None and not isinstance(input_history, list):
            raise TypeError(
                "summary_update.input_history must be a list when provided."
            )
        return cls(
            called=called,
            input_summary=input_summary,
            summary=summary,
            input_history=deepcopy(input_history),
        )


class HistorySummaryReactiveBenchmarkAgent(HistoryReactiveBenchmarkAgent):
    """Benchmark adapter for magma_agent's history-summary-reactive agent."""

    def __init__(
        self,
        agent_url: str,
        remote_agent_name: str = "history_summary_reactive",
        **args: Any,
    ) -> None:
        self.collect_model_logs = bool(args.pop("collect_model_logs", False))
        super().__init__(
            agent_url=agent_url,
            remote_agent_name=remote_agent_name,
            **args,
        )

    def compute_agent_results(
        self,
        batch_inputs: Dict[int, EpisodeSituation],
    ) -> List[BenchmarkAgentResult]:
        payloads = {
            env_idx: self._build_payload(situation)
            for env_idx, situation in batch_inputs.items()
        }
        responses = self.send_to_agent(payloads)

        results: List[BenchmarkAgentResult] = []
        for env_idx, situation in batch_inputs.items():
            response = responses[env_idx]
            current_summary = situation.memory.get("summary", "")
            if current_summary is None:
                current_summary = ""

            try:
                if not isinstance(current_summary, str):
                    raise TypeError(
                        "Situation memory['summary'] must be a string."
                    )
                update = SummaryUpdate.from_output(
                    response.output,
                    current_summary,
                )
            except (TypeError, ValueError) as error:
                answer: AgentAnswer = self._bad_answer(
                    env_idx,
                    str(error),
                    response.output,
                )
                diagnostics = self._contract_error_diagnostics(
                    payloads[env_idx],
                    response,
                    str(error),
                )
                results.append(BenchmarkAgentResult(
                    answer=answer,
                    situation=situation.snapshot(),
                    model_diagnostics=diagnostics,
                ))
                continue

            answer = self.normalize_model_response(response)
            effective = situation.snapshot()
            if answer.is_valid() and update.called:
                effective.memory = deepcopy(effective.memory)
                effective.memory["summary"] = update.summary
                effective.history = []

            diagnostics: List[Dict[str, Any]] = []
            if self.collect_model_logs:
                try:
                    diagnostics = self._model_diagnostics(
                        payloads[env_idx],
                        response,
                        update,
                    )
                except Exception as error:
                    diagnostics = [{
                        "component": "diagnostic_error",
                        "input": {
                            "instruction": payloads[env_idx]["instruction"],
                            "summary": current_summary,
                        },
                        "raw_output": deepcopy(response.output),
                        "error": str(error),
                    }]
            results.append(BenchmarkAgentResult(
                answer=answer,
                situation=self.update_conversation(effective, answer),
                model_diagnostics=diagnostics,
            ))
        return results

    def _build_payload(
        self,
        situation: EpisodeSituation,
    ) -> Dict[str, Any]:
        return {
            "instruction": self.stringify_content(
                situation.current_instruction.get_content()
            ),
            "instruction_role": situation.current_instruction.get_role(),
            "attributes": deepcopy(situation.attributes),
            "memory": deepcopy(situation.memory),
            "function": deepcopy(situation.tools),
            "history": deepcopy(situation.history),
            "prediction_mode": self.prediction_mode,
            "inference_mode": self.inference_mode,
        }

    def _model_diagnostics(
        self,
        request: Dict[str, Any],
        response: AgentOutput,
        update: SummaryUpdate,
    ) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        component = response.output.get("component")
        reason = response.output.get("reason")

        if update.called:
            diagnostics.append({
                "component": "summarizer",
                "input": {
                    "summary": update.input_summary,
                    "history": self._history_bodies(
                        update.input_history or []
                    ),
                },
                "raw_output": update.summary,
                "error": (
                    str(reason)
                    if not response.valid and component == "summarizer"
                    else None
                ),
            })

        effective_memory = deepcopy(request["memory"])
        effective_history = deepcopy(request["history"])
        if update.called and not (
            not response.valid and component == "summarizer"
        ):
            effective_memory["summary"] = update.summary
            effective_history = []

        commander_was_called = response.valid or component == "commander"
        if commander_was_called:
            if response.valid:
                commander_output = {
                    key: deepcopy(value)
                    for key, value in response.output.items()
                    if key != "summary_update"
                }
                commander_error = None
            else:
                commander_output = deepcopy(
                    response.output.get("raw_output")
                )
                commander_error = str(reason or "Invalid Commander output.")
            diagnostics.append({
                "component": "commander",
                "input": {
                    "instruction": request["instruction"],
                    "summary": effective_memory.get("summary", "") or "",
                    "permanent_rules": deepcopy(
                        effective_memory.get("memory_list", []) or []
                    ),
                    "attributes": deepcopy(request["attributes"]),
                    "history": self._history_bodies(effective_history),
                },
                "raw_output": commander_output,
                "error": commander_error,
            })
        if not diagnostics and not response.valid:
            diagnostics.append({
                "component": str(component or "hsr_error"),
                "input": {
                    "instruction": request["instruction"],
                    "summary": effective_memory.get("summary", "") or "",
                    "permanent_rules": deepcopy(
                        effective_memory.get("memory_list", []) or []
                    ),
                    "attributes": deepcopy(request["attributes"]),
                    "history": self._history_bodies(effective_history),
                },
                "raw_output": deepcopy(response.output.get("raw_output")),
                "error": str(reason or "Invalid HSR output."),
            })
        return diagnostics

    def _contract_error_diagnostics(
        self,
        request: Dict[str, Any],
        response: AgentOutput,
        reason: str,
    ) -> List[Dict[str, Any]]:
        if not self.collect_model_logs:
            return []
        return [{
            "component": "hsr_error",
            "input": {
                "instruction": request["instruction"],
                "summary": request["memory"].get("summary", "") or "",
                "permanent_rules": deepcopy(
                    request["memory"].get("memory_list", []) or []
                ),
                "attributes": deepcopy(request["attributes"]),
                "history": self._history_bodies(request["history"]),
            },
            "raw_output": deepcopy(response.output),
            "error": reason,
        }]

    @staticmethod
    def _history_bodies(history: List[Dict[str, Any]]) -> List[str]:
        bodies: List[str] = []
        for entry in history:
            content = entry.get("content", entry.get("sentence", ""))
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(
                    content,
                    ensure_ascii=False,
                    default=str,
                )
            bodies.append(" ".join(content.splitlines()))
        return bodies

    @staticmethod
    def _bad_answer(
        source_id: int,
        reason: str,
        raw_output: Any,
    ) -> BadAgentAnswer:
        return BadAgentAnswer(
            source_node_id=source_id,
            agent_step_id=0,
            say="",
            raw_action=json.dumps(
                raw_output,
                ensure_ascii=True,
                default=str,
            ),
            reason=reason,
        )
