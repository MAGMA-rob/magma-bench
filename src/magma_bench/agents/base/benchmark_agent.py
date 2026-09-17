from __future__ import annotations

from copy import deepcopy
import json
import threading
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests

from magma_core.domain.agent_call import Call
from magma_core.protocol.agent import (
    PROTOCOL_VERSION, AgentHealth, AgentInfo, AgentInput, AgentInstruction,
    AgentOutput, AgentRequest, AgentResponse, JsonObject,
)
from magma_core.simulation.agents import AgentAnswer, BadAgentAnswer, ValidAgentAnswer
from magma_core.simulation.data_structures import EmptyInstruction
from magma_bench.data_structures import BenchmarkAgentResult, EpisodeSituation


class BenchmarkAgent:
    """Batched HTTP client for a single protocol-v2 agent runtime."""

    def __init__(
        self,
        agent_url: str,
        agent_name: str | None = None,
        extra_keys: JsonObject | None = None,
        timeout: float = 360,
        collect_model_logs: bool = False,
    ) -> None:
        self.agent_url = agent_url.rstrip("/")
        self.timeout = timeout
        self.extra_keys = deepcopy(extra_keys) if extra_keys is not None else {}
        self.collect_model_logs = collect_model_logs
        health = requests.get(f"{self.agent_url}/health", timeout=timeout)
        health.raise_for_status()
        AgentHealth.model_validate(health.json())
        info = requests.get(f"{self.agent_url}/v1/info", timeout=timeout)
        info.raise_for_status()
        self.info = AgentInfo.model_validate(info.json())
        if self.info.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"Unsupported agent protocol {self.info.protocol_version!r}")
        if not self.info.capabilities.get("inference", False):
            raise ValueError("Agent runtime does not support inference")
        self.agent_name = agent_name or self.info.agent_id
        self.lock = threading.Lock()
        self.completed_results: List[BenchmarkAgentResult] = []
        self.waiting_inputs: Dict[int, EpisodeSituation] = {}
        self.thread_exception: Optional[Exception] = None
        self.thread_running = True
        self.thread = threading.Thread(
            target=self._periodic_computing_of_answers, args=(4,), daemon=True,
        )
        self.thread.start()

    def compute_agent_results(
        self, batch_inputs: Dict[int, EpisodeSituation],
    ) -> List[BenchmarkAgentResult]:
        if not batch_inputs:
            return []
        inputs: list[AgentInput] = []
        for env_idx, situation in batch_inputs.items():
            instruction = situation.current_instruction
            if isinstance(instruction, EmptyInstruction):
                raise ValueError("EmptyInstruction requires the suspended-skill resume path")
            role = instruction.get_role()
            if role not in {"USER", "SYSTEM"}:
                raise ValueError(f"Unsupported input role {role!r}")
            inputs.append(AgentInput(
                id=env_idx,
                instruction=AgentInstruction(
                    type="user" if role == "USER" else "env",
                    content=instruction.get_content(),
                ),
                tools=deepcopy(situation.tools),
                attributes=deepcopy(situation.attributes),
                memory=deepcopy(situation.memory),
                num_outputs=1,
                extra_keys=deepcopy(self.extra_keys),
            ))
        request = AgentRequest(request_id=uuid4().hex, inputs=inputs)
        response = self.send_to_agent(request)
        results: list[BenchmarkAgentResult] = []
        for output in response.root:
            updated = batch_inputs[output.source_id].snapshot()
            updated.memory = deepcopy(output.memory)
            diagnostics: list[dict[str, Any]] = []
            if self.collect_model_logs:
                diagnostics.append({
                    "internal_steps": deepcopy(output.internal_steps),
                    "error": (
                        output.error.model_dump(mode="json")
                        if output.error is not None
                        else None
                    ),
                })
            results.append(BenchmarkAgentResult(
                answer=self.normalize_model_response(output),
                situation=updated,
                model_diagnostics=diagnostics,
            ))
        return results

    def send_to_agent(self, request: AgentRequest) -> AgentResponse:
        response = requests.post(
            f"{self.agent_url}/v1/responses",
            json=request.model_dump(mode="json"), timeout=self.timeout,
        )
        response.raise_for_status()
        parsed = AgentResponse.model_validate(response.json())
        parsed.validate_request(request)
        return parsed

    def normalize_model_response(
        self, response: AgentOutput, agent_step_id: int = 0,
    ) -> AgentAnswer:
        if response.status == "error":
            assert response.error is not None
            return BadAgentAnswer(
                source_node_id=response.source_id, agent_step_id=agent_step_id, say="",
                raw_action=json.dumps(response.model_dump(mode="json"), ensure_ascii=False),
                reason=response.error.message,
            )
        assert response.output is not None
        calls = [
            Call(
                name=call.name,
                arguments=deepcopy(call.arguments),
                target_robot_name=call.target_robot_name,
            )
            for call in response.output.tool_calls
        ]
        return ValidAgentAnswer(
            source_node_id=response.source_id, agent_step_id=agent_step_id,
            say="" if calls else response.output.say,
            calls=calls,
        )

    def get_agent_card(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_name,
            "agent_id": self.info.agent_id,
            "agent_version": self.info.agent_version,
            "protocol_version": self.info.protocol_version,
            "extra_keys": deepcopy(self.extra_keys),
        }

    def get_pending_results(self) -> List[BenchmarkAgentResult]:
        with self.lock:
            if self.thread_exception is not None:
                raise RuntimeError(
                    "The benchmark-agent background thread failed"
                ) from self.thread_exception
            out = self.completed_results.copy()
            self.completed_results.clear()
        return out

    def add_inputs(self, inputs: Dict[int, EpisodeSituation]) -> None:
        with self.lock:
            if self.thread_exception is not None:
                raise RuntimeError(
                    "Cannot add inputs after the benchmark-agent thread failed"
                ) from self.thread_exception
            if not self.thread_running:
                raise RuntimeError("Cannot add inputs to a stopped benchmark agent")
            for idx, situation in inputs.items():
                if idx in self.waiting_inputs:
                    raise RuntimeError(
                        f"Environment {idx} already waits for an agent answer"
                    )
                self.waiting_inputs[idx] = situation.snapshot()

    def stop(self) -> None:
        with self.lock:
            self.thread_running = False
        self.thread.join(timeout=5)

    def _periodic_computing_of_answers(self, interval: float = 4.0) -> None:
        try:
            while self.thread_running:
                start_time = time.time()
                to_do: Dict[int, EpisodeSituation] = {}
                with self.lock:
                    if self.waiting_inputs:
                        to_do = self.waiting_inputs.copy()
                        self.waiting_inputs.clear()
                
                if to_do:
                    results = self.compute_agent_results(to_do)
                    with self.lock:
                        self.completed_results.extend(results)

                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)

        except Exception as error:
            with self.lock:
                self.thread_exception = error
        with self.lock:
            self.thread_running = False
