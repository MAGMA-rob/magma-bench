from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import threading
import time

import requests
from requests.exceptions import ConnectionError, RequestException

from magma_bench.data_structures import BenchmarkAgentResult, EpisodeSituation

from magma_core.base.agents import AgentAnswer, BadAgentAnswer, ValidAgentAnswer
from magma_core.protocol.agent import (
    AgentInput,
    AgentOutput,
    AgentRequest,
    AgentResponse,
)
from magma_core.utils.text_utils import (
    format_history_message,
    format_model_history_message,
)


class BenchmarkAgent(ABC):
    """
    Agent evaluated by magma_bench.

    A benchmark agent is a class that is the intermediate with the magma agent server.
    """

    agent_name: str

    waiting_inputs: Dict[int, EpisodeSituation]
    completed_results: List[BenchmarkAgentResult]

    def __init__(
        self,
        agent_url: str,
        remote_agent_name: str,
        prediction_mode: str = "tool_select",
        inference_mode: bool = False,
        timeout: float = 360,
        **args,
    ) -> None:
        if prediction_mode not in {"sequence", "tool_select"}:
            raise ValueError(
                f"prediction_mode should be sequence or tool_select. Got {prediction_mode}"
            )
        name_seg = args.get("agent_name", remote_agent_name).split("/")
        self.agent_name = name_seg[-1] if name_seg[-1].strip() != "" else name_seg[-2]
        self.agent_url = agent_url.rstrip("/")
        self.remote_agent_name = remote_agent_name
        self.prediction_mode = prediction_mode
        self.inference_mode = inference_mode
        self.timeout = timeout

        self.lock = threading.Lock()
        self.completed_results = []
        self.waiting_inputs = {}
        self.thread_exception: Optional[Exception] = None
        self.thread_running = True
        self.thread = threading.Thread(
            target=self._periodic_computing_of_answers,
            args=(4,),
            daemon=True,
        )
        self.thread.start()

    @staticmethod
    def stringify_content(content: Any) -> str:
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=True)
        if content is None:
            return ""
        return str(content)

    @abstractmethod
    def compute_agent_results(
        self,
        batch_inputs: Dict[int, EpisodeSituation],
    ) -> List[BenchmarkAgentResult]:
        raise NotImplementedError()

    @abstractmethod
    def get_candidate_counts(self) -> Dict[str, int]:
        raise NotImplementedError()

    def send_to_agent(
        self,
        input_payloads: Dict[int, Dict[str, Any]],
    ) -> Dict[int, AgentOutput]:
        """Send one real magma_agent batch while preserving environment IDs."""

        if not input_payloads:
            return {}

        request = AgentRequest(
            agent=self.remote_agent_name,
            inputs=[
                AgentInput(
                    id=env_idx,
                    input=input_payload,
                )
                for env_idx, input_payload in input_payloads.items()
            ],
            candidate_counts=self.get_candidate_counts(),
        )
        try:
            start_time = time.time()
            response = requests.post(
                f"{self.agent_url}/v1/responses",
                json=request.model_dump(mode="json"),
                timeout=self.timeout,
            )
            elapsed = time.time() - start_time
            response.raise_for_status()
        except ConnectionError as error:
            raise RuntimeError(f"[magma_bench] Connection error: {error}") from error
        except RequestException as error:
            response_text = ""
            if getattr(error, "response", None) is not None:
                response_text = f"\nResponse body: {error.response.text}"
            raise RuntimeError(f"[magma_bench] Request failed: {error}{response_text}") from error

        outputs = AgentResponse.model_validate(response.json()).root
        if len(outputs) != len(input_payloads):
            raise RuntimeError(
                f"Expected one output per input ({len(input_payloads)}), "
                f"got {len(outputs)} in {elapsed:.2f}s."
            )

        outputs_by_source: Dict[int, AgentOutput] = {}
        for output in outputs:
            if output.source_id not in input_payloads:
                raise RuntimeError(
                    f"Agent returned unknown source_id {output.source_id}."
                )
            if output.source_id in outputs_by_source:
                raise RuntimeError(
                    f"Agent returned multiple outputs for source_id "
                    f"{output.source_id}."
                )
            outputs_by_source[output.source_id] = output

        missing_sources = set(input_payloads) - set(outputs_by_source)
        if missing_sources:
            raise RuntimeError(
                f"Agent returned no output for source IDs "
                f"{sorted(missing_sources)}."
            )
        return outputs_by_source

    def normalize_model_response(
        self,
        response: AgentOutput,
        agent_step_id: int = 0,
    ) -> AgentAnswer:
        output = response.output
        if not response.valid:
            error = output.get("error", {})
            if not isinstance(error, dict):
                error = {}
            return BadAgentAnswer(
                source_node_id=response.source_id,
                agent_step_id=agent_step_id,
                say=str(output.get("say", "")),
                raw_action=json.dumps(
                    error.get(
                        "raw_output",
                        output.get("raw_output", output),
                    ),
                    ensure_ascii=True,
                    default=str,
                ),
                reason=str(
                    error.get(
                        "reason",
                        output.get("reason", "Invalid agent response."),
                    )
                ),
            )

        say = output.get("say", "")
        action = self.extract_action(output)

        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                return BadAgentAnswer(
                    source_node_id=response.source_id,
                    agent_step_id=agent_step_id,
                    say=str(say),
                    raw_action=action,
                    reason="Action field is not valid JSON.",
                )

        try:
            calls = self.parse_calls(action)
        except (TypeError, ValueError) as exc:
            return BadAgentAnswer(
                source_node_id=response.source_id,
                agent_step_id=agent_step_id,
                say=str(say),
                raw_action=json.dumps(action, ensure_ascii=True, default=str),
                reason=str(exc),
            )

        if str(say) != "" and calls:
            return BadAgentAnswer(
                source_node_id=response.source_id,
                agent_step_id=agent_step_id,
                say=str(say),
                raw_action=json.dumps(action, ensure_ascii=True, default=str),
                reason=(
                    "An answer cannot contain both a user-facing message "
                    "and tool calls."
                ),
            )

        return ValidAgentAnswer(
            source_node_id=response.source_id,
            agent_step_id=agent_step_id,
            say=str(say),
            calls=calls,
        )

    def update_conversation(
        self,
        situation: EpisodeSituation,
        answer: AgentAnswer,
    ) -> EpisodeSituation:
        """Return a snapshot containing the completed model exchange."""

        updated = situation.snapshot()
        if not answer.is_valid():
            return updated

        instruction = situation.current_instruction
        timestamp = instruction.get_timestamp()
        updated.history.append(
            format_history_message(
                instruction.get_role(),
                instruction.get_content(),
                timestamp,
            )
        )
        updated.history.append(
            format_model_history_message(
                answer.get_say(),
                answer.to_dict()["action"],
                timestamp + 1,
            )
        )
        return updated

    def extract_action(self, output: Dict[str, Any]) -> Any:
        return output.get("action", {})

    def parse_calls(self, action: Any) -> List:
        from magma_core.base.data_structures.agent_call import Call

        if action in ({}, None, []):
            return []

        if isinstance(action, list):
            calls = []
            for item in action:
                calls.extend(self.parse_calls(item))
            return calls

        if not isinstance(action, dict):
            raise TypeError(f"Action must be a dict, list, or empty value. Got {type(action)}.")

        if "name" in action:
            name = action.get("name")
            arguments = action.get("arguments", {})
            if not isinstance(name, str) or name == "":
                raise ValueError(f"Action name must be a non-empty string. Got {name!r}.")
            if not isinstance(arguments, dict):
                raise TypeError(f"Action arguments must be a dict. Got {type(arguments)}.")
            return [Call(name=name, arguments=arguments)]

        calls = []
        for target_robot_name, robot_action in action.items():
            if not isinstance(robot_action, dict):
                raise TypeError(
                    "Multi-robot action values must be dicts. "
                    f"Got {type(robot_action)} for robot {target_robot_name!r}."
                )
            name = robot_action.get("name")
            arguments = robot_action.get("arguments", {})
            if not isinstance(name, str) or name == "":
                raise ValueError(
                    f"Action name for robot {target_robot_name!r} must be a non-empty string."
                )
            if not isinstance(arguments, dict):
                raise TypeError(
                    f"Action arguments for robot {target_robot_name!r} must be a dict."
                )
            calls.append(
                Call(
                    name=name,
                    arguments=arguments,
                    target_robot_name=str(target_robot_name),
                )
            )
        return calls

    def get_agent_card(self) -> Dict:
        return {
            "agent": self.agent_name,
            "remote_agent": self.remote_agent_name,
            "agent_url": self.agent_url,
            "prediction_mode": self.prediction_mode,
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
