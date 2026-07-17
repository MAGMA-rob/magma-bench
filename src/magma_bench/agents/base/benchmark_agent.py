from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import copy
import json
import threading
import time
import requests
from requests.exceptions import ConnectionError, RequestException

from magma_bench.data_structures import EpisodeSituation

from magma_core.base.agents import AgentAnswer, BadAgentAnswer, ValidAgentAnswer
from magma_core.protocol.agent import AgentInput, AgentRequest, AgentResponse
from magma_core.utils.text_utils import (
    format_history_message,
    format_model_history_message,
)


class BenchmarkAgent(ABC):
    """
    Agent evaluated by magma_bench.

    A benchmark agent is the complete policy that plays in the environment. It
    can own explicit memory, history and task-state logic, but it talks to a
    single magma_agent deployment through the /v1/responses protocol.
    """

    agent_name: str

    def __init__(
        self,
        agent_url: str,
        remote_agent_name: str,
        prediction_mode: str = "tool_select",
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
        self.timeout = timeout

    @staticmethod
    def stringify_content(content: Any) -> str:
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=True)
        if content is None:
            return ""
        return str(content)

    @abstractmethod
    def compute_agent_answer(
        self,
        batch_inputs : List[Dict[int,EpisodeSituation]]
    ) -> List[AgentAnswer]:
        raise NotImplementedError()

    @abstractmethod
    def get_candidate_counts(self) -> Dict[str, int]:
        raise NotImplementedError()

    def send_to_agent(self, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        request = AgentRequest(
            agent=self.remote_agent_name,
            inputs=[
                AgentInput(
                    id=0,
                    input=input_payload,
                )
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
        if len(outputs) != 1:
            raise RuntimeError(
                f"Expected exactly one agent output, got {len(outputs)} in {elapsed:.2f}s."
            )
        output = outputs[0]
        if output.source_id != 0:
            raise RuntimeError(f"Agent returned unknown source_id {output.source_id}.")
        return {
            "valid": output.valid,
            "output": output.output,
        }

    def normalize_model_response(
        self,
        response: Dict[str, Any],
        source_node_id: int = 0,
        agent_step_id: int = 0,
    ) -> AgentAnswer:
        if not response.get("valid", True):
            output = response.get("output", {})
            return BadAgentAnswer(
                source_node_id=source_node_id,
                agent_step_id=agent_step_id,
                say=str(output.get("say", "")),
                raw_action=json.dumps(output.get("raw_output", output), ensure_ascii=True, default=str),
                reason=str(output.get("reason", "Invalid agent response.")),
            )

        output = response.get("output", response)
        say = output.get("say", "")
        action = self.extract_action(output)

        if isinstance(action, str):
            try:
                action = json.loads(action)
            except json.JSONDecodeError:
                return BadAgentAnswer(
                    source_node_id=source_node_id,
                    agent_step_id=agent_step_id,
                    say=str(say),
                    raw_action=action,
                    reason="Action field is not valid JSON.",
                )

        try:
            calls = self.parse_calls(action)
        except (TypeError, ValueError) as exc:
            return BadAgentAnswer(
                source_node_id=source_node_id,
                agent_step_id=agent_step_id,
                say=str(say),
                raw_action=json.dumps(action, ensure_ascii=True, default=str),
                reason=str(exc),
            )

        return ValidAgentAnswer(
            source_node_id=source_node_id,
            agent_step_id=agent_step_id,
            say=str(say),
            calls=calls,
        )

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

    def reset_task(self) -> None:
        self.tools = []
        self.reset_step()

    def reset_step(self) -> None:
        pass

    def get_listed_memory(self) -> List:
        return []


class ManagedBenchmarkAgent(BenchmarkAgent, ABC):
    """
    Base class for agents with local benchmark state.
    """

    memory: Any
    preserved_memory_indices: List[int]
    message_history: List[Dict]
    max_history_length: int = 4
    time_window: float = 90

    def __init__(
        self,
        agent_url: str,
        remote_agent_name: str,
        prediction_mode: str = "tool_select",
        max_time_window: float = 90,
        timeout: float = 360,
        **args,
    ) -> None:
        super().__init__(
            agent_url=agent_url,
            remote_agent_name=remote_agent_name,
            prediction_mode=prediction_mode,
            timeout=timeout,
            **args,
        )
        self.time_window = max_time_window
        self.message_history = []
        self.memory = self.empty_memory()
        self.preserved_memory_indices = []
        self.memory_update_lock = threading.Lock()

    def empty_memory(self) -> Any:
        return []

    def init_task(self, inputs: Dict) -> None:
        super().init_task(inputs)
        self.memory = copy.deepcopy(inputs.get("memory", self.empty_memory()))
        self.preserved_memory_indices = inputs.get("preserved_memory_indices", [])

    def add_message_to_history(
        self,
        query: Dict,
        model_answer: Any,
        model_action: Any = None,
        model_answer_timestamps=None,
    ) -> None:
        if not model_answer_timestamps or not isinstance(model_answer_timestamps, float):
            model_answer_timestamps = query["timestamp"] + 10

        self.message_history.extend(
            [
                format_history_message(
                    query.get("author", "USER"),
                    query.get("content"),
                    query["timestamp"],
                ),
                format_model_history_message(
                    model_answer,
                    model_action,
                    model_answer_timestamps,
                ),
            ]
        )

        while len(self.message_history) > self.max_history_length:
            self.message_history.pop(0)

    def get_recent_messages(self, current_time: float) -> List[Dict]:
        self.message_history = [
            message
            for message in self.message_history
            if current_time - message["timestamp"] <= self.time_window
        ]
        return self.message_history

    def get_listed_memory(self) -> List:
        if isinstance(self.memory, list):
            return copy.deepcopy(self.memory)
        if isinstance(self.memory, dict):
            return copy.deepcopy(self.memory.get("memory", []))
        return []

    def reset_step(self) -> None:
        self.memory = self.empty_memory()
        self.message_history = []
