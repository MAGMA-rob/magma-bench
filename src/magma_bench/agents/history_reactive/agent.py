from typing import Any, Dict, List
import time

from magma_bench.data_structures import EpisodeSituation
from magma_core.base.agents import AgentAnswer
from magma_bench.agents.base import BenchmarkAgent


class HistoryReactiveBenchmarkAgent(BenchmarkAgent):
    """
    History-reactive benchmark agent backed by magma_agent /v1/responses.
    """

    def __init__(
        self,
        agent_url: str,
        remote_agent_name: str = "history_reactive",
        **args,
    ) -> None:
        super().__init__(
            agent_url=agent_url,
            remote_agent_name=remote_agent_name,
            prediction_mode=args.pop("prediction_mode", "tool_select"),
            **args,
        )

    def get_candidate_counts(self) -> Dict[str, int]:
        return {"commander": 1}

    def compute_agent_answer(self, batch_inputs: Dict[int, EpisodeSituation]) -> List[AgentAnswer]:
        if not task_attributes or not query:
            raise ValueError("Inputs must contain task attributes and a query.")

        timestamp = query.get("timestamp", 0)
        payload = {
            "instruction": self.stringify_content(query.get("content", "")),
            "instruction_role": query.get("author", "USER"),
            "attributes": task_attributes,
            "memory": self._memory_payload(),
            "function": self.tools,
            "history": self.get_recent_messages(timestamp),
            "prediction_mode": self.prediction_mode,
        }

        start_time = time.time()
        response = self.send_to_agent(payload)
        answer = self.normalize_model_response(response)
        elapsed = time.time() - start_time

        if response["valid"]:
            output = response["output"]
            self.add_message_to_history(
                query,
                output.get("say", ""),
                output.get("action", {}),
                timestamp + elapsed,
            )
        return answer

    def _memory_payload(self) -> Dict[str, Any]:
        if isinstance(self.memory, dict):
            return self.memory
        return {"memory_list": self.memory}
