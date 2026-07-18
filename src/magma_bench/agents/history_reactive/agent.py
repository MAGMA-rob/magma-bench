from typing import Dict, List

from magma_bench.agents.base import BenchmarkAgent
from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeSituation,
)


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

    def compute_agent_results(
        self,
        batch_inputs: Dict[int, EpisodeSituation],
    ) -> List[BenchmarkAgentResult]:
        payloads = {
            env_idx: {
                "instruction": self.stringify_content(
                    situation.current_instruction.get_content()
                ),
                "instruction_role": situation.current_instruction.get_role(),
                "attributes": situation.attributes,
                "memory": situation.memory,
                "function": situation.tools,
                "history": situation.history,
                "prediction_mode": self.prediction_mode,
            }
            for env_idx, situation in batch_inputs.items()
        }
        responses = self.send_to_agent(payloads)

        results = []
        for env_idx, situation in batch_inputs.items():
            answer = self.normalize_model_response(responses[env_idx])
            results.append(
                BenchmarkAgentResult(
                    answer=answer,
                    situation=self.update_conversation(situation, answer),
                )
            )
        return results
