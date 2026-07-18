from typing import Any, Dict, List

from magma_core.base.agents import AgentAnswer
from magma_core.protocol.agent import AgentOutput

from magma_bench.agents.base import BenchmarkAgent
from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeSituation,
)


class TaskStateReactiveBenchmarkAgent(BenchmarkAgent):
    """
    Task-state-reactive benchmark agent backed by magma_agent /v1/responses.
    """

    def __init__(
        self,
        agent_url: str,
        remote_agent_name: str = "task_state_reactive",
        **args,
    ) -> None:
        super().__init__(
            agent_url=agent_url,
            remote_agent_name=remote_agent_name,
            prediction_mode=args.pop("prediction_mode", "tool_select"),
            **args,
        )

    def get_candidate_counts(self) -> Dict[str, int]:
        return {
            "tsm": 1,
            "dispatcher": 1,
        }

    def compute_agent_results(
        self,
        batch_inputs: Dict[int, EpisodeSituation],
    ) -> List[BenchmarkAgentResult]:
        payloads = {
            env_idx: {
                "memory": situation.memory,
                "attributes": situation.attributes,
                "history": situation.history,
                "function": situation.tools,
                "instruction": {
                    "type": (
                        "tool_result"
                        if situation.current_instruction.get_role() == "SYSTEM"
                        else "message"
                    ),
                    "content": self.stringify_content(
                        situation.current_instruction.get_content()
                    ),
                },
                "completed_todos": situation.agent_state.get(
                    "completed_todos",
                    [],
                ),
                "completed_goals": situation.agent_state.get(
                    "completed_goals",
                    [],
                ),
            }
            for env_idx, situation in batch_inputs.items()
        }
        responses = self.send_to_agent(payloads)

        results = []
        for env_idx, situation in batch_inputs.items():
            response = responses[env_idx]
            answer = self.normalize_model_response(response)
            updated = self.update_conversation(situation, answer)
            if response.valid:
                output = response.output
                tsm = output.get("tsm", {})
                if isinstance(tsm, dict):
                    representation = tsm.get("representation")
                    if isinstance(representation, dict):
                        updated.memory = representation.copy()

                completed_goals = output.get("completed_goals", [])
                if isinstance(completed_goals, list):
                    updated.agent_state["completed_goals"] = [
                        str(goal)
                        for goal in completed_goals
                    ]

                dispatcher = output.get("dispatcher", {})
                if isinstance(dispatcher, dict):
                    completed_todos = dispatcher.get("completed_todos", [])
                    if isinstance(completed_todos, list):
                        updated.agent_state["completed_todos"] = [
                            str(todo)
                            for todo in completed_todos
                        ]

            results.append(
                BenchmarkAgentResult(
                    answer=answer,
                    situation=updated,
                )
            )
        return results

    def extract_action(self, output: Dict[str, Any]) -> Any:
        dispatcher = output.get("dispatcher", {})
        if not isinstance(dispatcher, dict):
            return {}
        tools = dispatcher.get("tools", [])
        if not isinstance(tools, list):
            return {}
        return {
            call["robot"]: {
                "name": call["name"],
                "arguments": call.get("arguments", {}),
            }
            for call in tools
        }

    def normalize_model_response(
        self,
        response: AgentOutput,
        agent_step_id: int = 0,
    ) -> AgentAnswer:
        if response.valid:
            output = response.output
            dispatcher = output.get("dispatcher", {})
            if isinstance(dispatcher, dict) and "message" in dispatcher:
                message = dispatcher["message"]
                if (
                    isinstance(message, dict)
                    and message.get("recipient") == "user"
                ):
                    output = dict(output)
                    output["say"] = message.get("content", "")
                    response = response.model_copy(update={"output": output})
        return super().normalize_model_response(response, agent_step_id)
