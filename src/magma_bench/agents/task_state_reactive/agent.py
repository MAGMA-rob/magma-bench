from typing import Any, Dict, List

from magma_core.base.agents import AgentAnswer

from magma_bench.agents.base import ManagedBenchmarkAgent


class TaskStateReactiveBenchmarkAgent(ManagedBenchmarkAgent):
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
        self.completed_todos: List[str] = []
        self.completed_goals: List[str] = []

    def empty_memory(self) -> Dict[str, List[str]]:
        return {
            "rules": [],
            "goals": [],
            "todo": [],
        }

    def get_candidate_counts(self) -> Dict[str, int]:
        return {
            "tsm": 1,
            "dispatcher": 1,
        }

    def compute_agent_answer(self, query: Dict, task_attributes: Dict) -> AgentAnswer:
        if not task_attributes or not query:
            raise ValueError("Inputs must contain task attributes and a query.")

        payload = {
            "memory": self._memory_payload(),
            "attributes": task_attributes,
            "history": [],
            "function": self.tools,
            "instruction": {
                "type": self._instruction_type(query),
                "content": self.stringify_content(query.get("content", "")),
            },
            "completed_todos": self.completed_todos,
            "completed_goals": self.completed_goals,
        }

        response = self.send_to_agent(payload)
        answer = self.normalize_model_response(response)
        if response["valid"]:
            output = response["output"]
            tsm = output.get("tsm", {})
            if isinstance(tsm, dict) and isinstance(tsm.get("representation"), dict):
                self.memory = tsm["representation"]
            completed_goals = output.get("completed_goals", [])
            if isinstance(completed_goals, list):
                self.completed_goals = [str(goal) for goal in completed_goals]
            dispatcher = output.get("dispatcher", {})
            completed_todos = dispatcher.get("completed_todos", [])
            if isinstance(completed_todos, list):
                self.completed_todos = [str(todo) for todo in completed_todos]
        return answer

    def extract_action(self, output: Dict[str, Any]) -> Any:
        dispatcher = output.get("dispatcher", {})
        if not isinstance(dispatcher, dict):
            return {}
        tools = dispatcher.get("tools", [])
        if tools:
            return {
                call["robot"]: {
                    "name": call["name"],
                    "arguments": call.get("arguments", {}),
                }
                for call in tools
            }
        return {}

    def normalize_model_response(
        self,
        response: Dict[str, Any],
        source_node_id: int = 0,
        agent_step_id: int = 0,
    ) -> AgentAnswer:
        if response.get("valid", True):
            output = response.get("output", {})
            dispatcher = output.get("dispatcher", {})
            if isinstance(dispatcher, dict) and "message" in dispatcher:
                message = dispatcher["message"]
                if isinstance(message, dict) and message.get("recipient") == "user":
                    output = dict(output)
                    output["say"] = message.get("content", "")
                    response = dict(response)
                    response["output"] = output
        return super().normalize_model_response(response, source_node_id, agent_step_id)

    def reset_step(self) -> None:
        super().reset_step()
        self.completed_todos = []
        self.completed_goals = []

    def _memory_payload(self) -> Dict[str, Any]:
        if not isinstance(self.memory, dict):
            return self.empty_memory()
        return self.memory

    def _instruction_type(self, query: Dict) -> str:
        if query.get("author") == "SYSTEM":
            return "tool_result"
        return "message"
