from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Dict, List

from magma_core.simulation.agents import (
    AgentAnswer,
    BadAgentAnswer,
    ValidAgentAnswer,
)
from magma_core.simulation.data_structures import EmptyInstruction, StatusReturn
from magma_core.protocol.agent import AgentOutput
from magma_core.protocol.tsr import (
    TSRInstruction,
    TaskStateReactiveInput,
    TaskStateReactiveResult,
)
from magma_core.tsr_engine import (
    DispatcherMode,
    ReactiveTaskState,
    normalize_dispatcher_history,
    prepare_dispatcher_turn,
    render_dispatcher_view,
    render_tsm_view,
)

from magma_bench.agents.base import BenchmarkAgent
from magma_bench.data_structures import (
    BenchmarkAgentResult,
    EpisodeSituation,
)


class TaskStateReactiveBenchmarkAgent(BenchmarkAgent):
    """Benchmark adapter for the task-state-reactive magma_agent."""

    MAX_INTERNAL_RECALLS = 4
    RUNTIME_KEY = "task_state_reactive"

    def __init__(
        self,
        agent_url: str,
        remote_agent_name: str = "task_state_reactive",
        **args: Any,
    ) -> None:
        self.collect_model_logs = bool(args.pop("collect_model_logs", False))
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
        task_states: Dict[int, Dict[str, Any]] = {}
        dispatcher_histories: Dict[int, List[Dict[str, Any]]] = {}
        pending_payloads: Dict[int, Dict[str, Any]] = {}
        recall_counts = {env_idx: 0 for env_idx in batch_inputs}
        final_responses: Dict[int, AgentOutput] = {}
        failed_answers: Dict[int, BadAgentAnswer] = {}
        model_diagnostics: Dict[int, List[Dict[str, Any]]] = {
            env_idx: [] for env_idx in batch_inputs
        }

        for env_idx, situation in batch_inputs.items():
            runtime = situation.agent_state.get(self.RUNTIME_KEY, {})
            if not isinstance(runtime, dict):
                raise TypeError("TSR runtime must be an object.")

            initial_task_state = runtime.get(
                "task_state",
                situation.memory.get(
                    "task_state",
                    ReactiveTaskState.empty().to_dict(),
                ),
            )
            task_state = ReactiveTaskState.from_dict(initial_task_state).to_dict()
            dispatcher_history = normalize_dispatcher_history(
                runtime.get("dispatcher_history", [])
            )
            task_states[env_idx] = task_state
            dispatcher_histories[env_idx] = dispatcher_history

            instruction = situation.current_instruction
            if isinstance(instruction, EmptyInstruction):
                raise RuntimeError(
                    "An EmptyInstruction must be handled by the suspended-skill "
                    "resume path."
                )

            if isinstance(instruction, StatusReturn):
                environment_feedback: List[str] = []
                for field_name in ("infos", "error", "failure_reason"):
                    value = instruction.status.get(field_name)
                    if isinstance(value, str) and value:
                        environment_feedback = [value]
                        break
                    if (
                        isinstance(value, list)
                        and value
                        and all(
                            isinstance(message, str) and message
                            for message in value
                        )
                    ):
                        environment_feedback = value.copy()
                        break
                pending_payloads[env_idx] = self._build_input(
                    situation,
                    task_state,
                    dispatcher_history,
                    call_tsm=False,
                    environment_feedback=environment_feedback,
                ).model_dump(mode="python")
                continue

            instruction_role = instruction.get_role()
            if instruction_role not in {"USER", "SYSTEM"}:
                raise ValueError(
                    "A TSR turn requires a USER or SYSTEM instruction, or a "
                    "StatusReturn."
                )
            pending_payloads[env_idx] = self._build_input(
                situation,
                task_state,
                dispatcher_history,
                call_tsm=True,
                instruction=TSRInstruction(
                    role="user" if instruction_role == "USER" else "system",
                    content=self.stringify_content(instruction.get_content()),
                ),
            ).model_dump(mode="python")

        while pending_payloads:
            responses = self.send_to_agent(pending_payloads)
            next_payloads: Dict[int, Dict[str, Any]] = {}

            for env_idx, response in responses.items():
                if getattr(self, "collect_model_logs", False):
                    try:
                        records = self._extract_model_diagnostics(
                            pending_payloads[env_idx], response
                        )
                    except Exception as error:
                        records = [{
                            "component": "diagnostic_error",
                            "input": deepcopy(pending_payloads[env_idx]),
                            "raw_output": deepcopy(response.output),
                            "error": str(error),
                        }]
                    model_diagnostics[env_idx].extend(records)
                if not response.valid:
                    final_responses[env_idx] = response
                    continue

                try:
                    result = TaskStateReactiveResult.model_validate(
                        response.output
                    )
                    if result.error is not None:
                        raise ValueError(result.error.reason)
                    if result.task_state.before != task_states[env_idx]:
                        raise ValueError(
                            "TSR response task_state.before does not match the "
                            "runtime sent by the benchmark adapter."
                        )
                    task_states[env_idx] = ReactiveTaskState.from_dict(
                        result.task_state.final
                    ).to_dict()
                    dispatcher_histories[env_idx] = (
                        normalize_dispatcher_history(
                            result.dispatcher_history
                        )
                    )
                except (TypeError, ValueError, IndexError) as error:
                    failed_answers[env_idx] = self._bad_answer(
                        env_idx,
                        str(error),
                        response.output,
                    )
                    continue

                dispatcher_output = result.dispatcher.output
                continuation: TaskStateReactiveInput | None = None
                if result.dispatcher.called and dispatcher_output is not None:
                    message = dispatcher_output.message
                    if message is not None and message.recipient == "tsm":
                        continuation = self._build_input(
                            batch_inputs[env_idx],
                            task_states[env_idx],
                            dispatcher_histories[env_idx],
                            call_tsm=True,
                            instruction=TSRInstruction(
                                role="system",
                                content=message.content,
                            ),
                        )
                    elif not dispatcher_output.tools:
                        next_mode = prepare_dispatcher_turn(
                            ReactiveTaskState.from_dict(task_states[env_idx])
                        ).mode
                        if next_mode is DispatcherMode.EXECUTION_REPORT:
                            continuation = self._build_input(
                                batch_inputs[env_idx],
                                task_states[env_idx],
                                dispatcher_histories[env_idx],
                                call_tsm=False,
                            )

                if continuation is None:
                    final_responses[env_idx] = response
                    continue

                if recall_counts[env_idx] >= self.MAX_INTERNAL_RECALLS:
                    failed_answers[env_idx] = self._bad_answer(
                        env_idx,
                        (
                            "TSR exceeded "
                            f"{self.MAX_INTERNAL_RECALLS} internal recalls."
                        ),
                        response.output,
                    )
                    continue

                recall_counts[env_idx] += 1
                next_payloads[env_idx] = continuation.model_dump(
                    mode="python"
                )

            pending_payloads = next_payloads

        results: List[BenchmarkAgentResult] = []
        for env_idx, situation in batch_inputs.items():
            answer: AgentAnswer
            if env_idx in failed_answers:
                answer = failed_answers[env_idx]
            else:
                answer = self.normalize_model_response(
                    final_responses[env_idx]
                )

            updated = self.update_conversation(situation, answer)
            if answer.is_valid():
                updated.agent_state[self.RUNTIME_KEY] = {
                    "task_state": deepcopy(task_states[env_idx]),
                    "dispatcher_history": deepcopy(
                        dispatcher_histories[env_idx]
                    ),
                }
            results.append(
                BenchmarkAgentResult(
                    answer=answer,
                    situation=updated,
                    model_diagnostics=model_diagnostics[env_idx],
                )
            )
        return results

    @staticmethod
    def _extract_model_diagnostics(
        request: Dict[str, Any],
        response: AgentOutput,
    ) -> List[Dict[str, Any]]:
        """Return ordered, human-readable records for one TSR server turn."""

        try:
            result = TaskStateReactiveResult.model_validate(response.output)
        except (TypeError, ValueError) as error:
            return [{
                "component": "tsr_error",
                "input": deepcopy(request),
                "raw_output": deepcopy(response.output),
                "error": str(error),
            }]

        records: List[Dict[str, Any]] = []
        if result.tsm.called:
            tsm_view = render_tsm_view(result.tsm.view)
            records.append({
                "component": "tsm",
                "input": {
                    "permanent_rules": deepcopy(
                        request.get("persistent_rules", [])
                    ),
                    "rules": list(tsm_view.rules),
                    "goals": list(tsm_view.goals),
                    "instruction": result.tsm.instruction.content,
                },
                "raw_output": deepcopy(result.tsm.raw_output),
                "error": result.tsm.error,
            })

        if result.dispatcher.called:
            dispatcher_representation = deepcopy(result.dispatcher.view)
            dispatcher_representation.setdefault(
                "mode", result.dispatcher.mode
            )
            dispatcher_view = render_dispatcher_view(
                dispatcher_representation
            )
            records.append({
                "component": "dispatcher",
                "input": {
                    "permanent_rules": deepcopy(
                        request.get("persistent_rules", [])
                    ),
                    "rules": list(dispatcher_view.rules),
                    "goals": list(dispatcher_view.goals),
                    "attributes": deepcopy(request.get("attributes", {})),
                    "history": (
                        TaskStateReactiveBenchmarkAgent
                        ._dispatcher_history_bodies(
                            result.dispatcher.input_history
                        )
                    ),
                },
                "raw_output": deepcopy(result.dispatcher.raw_output),
                "error": result.dispatcher.error,
            })

        if not records and result.error is not None:
            records.append({
                "component": f"{result.error.component}_error",
                "input": deepcopy(request),
                "raw_output": deepcopy(result.error.raw_output),
                "error": result.error.reason,
            })
        return records

    @staticmethod
    def _dispatcher_history_bodies(
        history: List[Dict[str, Any]],
    ) -> List[str]:
        """Flatten Dispatcher history to one unlabeled message per line."""

        bodies: List[str] = []
        for entry in history:
            payload = json.loads(entry["content"])
            if entry["author"] == "MODEL":
                bodies.extend(
                    json.dumps(call, ensure_ascii=False)
                    for call in payload["tools"]
                )
                continue
            infos = payload["infos"]
            if isinstance(infos, list):
                bodies.extend(str(message) for message in infos)
            else:
                bodies.append(str(infos))
        return bodies

    def _build_input(
        self,
        situation: EpisodeSituation,
        task_state: Dict[str, Any],
        dispatcher_history: List[Dict[str, Any]],
        *,
        call_tsm: bool,
        instruction: TSRInstruction | None = None,
        environment_feedback: List[str] | None = None,
    ) -> TaskStateReactiveInput:
        persistent_rules = situation.memory.get("memory_list", [])
        if not isinstance(persistent_rules, list) or any(
            not isinstance(rule, str) or not rule
            for rule in persistent_rules
        ):
            raise TypeError("Situation memory_list must contain strings.")

        return TaskStateReactiveInput(
            task_state=deepcopy(task_state),
            call_tsm=call_tsm,
            instruction=instruction,
            environment_feedback=(
                []
                if environment_feedback is None
                else environment_feedback.copy()
            ),
            persistent_rules=persistent_rules.copy(),
            attributes=deepcopy(situation.attributes),
            dispatcher_history=deepcopy(dispatcher_history),
            tools=deepcopy(situation.tools),
            inference_mode=self.inference_mode,
        )

    def extract_action(self, output: Dict[str, Any]) -> Any:
        result = TaskStateReactiveResult.model_validate(output)
        dispatcher_output = result.dispatcher.output
        if dispatcher_output is None or not dispatcher_output.tools:
            return {}
        return {
            call.robot: {
                "name": call.name,
                "arguments": deepcopy(call.arguments),
            }
            for call in dispatcher_output.tools
        }

    def normalize_model_response(
        self,
        response: AgentOutput,
        agent_step_id: int = 0,
    ) -> AgentAnswer:
        if response.valid:
            try:
                result = TaskStateReactiveResult.model_validate(
                    response.output
                )
            except (TypeError, ValueError) as error:
                return self._bad_answer(
                    response.source_id,
                    str(error),
                    response.output,
                    agent_step_id,
                )
            if result.error is not None:
                return self._bad_answer(
                    response.source_id,
                    result.error.reason,
                    response.output,
                    agent_step_id,
                )

            dispatcher_output = result.dispatcher.output
            if (
                dispatcher_output is not None
                and dispatcher_output.message is not None
            ):
                message = dispatcher_output.message
                if message.recipient == "tsm":
                    return self._bad_answer(
                        response.source_id,
                        "An internal Dispatcher-to-TSM message reached the runner.",
                        response.output,
                        agent_step_id,
                    )
                return ValidAgentAnswer(
                    source_node_id=response.source_id,
                    agent_step_id=agent_step_id,
                    say=message.content,
                    calls=[],
                )
        return super().normalize_model_response(response, agent_step_id)

    @staticmethod
    def _bad_answer(
        source_id: int,
        reason: str,
        raw_output: Any,
        agent_step_id: int = 0,
    ) -> BadAgentAnswer:
        return BadAgentAnswer(
            source_node_id=source_id,
            agent_step_id=agent_step_id,
            say="",
            raw_action=json.dumps(
                raw_output,
                ensure_ascii=True,
                default=str,
            ),
            reason=reason,
        )
