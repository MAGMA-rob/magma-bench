import json
from typing import Any, Dict, List

import pytest


pytest.importorskip("sapien")
pytest.importorskip("mani_skill")

from magma_core.base.data_structures import StatusReturn
from magma_core.protocol.agent import AgentOutput
from magma_core.protocol.tsr import TaskStateReactiveResult

from magma_bench.agents.task_state_reactive.agent import (
    TaskStateReactiveBenchmarkAgent,
)
from magma_bench.data_structures import EpisodeSituation


EMPTY_STATE = {"rules": [], "goals": []}
ACTIVE_STATE = {
    "rules": [],
    "goals": [
        {
            "description": "Move the object",
            "completed": False,
            "todos": [
                {"description": "Pick the object", "completed": False}
            ],
        }
    ],
}
COMPLETED_STATE = {
    "rules": [],
    "goals": [
        {
            "description": "Move the object",
            "completed": True,
            "todos": [
                {"description": "Pick the object", "completed": True}
            ],
        }
    ],
}
TOOL_HISTORY = [
    {
        "author": "MODEL",
        "content": json.dumps(
            {
                "tools": [
                    {
                        "robot": "robot",
                        "name": "pick",
                        "arguments": {"object": "cube"},
                    }
                ]
            }
        ),
        "timestamp": 0,
    }
]


class FakeInstruction:
    def __init__(self, content: str, role: str = "USER") -> None:
        self.content = content
        self.role = role

    def get_content(self) -> str:
        return self.content

    def get_role(self) -> str:
        return self.role

    def get_timestamp(self) -> int:
        return 0


def build_situation(
    instruction: Any | None = None,
    *,
    memory: Dict[str, Any] | None = None,
    agent_state: Dict[str, Any] | None = None,
) -> EpisodeSituation:
    return EpisodeSituation(
        tools=[{"name": "pick"}],
        attributes={"known_robots": ["robot"]},
        memory={"memory_list": ["Handle objects carefully"]}
        if memory is None
        else memory,
        current_instruction=(
            FakeInstruction("Move the cube")
            if instruction is None
            else instruction
        ),
        agent_state={} if agent_state is None else agent_state,
    )


def build_response(
    source_id: int,
    before: Dict[str, Any],
    final: Dict[str, Any],
    *,
    message: Dict[str, str] | None = None,
    tools: List[Dict[str, Any]] | None = None,
    completed_todos: List[str] | None = None,
    mode: str = "execution",
    history: List[Dict[str, Any]] | None = None,
    tsm_called: bool = False,
) -> AgentOutput:
    dispatcher_output: Dict[str, Any] = {
        "completed_todos": (
            [] if completed_todos is None else completed_todos
        )
    }
    if message is not None:
        dispatcher_output["message"] = message
    else:
        dispatcher_output["tools"] = [] if tools is None else tools

    tsm: Dict[str, Any] = {
        "called": tsm_called,
        "view": None,
        "raw_output": None,
        "actions": [],
    }
    if tsm_called:
        tsm.update({
            "instruction": {"role": "user", "content": "instruction"},
            "view": {},
            "raw_output": "[]",
        })

    payload = TaskStateReactiveResult.model_validate({
        "task_state": {
            "before": before,
            "after_tsm": final if tsm_called else before,
            "final": final,
        },
        "tsm": tsm,
        "dispatcher": {
            "called": True,
            "mode": mode,
            "view": {},
            "input_history": [],
            "raw_output": dispatcher_output,
            "output": dispatcher_output,
        },
        "dispatcher_history": [] if history is None else history,
    }).model_dump(mode="python")
    return AgentOutput(
        source_id=source_id,
        candidate_index=0,
        valid=True,
        output=payload,
    )


def build_agent(
    response_batches: List[Dict[int, AgentOutput]],
) -> tuple[TaskStateReactiveBenchmarkAgent, List[Dict[int, Dict[str, Any]]]]:
    agent = object.__new__(TaskStateReactiveBenchmarkAgent)
    agent.inference_mode = False
    captured_batches: List[Dict[int, Dict[str, Any]]] = []
    remaining_batches = iter(response_batches)

    def send_to_agent(
        payloads: Dict[int, Dict[str, Any]],
    ) -> Dict[int, AgentOutput]:
        captured_batches.append(payloads)
        return next(remaining_batches)

    agent.send_to_agent = send_to_agent
    return agent, captured_batches


def test_initial_payload_uses_current_protocol_and_persists_runtime() -> None:
    response = build_response(
        3,
        EMPTY_STATE,
        ACTIVE_STATE,
        tools=[
            {
                "robot": "robot",
                "name": "pick",
                "arguments": {"object": "cube"},
            }
        ],
        history=TOOL_HISTORY,
        tsm_called=True,
    )
    agent, batches = build_agent([{3: response}])

    result = agent.compute_agent_results({3: build_situation()})[0]

    assert set(batches[0][3]) == {
        "task_state",
        "call_tsm",
        "instruction",
        "environment_feedback",
        "persistent_rules",
        "attributes",
        "dispatcher_history",
        "tools",
        "inference_mode",
    }
    assert batches[0][3]["call_tsm"] is True
    assert batches[0][3]["instruction"] == {
        "role": "user",
        "content": "Move the cube",
    }
    assert batches[0][3]["persistent_rules"] == [
        "Handle objects carefully"
    ]
    assert result.answer.get_action()[0].name == "pick"
    assert result.situation.agent_state[agent.RUNTIME_KEY] == {
        "task_state": ACTIVE_STATE,
        "dispatcher_history": TOOL_HISTORY,
    }


def test_model_diagnostics_follow_tsm_dispatcher_recall_order() -> None:
    first = build_response(
        3,
        EMPTY_STATE,
        ACTIVE_STATE,
        message={"recipient": "tsm", "content": "Clarify the task"},
        tsm_called=True,
    )
    second = build_response(
        3,
        ACTIVE_STATE,
        ACTIVE_STATE,
        tools=[{
            "robot": "robot",
            "name": "pick",
            "arguments": {"object": "cube"},
        }],
        tsm_called=True,
    )
    agent, _ = build_agent([{3: first}, {3: second}])
    agent.collect_model_logs = True

    result = agent.compute_agent_results({3: build_situation()})[0]

    assert [
        diagnostic["component"]
        for diagnostic in result.model_diagnostics
    ] == ["tsm", "dispatcher", "tsm", "dispatcher"]
    assert result.model_diagnostics[0]["input"]["instruction"] == "instruction"
    assert result.model_diagnostics[0]["input"]["permanent_rules"] == [
        "Handle objects carefully"
    ]
    assert "tools" not in result.model_diagnostics[-1]["input"]
    assert "parsed_output" not in result.model_diagnostics[-1]
    assert "state" not in result.model_diagnostics[-1]

    assert agent._dispatcher_history_bodies(TOOL_HISTORY) == [
        '{"robot": "robot", "name": "pick", '
        '"arguments": {"object": "cube"}}'
    ]


def test_dispatcher_to_tsm_recalls_are_rebatched() -> None:
    first_batch = {
        env_idx: build_response(
            env_idx,
            EMPTY_STATE,
            ACTIVE_STATE,
            message={"recipient": "tsm", "content": f"clarify {env_idx}"},
            tsm_called=True,
        )
        for env_idx in (1, 2)
    }
    first_batch[3] = build_response(
        3,
        EMPTY_STATE,
        ACTIVE_STATE,
        tools=[
            {
                "robot": "robot",
                "name": "pick",
                "arguments": {"object": "cube"},
            }
        ],
        history=TOOL_HISTORY,
        tsm_called=True,
    )
    second_batch = {
        env_idx: build_response(
            env_idx,
            ACTIVE_STATE,
            ACTIVE_STATE,
            tools=[
                {
                    "robot": "robot",
                    "name": "pick",
                    "arguments": {"object": "cube"},
                }
            ],
            history=TOOL_HISTORY,
            tsm_called=True,
        )
        for env_idx in (1, 2)
    }
    agent, batches = build_agent([first_batch, second_batch])

    results = agent.compute_agent_results({
        1: build_situation(),
        2: build_situation(),
        3: build_situation(),
    })

    assert len(batches) == 2
    assert set(batches[1]) == {1, 2}
    assert batches[1][1]["instruction"] == {
        "role": "system",
        "content": "clarify 1",
    }
    assert batches[1][2]["task_state"] == ACTIVE_STATE
    assert all(result.answer.get_action() for result in results)


def test_non_tool_completion_recalls_report_and_discards_first_say() -> None:
    execution = build_response(
        4,
        ACTIVE_STATE,
        COMPLETED_STATE,
        message={"recipient": "user", "content": "intermediate"},
        completed_todos=["t0"],
    )
    report = build_response(
        4,
        COMPLETED_STATE,
        EMPTY_STATE,
        message={"recipient": "user", "content": "Task completed"},
        mode="execution_report",
    )
    situation = build_situation(
        agent_state={
            TaskStateReactiveBenchmarkAgent.RUNTIME_KEY: {
                "task_state": ACTIVE_STATE,
                "dispatcher_history": [],
            }
        }
    )
    agent, batches = build_agent([{4: execution}, {4: report}])

    result = agent.compute_agent_results({4: situation})[0]

    assert len(batches) == 2
    assert batches[1][4]["call_tsm"] is False
    assert batches[1][4]["instruction"] is None
    assert result.answer.get_say() == "Task completed"
    assert result.situation.agent_state[agent.RUNTIME_KEY] == {
        "task_state": EMPTY_STATE,
        "dispatcher_history": [],
    }


def test_tool_completion_waits_for_feedback_before_report() -> None:
    execution = build_response(
        5,
        ACTIVE_STATE,
        COMPLETED_STATE,
        tools=[
            {
                "robot": "robot",
                "name": "pick",
                "arguments": {"object": "cube"},
            }
        ],
        completed_todos=["t0"],
        history=TOOL_HISTORY,
    )
    initial_situation = build_situation(
        agent_state={
            TaskStateReactiveBenchmarkAgent.RUNTIME_KEY: {
                "task_state": ACTIVE_STATE,
                "dispatcher_history": [],
            }
        }
    )
    agent, batches = build_agent([{5: execution}])

    execution_result = agent.compute_agent_results({5: initial_situation})[0]

    assert len(batches) == 1
    assert execution_result.answer.get_action()[0].name == "pick"

    report = build_response(
        5,
        COMPLETED_STATE,
        EMPTY_STATE,
        message={"recipient": "user", "content": "Task completed"},
        mode="execution_report",
    )
    feedback_situation = execution_result.situation.snapshot()
    feedback_situation.set_current_instruction(
        StatusReturn({"infos": "The cube was picked"})
    )
    agent, feedback_batches = build_agent([{5: report}])

    report_result = agent.compute_agent_results({5: feedback_situation})[0]

    assert feedback_batches[0][5]["call_tsm"] is False
    assert feedback_batches[0][5]["environment_feedback"] == [
        "The cube was picked"
    ]
    assert feedback_batches[0][5]["dispatcher_history"] == TOOL_HISTORY
    assert report_result.answer.get_say() == "Task completed"
    assert report_result.situation.agent_state[agent.RUNTIME_KEY][
        "dispatcher_history"
    ] == []

    next_execution = build_response(
        5,
        EMPTY_STATE,
        ACTIVE_STATE,
        tools=[
            {
                "robot": "robot",
                "name": "pick",
                "arguments": {"object": "cube"},
            }
        ],
        tsm_called=True,
    )
    next_situation = report_result.situation.snapshot()
    next_situation.set_current_instruction(FakeInstruction("Continue"))
    agent, next_batches = build_agent([{5: next_execution}])

    agent.compute_agent_results({5: next_situation})

    assert next_batches[0][5]["call_tsm"] is True
    assert next_batches[0][5]["dispatcher_history"] == []


def test_new_episode_does_not_reuse_previous_runtime() -> None:
    first_response = build_response(
        6,
        EMPTY_STATE,
        ACTIVE_STATE,
        tools=[
            {
                "robot": "robot",
                "name": "pick",
                "arguments": {},
            }
        ],
        tsm_called=True,
    )
    second_response = build_response(
        6,
        EMPTY_STATE,
        ACTIVE_STATE,
        tools=[
            {
                "robot": "robot",
                "name": "pick",
                "arguments": {},
            }
        ],
        tsm_called=True,
    )
    agent, batches = build_agent([
        {6: first_response},
        {6: second_response},
    ])

    agent.compute_agent_results({6: build_situation()})
    agent.compute_agent_results({6: build_situation()})

    assert batches[1][6]["task_state"] == EMPTY_STATE
    assert batches[1][6]["dispatcher_history"] == []


def test_runtime_discontinuity_returns_bad_answer() -> None:
    response = build_response(
        7,
        ACTIVE_STATE,
        ACTIVE_STATE,
        tools=[],
    )
    agent, _ = build_agent([{7: response}])

    result = agent.compute_agent_results({7: build_situation()})[0]

    assert result.answer.is_valid() is False
    assert "task_state.before" in result.answer.reason


def test_internal_recall_limit_returns_bad_answer() -> None:
    response_batches = [
        {
            8: build_response(
                8,
                EMPTY_STATE,
                EMPTY_STATE,
                message={"recipient": "tsm", "content": "retry"},
                tsm_called=True,
            )
        }
        for _ in range(5)
    ]
    agent, batches = build_agent(response_batches)

    result = agent.compute_agent_results({8: build_situation()})[0]

    assert len(batches) == 5
    assert result.answer.is_valid() is False
    assert result.answer.reason == "TSR exceeded 4 internal recalls."
