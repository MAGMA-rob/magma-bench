from copy import deepcopy
from typing import Any, Dict, List

import pytest


pytest.importorskip("sapien")
pytest.importorskip("mani_skill")

from magma_core.protocol.agent import AgentOutput

from magma_bench.agents.history_summary_reactive.agent import (
    HistorySummaryReactiveBenchmarkAgent,
)
from magma_bench.agents.registry import get_agent_mode
from magma_bench.data_structures import EpisodeSituation


class FakeInstruction:
    def __init__(self, content: str, role: str = "USER") -> None:
        self.content = content
        self.role = role

    def get_content(self) -> str:
        return self.content

    def get_role(self) -> str:
        return self.role

    def get_timestamp(self) -> int:
        return 10


def build_situation(
    env_idx: int,
    *,
    summary: str | None = None,
    history: List[Dict[str, Any]] | None = None,
) -> EpisodeSituation:
    memory: Dict[str, Any] = {
        "memory_list": [f"permanent rule {env_idx}"]
    }
    if summary is not None:
        memory["summary"] = summary
    return EpisodeSituation(
        tools=[{"name": "pick"}],
        attributes={"known_robots": [f"robot_{env_idx}"]},
        memory=memory,
        current_instruction=FakeInstruction(f"instruction {env_idx}"),
        history=[] if history is None else deepcopy(history),
    )


def summary_update(
    *,
    called: bool,
    input_summary: str,
    summary: str,
    input_history: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "called": called,
        "input_summary": input_summary,
        "summary": summary,
    }
    if input_history is not None:
        result["input_history"] = deepcopy(input_history)
    return result


def valid_response(
    source_id: int,
    update: Dict[str, Any],
    *,
    say: str = "done",
) -> AgentOutput:
    return AgentOutput(
        source_id=source_id,
        candidate_index=0,
        valid=True,
        output={
            "say": say,
            "action": {},
            "think": "internal",
            "summary_update": update,
        },
    )


def invalid_response(
    source_id: int,
    component: str,
    update: Dict[str, Any],
) -> AgentOutput:
    return AgentOutput(
        source_id=source_id,
        candidate_index=0,
        valid=False,
        output={
            "component": component,
            "reason": f"{component} failed",
            "raw_output": "bad output",
            "summary_update": update,
        },
    )


def build_agent(
    response_batches: List[Dict[int, AgentOutput]],
    *,
    logs: bool = False,
) -> tuple[
    HistorySummaryReactiveBenchmarkAgent,
    List[Dict[int, Dict[str, Any]]],
]:
    agent = object.__new__(HistorySummaryReactiveBenchmarkAgent)
    agent.prediction_mode = "tool_select"
    agent.inference_mode = False
    agent.collect_model_logs = logs
    captured: List[Dict[int, Dict[str, Any]]] = []
    remaining = iter(response_batches)

    def send_to_agent(
        payloads: Dict[int, Dict[str, Any]],
    ) -> Dict[int, AgentOutput]:
        captured.append(deepcopy(payloads))
        return next(remaining)

    agent.send_to_agent = send_to_agent
    return agent, captured


def test_hsr_payload_and_skipped_summary_preserve_history() -> None:
    history = [{
        "author": "USER",
        "content": "earlier instruction",
        "timestamp": 0,
    }]
    response = valid_response(
        3,
        summary_update(
            called=False,
            input_summary="stored summary",
            summary="stored summary",
        ),
    )
    agent, batches = build_agent([{3: response}])
    situation = build_situation(
        3,
        summary="stored summary",
        history=history,
    )

    result = agent.compute_agent_results({3: situation})[0]

    assert agent.get_candidate_counts() == {"commander": 1}
    assert batches[0][3] == {
        "instruction": "instruction 3",
        "instruction_role": "USER",
        "attributes": {"known_robots": ["robot_3"]},
        "memory": {
            "memory_list": ["permanent rule 3"],
            "summary": "stored summary",
        },
        "function": [{"name": "pick"}],
        "history": history,
        "prediction_mode": "tool_select",
        "inference_mode": False,
    }
    assert result.answer.get_say() == "done"
    assert result.situation.memory["summary"] == "stored summary"
    assert result.situation.history[:1] == history
    assert len(result.situation.history) == 3


def test_hsr_called_summary_replaces_old_history_before_current_exchange() -> None:
    old_history = [
        {"author": "USER", "content": "old request", "timestamp": 0},
        {"author": "MODEL", "content": "old answer", "timestamp": 1},
    ]
    response = valid_response(
        4,
        summary_update(
            called=True,
            input_summary="old summary",
            summary="new summary",
            input_history=old_history,
        ),
    )
    agent, _ = build_agent([{4: response}], logs=True)

    result = agent.compute_agent_results({
        4: build_situation(
            4,
            summary="old summary",
            history=old_history,
        )
    })[0]

    assert result.situation.memory["summary"] == "new summary"
    assert len(result.situation.history) == 2
    assert result.situation.history[0]["content"] == "instruction 4"
    assert result.situation.history[1]["content"] == "done"
    assert [
        item["component"] for item in result.model_diagnostics
    ] == ["summarizer", "commander"]
    assert result.model_diagnostics[0]["input"]["history"] == [
        "old request",
        "old answer",
    ]
    assert result.model_diagnostics[1]["input"]["summary"] == "new summary"
    assert result.model_diagnostics[1]["input"]["history"] == []
    assert result.model_diagnostics[1]["raw_output"]["think"] == "internal"


def test_hsr_batch_keeps_summary_updates_isolated() -> None:
    first_history = [{
        "author": "USER",
        "content": "first old message",
        "timestamp": 0,
    }]
    responses = {
        1: valid_response(
            1,
            summary_update(
                called=True,
                input_summary="summary 1",
                summary="updated 1",
                input_history=first_history,
            ),
        ),
        2: valid_response(
            2,
            summary_update(
                called=False,
                input_summary="summary 2",
                summary="summary 2",
            ),
        ),
    }
    agent, _ = build_agent([responses])

    results = agent.compute_agent_results({
        1: build_situation(1, summary="summary 1", history=first_history),
        2: build_situation(2, summary="summary 2", history=[]),
    })
    by_env = {result.answer.source_node_id: result for result in results}

    assert by_env[1].situation.memory["summary"] == "updated 1"
    assert by_env[2].situation.memory["summary"] == "summary 2"
    assert by_env[1].situation.memory["memory_list"] == ["permanent rule 1"]
    assert by_env[2].situation.memory["memory_list"] == ["permanent rule 2"]


@pytest.mark.parametrize("component", ["summarizer", "context_budget", "commander"])
def test_hsr_component_failures_do_not_update_state(component: str) -> None:
    old_history = [{
        "author": "USER",
        "content": "old request",
        "timestamp": 0,
    }]
    called = component in {"summarizer", "commander"}
    response = invalid_response(
        5,
        component,
        summary_update(
            called=called,
            input_summary="old summary",
            summary=(
                "new summary" if component == "commander" else "old summary"
            ),
            input_history=old_history if called else None,
        ),
    )
    agent, _ = build_agent([{5: response}], logs=True)
    situation = build_situation(
        5,
        summary="old summary",
        history=old_history,
    )

    result = agent.compute_agent_results({5: situation})[0]

    assert not result.answer.is_valid()
    assert result.situation.memory == situation.memory
    assert result.situation.history == situation.history
    assert result.model_diagnostics
    if component == "context_budget":
        assert [
            item["component"] for item in result.model_diagnostics
        ] == ["context_budget"]
    elif component == "summarizer":
        assert [
            item["component"] for item in result.model_diagnostics
        ] == ["summarizer"]
    else:
        assert [
            item["component"] for item in result.model_diagnostics
        ] == ["summarizer", "commander"]


@pytest.mark.parametrize(
    "output, reason",
    [
        ({"say": "done", "action": {}}, "missing summary_update"),
        (
            {
                "say": "done",
                "action": {},
                "summary_update": {
                    "called": "yes",
                    "input_summary": "",
                    "summary": "",
                },
            },
            "called must be a boolean",
        ),
        (
            {
                "say": "done",
                "action": {},
                "summary_update": {
                    "called": False,
                    "input_summary": "wrong",
                    "summary": "wrong",
                },
            },
            "does not match",
        ),
    ],
)
def test_hsr_malformed_summary_update_becomes_bad_answer(
    output: Dict[str, Any],
    reason: str,
) -> None:
    response = AgentOutput(
        source_id=6,
        candidate_index=0,
        valid=True,
        output=output,
    )
    agent, _ = build_agent([{6: response}], logs=True)
    situation = build_situation(6)

    result = agent.compute_agent_results({6: situation})[0]

    assert not result.answer.is_valid()
    assert reason in result.answer.to_string()
    assert result.situation.memory == situation.memory
    assert result.situation.history == situation.history
    assert [
        item["component"] for item in result.model_diagnostics
    ] == ["hsr_error"]


def test_hsr_logs_skip_summarizer_when_not_called() -> None:
    response = valid_response(
        7,
        summary_update(
            called=False,
            input_summary="",
            summary="",
        ),
    )
    agent, _ = build_agent([{7: response}], logs=True)

    result = agent.compute_agent_results({7: build_situation(7)})[0]

    assert [
        item["component"] for item in result.model_diagnostics
    ] == ["commander"]
    assert "function" not in result.model_diagnostics[0]["input"]


def test_hsr_agent_is_registered() -> None:
    mode = get_agent_mode("history_summary_reactive")

    assert mode.name == "history_summary_reactive"
    assert mode.load_agent_class() is HistorySummaryReactiveBenchmarkAgent
