from copy import deepcopy
from types import SimpleNamespace
import json
import sys
import subprocess
import threading
from unittest.mock import Mock

import pytest
import requests

pytest.importorskip("sapien")
pytest.importorskip("mani_skill")

from magma_core.protocol.agent import AgentRequest
from magma_core.simulation.data_structures import EmptyInstruction, StatusReturn, UserInstruction
from magma_bench.agents import BenchmarkAgent
from magma_bench.data_structures import EpisodeSituation


@pytest.fixture
def agent(monkeypatch):
    def get(url, **kwargs):
        payload = {"status": "ready"} if url.endswith("/health") else {
            "agent_id": "example", "agent_version": "1.0", "protocol_version": "2.0",
            "capabilities": {"inference": True},
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)
    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    return BenchmarkAgent("http://example", extra_keys={"inference_mode": True}, collect_model_logs=True)


def situation(instruction=None, memory=None, history=None):
    return EpisodeSituation(
        tools=[], attributes={"known_robots": ["left", "right"]},
        memory={} if memory is None else memory,
        current_instruction=instruction if instruction is not None else UserInstruction("Do it"),
        history=history,
    )


def install_response(monkeypatch, transform=None):
    captured = []
    def post(url, *, json, timeout):
        request = AgentRequest.model_validate(json)
        captured.append(deepcopy(json))
        outputs = [{
            "request_id": request.request_id, "source_id": entry.id, "candidate_index": 0,
            "status": "completed", "memory": {"private": [entry.id]},
            "internal_steps": [{"component": "arbitrary", "output_raw": f"raw-{entry.id}"}],
            "output": {"say": "done", "tool_calls": []}, "error": None,
        } for entry in request.inputs]
        if transform is not None:
            transform(outputs)
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: outputs)
    monkeypatch.setattr(requests, "post", post)
    return captured


def test_batch_memory_and_full_feedback(agent, monkeypatch):
    captured = install_response(monkeypatch)
    feedback = StatusReturn({"error": "blocked", "previous_tool_call": {"left": "pick"}})
    original = {7: situation(), 3: situation(feedback)}
    results = agent.compute_agent_results(original)
    assert [result.answer.source_node_id for result in results] == [7, 3]
    assert [item["instruction"]["type"] for item in captured[0]["inputs"]] == ["user", "env"]
    assert captured[0]["inputs"][1]["instruction"]["content"] == feedback.get_content()
    assert results[0].situation.memory == {"private": [7]}
    assert original[7].memory == {"history": []}
    assert results[0].model_diagnostics == [{
        "internal_steps": [{"component": "arbitrary", "output_raw": "raw-7"}],
        "error": None,
    }]
    assert "raw-3" not in str(results[0].model_diagnostics)
    agent.compute_agent_results({7: results[0].situation})
    assert captured[1]["inputs"][0]["memory"] == {"private": [7]}
    assert captured[1]["request_id"] != captured[0]["request_id"]


def test_memory_initialization_and_isolation():
    memory, history = {"rules": ["careful"]}, [{"author": "USER", "content": "initial"}]
    first = situation(memory=memory, history=history)
    second = situation(memory=memory, history=history)
    first.memory["history"].append({"content": "later"})
    first.memory["rules"].append("new")
    assert second.memory == {"rules": ["careful"], "history": history}
    assert memory == {"rules": ["careful"]}
    assert "history" not in first.to_dict()
    assert "agent_state" not in first.to_dict()


def test_tool_order_and_repeated_robot_preserved(agent, monkeypatch):
    def transform(outputs):
        outputs[0]["output"] = {"tool_calls": [
            {"name": name, "arguments": {"n": index}, "target_robot_name": robot}
            for index, (name, robot) in enumerate((("pick", "left"), ("place", "right"), ("wait", "left")))
        ]}
    install_response(monkeypatch, transform)
    calls = agent.compute_agent_results({0: situation()})[0].answer.get_action()
    assert [(call.name, call.target_robot_name, call.arguments) for call in calls] == [
        ("pick", "left", {"n": 0}), ("place", "right", {"n": 1}), ("wait", "left", {"n": 2}),
    ]


def test_candidate_error_keeps_memory_and_logs(agent, monkeypatch):
    def transform(outputs):
        outputs[0].update(status="error", output=None, internal_steps=[],
                          error={"code": "invalid_output", "message": "bad output"})
    install_response(monkeypatch, transform)
    result = agent.compute_agent_results({0: situation()})[0]
    assert not result.answer.is_valid()
    assert result.answer.reason == "bad output"
    assert result.situation.memory == {"private": [0]}
    assert result.model_diagnostics == [{
        "internal_steps": [],
        "error": {
            "code": "invalid_output",
            "message": "bad output",
            "component": None,
        },
    }]


@pytest.mark.parametrize("change", ["request", "source", "candidate", "missing", "duplicate", "order", "decision"])
def test_invalid_wire_response_is_infrastructure_error(agent, monkeypatch, change):
    def transform(outputs):
        if change == "request": outputs[0]["request_id"] = "wrong"
        elif change == "source": outputs[0]["source_id"] = 99
        elif change == "candidate": outputs[0]["candidate_index"] = 1
        elif change == "missing": outputs.pop()
        elif change == "duplicate": outputs.append(outputs[0])
        elif change == "order": outputs.reverse()
        else: outputs[0]["output"] = {"say": "x", "tool_calls": [{"name": "x", "target_robot_name": "left"}]}
    install_response(monkeypatch, transform)
    with pytest.raises(ValueError):
        agent.compute_agent_results({0: situation(), 1: situation()})


def test_http_error_propagates(agent, monkeypatch):
    def post(*args, **kwargs):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(requests.ConnectionError):
        agent.compute_agent_results({0: situation()})


def test_invalid_instruction_never_calls_server(agent, monkeypatch):
    captured = install_response(monkeypatch)
    for instruction in (EmptyInstruction(), SimpleNamespace(get_role=lambda: "MODEL")):
        with pytest.raises(ValueError):
            agent.compute_agent_results({0: situation(instruction)})
    assert captured == []


def test_identity_and_generic_imports(agent):
    assert agent.get_agent_card() == {
        "agent": "example", "agent_id": "example", "agent_version": "1.0",
        "protocol_version": "2.0", "extra_keys": {"inference_mode": True},
    }
    code = """
import sys
import importlib.abc
class BlockAgentImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0].endswith('_agent'):
            raise ImportError(fullname)
sys.meta_path.insert(0, BlockAgentImports())
from types import SimpleNamespace
from unittest.mock import patch
from magma_bench.agents import BenchmarkAgent
from magma_bench.data_structures import EpisodeSituation
from magma_core.simulation.data_structures import UserInstruction

def get(url, **kwargs):
    payload = {"status": "ready"} if url.endswith("/health") else {
        "agent_id": "independent", "agent_version": "1", "protocol_version": "2.0",
        "capabilities": {"inference": True},
    }
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

def post(url, *, json, timeout):
    output = [{"request_id": json["request_id"], "source_id": 0, "candidate_index": 0,
               "status": "completed", "memory": {"opaque": True}, "output": {"say": "done"}}]
    return SimpleNamespace(raise_for_status=lambda: None, json=lambda: output)

with patch("requests.get", get), patch("requests.post", post), patch("threading.Thread.start"):
    client = BenchmarkAgent("http://simulated")
    situation = EpisodeSituation([], {}, {}, UserInstruction("Do it"))
    result = client.compute_agent_results({0: situation})[0]
    assert result.answer.get_say() == "done"
    assert result.situation.memory == {"opaque": True}
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)



@pytest.mark.parametrize("payload", [
    {"status": "starting"},
    {"agent_id": "x", "agent_version": "1", "protocol_version": "1.0", "capabilities": {"inference": True}},
    {"agent_id": "x", "agent_version": "1", "protocol_version": "2.0", "capabilities": {}},
])
def test_startup_rejects_unready_or_incompatible_server(monkeypatch, payload):
    def get(url, **kwargs):
        data = payload if "status" in payload or url.endswith("/v1/info") else {"status": "ready"}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(ValueError):
        BenchmarkAgent("http://example")


def test_cli_uses_generic_options(monkeypatch):
    from magma_bench.launch import build_override_dict, parse_args
    monkeypatch.setattr(sys, "argv", [
        "magma-bench", "--benchmark-root", "/tmp/benchmark",
        "--agent-address", "http://example", "--run-name", "experiment",
        "--planner-address", "http://planner", "--verifier-backend", "judge",
        "--agent-timeout", "120", "--nb-env", "4",
        "--extra-keys", '{"inference_mode": false}', "--no-deterministic-decoding",
    ])
    args = parse_args()
    assert args.run_name == "experiment"
    assert args.planner_address == "http://planner"
    assert args.extra_keys == {"inference_mode": False}
    assert build_override_dict(args) == {
        "magma_agent_address": "http://example",
        "benchmark": {
            "agent_timeout": 120,
            "backend_verifier": "judge",
            "deterministic_decoding": False,
            "nb_env": 4,
        },
    }
    monkeypatch.setattr(sys, "argv", [
        "magma-bench", "--benchmark-root", "/tmp/benchmark",
        "--unknown-adapter-option", "x",
    ])
    with pytest.raises(SystemExit):
        parse_args()

    for video_mode in ("off", "all", "planner-failure"):
        monkeypatch.setattr(sys, "argv", [
            "magma-bench", "--benchmark-root", "/tmp/benchmark",
            "--videos", video_mode,
        ])
        assert parse_args().videos == video_mode

    for invalid_args in (
        [],
        ["--benchmark-root", "/tmp/benchmark", "--save-dir", "a", "--results-path", "b"],
        ["--benchmark-root", "/tmp/benchmark", "--skip-judge", "--verifier-backend", "judge"],
        ["--benchmark-root", "/tmp/benchmark", "--agent-timeout", "0"],
        ["--benchmark-root", "/tmp/benchmark", "--nb-env", "0"],
        ["--benchmark-root", "/tmp/benchmark", "--videos", "true"],
        ["--benchmark-root", "/tmp/benchmark", "--videos"],
    ):
        monkeypatch.setattr(sys, "argv", ["magma-bench", *invalid_args])
        with pytest.raises(SystemExit):
            parse_args()


def test_conflicting_decoding_options_rejected_before_startup():
    from magma_bench.runner.runner import BenchmarkRunner
    config = SimpleNamespace(benchmark={"deterministic_decoding": True})
    with pytest.raises(ValueError, match="conflicts"):
        BenchmarkRunner(config, extra_keys={"inference_mode": False}, skip_judge=True)


def test_missing_verifier_backend_is_reported_before_agent_startup():
    from magma_bench.runner.runner import BenchmarkRunner
    config = SimpleNamespace(
        benchmark={},
        backends={},
        magma_agent_address="http://example",
        magma_planner_address="http://planner",
    )
    with pytest.raises(ValueError, match="--verifier-backend.*--skip-judge"):
        BenchmarkRunner(config)


def test_episode_completion_logs_infrastructure_failure_once_as_warning():
    from magma_bench.runner.runner import BenchmarkRunner

    runner = BenchmarkRunner.__new__(BenchmarkRunner)
    runner.result_manager = SimpleNamespace(record_episode=Mock())
    runner._episode_started_at = {"infra": 0.0, "success": 0.0}
    runner._progress_logger = SimpleNamespace(info=Mock(), warning=Mock())

    infrastructure_failure = SimpleNamespace(
        episode_id="infra",
        success=False,
        terminal=SimpleNamespace(
            status="infrastructure_failure",
            reason="planner retries exhausted",
        ),
    )
    success = SimpleNamespace(
        episode_id="success",
        success=True,
        terminal=SimpleNamespace(status="success", reason=None),
    )

    runner._record_episode_outcome(infrastructure_failure)
    runner._record_episode_outcome(success)

    runner._progress_logger.warning.assert_called_once()
    runner._progress_logger.info.assert_called_once()
    assert runner._progress_logger.warning.call_args.args[0].startswith(
        "EPISODE_COMPLETED"
    )


def test_disabled_logs_do_not_collect_exchanges(agent, monkeypatch):
    install_response(monkeypatch)
    agent.collect_model_logs = False
    result = agent.compute_agent_results({0: situation()})[0]
    assert result.model_diagnostics == []
