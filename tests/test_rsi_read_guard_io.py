"""E-030 guarded real-tool trace and model-I/O safety tests; synthetic policy only."""
import copy
import json

import pytest

from sidra.agent import AgentConfig, SearchAgent
from sidra.rsi_model_io_audit import audit_model_action_io
from sidra.rsi_read_before_finish import ObservedReadBeforeFinishPolicy
from sidra.rsi_trace_audit import audit_tool_trace
from test_rsi_read_before_finish import BasicFakePolicy, make_docs, question


def recorded_model_calls(trace, cfg, *, seed=43):
    """Shape-compatible synthetic API messages, NOT actual model inference."""
    obs = []
    calls = []
    for step_no, step in enumerate(trace.steps):
        if step.get("role") != "observed_read_before_finish":
            calls.append({
                "messages": [
                    {"role": "system", "content": "synthetic fixture only"},
                    {"role": "user", "content": json.dumps({
                        "question": trace.question,
                        "remaining_steps": cfg.max_steps - step_no,
                        "tool_history": obs,
                    })},
                ],
                "constrained_schema": True,
                "generation_seed": seed + len(calls),
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "raw_completion": json.dumps(step["action"]),
            })
        if "observation" in step:
            obs.append(step["observation"])
    return [{"challenge_id": trace.question_id, "calls": calls}]


def treated_case():
    cfg = AgentConfig(max_steps=6, max_tool_calls=5, search_top_k=3)
    guard = ObservedReadBeforeFinishPolicy(BasicFakePolicy())
    trace = SearchAgent(make_docs(), guard, cfg).run(question())
    assert trace.termination == "finish"
    assert trace.read_calls == 2
    return trace, cfg, recorded_model_calls(trace, cfg)


def test_guarded_trace_is_replayable_with_one_verifiable_rule_step():
    trace, cfg, io = treated_case()
    verify = audit_model_action_io(
        [trace], io, seed=43, config=cfg, allow_observed_read_guard=True)
    assert verify["verified_deterministic_observed_READs"] == 1
    assert verify["verified_response_count"] == trace.model_calls == 3
    assert verify["all_model_contexts_and_seeds_match_tool_replay"] is True
    tool = audit_tool_trace(question(), trace, make_docs(), config=cfg)
    assert tool.structurally_valid is True
    assert tool.checked_reads == 2
    assert tool.checked_tool_calls == trace.tool_calls == 3


def test_guarded_trace_cannot_be_laundered_as_ordinary_model_actions():
    trace, cfg, io = treated_case()
    with pytest.raises(ValueError, match="guarded policy identity"):
        audit_model_action_io([trace], io, seed=43, config=cfg)
    guard_clone = copy.deepcopy(trace)
    guard_clone.policy = "synthetic_test_fixture_only"
    with pytest.raises(ValueError, match="guarded policy identity"):
        audit_model_action_io([guard_clone], io, seed=43, config=cfg,
                              allow_observed_read_guard=True)


@pytest.mark.parametrize("tamper,expected", [
    ("wrong_doc", "READ guard did not choose"),
    ("wrong_role_calls", "READ guard attestation"),
    ("wrong_gold_label", "READ guard attestation"),
    ("wrong_frontier", "READ guard frontier"),
    ("wrong_model_context", "model prompt deviates"),
    ("wrong_server_action", "model output/action"),
])
def test_guard_and_model_io_tampering_fails_closed(tamper, expected):
    trace, cfg, io = treated_case()
    trace = copy.deepcopy(trace)
    io = copy.deepcopy(io)
    rule = trace.steps[2]
    if tamper == "wrong_doc":
        rule["action"]["doc_id"] = "d-irrelevant"
    elif tamper == "wrong_role_calls":
        rule["role_model_calls"] = 1
    elif tamper == "wrong_gold_label":
        rule["verification"]["gold_used"] = True
    elif tamper == "wrong_frontier":
        rule["verification"]["legal_frontier_size"] = 6
    elif tamper == "wrong_model_context":
        io[0]["calls"][-1]["messages"][1]["content"] = json.dumps({
            "question": trace.question, "tool_history": [],
            "remaining_steps": 3,
        })
    elif tamper == "wrong_server_action":
        io[0]["calls"][0]["raw_completion"] = json.dumps({
            "kind": "finish", "answer": "forge", "citations": []})
    with pytest.raises(ValueError, match=expected):
        audit_model_action_io([trace], io, seed=43, config=cfg,
                              allow_observed_read_guard=True)


def test_legacy_true_model_action_sequence_still_passes_unchanged():
    cfg = AgentConfig(max_steps=6, max_tool_calls=5, search_top_k=3)
    trace = SearchAgent(make_docs(), BasicFakePolicy(), cfg).run(question())
    assert trace.read_calls == 1
    io = recorded_model_calls(trace, cfg)
    result = audit_model_action_io([trace], io, seed=43, config=cfg)
    assert result["verified_response_count"] == len(trace.steps) == 3
    assert result["verified_deterministic_observed_READs"] == 0
