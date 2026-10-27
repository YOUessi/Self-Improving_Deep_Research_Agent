"""E-029 deterministic CPU tests, never used as real RSI model evaluation."""
import copy
import json
from pathlib import Path

import pytest

from sidra.rsi_observed_read import choose_observed_second_read
from sidra.rsi_observed_read_audit import shadow_slot, paired_read_delta


def hit(doc_id, title, score=2.0):
    return {
        "doc_id": doc_id, "title": title,
        "snippet": f"{title} includes background on the subject.",
        "score": score,
    }


def observed():
    return [
        {"kind": "search", "query": "Founder and partners",
         "hits": [
             hit("d-old", "Old Mill Group", 7),
             hit("d-bridge", "Blue Garden Foundation", 3),
             hit("d-irrelevant", "Mountain Cartography", 1),
         ]},
        {"kind": "read", "doc_id": "d-old", "title": "Old Mill Group",
         "text": ("Old Mill Group was established in Harbor. "
                  "Its partner was Blue Garden Foundation."),
         "truncated": False},
    ]


def test_gold_blind_observed_link_prefers_partner_not_initial_first_hit():
    proposal = choose_observed_second_read(
        "Which organization partnered with Old Mill Group?", observed())
    assert proposal.doc_id == "d-bridge"
    assert proposal.visible_candidates == 2


def test_no_read_no_second_read_and_no_unused_frontier():
    assert choose_observed_second_read(
        "Some question", observed()[:1]) is None
    o = observed()
    o[0]["hits"] = o[0]["hits"][:2]
    assert choose_observed_second_read("Some question", o) is None


@pytest.mark.parametrize("steps,calls", [(2, 5), (3, 2), (1, 3)])
def test_no_budget_for_read_plus_finish(steps, calls):
    assert choose_observed_second_read(
        "Which organization partnered with Old Mill Group?",
        observed(), max_steps=steps, max_tool_calls=calls) is None


@pytest.mark.parametrize("bad", [
    {"kind": "search", "query": "x", "hits": [], "supporting_doc_ids": ["d-bridge"]},
    {"kind": "search", "query": "x", "hits": [
        {**hit("d-bridge", "Blue Garden Foundation"), "gold_answer": "Blue Garden"}]},
    {"kind": "read", "doc_id": "d-old", "title": "Old Mill Group",
     "text": "hidden", "truncated": False, "reward": 1},
    {"kind": "read", "doc_id": "never-searched", "title": "Fake",
     "text": "hidden", "truncated": False},
    {"kind": "search", "query": "x", "hits": [
        {**hit("d-bridge", "Blue Garden Foundation"), "score": float("nan")}]},
])
def test_forbid_gold_fields_and_forged_unobserved_tool_state(bad):
    obs = observed()
    obs.append(bad)
    with pytest.raises(ValueError):
        choose_observed_second_read("Which organization partnered?", obs)


def test_same_doc_in_conflicting_search_sources_rejected():
    o = observed()
    o.append({"kind": "search", "query": "other",
              "hits": [hit("d-bridge", "Incorrect changed title", 3)]})
    with pytest.raises(ValueError, match="conflicting"):
        choose_observed_second_read("Which organization partnered?", o)


def test_shadow_replaces_only_one_real_read_slot_without_extra_tools():
    o = observed()
    raw = {"question_id": "c-" + "1" * 24,
           "question": "Which organization partnered with Old Mill Group?",
           "steps": [
               {"step": 0, "action": {"kind": "search"}, "observation": o[0]},
               {"step": 1, "action": {"kind": "read", "doc_id": "d-old"}, "observation": o[1]},
               {"step": 2, "action": {"kind": "read", "doc_id": "d-irrelevant"},
                "observation": {
                    "kind": "read", "doc_id": "d-irrelevant",
                    "title": "Mountain Cartography", "text": "unrelated", "truncated": False}},
               {"step": 3, "action": {"kind": "finish", "answer": "X", "citations": []}},
           ]}
    row = shadow_slot(raw["question_id"], raw["question"], raw)
    assert row["eligible"] is True
    assert row["step"] == 2
    assert row["actual_read_doc_id"] == "d-irrelevant"
    assert row["shadow_read_doc_id"] == "d-bridge"
    assert row["previous_read_ids"] == ["d-old"]
    assert len(row["observation_prefix_sha256"]) == 64
    assert set(row) == {
        "challenge_id", "eligible", "step", "actual_read_doc_id",
        "shadow_read_doc_id", "frontier_size", "previous_read_ids",
        "observation_prefix_sha256"}
    forged = copy.deepcopy(raw)
    forged["steps"][2]["action"]["doc_id"] = "not-observed"
    with pytest.raises(ValueError, match="not in observed"):
        shadow_slot(raw["question_id"], raw["question"], forged)


def test_shadow_no_action_rewrite_if_no_real_second_read():
    raw = {"question_id": "c-" + "2" * 24, "question": "Question",
           "steps": [{"step": 0, "action": {"kind": "finish"}}]}
    assert shadow_slot(raw["question_id"], "Question", raw) == {
        "challenge_id": raw["question_id"], "eligible": False}


def test_bootstrap_pairs_by_question_not_per_observation():
    assert paired_read_delta([]) == {
        "mean": None, "ci95": None, "n_questions": 0}
    v = paired_read_delta([(0, 1), (1, 0), (1, 1), (0, 0)], seed=829, draws=300)
    assert v["mean"] == 0
    assert v["n_questions"] == 4
    assert v == paired_read_delta(
        [(0, 1), (1, 0), (1, 1), (0, 0)], seed=829, draws=300)
    with pytest.raises(ValueError, match="nonbinary"):
        paired_read_delta([(0, 3)])


def test_endpoints_not_needed_and_train_gold_never_accepted_as_arguments():
    from inspect import signature
    sig = str(signature(choose_observed_second_read))
    assert "gold" not in sig and "support" not in sig
    assert "train" not in sig and "corpus" not in sig


def test_cli_role_separation_no_train_input_to_model_only():
    source = Path(__file__).parents[1] / "scripts" / "audit_rsi_observed_read_shadow.py"
    text = source.read_text(encoding="utf-8")
    assert 'if name == "verify":' in text
    assert 'cmd.add_argument("--train-data"' in text
    assert 'if args.role == "model-only":' in text
