"""Offline failure taxonomy regression tests and heldout export limits."""
import json

import pytest

from scripts.diagnose_traces import audit_run, classify
from sidra.io import save_jsonl
from sidra.schema import Action, AgentTrace, Document, Question


def scenario(*, finish_answer="", cited=(), support=("a", "b")):
    docs = [Document("a", "A", "Alpha lives in Port"),
            Document("b", "B", "Port is in North"),
            Document("c", "C", "Unrelated")]
    q = Question("q", "Where does Alpha live?", "Port",
                 supporting_doc_ids=support, split="dev")
    trace = AgentTrace("q", q.question, "test", answer=finish_answer,
                       citations=list(cited), termination="finish")
    trace.steps = [
        {"step": 0, "action": Action("search", query="Alpha").to_dict(),
         "observation": {"kind": "search", "query": "Alpha",
                         "hits": [{"doc_id":"a"}, {"doc_id":"b"}, {"doc_id":"c"}]}},
        {"step": 1, "action": Action("read", doc_id="a").to_dict(),
         "observation": {"kind":"read","doc_id":"a","text":docs[0].text}},
    ]
    return docs, q, trace


def test_diagnosis_distinguishes_retrieval_vs_read_vs_citation():
    docs, q, t = scenario(finish_answer="Port", cited=("a",))
    row = classify(q, t, {d.doc_id:d for d in docs})
    assert row["exact_match"] == 1
    assert row["support_exposed_count"] == 2
    assert row["support_read_count"] == 1
    assert row["support_cited_count"] == 1
    assert row["bottleneck"] == "tool_read_selection"
    assert row["verified_success"] == 0
    assert row["read_before_cite_ok"] is True


def test_diagnosis_reports_repeated_reads_and_malformed_actions():
    docs, q, t = scenario()
    t.steps.append({"step":2,"action":Action("read", doc_id="a").to_dict(),
                    "observation":{"kind":"read","doc_id":"a","text":"Again"}})
    t.termination = "error"
    t.error = "ValueError: Model output did not contain a valid agent action JSON object"
    row = classify(q, t, {d.doc_id:d for d in docs})
    assert row["repeated_read_count"] == 1
    assert row["reason"] == "malformed_action_json"


def test_holdout_summary_contains_no_per_case_gold(tmp_path):
    docs, q, t = scenario(finish_answer="Port", cited=("a",))
    directory=tmp_path/"data";directory.mkdir()
    save_jsonl(directory/"documents.jsonl",[d.__dict__ for d in docs])
    save_jsonl(directory/"questions.jsonl",[q.__dict__])
    traces=tmp_path/"traces.jsonl"
    save_jsonl(traces,[t.as_dict()])
    pilot=audit_run(directory,traces,tmp_path/"pilot_report",mode="pilot")
    assert pilot["support_read_full"] == 0
    assert (tmp_path/"pilot_report"/"per_example.jsonl").exists()
    heldout=audit_run(directory,traces,tmp_path/"holdout_report",mode="heldout")
    assert heldout["answer_exact"] == 1
    assert not (tmp_path/"holdout_report"/"per_example.jsonl").exists()
    assert "gold_answer" not in (tmp_path/"holdout_report"/"summary.json").read_text()
