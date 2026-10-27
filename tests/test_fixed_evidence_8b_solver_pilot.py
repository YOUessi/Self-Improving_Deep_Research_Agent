"""Fixed-evidence 8B rescoring tests: no DEV gold in model input."""
import json

import pytest

from scripts.m7_fixed_evidence_8b_solver_pilot import (
    read_sources,solver_request,validate_prediction,evaluate_counterfactual
)
from sidra.schema import AgentTrace,Action


def test_only_actually_read_tool_text_reaches_solver():
    t=AgentTrace(
        question_id="p1",question="Which country?",policy="finite",
        answer="",citations=[],termination="abstained",steps=[
            {"action":Action("search",query="Country").to_dict(),
             "observation":{"kind":"search","hits":[
                 {"doc_id":"seen-not-read","title":"NOT_READ",
                  "snippet":"gold_answer: HIDDEN"}]}},
            {"action":Action("read",doc_id="a").to_dict(),
             "observation":{"kind":"read","doc_id":"a","title":"A",
                            "text":"The country is Canada."}},
            {"action":Action("read",doc_id="b").to_dict(),
             "observation":{"kind":"read","doc_id":"b","title":"B",
                            "text":"Canada lies in North America."}},
        ])
    read=read_sources(t)
    assert [x["doc_id"] for x in read]==["a","b"]
    assert "seen-not-read" not in json.dumps(read)
    assert "gold_answer" not in json.dumps(read)
    prompt=solver_request(t.question,read,model="sidra-qwen3-8b-awq")
    text=json.dumps(prompt)
    assert "gold_answer" not in text
    assert "supporting_doc_ids" not in text
    assert "seen-not-read" not in text
    assert prompt["response_format"]["json_schema"]["schema"][
        "properties"]["citations"]["items"]["enum"] == ["a","b"]


def test_solver_response_cannot_fake_unread_citations():
    with pytest.raises(ValueError,match="unread"):
        validate_prediction({"answer":"Canada","citations":["a","unread"]},["a","b"])
    with pytest.raises(ValueError,match="duplicate"):
        validate_prediction({"answer":"Canada","citations":["a","a"]},["a","b"])
    with pytest.raises(ValueError,match="empty"):
        validate_prediction({"answer":"","citations":["a"]},["a","b"])
    assert validate_prediction({"answer":"Canada ","citations":["a","b"]},
                               ["a","b"])==("Canada",["a","b"])


def test_counterfactual_refuses_train64_and_sealed_dev_paths(tmp_path):
    for typ in ("heldout","train64","final_b160","eval_a"):
        path=tmp_path/typ/"converted"
        path.mkdir(parents=True,exist_ok=True)
        with pytest.raises(ValueError,match="DEV pilot"):
            evaluate_counterfactual(path,tmp_path/"fake_traces.jsonl",
                                    tmp_path/"output",endpoint="http://127.0.0.1",
                                    model="no_model")


def test_empty_evidence_cannot_get_solver_request():
    with pytest.raises(ValueError,match=">=2"):
        solver_request("Q",[],model="m")
