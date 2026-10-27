"""Sentence-ID evidence selection cannot reference unread or invented evidence."""
import json

import pytest

from scripts.m7_sentence_id_solver_pilot import (
    sentence_candidates,sentence_prompt,validate_selection,compare
)


SOURCES=[
    {"doc_id":"doc-a","title":"A",
     "text":"Mara was born in River City in the year 1950. "
            "Another unrelated person was born in 1971."},
    {"doc_id":"doc-b","title":"B",
     "text":"River City is located in the province of Island Green. "
            "Island Green is in the coastal region."},
]


def test_catalog_comes_only_from_observed_sentences():
    rows=sentence_candidates("Where was Mara born?",SOURCES,max_per_doc=9)
    assert rows
    assert {x["doc_id"] for x in rows}=={"doc-a","doc-b"}
    assert len({x["id"] for x in rows})==len(rows)
    assert all(x["sentence"] in next(d["text"] for d in SOURCES
                                     if d["doc_id"]==x["doc_id"]) for x in rows)


def test_constrained_answer_uses_only_observed_sentence_ids():
    catalog=sentence_candidates("Where was Mara born?",SOURCES)
    req=sentence_prompt("Where was Mara born?",catalog,model="sidra-qwen3-8b-awq")
    allowed=req["response_format"]["json_schema"]["schema"][
        "properties"]["sentence_ids"]["items"]["enum"]
    assert set(allowed)=={s["id"] for s in catalog}
    assert "gold_answer" not in json.dumps(req)
    assert "supporting_doc_ids" not in json.dumps(req)
    from_a=next(x["id"] for x in catalog if x["doc_id"]=="doc-a")
    from_b=next(x["id"] for x in catalog if x["doc_id"]=="doc-b")
    gated=validate_selection(
        {"answer":"River City","sentence_ids":[from_a,from_b]},catalog)
    assert gated["evidence_gate_passed"]
    assert gated["gated_citations"]==["doc-a","doc-b"]


def test_invalid_unread_evidence_ids_rejected():
    catalog=sentence_candidates("Where was Mara born?",SOURCES)
    with pytest.raises(ValueError,match="unobserved"):
        validate_selection({"answer":"River City","sentence_ids":["fabricated"]},catalog)
    first=catalog[0]["id"]
    with pytest.raises(ValueError,match="duplicate"):
        validate_selection({"answer":"River City","sentence_ids":[first,first]},catalog)
    same_doc=[x["id"] for x in catalog if x["doc_id"]=="doc-a"]
    if len(same_doc)>=2:
        result=validate_selection(
            {"answer":"River City","sentence_ids":same_doc[:2]},catalog)
        assert not result["evidence_gate_passed"]
        assert result["gated_answer"]==""


def test_only_opened_dev_pilot_can_be_evaluated(tmp_path):
    for part in ("heldout","eval_a","final_b160"):
        input_dir=tmp_path/part/"converted"
        input_dir.mkdir(parents=True)
        with pytest.raises(ValueError,match="DEV pilot"):
            compare(input_dir,tmp_path/"no-traces",tmp_path/"report",
                    endpoint="http://127.0.0.1",model="not-real")
