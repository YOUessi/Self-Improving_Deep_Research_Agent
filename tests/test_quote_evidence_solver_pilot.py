"""Quote-backed evidence is verified against READ text and cannot fake support."""
import json

import pytest

from scripts.m7_quote_evidence_solver_pilot import (
    canonical_quote,quote_prompt,validate_evidence,run
)


SOURCES=[
    {"doc_id":"a","title":"A","text":"Mara was born in River City in 1950."},
    {"doc_id":"b","title":"B","text":"River City is located within Island Green."},
]


def test_exact_quotes_from_two_independent_reads_are_accepted():
    model_output={
        "answer":"River City",
        "evidence":[
            {"doc_id":"a","quote":"Mara was born in River City in 1950."},
            {"doc_id":"b","quote":"River City is located within Island Green."},
        ],
    }
    verified=validate_evidence(model_output,SOURCES)
    assert verified["quote_gate_passed"] is True
    assert verified["gated_answer"]=="River City"
    assert verified["gated_citations"]==["a","b"]
    assert not verified["bad_quotes_or_ids"]


def test_single_quote_or_fabricated_quote_cannot_pass():
    one=validate_evidence({"answer":"River City","evidence":[
        {"doc_id":"a","quote":"Mara was born in River City in 1950."}]},SOURCES)
    assert one["raw_answer"]=="River City"
    assert not one["quote_gate_passed"]
    assert one["gated_answer"]==""
    bad=validate_evidence({"answer":"River City","evidence":[
        {"doc_id":"a","quote":"Mara was born in River City in 1950."},
        {"doc_id":"b","quote":"This town is in France and not Island Green."},
    ]},SOURCES)
    assert bad["gated_citations"]==[]
    assert "fabricated_or_nonverbatim_quote" in bad["bad_quotes_or_ids"]


def test_unread_duplicate_or_short_quote_are_rejected():
    bad=validate_evidence({"answer":"River City","evidence":[
        {"doc_id":"a","quote":"Mara was born in River City in 1950."},
        {"doc_id":"a","quote":"Mara was born in River City in 1950."},
        {"doc_id":"not-read","quote":"Mara was born in River City in 1950."},
        {"doc_id":"b","quote":"City"},
    ]},SOURCES)
    assert not bad["quote_gate_passed"]
    assert "duplicate_document_evidence" in bad["bad_quotes_or_ids"]
    assert "unread_doc_citation" in bad["bad_quotes_or_ids"]
    assert "unsupported_short_quote" in bad["bad_quotes_or_ids"]


def test_generator_receives_question_and_source_text_not_gold():
    request=quote_prompt("Where was Mara born?",SOURCES,model="sidra-qwen3-8b-awq")
    s=json.dumps(request)
    assert "gold_answer" not in s
    assert "supporting_doc_ids" not in s
    evidence=request["response_format"]["json_schema"]["schema"][
        "properties"]["evidence"]
    assert evidence["items"]["properties"]["doc_id"]["enum"]==["a","b"]
    assert request["chat_template_kwargs"]=={"enable_thinking":False}


def test_strictly_pilot_only_dataset(tmp_path):
    for label in ("heldout","eval_a","final_b160"):
        x=tmp_path/label/"converted"
        x.mkdir(parents=True)
        with pytest.raises(ValueError,match="pilot"):
            run(x,tmp_path/"not_there",tmp_path/"out",
                endpoint="http://localhost",model="no-model")
