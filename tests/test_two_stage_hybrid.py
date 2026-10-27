"""Two-stage hybrid: READ-only 8B prompt, score separation, eval-only gates."""
import json

import pytest

from sidra.evidence_ranker import EvidenceRanker,FEATURE_NAMES
from sidra.schema import AgentTrace,Action
from scripts.eval_two_stage_hybrid import compose,evaluate


def fake_ranker():
    return EvidenceRanker(
        weights=[0.0]*len(FEATURE_NAMES),bias=0.3,
        means=[0.0]*len(FEATURE_NAMES),
        scales=[1.0]*len(FEATURE_NAMES),
        threshold=0.2,
        train_data_sha256="a"*64)


def trace(reads=2):
    t=AgentTrace(question_id="train-new",question="Where was River City established?",
                 policy="finite_v4",termination="finish")
    for i in range(reads):
        t.steps.append({
            "step":i,"action":Action("read",doc_id=f"d{i}").to_dict(),
            "observation":{"kind":"read","doc_id":f"d{i}",
                "title":f"River District {i}",
                "text":f"River City was established in 1957, according to source {i}."},
        })
    return t


def test_solver_receives_only_actual_read_text_and_ranker_provides_citations(monkeypatch):
    received=[]
    def fake_http(endpoint,body,timeout=75):
        received.append((endpoint,body))
        return ({"answer":"1957","citations":["d0"]},
                {"prompt_tokens":100,"completion_tokens":18},0.025)
    monkeypatch.setattr("scripts.eval_two_stage_hybrid.get_solver_output",fake_http)
    t=trace(reads=2)
    response=compose(t.question,t,fake_ranker(),
                     endpoint="http://127.0.0.1:18988/v1/chat/completions",
                     model="sidra-qwen3-8b-awq")
    assert response["answer"]=="1957"
    assert set(response["citations"])=={"d0","d1"}
    assert response["llm_calls"]==1
    assert response["prompt_tokens"]==100
    assert response["completion_tokens"]==18
    serialized=json.dumps(received)
    assert "gold_answer" not in serialized
    assert "supporting_doc_ids" not in serialized
    assert set(response["citations"]).issubset(
        {step["observation"]["doc_id"] for step in t.steps})


def test_no_second_read_means_no_llm_call(monkeypatch):
    def forbid(*args,**kwargs):
        raise AssertionError("no solver call without independent READ context")
    monkeypatch.setattr("scripts.eval_two_stage_hybrid.get_solver_output",forbid)
    t=trace(reads=1)
    response=compose(t.question,t,fake_ranker(),endpoint="http://localhost",model="m")
    assert response["answer"]=="" and not response["citations"]
    assert response["llm_calls"]==0


def test_unseen_train64_scope_is_mandatory(tmp_path):
    for name in ("pilot","final_b160","frozen_seed42_32p_224h"):
        with pytest.raises(ValueError,match="preregistered"):
            evaluate(tmp_path/name,tmp_path/"baseline",
                     tmp_path/"MODEL.json",tmp_path/"report",
                     endpoint="http://localhost",model="m",
                     expected_ranker_sha="b"*64)


def test_ranker_checksum_prevents_posthoc_threshold_changes(tmp_path):
    root=tmp_path/"frozen_unseen_train64_s71212026"
    root.mkdir()
    (root/"FREEZE_MANIFEST.json").write_text(json.dumps({
        "stage":"M7_NEW_QUESTION_TRAIN64_EVAL_ONLY",
        "optimizer_train_eligible":False,"n":64,
        "question_id_exclusion_verified":True,
        "converted_questions_sha256":"q"*64,
        "converted_documents_sha256":"d"*64,
        "prior_train1024_questions_sha256":"a"*64,
    }))
    source=root/"converted";source.mkdir()
    (source/"questions.jsonl").write_text("sample")
    (source/"documents.jsonl").write_text("sample")
    with pytest.raises(ValueError,match="modified"):
        evaluate(root,tmp_path/"baseline",tmp_path/"MODEL.json",
                 tmp_path/"report",endpoint="http://localhost",model="m",
                 expected_ranker_sha="b"*64)
