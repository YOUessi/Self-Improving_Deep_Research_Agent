"""RSI cross-role checks: novel challenges are NOT auto-labelled with source answers."""
import json

import pytest

from sidra.rsi_ledger import (
    Challenge,freeze_train_challenges,verify_train_traces,record_generation,
)
from sidra.schema import Action,AgentTrace,Document,Question


SOURCE_SHA="a"*40


def train_set():
    return [Question("train-1","What city is home to the Mason Bridge?","Port",
                     supporting_doc_ids=("a","b"),split="train")]


def test_challenger_freeze_deduplicates_and_excludes_dev(tmp_path):
    qs=train_set()
    proposals=[
        {"source_train_question_id":"train-1",
         "challenge_text":"What city is home to the Mason Bridge?",
         "difficulty_hint":"single bridge"},
        {"source_train_question_id":"train-1",
         "challenge_text":" what  city is HOME to the mason bridge? ",
         "difficulty_hint":"duplicate"},
    ]
    out=tmp_path/"frozen"
    report=freeze_train_challenges(
        qs,{"dev-other"},proposals,out,
        challenger_model="challenger_train_only",source_commit=SOURCE_SHA)
    assert report["frozen_count"]==1 and report["deduplicated_count"]==1
    solver=json.loads((out/"SOLVER_INPUT.jsonl").read_text().splitlines()[0])
    assert set(solver)=={"challenge_id","question"}
    assert "gold_answer" not in json.dumps(solver)
    with pytest.raises(FileExistsError):
        freeze_train_challenges(qs,set(),proposals,out,
                                challenger_model="c",source_commit=SOURCE_SHA)


def test_challenger_proposals_must_not_contain_gold_fields(tmp_path):
    proposal={"source_train_question_id":"train-1",
              "challenge_text":"What city is home to the Mason Bridge?",
              "difficulty_hint":"seed",
              "gold_answer":"Port"}
    with pytest.raises(ValueError,match="fields"):
        freeze_train_challenges(train_set(),set(),[proposal],tmp_path/"bad",
                                challenger_model="c",source_commit=SOURCE_SHA)


def test_novel_prompts_cannot_inherit_labels_from_original_train(tmp_path):
    candidate=Challenge(
        challenge_id="c1",source_train_question_id="train-1",
        challenge_text="Is the Mason Bridge in Port or a different city?",
        source_commit=SOURCE_SHA,challenger_model="c",
        difficulty_hint="comparison",text_sha256="b"*64)
    trace=AgentTrace("c1",candidate.challenge_text,"agent",
                     answer="Port",citations=["a","b"],termination="finish")
    docs={"a":Document("a","A","Mason Bridge stands in Port"),
          "b":Document("b","B","Port is a city")}
    result=verify_train_traces(
        [candidate],[trace],{q.question_id:q for q in train_set()},docs)
    assert result["approved_count"]==0
    assert result["evaluated"][0]["verdict"]=="unlabelled_challenge"


def report(before=0.0,after=0.15):
    return {"n":64,"before_verified":before,"after_verified":after,
            "before_answer_em":0.2,"after_answer_em":0.3,
            "frozen_dev_sha256":"c"*64}


def test_generation_ledger_requires_changed_checkpoint_and_positive_evidence(tmp_path):
    r=record_generation(generation=1,parent_adapter_sha256="1"*64,
        new_adapter_sha256="2"*64,source_train_rollouts_sha256="3"*64,
        evaluator_report=report(),out=tmp_path/"g1.json")
    assert not r["promoted"]
    assert r["status"]=="pilot_pending_independent_eval"
    assert r["single_generation_dev_signal_observed"]
    assert r["next_selected_checkpoint_sha256"]=="1"*64
    assert r["actual_model_update_file_verified"] is False
    assert r["multi_generation_RSI_completed"] is False
    with pytest.raises(FileExistsError):
        record_generation(generation=1,parent_adapter_sha256="1"*64,
            new_adapter_sha256="2"*64,source_train_rollouts_sha256="3"*64,
            evaluator_report=report(),out=tmp_path/"g1.json")
    with pytest.raises(ValueError,match="no real"):
        record_generation(generation=2,parent_adapter_sha256="1"*64,
            new_adapter_sha256="1"*64,source_train_rollouts_sha256="3"*64,
            evaluator_report=report(),out=tmp_path/"bad.json")
    bad=record_generation(generation=1,parent_adapter_sha256="1"*64,
        new_adapter_sha256="4"*64,source_train_rollouts_sha256="3"*64,
        evaluator_report=report(0.2,0.1),out=tmp_path/"negative.json")
    assert not bad["promoted"]
    assert bad["status"]=="kept_for_negative_analysis"
