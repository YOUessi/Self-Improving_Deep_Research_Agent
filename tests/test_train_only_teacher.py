"""Teacher gold is isolated from student prompts; successful actions are real."""
import json
from pathlib import Path

import pytest

from scripts.build_and_fit_train_teacher import (
    TrainOnlyOracleTeacher,make_teacher_data,require_train_only,
    _encode_teacher_rows,
)
from sidra.agent import SearchAgent,AgentConfig
from sidra.finite_action import FiniteActionConfig,build_legal_actions,action_prompt
from sidra.io import save_jsonl,iter_jsonl
from sidra.retrieval import BM25Index
from sidra.schema import Document,Question
from sidra.train_reward import score_train_rollout


def _dataset(tmp_path):
    data=tmp_path/"data"
    data.mkdir()
    docs=[
        Document("d0","Mason Bridge","Mason Bridge opened in 1985.",scope="q0"),
        Document("d1","North Bridge","North Bridge opened in 1997.",scope="q1"),
        Document("d2","South Bridge","South Bridge opened in 2004.",scope="q2"),
        Document("d3","West Bridge","West Bridge opened in 2007.",scope="q3"),
    ]
    qs=[
        Question(f"q{i}",f"When did {bridge} open?",year,
                 supporting_doc_ids=(f"d{i}",),split="train",scope=f"q{i}")
        for i,(bridge,year) in enumerate([
            ("Mason Bridge","1985"),("North Bridge","1997"),
            ("South Bridge","2004"),("West Bridge","2007")])
    ]
    save_jsonl(data/"documents.jsonl",[d.__dict__ for d in docs])
    save_jsonl(data/"questions.jsonl",[q.__dict__ for q in qs])
    manifest=tmp_path/"frozen_dev.json"
    manifest.write_text(json.dumps({"pilot":{"selected_ids":["devA"]},
                                    "heldout":{"selected_ids":["devB"]}}))
    return data,manifest,qs,docs


def test_teacher_only_executes_real_legal_actions_and_strict_support(tmp_path):
    _,_,qs,docs=_dataset(tmp_path)
    q=qs[0]
    cfg=AgentConfig(max_steps=6,max_tool_calls=5,search_top_k=3)
    finite_cfg=FiniteActionConfig(answer_proposer="v2")
    oracle=TrainOnlyOracleTeacher(
        q,BM25Index(docs),config=cfg,finite_cfg=finite_cfg)
    trace=SearchAgent(docs,oracle,cfg).run(q)
    assert trace.termination=="finish",trace.error
    reward=score_train_rollout(q,trace,{d.doc_id:d for d in docs},
                              max_tool_calls=5)
    assert reward.strict_outcome==1
    assert trace.search_calls>=1 and trace.read_calls>=1
    assert all(trace.steps[i]["action"]==step["allowed_actions"][step["target_index"]]
               for i,step in enumerate(oracle.steps))
    assert all(step["student_model_gold_injected"] is False
               for step in oracle.steps)
    assert all(step["teacher_gold_used_for_target"] is True
               for step in oracle.steps)


def test_teacher_generation_split_and_no_hidden_label_field(tmp_path):
    data,dev_manifest,qs,docs=_dataset(tmp_path)
    out=tmp_path/"teacher"
    report=make_teacher_data(
        data,dev_manifest,out,question_limit=4,
        answer_proposer="v2")
    assert report["question_count_attempted"]==4
    assert report["strict_success_trajectories"]>=3
    assert report["train_action_labels"]>0
    assert report["heldout_train_action_labels"]>0
    train=list(iter_jsonl(out/"TEACHER_ACTIONS_TRAIN.jsonl"))
    held=list(iter_jsonl(out/"TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl"))
    assert set(x["question_id"] for x in train).isdisjoint(
        x["question_id"] for x in held)
    assert all("gold_answer" not in x and "supporting_doc_ids" not in x
               for x in train+held)
    with pytest.raises(FileExistsError):
        make_teacher_data(data,dev_manifest,out)
    with pytest.raises(ValueError,match="overlap"):
        require_train_only(qs,{"q0"})


def test_teacher_rejects_non_train_gold_and_action_catalog_cheats(tmp_path):
    _,_,qs,docs=_dataset(tmp_path)
    other=Question("dev","When did Mason Bridge open?","1985",
                   supporting_doc_ids=("d0",),split="dev",scope="q0")
    with pytest.raises(ValueError,match="TRAIN"):
        TrainOnlyOracleTeacher(
            other,BM25Index(docs),config=AgentConfig(),
            finite_cfg=FiniteActionConfig(answer_proposer="v2"))
    oracle=TrainOnlyOracleTeacher(
        qs[0],BM25Index(docs),config=AgentConfig(),
        finite_cfg=FiniteActionConfig(answer_proposer="v2"))
    with pytest.raises(ValueError,match="identity"):
        oracle.decide("another question",[],6)
    # No model call can peek at supporting_doc_ids / gold_answer.
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            serialized=json.dumps(messages)
            assert "gold_answer" not in serialized
            assert "supporting_doc_ids" not in serialized
            return serialized
        def __call__(self,s,add_special_tokens=False):
            return {"input_ids":[10,11,12]}
    opts=build_legal_actions(qs[0].question,[],remaining=6,max_tool_calls=5)
    encoded=action_prompt(
        Tokenizer(),qs[0].question,[],6,opts,
        cfg=FiniteActionConfig(answer_proposer="v2"))
    assert encoded==[10,11,12]


def test_teacher_dataset_never_uses_dev_gold_in_label_generation(tmp_path):
    data,manifest,qs,docs=_dataset(tmp_path)
    # Deliberate train/dev overlap must raise before any label is generated.
    manifest.write_text(json.dumps({"pilot":{"selected_ids":["q1"]},
                                    "heldout":{"selected_ids":[]}}))
    with pytest.raises(ValueError,match="overlap"):
        make_teacher_data(data,manifest,tmp_path/"blocked",question_limit=4)
    assert not (tmp_path/"blocked").exists()
