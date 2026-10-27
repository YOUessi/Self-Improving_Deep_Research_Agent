"""TRAIN teacher-reachable heldout is not public DEV or online reward data."""
import json
from pathlib import Path

import pytest

from scripts.audit_train_teacher_holdout import audit_holdout
from scripts.materialize_train_teacher_holdout import materialize
from sidra.experiment import sha256_file
from sidra.io import save_jsonl,iter_jsonl


def _fixture(tmp_path):
    d=tmp_path/"TRAIN";d.mkdir()
    save_jsonl(d/"documents.jsonl",[
        {"doc_id":"a","title":"A","text":"A in 2015","scope":"global"},
        {"doc_id":"b","title":"B","text":"B in 2016","scope":"global"},
    ])
    save_jsonl(d/"questions.jsonl",[
        {"question_id":"train1","question":"When A?","gold_answer":"2015",
         "supporting_doc_ids":["a"],"split":"train"},
        {"question_id":"train2","question":"When B?","gold_answer":"2016",
         "supporting_doc_ids":["b"],"split":"train"},
    ])
    dev=tmp_path/"DEV_MANIFEST.json"
    dev.write_text(json.dumps({
        "pilot":{"selected_ids":["dev1"]},
        "heldout":{"selected_ids":["dev2"]}
    }))
    t=tmp_path/"TEACHER";t.mkdir()
    save_jsonl(t/"TEACHER_ACTIONS_TRAIN.jsonl",[{"question_id":"train1"}])
    save_jsonl(t/"TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl",[
        {"question_id":"train2"}])
    summary={
        "stage":"TRAIN_ONLY_GOLD_SUPERVISED_TEACHER_NO_RL",
        "teacher_uses_train_gold":True,
        "student_model_sees_gold_answer":False,
        "includes_only_real_strict_successes":True,
        "train_question_sha256":sha256_file(d/"questions.jsonl"),
        "train_document_sha256":sha256_file(d/"documents.jsonl"),
        "dev_manifest_sha256":sha256_file(dev),
        "heldout_train_trajectories":1,
    }
    (t/"TEACHER_SUMMARY.json").write_text(json.dumps(summary))
    return d,t,dev


def test_materialized_teacher_train_holdout_preserves_retrieval_and_disjoint_ids(tmp_path):
    source,teacher,dev=_fixture(tmp_path)
    out=tmp_path/"TRAIN_HOLDOUT"
    h=materialize(source,teacher,dev,out)
    assert h["n"]==1 and h["split"]=="train"
    assert h["ids"]==["train2"]
    assert h["none_in_teacher_fit_partition"]
    assert (source/"documents.jsonl").read_bytes()==(out/"documents.jsonl").read_bytes()
    qs=list(iter_jsonl(out/"questions.jsonl"))
    assert len(qs)==1 and qs[0]["split"]=="train"
    assert qs[0]["question_id"]=="train2"
    with pytest.raises(FileExistsError):
        materialize(source,teacher,dev,out)


def test_teacher_holdout_cannot_include_fit_or_dev_train_ids(tmp_path):
    source,t,dev=_fixture(tmp_path)
    save_jsonl(t/"TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl",[
        {"question_id":"train1"}])
    with pytest.raises(ValueError,match="overlap"):
        materialize(source,t,dev,tmp_path/"bad")
    save_jsonl(t/"TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl",[
        {"question_id":"dev1"}])
    with pytest.raises(ValueError,match="overlap"):
        materialize(source,t,dev,tmp_path/"bad2")


def _paired_fixture(tmp_path):
    source,t,dev=_fixture(tmp_path)
    target=tmp_path/"HOLD"
    hold=materialize(source,t,dev,target)
    trained=tmp_path/"trained";trained.mkdir()
    oldsha="1"*64
    newsha="2"*64
    (trained/"TRAIN_SUMMARY.json").write_text(json.dumps({
        "split":"train","dev_gold_used_for_reward":False,
        "no_model_oracle_labels":True,
        "sampling_and_loss_distribution_match":True,
        "training_questions_sha256":hold["original_questions_sha256"],
        "dev_manifest_sha256":hold["frozen_public_dev_manifest_sha256"],
        "selected_training_ids":["train1"],"num_updates":2,
        "old_adapter_sha256":oldsha,"new_adapter_sha256":newsha,
        "answer_proposer":"v3",
    }))
    arms=[]
    for label,adaptersha,scored in (("before",oldsha,0),("after",newsha,1)):
        arm=tmp_path/label;arm.mkdir()
        m={
            "split":"train","no_weight_updates_in_this_evaluation":True,
            "synthetic_fixture":False,"sample_count":1,
            "data_sha256":{
                "documents.jsonl":hold["original_documents_sha256"],
                "questions.jsonl":hold["holdout_questions_sha256"]
            },
            "seed":63,"agent_config":{"max_tool_calls":5,"max_steps":6},
            "git_commit":"a"*40,"model_id":"qwen3-test",
            "model_or_policy":"finite_constrained_categorical_agent",
            "evaluation_offset":0,"generation_max_new_tokens":None,
        }
        (arm/"manifest.json").write_text(json.dumps(m))
        policy={
            "policy_class":"finite_constrained_categorical_agent",
            "selection_mode":"sample",
            "answer_proposer":"v3",
            "categorical_distribution":"legal-action-only-softmax",
            "random_seed_keying":"state",
            "trained_on_eval_gold":False,
            "test_scope":"TRAIN-only teacher-heldout, NOT DEV/TEST/final-B160",
            "tool_budget_max":5,"step_budget_max":6,
            "source_adapter_sha256":adaptersha,
        }
        (arm/"FINITE_ACTION_POLICY.json").write_text(json.dumps(policy))
        save_jsonl(arm/"per_example.jsonl",[{
            "question_id":"train2","split":"train",
            "exact_match":scored,"verified_success":scored,
            "evidence_recall":float(scored),"token_f1":float(scored),
            "tool_calls":4,"search_calls":2,"read_calls":2,
        }])
        arms.append(arm)
    return arms[0],arms[1],target,trained


def test_train_holdout_auditor_requires_unchanged_decoder_and_cannot_promote(tmp_path):
    before,after,source,trained=_paired_fixture(tmp_path)
    r=audit_holdout(before,after,source,trained,tmp_path/"pair.json",draws=200)
    assert r["n"]==1
    assert r["metrics"]["verified_success"]["after_count"]==1
    assert r["eligible_to_promote_checkpoint"] is False
    assert r["no_external_generalization_claim"]


def test_train_holdout_auditor_rejects_training_on_holdout_ids(tmp_path):
    before,after,source,trained=_paired_fixture(tmp_path)
    path=trained/"TRAIN_SUMMARY.json"
    tr=json.loads(path.read_text())
    tr["selected_training_ids"]=["train2"]
    path.write_text(json.dumps(tr))
    with pytest.raises(ValueError,match="trained on heldout"):
        audit_holdout(before,after,source,trained,tmp_path/"bad.json",draws=200)


def test_train_holdout_audit_standalone_cli_works_with_only_src_pythonpath():
    """Previously CI imports passed but `python scripts/X.py` crashed in Tang."""
    import os
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(
        [sys.executable,str(root/"scripts"/"audit_train_teacher_holdout.py"),
         "--help"],
        cwd=root,env={**os.environ,"PYTHONPATH":str(root/"src")},
        capture_output=True,text=True,timeout=10,
    )
    assert result.returncode==0,result.stderr
    assert "--trained" in result.stdout


def test_train_holdout_audit_accepts_exact_matched_v4_policy(tmp_path):
    before,after,source,trained=_paired_fixture(tmp_path)
    train=trained/"TRAIN_SUMMARY.json"
    record=json.loads(train.read_text())
    record["answer_proposer"]="v4"
    train.write_text(json.dumps(record))
    for arm in (before,after):
        policy_path=arm/"FINITE_ACTION_POLICY.json"
        policy=json.loads(policy_path.read_text())
        policy["answer_proposer"]="v4"
        policy_path.write_text(json.dumps(policy))
    report=audit_holdout(before,after,source,trained,tmp_path/"v4pair.json",
                         draws=200)
    assert report["n"]==1
    assert report["same_masked_behavior_policy"]
    assert report["eligible_to_promote_checkpoint"] is False


def test_train_holdout_audit_refuses_v3_vs_v4_model_policy_mismatch(tmp_path):
    before,after,source,trained=_paired_fixture(tmp_path)
    train=trained/"TRAIN_SUMMARY.json"
    record=json.loads(train.read_text())
    record["answer_proposer"]="v4"
    train.write_text(json.dumps(record))
    # Evaluated V3 decoder cannot be reported as V4 training evidence.
    with pytest.raises(ValueError,match="checkpoint or actual"):
        audit_holdout(before,after,source,trained,tmp_path/"wrong.json",draws=200)
