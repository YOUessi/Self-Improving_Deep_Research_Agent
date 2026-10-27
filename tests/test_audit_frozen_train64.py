"""The internal TRAIN64 paired gate rejects label/decoder/weight-source changes."""
import hashlib
import json

import pytest

from scripts.audit_frozen_train64 import audit_pair_train64


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path,contents):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(contents,ensure_ascii=False)+"\n")


def _jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def _fixture(tmp_path):
    frozen=tmp_path/"frozen"
    frozen.mkdir(parents=True)
    qs=[{"question_id":f"train-{i:03d}","split":"train",
         "question":"Where?","gold_answer":"Y",
         "supporting_doc_ids":["a","b"]} for i in range(64)]
    _jsonl(frozen/"questions.jsonl",qs)
    _jsonl(frozen/"documents.jsonl",[
        {"doc_id":"a","scope":"global","title":"A","text":"Y"},
        {"doc_id":"b","scope":"global","title":"B","text":"Y"}])
    docs=_sha(frozen/"documents.jsonl")
    qhash=_sha(frozen/"questions.jsonl")
    _write(frozen/"FREEZE_MANIFEST.json",{
        "split":"train","count":64,"model_training_labels_allowed":False,
        "frozen_documents_sha256":docs,"frozen_questions_sha256":qhash,
        "ordered_qid_sha256":"c"*64})
    ts=tmp_path/"train.json"
    _write(ts,{
        "split":"train","dev_gold_used_for_reward":False,
        "sampling_and_loss_distribution_match":True,
        "multi_turn_Search_Read_Finish":True,"native_TRL_GRPOTrainer":False,
        "old_adapter_sha256":"a"*64,"new_adapter_sha256":"b"*64,
        "selected_training_ids":["fit-001","fit-002"],"any_aux_supervision_is_not_pure_GRPO":True,
        "updates":[{"updated":True,"nonconstant_groups":1,
                    "behavior_replay_max_logprob_error":0,
                    "tool_text_token_targets":0}]})
    for name,hash_,gain in (("old","a"*64,0),("new","b"*64,6)):
        path=tmp_path/name
        rows=[{"question_id":f"train-{i:03d}","split":"train",
               "exact_match":int(i<4+gain),
               "verified_success":int(i<3+gain),
               "evidence_recall":0.5+0.5*int(i<3+gain),
               "tool_calls":5,"read_calls":2,"model_calls":6,
               "abstained":int(i>=4+gain)} for i in range(64)]
        _jsonl(path/"per_example.jsonl",rows)
        _write(path/"manifest.json",{
            "split":"train","sample_count":64,"data_sha256":{
                "documents.jsonl":docs,"questions.jsonl":qhash},
            "git_commit":"z"*40,"model_id":"base",
            "agent_config":{"max_steps":6,"max_tool_calls":5},
            "seed":171,"evaluation_offset":0,
            "model_or_policy":"finite_constrained_categorical_agent"})
        _write(path/"summary.json",{"n":64,"split":"train"})
        _write(path/"FINITE_ACTION_POLICY.json",{
            "source_adapter_sha256":hash_,"trained_on_eval_gold":False,
            "policy_class":"finite_constrained_categorical_agent",
            "answer_proposer":"v4","selection_mode":"sample",
            "tool_budget_max":5,"step_budget_max":6,
            "categorical_distribution":"full current legal single-token action softmax",
            "random_seed_keying":"state keyed"})
    return frozen,ts,tmp_path/"old",tmp_path/"new",tmp_path/"result.json"


def test_audit_train64_expected_matched_pair(tmp_path):
    frozen,ts,baseline,candidate,out=_fixture(tmp_path)
    result=audit_pair_train64(
        frozen=frozen,train_summary=ts,baseline=baseline,
        candidate=candidate,out=out,bootstrap=300,seed=919)
    assert result["n_independent_original_train_questions"]==64
    assert result["metrics"]["exact_match"]["paired_delta"]==pytest.approx(6/64)
    assert result["metrics"]["verified_success"]["paired_delta"]==pytest.approx(6/64)
    assert result["hybrid_supervision_not_pure_GRPO"] is True
    assert result["promotion_allowed"] is False
    assert out.exists()
    with pytest.raises(FileExistsError):
        audit_pair_train64(frozen=frozen,train_summary=ts,baseline=baseline,
                           candidate=candidate,out=out,bootstrap=300)


def test_audit_train64_rejects_optimizer_overlap_and_changed_checkpoint(tmp_path):
    frozen,ts,baseline,candidate,out=_fixture(tmp_path)
    d=json.loads(ts.read_text())
    d["selected_training_ids"].append("train-001")
    _write(ts,d)
    with pytest.raises(ValueError,match="leaked"):
        audit_pair_train64(frozen=frozen,train_summary=ts,baseline=baseline,
                           candidate=candidate,out=out,bootstrap=300)
    d["selected_training_ids"]=["fit-001"]
    _write(ts,d)
    bad=json.loads((candidate/"FINITE_ACTION_POLICY.json").read_text())
    bad["source_adapter_sha256"]="a"*64
    _write(candidate/"FINITE_ACTION_POLICY.json",bad)
    with pytest.raises(ValueError,match="adapter/decoder"):
        audit_pair_train64(frozen=frozen,train_summary=ts,baseline=baseline,
                           candidate=candidate,out=out,bootstrap=300)


def test_audit_train64_rejects_changed_data_and_test_split(tmp_path):
    frozen,ts,baseline,candidate,out=_fixture(tmp_path)
    (frozen/"questions.jsonl").write_text((frozen/"questions.jsonl").read_text()+"\n")
    with pytest.raises(ValueError,match="dataset changed"):
        audit_pair_train64(frozen=frozen,train_summary=ts,baseline=baseline,
                           candidate=candidate,out=out,bootstrap=300)
    frozen,ts,baseline,candidate,out=_fixture(tmp_path/"fresh")
    meta=json.loads((candidate/"manifest.json").read_text())
    meta["split"]="test"
    _write(candidate/"manifest.json",meta)
    with pytest.raises(ValueError,match="64-example TRAIN"):
        audit_pair_train64(frozen=frozen,train_summary=ts,baseline=baseline,
                           candidate=candidate,out=out,bootstrap=300)
