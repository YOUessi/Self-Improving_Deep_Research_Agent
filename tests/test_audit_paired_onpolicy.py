"""Fail-closed paired DEV gate tests — synthetic metrics only, no model/GPU."""
from __future__ import annotations

import copy
import json

import pytest
from scripts.audit_paired_onpolicy import audit_pair,_sign_p_improvement


def _make_runs(tmp_path, *,finite=False,n=32):
    roots=[tmp_path/"baseline",tmp_path/"candidate"]
    rows=[]
    for k in range(n):
        rows.append({
            "question_id":f"dev-{k:03d}","split":"dev",
            "exact_match":0,"verified_success":0,
            "evidence_recall":0,"token_f1":0,
            "tool_calls":2,"model_calls":3,
            "completion_tokens":123,"elapsed_ms":100,
        })
    manifest={
        "split":"dev","dataset_dir":str(tmp_path/"frozen_dev_pilot"),
        "synthetic_fixture":False,
        "no_weight_updates_in_this_evaluation":True,
        "data_sha256":{"documents.jsonl":"a"*64,"questions.jsonl":"b"*64},
        "agent_config":{"max_steps":6,"max_tool_calls":5,
                        "search_top_k":3,"max_read_chars":3500,
                        "title_boost":0},
        "seed":42,"sample_count":n,"evaluation_offset":0,
        "generation_max_new_tokens":128,
        "model_id":"/home/you/models/Qwen3-0.6B-Base",
        "model_or_policy":(
            "finite_constrained_categorical_agent"
            if finite else "transformers_pretrained_agent"),
        "agent_roles":["single_policy"],"git_commit":"c"*40,
    }
    for i,r in enumerate(roots):
        r.mkdir(parents=True)
        m=copy.deepcopy(manifest)
        m["adapter_path"]=f"/adapter/{i}"
        (r/"manifest.json").write_text(json.dumps(m)+"\n")
        (r/"per_example.jsonl").write_text(
            "".join(json.dumps(row)+"\n" for row in rows))
        if finite:
            companion={
                "policy_class":"finite_constrained_categorical_agent",
                "selection_mode":"sample",
                "categorical_distribution":"masked single-digit exact",
                "random_seed_keying":"hash local state",
                "trained_on_eval_gold":False,
                "test_scope":"public DEV pilot, not sealed final-B160",
                "tool_budget_max":5,"step_budget_max":6,
                "source_adapter_sha256":str(i+1)*64,
            }
            (r/"FINITE_ACTION_POLICY.json").write_text(json.dumps(companion))
    return roots


def _set_row(directory, index, **fields):
    file=directory/"per_example.jsonl"
    data=[json.loads(x) for x in file.read_text().splitlines()]
    data[index].update(fields)
    file.write_text("".join(json.dumps(r)+"\n" for r in data))


def _set_manifest(directory, **fields):
    f=directory/"manifest.json"
    row=json.loads(f.read_text())
    row.update(fields)
    f.write_text(json.dumps(row)+"\n")


def _set_policy(directory, **fields):
    f=directory/"FINITE_ACTION_POLICY.json"
    row=json.loads(f.read_text())
    row.update(fields)
    f.write_text(json.dumps(row)+"\n")


@pytest.mark.parametrize("finite",[False,True])
def test_no_qa_gain_is_never_reported_as_improvement(tmp_path,finite):
    old,new=_make_runs(tmp_path,finite=finite)
    result=audit_pair(old,new,tmp_path/"audit.json",bootstrap=200)
    assert result["n"]==32
    assert result["metrics"]["exact_match"]["paired_delta_mean"]==0
    assert result["metrics"]["verified_success"]["after_count"]==0
    assert result["metrics"]["verified_success"]["paired_delta_ci95"]==[0,0]
    assert result["pilot_positive_gate"] is False
    assert result["promote_to_production"] is False
    assert result["independent_generalization_established"] is False
    assert result["metrics"]["exact_match"]["discordant_exact_sign_p_one_sided"]==1
    with pytest.raises(FileExistsError):
        audit_pair(old,new,tmp_path/"audit.json",bootstrap=200)


def test_positive_paired_improvement_requires_both_answers_and_evidence(tmp_path):
    old,new=_make_runs(tmp_path,n=64)
    for i in range(16):
        _set_row(new,i,exact_match=1,verified_success=1)
    report=audit_pair(old,new,tmp_path/"audit.json",bootstrap=2000)
    assert report["metrics"]["exact_match"]["after_count"]==16
    assert report["metrics"]["exact_match"]["discordant_exact_sign_p_one_sided"]<1e-4
    assert report["metrics"]["exact_match"]["paired_delta_ci95"][0]>0
    assert report["pilot_positive_gate"] is True
    assert report["promote_to_production"] is False


def test_strict_evidence_zero_blocks_promoting_answer_only(tmp_path):
    old,new=_make_runs(tmp_path)
    for i in range(16):
        _set_row(new,i,exact_match=1)
    report=audit_pair(old,new,tmp_path/"audit.json",bootstrap=200)
    assert report["metrics"]["exact_match"]["paired_delta_mean"]>0
    assert report["metrics"]["verified_success"]["paired_delta_mean"]==0
    assert not report["pilot_positive_gate"]


@pytest.mark.parametrize("key,value",[
    ("seed",43),
    ("data_sha256",{"documents.jsonl":"a"*64,"questions.jsonl":"f"*64}),
    ("model_id","Qwen3-8B"),
    ("agent_config",{"max_steps":8,"max_tool_calls":99}),
    ("generation_max_new_tokens",512),
    ("sample_count",31),
    ("git_commit","different"),
    ("split","test"),
])
def test_mismatched_model_dataset_seed_decoder_or_budget_block(tmp_path,key,value):
    old,new=_make_runs(tmp_path)
    _set_manifest(new,**{key:value})
    with pytest.raises(ValueError):
        audit_pair(old,new,tmp_path/"no.json",bootstrap=200)


def test_question_id_mismatch_and_duplicate_block(tmp_path):
    old,new=_make_runs(tmp_path)
    _set_row(new,0,question_id="not-the-frozen-question")
    with pytest.raises(ValueError,match="nonidentical"):
        audit_pair(old,new,tmp_path/"audit.json",bootstrap=200)
    _set_row(new,0,question_id="dev-001")
    with pytest.raises(ValueError,match="duplicate"):
        audit_pair(old,new,tmp_path/"audit.json",bootstrap=200)


def test_sealed_final_is_never_a_dev_pilot(tmp_path):
    old,new=_make_runs(tmp_path)
    _set_manifest(new,dataset_dir="/home/you/experiments/sidra_m4/data/final-B160")
    with pytest.raises(ValueError,match="sealed final"):
        audit_pair(old,new,tmp_path/"audit.json",bootstrap=200)


def test_finite_policy_mismatch_blocks(tmp_path):
    old,new=_make_runs(tmp_path,finite=True)
    _set_policy(new,selection_mode="greedy")
    with pytest.raises(ValueError,match="decoder/selection"):
        audit_pair(old,new,tmp_path/"audit.json",bootstrap=200)


def test_finite_companion_absent_in_one_arm_blocks(tmp_path):
    old,new=_make_runs(tmp_path,finite=True)
    (new/"FINITE_ACTION_POLICY.json").unlink()
    with pytest.raises(ValueError,match="only one"):
        audit_pair(old,new,tmp_path/"audit.json",bootstrap=200)


def test_signed_discordant_pair_exact_p():
    assert _sign_p_improvement(0,0)==1
    assert _sign_p_improvement(5,0)==pytest.approx(1/32)
    assert _sign_p_improvement(0,5)==1
    assert _sign_p_improvement(3,3)==pytest.approx(0.65625)


def test_reject_unbounded_bootstrap_and_nonfinite_metrics(tmp_path):
    old,new=_make_runs(tmp_path)
    with pytest.raises(ValueError,match="bootstrap"):
        audit_pair(old,new,tmp_path/"report.json",bootstrap=10)
    _set_row(new,0,token_f1=float("nan"))
    with pytest.raises(ValueError,match="nonfinite"):
        audit_pair(old,new,tmp_path/"report.json",bootstrap=200)
