"""Paired DEV screening is aggregate-only, fail-closed and never auto-promotes."""
from pathlib import Path
import json

import pytest

from sidra.rsi_generation_protocol import (
    screen_paired_dev_pilot,make_candidate_record,multi_seed_gate,sha256_file,
)


def write_eval(tmp_path,name,bits,*,tools=2,config=None,dataset="pilot/converted"):
    path=tmp_path/name
    path.mkdir()
    manifest={
        "dataset_dir":dataset,
        "data_sha256":{"documents.jsonl":"x"*64,"questions.jsonl":"y"*64},
        "split":"dev","model_id":"Qwen-0.6B","agent_config":config or {
            "max_steps":6,"max_tool_calls":5,"search_top_k":3},
        "git_commit":"a"*40,"sample_count":len(bits),
        "evaluation_offset":0,"policy":"transformers_pretrained_agent",
        "model_or_policy":"transformers_pretrained_agent",
        "generation_max_new_tokens":128,
        "adapter_path":name,
    }
    (path/"manifest.json").write_text(json.dumps(manifest))
    with (path/"per_example.jsonl").open("w") as f:
        for i,v in enumerate(bits):
            f.write(json.dumps({
                "question_id":f"dev-{i:03d}","verified_success":v,
                "exact_match":v,"tool_calls":tools,
            })+"\n")
    return path


def test_negative_pilot_has_zero_improvement_and_rolls_back(tmp_path):
    old=write_eval(tmp_path,"start",[0]*32)
    new=write_eval(tmp_path,"post",[0]*32)
    result=screen_paired_dev_pilot(old,new,seed=10,draws=1000)
    assert result["n"]==32
    assert result["paired_verified_delta"]==0
    assert result["paired_verified_delta_ci95"]==[0.0,0.0]
    assert result["screen_reason"]=="no_strict_verified_gain"
    assert result["candidate_for_independent_eval"] is False
    assert result["promoted"] is False
    assert "question_id" not in result


def test_pilot_improvement_never_automatically_promotes(tmp_path):
    old=write_eval(tmp_path,"before",[0]*32)
    new=write_eval(tmp_path,"after",[1]*16+[0]*16)
    result=screen_paired_dev_pilot(old,new,seed=1,draws=2000)
    assert result["paired_verified_delta"]==0.5
    assert result["paired_verified_delta_ci95"][0]>0
    assert result["candidate_for_independent_eval"]
    assert result["promoted"] is False
    assert result["requires_independent_eval_A"]


def test_unmatched_decoder_tool_config_or_data_fails(tmp_path):
    old=write_eval(tmp_path,"old",[0]*32)
    new=write_eval(tmp_path,"new",[0]*32,
                   config={"max_steps":4,"max_tool_calls":5,"search_top_k":3})
    with pytest.raises(ValueError,match="agent_config"):
        screen_paired_dev_pilot(old,new)
    meta=json.loads((new/"manifest.json").read_text())
    meta["agent_config"]={"max_steps":6,"max_tool_calls":5,"search_top_k":3}
    meta["data_sha256"]["questions.jsonl"]="changed"
    (new/"manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError,match="data_sha256"):
        screen_paired_dev_pilot(old,new)


def test_refuses_final_b_even_if_report_claims_dev(tmp_path):
    old=write_eval(tmp_path,"old",[0]*32,dataset="sealed/final-B160/converted")
    new=write_eval(tmp_path,"new",[0]*32,dataset="sealed/final-B160/converted")
    with pytest.raises(ValueError,match="DEV PILOT"):
        screen_paired_dev_pilot(old,new)


def test_refuses_nonpaired_question_ids(tmp_path):
    old=write_eval(tmp_path,"old",[0]*32)
    new=write_eval(tmp_path,"new",[0]*32)
    fn=new/"per_example.jsonl"
    text=fn.read_text().replace("dev-000","different-000")
    fn.write_text(text)
    with pytest.raises(ValueError,match="IDs differ"):
        screen_paired_dev_pilot(old,new)


def test_checkpoint_file_hashes_and_rollback_are_real(tmp_path):
    old=write_eval(tmp_path,"old",[0]*32)
    new=write_eval(tmp_path,"new",[0]*32)
    report=screen_paired_dev_pilot(old,new,draws=1000)
    baseline=tmp_path/"old.safetensors"
    latest=tmp_path/"candidate.safetensors"
    train=tmp_path/"TRAIN-only.jsonl"
    baseline.write_bytes(b"baseline weights")
    latest.write_bytes(b"different model weights")
    train.write_bytes(b"sampled TRAIN episodes")
    record=make_candidate_record(
        generation=1,parent_adapter=baseline,candidate_adapter=latest,
        train_rollouts=train,pilot_report=report,
        source_git_sha="a"*40,out=tmp_path/"ledger/g1.json")
    assert record["state"]=="REJECTED_ROLLBACK_TO_PARENT"
    assert record["selected_for_next_generation_sha256"]==sha256_file(baseline)
    assert not record["promoted"] and not record["multi_generation_RSI_completed"]
    with pytest.raises(FileExistsError):
        make_candidate_record(
            generation=1,parent_adapter=baseline,candidate_adapter=latest,
            train_rollouts=train,pilot_report=report,
            source_git_sha="a"*40,out=tmp_path/"ledger/g1.json")
    with pytest.raises(ValueError,match="no checkpoint update"):
        make_candidate_record(
            generation=1,parent_adapter=baseline,candidate_adapter=baseline,
            train_rollouts=train,pilot_report=report,
            source_git_sha="a"*40,out=tmp_path/"ledger/bad.json")
    with pytest.raises(ValueError,match="parent checkpoint"):
        make_candidate_record(
            generation=2,parent_adapter=latest,candidate_adapter=baseline,
            train_rollouts=train,pilot_report=report,
            source_git_sha="a"*40,out=tmp_path/"ledger/g2_bad.json",
            previous_record=tmp_path/"ledger/g1.json")


def test_three_training_seeds_do_not_substitute_independent_eval(tmp_path):
    old=write_eval(tmp_path,"old",[0]*32)
    new=write_eval(tmp_path,"new",[1]*16+[0]*16)
    report=screen_paired_dev_pilot(old,new,draws=1000)
    assert not multi_seed_gate({101:report,102:report})["passed"]
    r=multi_seed_gate({101:report,102:report,103:report})
    assert not r["passed"]
    assert r["reason"]=="all_pilot_seeds_positive_but_independent_eval_A_still_required"
