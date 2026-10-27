"""Two-seed pilot comparison clusters by question, not by run."""
import json
from pathlib import Path

import pytest

from scripts.compare_v4_v5 import report_pair,MEASURES


def make_run(root:Path,arm:str,seed:int,*,change=0,drift=False)->Path:
    run=root/(arm+"_"+str(seed));run.mkdir()
    manifest={
        "seed":seed,"split":"dev","sample_count":32,
        "data_sha256":{"questions.jsonl":"q"*64,"documents.jsonl":"d"*64},
        "git_commit":"a"*40,
        "model_id":"Qwen3-0.6B","agent_config":{"max_steps":6,"max_tool_calls":5},
        "adapter_path":"/base/adapter",
    }
    if drift:manifest["agent_config"]["max_steps"]=8
    policy={
        "answer_proposer":arm,"policy_class":"finite_constrained_categorical_agent",
        "selection_mode":"sample","source_adapter_sha256":"b"*64,
        "tool_budget_max":5,"step_budget_max":6,
        "categorical_distribution":"full categorical masked softmax",
    }
    (run/"manifest.json").write_text(json.dumps(manifest))
    (run/"FINITE_ACTION_POLICY.json").write_text(json.dumps(policy))
    rows=[]
    for i in range(32):
        row={"question_id":f"q{i:03d}","split":"dev"}
        row.update({key:0.0 for key in MEASURES})
        row.update({"exact_match":int(i<3)+change if i==2 else int(i<3),
                    "verified_success":int(i<2)+change if i==1 else int(i<2),
                    "evidence_recall":0.3,"tool_calls":3.0,
                    "model_calls":4.0,"elapsed_ms":200.0})
        rows.append(json.dumps(row))
    (run/"per_example.jsonl").write_text("\n".join(rows)+"\n")
    return run


def test_report_aggregates_by_32_questions_and_provenance(tmp_path):
    runs={}
    for seed in (151,157):
        runs[(seed,"v4")]=make_run(tmp_path,"v4",seed)
        runs[(seed,"v5")]=make_run(tmp_path,"v5",seed,change=1)
    result=report_pair(runs,tmp_path/"report.json",bootstrap=250)
    assert result["unique_question_id_clusters"]==32
    assert result["episodes_per_arm"]==64
    assert result["metrics"]["exact_match"]["v5_minus_v4"]==pytest.approx(1/32)
    assert result["metrics"]["verified_success"]["v5_minus_v4"]==pytest.approx(1/32)
    assert result["promote_v5_for_training"]
    with pytest.raises(FileExistsError):
        report_pair(runs,tmp_path/"report.json",bootstrap=250)


def test_report_rejects_mismatched_dataset_and_tool_budget(tmp_path):
    runs={(s,a):make_run(tmp_path,a,s,drift=(s==157 and a=="v5"))
          for s in (151,157) for a in ("v4","v5")}
    with pytest.raises(ValueError,match="agent_config"):
        report_pair(runs,tmp_path/"should_not_write.json",bootstrap=250)
    assert not (tmp_path/"should_not_write.json").exists()
