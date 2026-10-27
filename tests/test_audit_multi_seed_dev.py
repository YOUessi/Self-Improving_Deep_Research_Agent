"""Multi-seed paired DEV pilot cannot silently count repeated RNG as new QA."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.audit_multi_seed_dev import audit_three_arm_dev


BASE_SHA="1"*64
CONTROL_SHA="2"*64
HYBRID_SHA="3"*64
QUESTION_IDS=("q1","q2")


def _write_json(p,data):
    p.write_text(json.dumps(data),encoding="utf-8")


def _make_seed(root:Path,seed:int,*,mode:str="good"):
    dirs={}
    for arm,sha in (
        ("baseline",BASE_SHA),("control",CONTROL_SHA),("hybrid",HYBRID_SHA)
    ):
        path=root/f"seed-{seed}-{arm}"
        path.mkdir(parents=True)
        manifest={
            "split":"dev","dataset_dir":"/safe/frozen/public_dev",
            "data_sha256":{"documents.jsonl":"a"*64,"questions.jsonl":"b"*64},
            "seed":seed,"sample_count":2,"evaluation_offset":0,
            "generation_max_new_tokens":None,
            "model_id":"local-qwen3","model_or_policy":"finite_constrained_categorical_agent",
            "agent_config":{"max_steps":6,"max_tool_calls":5},
            "git_commit":"e"*40,"agent_roles":["single_policy"],
            "synthetic_fixture":False,"no_weight_updates_in_this_evaluation":True,
        }
        if mode=="drift" and seed==157:
            manifest["git_commit"]="f"*40
        if mode=="sealed" and arm=="hybrid" and seed==151:
            manifest["dataset_dir"]="/safe/final-B160"
        if mode=="train" and arm=="hybrid":
            manifest["split"]="train"
        _write_json(path/"manifest.json",manifest)
        policy={
            "policy_class":"finite_constrained_categorical_agent",
            "selection_mode":"sample","answer_proposer":"v4",
            "categorical_distribution":"exact legal masked softmax",
            "random_seed_keying":"sha256",
            "trained_on_eval_gold":False,
            "test_scope":"public DEV pilot, not sealed final-B160",
            "tool_budget_max":5,"step_budget_max":6,
            "source_adapter_sha256":sha,
        }
        if mode=="leak" and arm=="hybrid":
            policy["trained_on_eval_gold"]=True
        _write_json(path/"FINITE_ACTION_POLICY.json",policy)
        records=[]
        for qid in QUESTION_IDS:
            win=int(
                arm=="hybrid" and (
                    seed==151 and qid=="q1" or seed==157 and qid=="q2"
                )
            )
            records.append({
                "split":"dev","question_id":qid,
                "exact_match":win,"verified_success":win,
                "evidence_recall":0.5+win*0.5,
                "token_f1":float(win),
                "tool_calls":4,"model_calls":5,
                "completion_tokens":1,"elapsed_ms":21.5,
            })
        (path/"per_example.jsonl").write_text(
            "\n".join(json.dumps(x) for x in records)+"\n",encoding="utf-8")
        dirs[arm]=path
    return {"seed":seed,**dirs}


def test_clustered_bootstrap_two_seeds_are_two_not_four_questions(tmp_path):
    runs=[_make_seed(tmp_path,151),_make_seed(tmp_path,157)]
    output=tmp_path/"result.json"
    result=audit_three_arm_dev(runs,output,bootstrap=300,rng_seed=9)
    assert result["stochastic_eval_seeds"]==[151,157]
    assert result["total_episodes_per_arm"]==4
    assert result["independent_question_count"]==2
    assert result["any_adapter_promoted"] is False
    assert result["independent_generalization_established"] is False
    hybrid=result["metrics"]["hybrid"]["exact_match"]
    assert hybrid["candidate_mean"]==0.5
    assert hybrid["delta_mean"]==0.5
    assert hybrid["question_cluster_bootstrap_ci95"]==[0.5,0.5]
    assert hybrid["counts_per_seed"]=={
        "151":{"baseline":0,"candidate":1},
        "157":{"baseline":0,"candidate":1},
    }
    assert result["metrics"]["control"]["verified_success"]["delta_mean"]==0
    assert json.loads(output.read_text())["pseudo_replication_prevented"]
    with pytest.raises(FileExistsError):
        audit_three_arm_dev(runs,output,bootstrap=300)


@pytest.mark.parametrize("case,match",[
    ("drift","drift"),
    ("leak","trained_on_eval_gold"),
    ("sealed","sealed"),
    ("train","DEV-only"),
])
def test_fail_closed_for_protocol_source_or_gold_contamination(tmp_path,case,match):
    runs=[_make_seed(tmp_path,151,mode=case),
          _make_seed(tmp_path,157,mode=case)]
    with pytest.raises(ValueError,match=match):
        audit_three_arm_dev(runs,tmp_path/"result.json",bootstrap=300)


def test_duplicate_seed_or_missing_replication_is_rejected(tmp_path):
    a=_make_seed(tmp_path,151)
    with pytest.raises(ValueError,match="two"):
        audit_three_arm_dev([a],tmp_path/"result.json",bootstrap=300)
    with pytest.raises(ValueError,match="duplicate"):
        audit_three_arm_dev([a,a],tmp_path/"result.json",bootstrap=300)


def test_same_adapter_in_control_and_hybrid_is_rejected(tmp_path):
    a=_make_seed(tmp_path,151)
    b=_make_seed(tmp_path,157)
    path=a["hybrid"]/"FINITE_ACTION_POLICY.json"
    x=json.loads(path.read_text())
    x["source_adapter_sha256"]=CONTROL_SHA
    _write_json(path,x)
    with pytest.raises(ValueError,match="3 distinct"):
        audit_three_arm_dev([a,b],tmp_path/"result.json",bootstrap=300)
