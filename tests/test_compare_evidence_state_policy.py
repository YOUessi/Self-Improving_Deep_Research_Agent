"""Question-clustered same-model evidence state evaluation and integrity tests."""
import json

import pytest

from scripts.compare_evidence_state_policy import compare,METRICS
from sidra.io import save_jsonl
from sidra.schema import Document,Question


def fixture(tmp_path):
    data=tmp_path/"pilot"/"converted";data.mkdir(parents=True)
    docs=[]
    qs=[]
    for i in range(32):
        qid=f"dev{i:03d}"
        docs.extend([
            Document(f"{qid}::a","River City",
                     "River City is part of Coastal Province.",scope=qid),
            Document(f"{qid}::b","Coastal Province",
                     "Coastal Province includes River City.",scope=qid),
        ])
        qs.append(Question(qid,"Which province includes River City?",
                    "Coastal Province",supporting_doc_ids=(
                        f"{qid}::a",f"{qid}::b"),scope=qid,split="dev"))
    save_jsonl(data/"documents.jsonl",[d.__dict__ for d in docs])
    save_jsonl(data/"questions.jsonl",[q.__dict__ for q in qs])
    runs={}
    for seed in (151,157):
        for mode in ("off","v1"):
            out=tmp_path/f"{mode}_{seed}";out.mkdir()
            common={
                "split":"dev","sample_count":32,"seed":seed,
                "git_commit":"f"*40,"model_id":"Qwen3-0.6B",
                "agent_config":{"max_steps":6,"max_tool_calls":5},
                "data_sha256":{"questions.jsonl":"a"*64,
                               "documents.jsonl":"b"*64},
                "adapter_path":"/models/sft_adapter",
            }
            policy={
                "answer_proposer":"v4","evidence_state_mode":mode,
                "selection_mode":"sample","source_adapter_sha256":"c"*64,
                "tool_budget_max":5,"step_budget_max":6,
                "categorical_distribution":"masked softmax",
            }
            rows=[]
            traces=[]
            for q in qs:
                n=int(q.question_id[-3:])
                score=int(n<4 if mode=="off" else n<7)
                metric={
                    "question_id":q.question_id,
                    "exact_match":score,"verified_success":score,
                    "evidence_recall":1.0 if score else 0.5,
                    "search_calls":1,"read_calls":2,"tool_calls":3,
                    "model_calls":4,"prompt_tokens":400,
                    "completion_tokens":4,"elapsed_ms":300,
                }
                rows.append(metric)
                traces.append({
                    "question_id":q.question_id,"question":q.question,
                    "steps":[
                        {"observation":{"kind":"search","query":q.question,
                          "hits":[{"doc_id":q.supporting_doc_ids[0],
                                   "title":"River City","snippet":"River City"},
                                  {"doc_id":q.supporting_doc_ids[1],
                                   "title":"Coastal Province","snippet":"Coastal Province"}]}},
                        {"observation":{"kind":"read",
                         "doc_id":q.supporting_doc_ids[0],
                         "title":"River City",
                         "text":"River City is part of Coastal Province."}},
                        {"observation":{"kind":"read",
                         "doc_id":q.supporting_doc_ids[1],
                         "title":"Coastal Province",
                         "text":"Coastal Province includes River City."}},
                    ],
                })
            (out/"manifest.json").write_text(json.dumps(common))
            (out/"FINITE_ACTION_POLICY.json").write_text(json.dumps(policy))
            save_jsonl(out/"per_example.jsonl",rows)
            save_jsonl(out/"traces.jsonl",traces)
            runs[(seed,mode)]=out
    return data,runs


def test_question_clustered_report_and_no_hidden_data(tmp_path):
    pilot,runs=fixture(tmp_path)
    result=compare(pilot,runs,tmp_path/"paired.json",bootstrap=250)
    assert result["n_original_question_clusters"]==32
    assert result["episodes_per_arm"]==64
    assert result["metrics"]["verified_success"]["delta"]==pytest.approx(3/32)
    assert result["metrics"]["gold_read_pair"]["off_mean"]==1
    assert result["metrics"]["gold_read_pair"]["state_v1_mean"]==1
    assert result["state_model_gold_inputs"] is False
    assert result["grant_train_only_online_grpo_trial"] is True
    with pytest.raises(FileExistsError):
        compare(pilot,runs,tmp_path/"paired.json",bootstrap=250)


def test_same_model_tool_budget_required(tmp_path):
    pilot,runs=fixture(tmp_path)
    corrupted=runs[(157,"v1")]/"manifest.json"
    j=json.loads(corrupted.read_text())
    j["agent_config"]["max_steps"]=8
    corrupted.write_text(json.dumps(j))
    with pytest.raises(ValueError,match="agent_config"):
        compare(pilot,runs,tmp_path/"blocked.json",bootstrap=250)
    assert not (tmp_path/"blocked.json").exists()


def test_no_heldout_or_relabelled_dev_scoring(tmp_path):
    pilot,runs=fixture(tmp_path)
    heldout=tmp_path/"final_b160"/"converted"
    heldout.mkdir(parents=True)
    with pytest.raises(ValueError,match="DEV pilot"):
        compare(heldout,runs,tmp_path/"not_run.json",bootstrap=250)


def test_v2_paired_comparator_uses_same_question_cluster_bootstrap(tmp_path):
    pilot,runs=fixture(tmp_path)
    # Existing synthetic helper's treatment data have identical schema; only
    # rename the metadata to the v2 policy mode.
    for seed in (151,157):
        source=runs[(seed,"v1")]
        p=source/"FINITE_ACTION_POLICY.json"
        payload=json.loads(p.read_text())
        payload["evidence_state_mode"]="v2"
        p.write_text(json.dumps(payload))
        runs[(seed,"v2")]=source
    out=tmp_path/"v2_paired.json"
    result=compare(pilot,runs,out,treatment_mode="v2",bootstrap=250)
    assert result["policy"]["treatment"]=="v2"
    assert result["n_original_question_clusters"]==32
    assert result["metrics"]["exact_match"]["state_v2_mean"]>result[
        "metrics"]["exact_match"]["off_mean"]
    with pytest.raises(ValueError,match="treatment"):
        compare(pilot,runs,tmp_path/"invalid.json",treatment_mode="unknown")


def test_v3_paired_comparison_reconstructs_the_same_v3_state_in_both_arms(tmp_path):
    pilot,runs=fixture(tmp_path)
    for seed in (151,157):
        source=runs[(seed,"v1")]
        p=source/"FINITE_ACTION_POLICY.json"
        info=json.loads(p.read_text())
        info["evidence_state_mode"]="v3"
        p.write_text(json.dumps(info))
        runs[(seed,"v3")]=source
    report=compare(pilot,runs,tmp_path/"v3_paired.json",
                   treatment_mode="v3",bootstrap=250)
    assert report["policy"]["treatment"]=="v3"
    assert report["state_reconstruction_version_for_both_arms"]=="v3"
    assert report["n_original_question_clusters"]==32
    assert report["metrics"]["observed_cross_links"]["off_mean"]==1
    assert report["metrics"]["observed_cross_links"]["state_v3_mean"]==1
    assert report["metrics"]["verified_success"]["delta"]==pytest.approx(3/32)
