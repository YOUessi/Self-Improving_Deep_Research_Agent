"""Matched-run reporting never silently compares different datasets."""
import json
import pytest

from scripts.paired_compare import compare


def write(path,ids,values):
    fields={
        "exact_match":0,"verified_success":0,
        "token_f1":0.0,"evidence_recall":0.0,"model_calls":1,
        "prompt_tokens":100,"completion_tokens":30,
        "tool_calls":1,"elapsed_ms":500,
    }
    with path.open("w") as fd:
        for qid,val in zip(ids,values):
            fd.write(json.dumps({"question_id":qid,**fields,
                                 "verified_success":val,
                                 "exact_match":val})+"\n")


def test_paired_ci_and_outcome_counts(tmp_path):
    b=tmp_path/"b.jsonl";t=tmp_path/"t.jsonl"
    write(b,["a","b","c","d"],[0,1,0,0])
    write(t,["a","b","c","d"],[1,1,0,1])
    x=compare(b,t,tmp_path/"out.json",seed=43,bootstrap=100)
    v=x["metrics"]["verified_success"]
    assert v["baseline_count"]==1
    assert v["treatment_count"]==3
    assert v["delta_mean"]==0.5
    assert v["improved_cases"]==2 and v["worsened_cases"]==0
    with pytest.raises(FileExistsError):
        compare(b,t,tmp_path/"out.json",seed=43,bootstrap=100)


def test_mismatched_question_sets_rejected(tmp_path):
    b=tmp_path/"b.jsonl";t=tmp_path/"t.jsonl"
    write(b,["a"],[1]);write(t,["b"],[1])
    with pytest.raises(ValueError,match="identical"):
        compare(b,t,tmp_path/"out.json",bootstrap=20)
