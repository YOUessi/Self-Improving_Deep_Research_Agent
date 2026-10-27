"""M7 E-028 tests: immutable TRAIN provenance, truly invariant evidence set,
question-level uncertainty, and detection of model-only ordering sensitivity."""
from __future__ import annotations

import copy

import pytest

from sidra.premise_order_audit import cluster_bootstrap,run_order_audit,validate_inputs


def fixture():
    rows=[]
    for i in range(12):
        q=f"train-{i:02d}"
        for name,label in (
            ("positive","entailment"),("negative","contradiction"),
            ("omission","neutral")):
            keys=[["Wiki Alpha",0],["Wiki Beta",1]]
            source="Wiki Alpha: Country Alpha was established in year 1963.\nWiki Beta: The country is Alpha."
            if name=="omission":
                keys=[keys[0]]
                source=source.split("\n")[0]
            probs={
                "positive":{"entailment":.94,"neutral":.04,"contradiction":.02},
                "negative":{"entailment":.02,"neutral":.03,"contradiction":.95},
                "omission":{"entailment":.05,"neutral":.90,"contradiction":.05},
            }[name]
            rows.append({
                "case_id":f"{q}-{name}","question_id":q,
                "expected_label":label,
                "phenomenon":name,
                "support_fact_keys":keys,"source_premise":source,
                "hypothesis":"Country Alpha was established in 1963.",
                "model_posterior":probs,
                "model_predicted_label":max(probs,key=probs.get),
            })
    summary={
        "annotation_status":"researcher_curated_not_independent_human_adjudication",
        "n_cases":36,"independent_question_clusters":12,
        "dev_gold_in_scorer_or_policy":False,
        "eval_A64_or_final_B160_accessed":False,
        "input_sha256":{k:"a"*64 for k in (
            "train_questions","train_documents","original_train","controlled_cases")},
        "model_manifest":{"weights_sha256":"b"*64},
    }
    return rows,summary


class OrderSensitiveFakeNLI:
    max_length=384
    def __init__(self):self.calls=[]
    def score(self,premises,hypotheses):
        self.calls.extend(zip(premises,hypotheses))
        assert len(premises)==1
        # Intentionally ALWAYS predicts neutral after reversed evidence order.
        return [{"entailment":.05,"neutral":.93,"contradiction":.02}]


def test_clean_fixture_filters_one_line_and_reverses_only_source_lines(tmp_path):
    rows,summary=fixture()
    eligible=validate_inputs(rows,summary)
    assert len(eligible)==24
    assert len({x["question_id"] for x in eligible})==12
    nli=OrderSensitiveFakeNLI()
    result=run_order_audit(rows,summary,nli,out=tmp_path/"out",bootstrap=110)
    assert result["real_new_CPU_NLI_calls"]==24
    assert result["one_source_rows_excluded"]==12
    assert result["label_flip_rate"]["count"]==24
    assert result["acceptance_flip_rate"]["count"]==12
    assert result["acceptance_flip_rate"]["ci95"]==[.5,.5]
    assert result["dev_gold_used"] is False
    assert result["agent_tool_budget_changed"] is False
    assert len(nli.calls)==24
    for premise,hypothesis in nli.calls:
        assert premise.startswith("Wiki Beta: ")
        assert premise.endswith("Wiki Alpha: Country Alpha was established in year 1963.")
        assert hypothesis=="Country Alpha was established in 1963."
    with pytest.raises(FileExistsError):
        run_order_audit(rows,summary,nli,out=tmp_path/"out",bootstrap=110)


def test_provenance_rejects_dev_metadata_sentence_forgery_and_duplication():
    rows,summary=fixture()
    bad=copy.deepcopy(summary)
    bad["eval_A64_or_final_B160_accessed"]=True
    with pytest.raises(ValueError,match="TRAIN-only"):
        validate_inputs(rows,bad)
    bad=copy.deepcopy(rows)
    bad[0]["source_premise"]=bad[0]["source_premise"].replace(
        "Wiki Beta: ","Forged title: ",1)
    with pytest.raises(ValueError,match="exact labelled"):
        validate_inputs(bad,summary)
    bad=copy.deepcopy(rows)
    bad[1]["case_id"]=bad[0]["case_id"]
    with pytest.raises(ValueError,match="duplicate"):
        validate_inputs(bad,summary)
    bad=copy.deepcopy(rows)
    bad[0]["support_fact_keys"][1]=["Wiki Alpha",0]
    with pytest.raises(ValueError,match="exact labelled|duplicate"):
        validate_inputs(bad,summary)
    bad=copy.deepcopy(rows)
    bad[0]["model_posterior"]={"entailment":.9,"neutral":.2,"contradiction":-.1}
    with pytest.raises(ValueError,match="bad original"):
        validate_inputs(bad,summary)


def test_question_cluster_ci_not_same_as_row_independence_and_no_denominator():
    rows=[
        {"question_id":"one","flips":1,"one":1},
        {"question_id":"one","flips":1,"one":1},
        {"question_id":"two","flips":0,"one":1},
    ]
    x=cluster_bootstrap(rows,"flips","one",n=130,seed=8)
    assert x["value"]==pytest.approx(2/3,abs=1e-6)
    assert x["question_clusters"]==2
    assert x["denominator"]==3
    assert x["ci95"][0]<=x["value"]<=x["ci95"][1]
    for r in rows:r["none"]=0
    result=cluster_bootstrap(rows,"flips","none",n=130)
    assert result["value"] is None
    assert result["ci95"]==[None,None]


def test_refuses_model_truncation_before_inference(tmp_path):
    rows,summary=fixture()
    class TinyBudget(OrderSensitiveFakeNLI):
        max_length=64
        def tokenizer(self,premise,hypothesis,truncation):
            assert truncation is False
            return {"input_ids":list(range(65))}
    scorer=TinyBudget()
    with pytest.raises(ValueError,match="truncate"):
        run_order_audit(rows,summary,scorer,out=tmp_path/"x",bootstrap=100)
    assert scorer.calls==[]
    assert not (tmp_path/"x").exists()
