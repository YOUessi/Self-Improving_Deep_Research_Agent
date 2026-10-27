"""E027: paired TRAIN-only original sentence-ablation and false approval gates."""
import json
from copy import deepcopy

import pytest

from sidra.contrastive_reliability import compute_diagnostics,audit_file


def sample(qid, *, full=0.97,neg=0.01,neutral=0.12,category="missing_bridge_source"):
    facts=[["A",0],["B",0]]
    def probs(p):
        return {"entailment":p,"contradiction":round((1-p)*0.6,6),
                "neutral":round((1-p)*0.4,6)}
    # All hypothesis/provenance must be in actual already-scored research rows,
    # not generated from hidden gold inside a verifier.
    return [
        {"case_id":f"{qid}-e","question_id":qid,"expected_label":"entailment",
         "phenomenon":"bridge_entailment","model_posterior":probs(full),
         "model_predicted_label":"entailment",
         "hypothesis":"Bridge A connects to B.","support_fact_keys":facts},
        {"case_id":f"{qid}-c","question_id":qid,"expected_label":"contradiction",
         "phenomenon":"explicit_negation","model_posterior":probs(neg),
         "model_predicted_label":"contradiction",
         "hypothesis":"Bridge A does not connect to B.",
         "support_fact_keys":facts},
        {"case_id":f"{qid}-n","question_id":qid,"expected_label":"neutral",
         "phenomenon":category,"model_posterior":probs(neutral),
         "model_predicted_label":"neutral" if neutral<0.85 else "entailment",
         "hypothesis":("Bridge A connects to B." if category=="missing_bridge_source"
                       else "Bridge A connects to other C."),
         "support_fact_keys":([["A",0]] if category=="missing_bridge_source" else facts)},
    ]


def test_paired_support_removal_requires_actual_strict_subset_and_same_hypothesis():
    rows=sample("train_a")
    report=compute_diagnostics(rows,bootstrap=101)
    assert report["n_support_removal_pairs"]==1
    assert report["metrics"]["paired_support_removal_model_necessity_pass"]["value"]==1
    assert report["metrics"]["paired_support_removal_mean_entailment_delta"]["value"]==pytest.approx(.85)
    tampered=deepcopy(rows)
    tampered[2]["hypothesis"]="Another unrelated statement."
    with pytest.raises(ValueError,match="support pair"):
        compute_diagnostics(tampered,bootstrap=101)
    tampered=deepcopy(rows)
    tampered[2]["support_fact_keys"]=[["A",0],["B",0]]
    with pytest.raises(ValueError,match="support pair"):
        compute_diagnostics(tampered,bootstrap=101)


def test_confident_near_miss_is_separate_from_valid_support_removal():
    rows=sample("train_a",category="unsupported_bridge",neutral=0.95)
    report=compute_diagnostics(rows,bootstrap=101)
    assert report["n_support_removal_pairs"]==0
    assert report["neutral_case_false_accept_by_type"]["unsupported_bridge"]=={
        "confident_false_accepts":1,"count":1}
    assert report["metrics"]["neutral_confident_false_accept"]["value"]==1


def test_rejects_posterior_mislabel_or_duplicate_cases_before_metrics():
    rows=sample("a")
    bad=deepcopy(rows);bad[0]["model_posterior"]["neutral"]=float("nan")
    with pytest.raises(ValueError,match="posterior"):
        compute_diagnostics(bad,bootstrap=101)
    bad=deepcopy(rows);bad[1]["case_id"]=bad[0]["case_id"]
    with pytest.raises(ValueError,match="duplicate"):
        compute_diagnostics(bad,bootstrap=101)
    bad=deepcopy(rows);bad[2]["phenomenon"]="explicit_negation"
    with pytest.raises(ValueError,match="label"):
        compute_diagnostics(bad,bootstrap=101)


def test_question_cluster_resampling_is_deterministic_and_repeated_rows_not_independent():
    rows=sample("train_a")+sample("train_b",neutral=0.91,category="unsupported_bridge")
    a=compute_diagnostics(rows,bootstrap=301,seed=2027)
    b=compute_diagnostics(list(reversed(rows)),bootstrap=301,seed=2027)
    assert a==b
    assert a["metrics"]["neutral_confident_false_accept"]["value"]==.5
    assert a["metrics"]["neutral_confident_false_accept"]["question_clusters"]==2


def test_rejects_untrusted_dev_manifest_before_output(tmp_path):
    rows=sample("a")*12
    rowfile=tmp_path/"rows.jsonl"
    rowfile.write_text("".join(json.dumps(x)+"\n" for x in rows))
    summary=tmp_path/"summary.json"
    summary.write_text(json.dumps({"annotation_status":"wrong",
        "n_cases":36,"independent_question_clusters":12,
        "dev_gold_in_scorer_or_policy":False,
        "eval_A64_or_final_B160_accessed":False}))
    with pytest.raises(ValueError,match="untrusted"):
        audit_file(rowfile,summary,tmp_path/"out",bootstrap=101)
    assert not (tmp_path/"out").exists()
