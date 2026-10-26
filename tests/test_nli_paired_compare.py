"""No DEV, no mismatched model/data, no unpaired synthetic NLI ablation."""
import json
import pytest

from scripts.m7_nli_paired_compare import compare


def fixture_run(path,*,gate,accepted,gold_correct=0,sha="a",qid="train_q1"):
    path.mkdir()
    summary={
        "stage":"TRAIN-only offline CPU MNLI experimental screening",
        "no_dev_correct_answer_available_to_verifier":True,
        "no_eval_A64_or_final_B160_processed":True,
        "n_episodes":1,
        "cross_citation_title_link_required":gate,
        "source_sha256":{"train":"a"*64},
        "NLI_model":{"checkpoint_sha256":sha*64},
    }
    row={
        "question_id":qid,"source_experiment":"train01","rollout_index":0,
        "nli_scored":1,"model_predicted_entailment":accepted,
        "model_predicted_contradiction":0,
        "answer_em":gold_correct,"original_sentence_cited_complete":gold_correct,
        "tool_calls":3,"model_calls":4,
        "prompt_tokens":100,"completion_tokens":50,"elapsed_ms":700,
    }
    (path/"summary.json").write_text(json.dumps(summary))
    (path/"train_only_per_episode.jsonl").write_text(json.dumps(row)+"\n")


def test_question_cluster_paired_link_filter_counts_wrong_accepts(tmp_path):
    base=tmp_path/"base";after=tmp_path/"after"
    fixture_run(base,gate=False,accepted=1,gold_correct=0)
    fixture_run(after,gate=True,accepted=0,gold_correct=0)
    res=compare(base,after,tmp_path/"result",repeats=101,seed=2027)
    assert res["baseline_wrong_accepts"]==1
    assert res["candidate_wrong_accepts"]==0
    assert res["paired_metric_deltas"]["wrong_accept"]["mean"]==-1.0
    assert res["independent_question_clusters"]==1
    with pytest.raises(FileExistsError):
        compare(base,after,tmp_path/"result",repeats=101)


def test_refuse_unmatched_models_or_nontrain_or_unpaired(tmp_path):
    base=tmp_path/"base";after=tmp_path/"after"
    fixture_run(base,gate=False,accepted=1)
    fixture_run(after,gate=True,accepted=0,sha="b")
    with pytest.raises(ValueError,match="model mismatch"):
        compare(base,after,tmp_path/"bad1",repeats=101)
    fixture_run(tmp_path/"after2",gate=True,accepted=0,qid="train_other")
    with pytest.raises(ValueError,match="unpaired"):
        compare(base,tmp_path/"after2",tmp_path/"bad2",repeats=101)
    m=json.loads((base/"summary.json").read_text())
    m["no_dev_correct_answer_available_to_verifier"]=False
    (base/"summary.json").write_text(json.dumps(m))
    with pytest.raises(ValueError,match="TRAIN"):
        compare(base,tmp_path/"after2",tmp_path/"bad3",repeats=101)
