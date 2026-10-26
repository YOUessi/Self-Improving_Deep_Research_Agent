"""M7 evidence diagnostics: source-only privacy and real TRAIN-only split gates."""
import json

import pytest

from sidra.evidence_diagnostics import (
    _present_in_one_document,source_only_verdict,offline_train_diagnosis,
)
from sidra.schema import Question
from scripts.m7_evidence_diagnostics import (
    run_train,run_pilot,clustered_bootstrap,
)


def episode(*,qid="q1",answer="Paris",citations=("d1","d2"),
            end="finish",first="Paris is the capital of France",
            second="France is a European country",read2=True,
            search2=True):
    steps=[
        {"action":{"kind":"search","query":"France"},
         "observation":{"kind":"search","hits":[
             {"doc_id":"d1","title":"France"},
             *([{"doc_id":"d2","title":"Paris"}] if search2 else []),
         ]}},
        {"action":{"kind":"read","doc_id":"d1"},
         "observation":{"kind":"read","doc_id":"d1","title":"France",
                        "text":first}},
    ]
    if read2:
        steps.append({"action":{"kind":"read","doc_id":"d2"},
                      "observation":{"kind":"read","doc_id":"d2",
                                     "title":"Paris","text":second}})
    steps.append({"action":{"kind":"finish","answer":answer,
                            "citations":list(citations)}})
    return {
        "question_id":qid,"question":"Where is Paris?","answer":answer,
        "citations":list(citations),"termination":end,"steps":steps,
        "tool_calls":1+1+int(read2),"search_calls":1,
        "read_calls":1+int(read2),"elapsed_ms":200,
        "model_calls":4,"prompt_tokens":150,"completion_tokens":30,
    }


def official():
    return {"_id":"q1","question":"Where is Paris?","answer":"Paris",
            "context":[
                ["France",["Paris is the capital of France."]],
                ["Paris",["France is a European country."]],
            ],
            "supporting_facts":[["France",0],["Paris",0]]}


def question(split="train"):
    return Question("q1","Where is Paris?","Paris",
                    supporting_doc_ids=("d1","d2"),split=split)


def test_source_observed_only_never_entailment():
    tr=episode()
    v=source_only_verdict(tr)
    assert v.reason=="lexical_only_entailment_unverified"
    assert v.lexical_candidate
    assert v.semantic_entailment=="not_evaluated"
    # Cited negative evidence must NOT be interpreted as semantic proof.
    tr["steps"][1]["observation"]["text"]="Paris did not win."
    assert source_only_verdict(tr).reason=="lexical_only_entailment_unverified"


def test_cross_document_phantom_is_not_a_lexical_source():
    tr=episode(answer="new york",first="new",
               second="york",citations=("d1","d2"))
    v=source_only_verdict(tr)
    assert not v.lexical_candidate
    assert v.concat_only_lexical_candidate
    assert v.reason=="lexical_answer_not_in_any_cited_read"
    assert not _present_in_one_document("York", "New Yorkie is here")
    assert _present_in_one_document("New York", "the New York city")


def test_malformed_unread_duplicate_and_abstention_buckets():
    tr=episode(end="error")
    tr["error"]="ValueError: Unverified citations: ['d2']; must read before cite"
    assert source_only_verdict(tr).reason=="unread_citation_action_error"
    tr=episode(read2=False)
    assert source_only_verdict(tr).reason=="citation_not_read"
    tr=episode(citations=("d1","d1"))
    assert source_only_verdict(tr).reason=="duplicate_citation"
    tr=episode(citations=("d1",))
    assert source_only_verdict(tr).reason=="too_few_citations"
    tr=episode(end="abstained",answer="",citations=())
    assert source_only_verdict(tr).reason=="abstained"


def test_offline_original_sentence_stage_is_strictly_train():
    case=official()
    row=offline_train_diagnosis(question(),episode(),case)
    assert row["support_retrieval_complete"]==1
    assert row["support_read_complete"]==1
    assert row["original_sentence_read_complete"]==1
    assert row["original_sentence_cited_complete"]==1
    assert row["answer_em"]==1
    assert row["semantic_entailment_evaluated"]==0
    assert row["primary_evidence_bottleneck"]=="answer_and_sentence_exposed_semantics_unverified"
    with pytest.raises(ValueError,match="TRAIN-only"):
        offline_train_diagnosis(question("dev"),episode(),case)
    with pytest.raises(ValueError,match="IDs differ"):
        offline_train_diagnosis(question(),episode(qid="wrong"),case)
    bad=episode(search2=False,read2=False,citations=("d1",))
    missing=offline_train_diagnosis(question(),bad,case)
    assert missing["primary_evidence_bottleneck"]=="support_not_retrieved"
    assert missing["original_sentence_read_complete"]==0


def test_read_document_id_not_sufficient_if_sentence_truncated():
    tr=episode(second="Only short unrelated text")
    row=offline_train_diagnosis(question(),tr,official())
    assert row["support_read_complete"]==1
    assert row["original_sentence_read_complete"]==0
    assert row["primary_evidence_bottleneck"]=="support_read_but_gold_sentence_unseen"


def test_original_validation_blocks_inconsistent_training_labels():
    wrong=official()
    wrong["answer"]="Berlin"
    with pytest.raises(ValueError,match="gold answer mismatch"):
        offline_train_diagnosis(question(),episode(),wrong)
    wrong=official()
    wrong["supporting_facts"]=[["France",100],["Paris",0]]
    with pytest.raises(ValueError,match="sentence indices"):
        offline_train_diagnosis(question(),episode(),wrong)


def write_jsonl(path,items):
    path.write_text("".join(json.dumps(x)+"\n" for x in items))


def train_fixture(tmp_path):
    data=tmp_path/"converted"
    data.mkdir()
    write_jsonl(data/"documents.jsonl",[
        {"doc_id":"d1","scope":"q1","title":"France",
         "text":"Paris is the capital of France."},
        {"doc_id":"d2","scope":"q1","title":"Paris",
         "text":"France is a European country."},
    ])
    write_jsonl(data/"questions.jsonl",[
        {"question_id":"q1","scope":"q1","question":"Where is Paris?",
         "gold_answer":"Paris","split":"train",
         "supporting_doc_ids":["d1","d2"]},
    ])
    orig=tmp_path/"original_train_subset.json"
    orig.write_text(json.dumps([official()]))
    rollout=tmp_path/"rollouts_train.jsonl"
    write_jsonl(rollout,[
        {"question_id":"q1","rollout_index":0,
         "trace":episode(),"reward":{"total":1.0}},
        {"question_id":"q1","rollout_index":1,
         "trace":episode(read2=False),"reward":{"total":0.1}},
    ])
    return data,orig,rollout


def test_train_script_analyzes_true_original_train_sentences(tmp_path):
    data,orig,rollouts=train_fixture(tmp_path)
    report=run_train(data,orig,rollouts,tmp_path/"out",bootstrap=101)
    assert report["n_train_questions"]==1
    assert report["n_episodes"]==2
    assert report["semantic_entailment_evaluated"] is False
    assert report["question_cluster_bootstrap"]["answer_em"]["question_clusters"]==1
    assert report["primary_bottleneck_counts"]["support_retrieved_not_read"]==1
    assert (tmp_path/"out/train_only_per_episode.jsonl").is_file()
    with pytest.raises(FileExistsError):
        run_train(data,orig,rollouts,tmp_path/"out",bootstrap=101)
    invalid=json.loads((data/"questions.jsonl").read_text().splitlines()[0])
    invalid["split"]="dev"
    write_jsonl(data/"questions.jsonl",[invalid])
    with pytest.raises(ValueError,match="TRAIN-only"):
        run_train(data,orig,rollouts,tmp_path/"out2",bootstrap=101)


def test_pilot_manifest_fails_closed_before_looking_at_trace(tmp_path):
    invalid=tmp_path/"fake_manifest.json"
    invalid.write_text(json.dumps({
        "sample_count":32,"split":"dev",
        "data_sha256":{"documents.jsonl":"WRONG",
                       "questions.jsonl":"WRONG"}}))
    output=tmp_path/"pilot"
    with pytest.raises(ValueError,match="frozen pilot32"):
        run_pilot(tmp_path/"nonexistent_traces",invalid,output,bootstrap=101)
    assert not output.exists()


def test_cluster_bootstrap_does_not_treat_four_rollouts_as_four_tasks():
    rows=[{"question_id":"q1","x":1} for _ in range(4)]
    rows +=[{"question_id":"q2","x":0}]
    ci=clustered_bootstrap(rows,("x",),seed=5,repeats=150)
    assert ci["x"]["mean"]==0.8
    assert ci["x"]["question_clusters"]==2
    assert ci["x"]["episodes"]==5
    assert ci==clustered_bootstrap(rows,("x",),seed=5,repeats=150)
