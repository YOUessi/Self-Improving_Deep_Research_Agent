"""E-025 TRAIN-only end-to-end audit integrity and reproducibility tests."""
import json
import pytest

from scripts.m7_counterfactual_nli_audit import audit_counterfactual_train


def _jsonl(path,rows):
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def fixture(tmp_path,split="train"):
    data=tmp_path/"dataset"
    data.mkdir()
    _jsonl(data/"questions.jsonl",[{
        "question_id":"q1","question":"Where was Marie Curie born?",
        "gold_answer":"Warsaw","scope":"q1",
        "supporting_doc_ids":["d1","d2"],"split":split,
    }])
    _jsonl(data/"documents.jsonl",[
        {"doc_id":"d1","scope":"q1","title":"Marie Curie",
         "text":"Marie Curie was born in Warsaw."},
        {"doc_id":"d2","scope":"q1","title":"Poland",
         "text":"Warsaw is a city in Poland."},
    ])
    original=tmp_path/"original.json"
    original.write_text(json.dumps([{
        "_id":"q1","question":"Where was Marie Curie born?","answer":"Warsaw",
        "context":[
            ["Marie Curie",["Marie Curie was born in Warsaw."]],
            ["Poland",["Warsaw is a city in Poland."]],
        ],
        "supporting_facts":[["Marie Curie",0],["Poland",0]],
    }]))
    tr={"question_id":"q1","question":"Where was Marie Curie born?",
        "answer":"Warsaw","citations":["d1","d2"],"termination":"finish",
        "steps":[
            {"action":{"kind":"search","query":"Marie Curie"},
             "observation":{"kind":"search","hits":[
                 {"doc_id":"d1","title":"Marie Curie"},
                 {"doc_id":"d2","title":"Poland"}]}},
            {"action":{"kind":"read","doc_id":"d1"},
             "observation":{"kind":"read","doc_id":"d1","title":"Marie Curie",
                            "text":"Marie Curie was born in Warsaw."}},
            {"action":{"kind":"read","doc_id":"d2"},
             "observation":{"kind":"read","doc_id":"d2","title":"Poland",
                            "text":"Warsaw is a city in Poland."}},
            {"action":{"kind":"finish","answer":"Warsaw","citations":["d1","d2"]}},
        ],
        "tool_calls":3,"search_calls":1,"read_calls":2,
        "model_calls":4,"prompt_tokens":240,"completion_tokens":45,
        "elapsed_ms":1170}
    roll=tmp_path/"test_rollout.jsonl"
    _jsonl(roll,[{"question_id":"q1","rollout_index":0,"trace":tr}])
    return data,original,roll


class StubNLI:
    def __init__(self):
        self.inputs=[]
    def score(self,premises,hypotheses):
        self.inputs.extend(zip(premises,hypotheses))
        out=[]
        for prem in premises:
            p=0.98 if "Marie Curie:" in prem and "Poland:" in prem else .03
            out.append({"entailment":p,"neutral":.5*(1-p),
                        "contradiction":.5*(1-p)})
        return out


def test_real_episode_audit_separates_model_inputs_and_train_truth(tmp_path):
    dataset,original,roll=fixture(tmp_path)
    scorer=StubNLI()
    out=tmp_path/"run"
    report=audit_counterfactual_train(
        dataset=dataset,original_train=original,rollouts=[roll],
        scorer=scorer,out=out,bootstrap=100)
    assert report["n_episodes"]==1
    assert report["n_question_clusters"]==1
    assert report["counts"]["raw_nli_entailment"]==1
    assert report["counts"]["loco_accept"]==1
    assert report["counts"]["loco_additional_nli_calls"]==2
    assert report["model_weights_changed"] is False
    assert report["semantic_ground_truth_established"] is False
    assert report["no_dev_correct_answer_available_to_verifier"]
    assert report["no_eval_A64_or_final_B160_processed"]
    assert report["question_cluster_bootstrap"]["loco_accept"]["mean"]==1
    assert all("gold_answer" not in str(x) for x in scorer.inputs)
    assert (out/"summary.json").is_file()
    with pytest.raises(FileExistsError):
        audit_counterfactual_train(dataset=dataset,original_train=original,
            rollouts=[roll],scorer=scorer,out=out,bootstrap=100)


def test_dev_split_refused_before_semantic_model_called(tmp_path):
    dataset,original,roll=fixture(tmp_path,split="dev")
    scorer=StubNLI()
    with pytest.raises(ValueError,match="TRAIN-only"):
        audit_counterfactual_train(dataset=dataset,original_train=original,
            rollouts=[roll],scorer=scorer,out=tmp_path/"fail",bootstrap=100)
    assert scorer.inputs==[]


def test_trace_question_forgery_and_duplicate_episode_rejected(tmp_path):
    data,original,roll=fixture(tmp_path)
    entries=[json.loads(x) for x in roll.read_text().splitlines()]
    entries[0]["trace"]["question"]="What is the reference answer?"
    _jsonl(roll,entries)
    scorer=StubNLI()
    with pytest.raises(ValueError,match="does not match TRAIN"):
        audit_counterfactual_train(dataset=data,original_train=original,
            rollouts=[roll],scorer=scorer,out=tmp_path/"fail",bootstrap=100)
    assert scorer.inputs==[]
    entries[0]["trace"]["question"]="Where was Marie Curie born?"
    _jsonl(roll,entries*2)
    with pytest.raises(ValueError,match="duplicate trained rollout episode"):
        audit_counterfactual_train(dataset=data,original_train=original,
            rollouts=[roll],scorer=scorer,out=tmp_path/"fail2",bootstrap=100)
