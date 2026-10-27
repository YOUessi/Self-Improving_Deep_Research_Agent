"""M7 CPU-NLI experiment must refuse DEV/TEST, forged traces and overwritten outputs."""
import json

import pytest

from scripts.m7_train_semantic_nli_audit import (
    audit_train,question_cluster_bootstrap,
)


class FakeNLI:
    def __init__(self):
        self.calls=0
        self.inputs=[]
    def score(self,premises,hypotheses):
        self.calls+=1
        self.inputs.append((premises,hypotheses))
        return [{"entailment":0.98,"neutral":0.01,"contradiction":0.01}
                for _ in premises]


def write_jsonl(path,records):
    path.write_text("".join(json.dumps(x)+"\n" for x in records))


def train_fixture(tmp_path,split="train"):
    converted=tmp_path/"converted"
    converted.mkdir()
    write_jsonl(converted/"documents.jsonl",[
        {"doc_id":"d1","title":"Marie Curie","scope":"q1",
         "text":"Marie Curie was born in Warsaw."},
        {"doc_id":"d2","title":"Poland","scope":"q1",
         "text":"Warsaw is in Poland."},
    ])
    write_jsonl(converted/"questions.jsonl",[
        {"question_id":"q1","scope":"q1","question":"Where was Marie Curie born?",
         "gold_answer":"Warsaw","supporting_doc_ids":["d1","d2"],"split":split},
    ])
    orig=tmp_path/"original.json"
    orig.write_text(json.dumps([{
        "_id":"q1","question":"Where was Marie Curie born?","answer":"Warsaw",
        "context":[["Marie Curie",["Marie Curie was born in Warsaw."]],
                   ["Poland",["Warsaw is in Poland."]]],
        "supporting_facts":[["Marie Curie",0],["Poland",0]],
    }]))
    tr={
        "question_id":"q1","question":"Where was Marie Curie born?",
        "answer":"Warsaw","citations":["d1","d2"],"termination":"finish",
        "steps":[
            {"action":{"kind":"search","query":"Marie Curie"},
             "observation":{"kind":"search","hits":[
                 {"doc_id":"d1","title":"Marie Curie"},{"doc_id":"d2","title":"Poland"}]}},
            {"action":{"kind":"read","doc_id":"d1"},
             "observation":{"kind":"read","doc_id":"d1","title":"Marie Curie",
                            "text":"Marie Curie was born in Warsaw."}},
            {"action":{"kind":"read","doc_id":"d2"},
             "observation":{"kind":"read","doc_id":"d2","title":"Poland",
                            "text":"Warsaw is in Poland."}},
            {"action":{"kind":"finish","answer":"Warsaw","citations":["d1","d2"]}},
        ],
        "tool_calls":3,"search_calls":1,"read_calls":2,
        "prompt_tokens":120,"completion_tokens":36,"elapsed_ms":440,
        "model_calls":4,
    }
    roll=tmp_path/"rollouts.jsonl"
    write_jsonl(roll,[{"question_id":"q1","rollout_index":0,
                       "trace":tr,"reward":{"total":0.95}}])
    return converted,orig,roll


def test_train_only_real_trace_audit_and_separate_gold(tmp_path):
    data,original,rollout=train_fixture(tmp_path)
    scorer=FakeNLI()
    out=tmp_path/"result"
    report=audit_train(dataset=data,original_train=original,rollouts=[rollout],
                       scorer=scorer,out=out,bootstrap=101)
    assert report["n_episodes"]==1
    assert report["n_question_clusters"]==1
    assert report["nli_scored_count"]==1
    assert report["predicted_entailment_count"]==1
    assert report["no_dev_correct_answer_available_to_verifier"] is True
    assert report["semantic_entailment_ground_truth_measured"] is False
    assert scorer.calls==1
    raw=str(scorer.inputs)
    assert "gold_answer" not in raw and "supporting_facts" not in raw
    assert (out/"train_only_per_episode.jsonl").is_file()
    with pytest.raises(FileExistsError):
        audit_train(dataset=data,original_train=original,rollouts=[rollout],
                    scorer=scorer,out=out,bootstrap=101)


def test_dev_split_rejected_before_any_model_inference(tmp_path):
    data,original,rollout=train_fixture(tmp_path,split="dev")
    scorer=FakeNLI()
    with pytest.raises(ValueError,match="TRAIN-only"):
        audit_train(dataset=data,original_train=original,rollouts=[rollout],
                    scorer=scorer,out=tmp_path/"out",bootstrap=101)
    assert scorer.calls==0


def test_unknown_trace_id_rejected_before_any_model_inference(tmp_path):
    data,original,rollout=train_fixture(tmp_path)
    rows=[json.loads(x) for x in rollout.read_text().splitlines()]
    rows[0]["question_id"]="unknown"
    write_jsonl(rollout,rows)
    scorer=FakeNLI()
    with pytest.raises(ValueError,match="not in original TRAIN"):
        audit_train(dataset=data,original_train=original,rollouts=[rollout],
                    scorer=scorer,out=tmp_path/"out",bootstrap=101)
    assert scorer.calls==0


def test_question_cluster_bootstrap_handles_repeated_rollouts():
    rows=[{"question_id":"q1","x":1} for _ in range(4)]
    rows+=[{"question_id":"q2","x":0}]
    a=question_cluster_bootstrap(rows,("x",),repeats=101,seed=5)
    b=question_cluster_bootstrap(rows,("x",),repeats=101,seed=5)
    assert a==b
    assert a["x"]["question_clusters"]==2
    assert a["x"]["mean"]==0.8
