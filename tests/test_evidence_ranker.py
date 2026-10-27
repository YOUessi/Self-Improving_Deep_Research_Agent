"""Evidence ranker runtime reads only tool observations, not evaluator gold."""
import json
from pathlib import Path

import pytest

from sidra.evidence_ranker import (
    EvidenceRanker,FEATURE_NAMES,features,phrase_in,best_quote
)
from sidra.io import save_jsonl
from sidra.schema import Document,Question


SOURCE=[
    {"doc_id":"a","title":"Ridge River","text":
       "Ridge River was founded in 1956. It runs through the town."},
    {"doc_id":"b","title":"Coastal Province","text":
       "Coastal Province contains Ridge River and is located in Region Alpha."},
    {"doc_id":"d","title":"Unrelated place","text":
       "The District is located far away in another country."},
]


def make_ranker(threshold=0.2):
    return EvidenceRanker(
        weights=[0.1,0.1,0.2,0.1,0.1,1.0,1.0,0.5,0.1,0.1,0.0,0.0],
        bias=0.2,
        means=[0.0]*len(FEATURE_NAMES),
        scales=[1.0]*len(FEATURE_NAMES),
        threshold=threshold,train_data_sha256="f"*64,
    )


def test_runtime_scores_only_question_answer_and_read_text():
    q="Which region contains Ridge River?"
    a=features(q,"Region Alpha",SOURCE[1],SOURCE)
    assert len(a)==len(FEATURE_NAMES)
    assert phrase_in("Region Alpha",SOURCE[1]["text"])
    assert not phrase_in("Ridge River","Ridge Riverdale")
    assert best_quote(q,"Region Alpha",SOURCE[1]) in SOURCE[1]["text"]
    x=make_ranker()
    scores=x.rank(q,"Region Alpha",SOURCE)
    assert len(scores)==3
    assert {a.doc_id for a in scores}=={"a","b","d"}
    data=json.dumps(x.to_dict())
    assert "gold_answer" not in data and "supporting_doc_ids" not in data


def test_unread_documents_cannot_be_cited_and_insufficient_sources_abstain():
    model=make_ranker(threshold=0)
    pred=model.select("Which province?","Region Alpha",SOURCE[:2])
    assert set(pred["citations"])=={"a","b"}
    assert all(item["quote"] in next(d["text"] for d in SOURCE
                                     if d["doc_id"]==item["doc_id"])
               for item in pred["evidence"])
    single=model.select("Which province?","Region Alpha",SOURCE[:1])
    assert single["answer"]=="" and not single["citations"]
    unknown=model.select("Which province?","Imaginary Planet",SOURCE)
    assert unknown["answer"]=="" and not unknown["citations"]
    with pytest.raises(ValueError,match="duplicate"):
        model.select("Question","Region Alpha",SOURCE[:2]+SOURCE[:1])


def test_ranker_json_validates_dimensions_and_threshold(tmp_path):
    ranker=make_ranker()
    file=tmp_path/"MODEL.json"
    file.write_text(json.dumps(ranker.to_dict()))
    loaded=EvidenceRanker.from_file(file)
    assert loaded.threshold==ranker.threshold
    broken=json.loads(file.read_text())
    broken["features"][0]="other_features"
    file.write_text(json.dumps(broken))
    with pytest.raises(ValueError,match="features"):
        EvidenceRanker.from_file(file)
    with pytest.raises(ValueError,match="feature dimension"):
        EvidenceRanker(weights=[0.1],means=[0.1],scales=[1],bias=0.1,
                       threshold=0.2,train_data_sha256="c"*64)


def test_train_only_group_disjoint_fitting(tmp_path):
    pytest.importorskip("numpy")
    from scripts.train_evidence_ranker import train_ranker
    data=tmp_path/"train_synthetic"/"converted"
    data.mkdir(parents=True)
    docs=[]
    qs=[]
    for i in range(24):
        scope=f"train-{i:03d}"
        place=f"Town{i}"
        answer=f"Province{i}"
        docs.extend([
            Document(f"{scope}::a",place,
                     f"The city {place} lies in {answer}.",scope=scope),
            Document(f"{scope}::b",answer,
                     f"{answer} is an administrative province containing {place}.",scope=scope),
            Document(f"{scope}::c","Unrelated",
                     "A mountain from a different era and another country.",scope=scope),
        ])
        qs.append(Question(scope,f"Which province contains {place}?",
                    answer,supporting_doc_ids=(f"{scope}::a",f"{scope}::b"),
                    split="train",scope=scope))
    save_jsonl(data/"documents.jsonl",[d.__dict__ for d in docs])
    save_jsonl(data/"questions.jsonl",[q.__dict__ for q in qs])
    out=tmp_path/"model"
    meta=train_ranker(data,out,seed=712,steps=50)
    assert meta["model_weights_trained"]
    assert meta["train_questions"]>0
    assert meta["internal_validation_questions"]>0
    assert meta["train_questions"]+meta["internal_validation_questions"]==24
    assert (out/"MODEL.json").is_file()
    with pytest.raises(FileExistsError):
        train_ranker(data,out)
    dev=tmp_path/"heldout"/"converted"
    dev.mkdir(parents=True)
    with pytest.raises(ValueError,match="TRAIN"):
        train_ranker(dev,tmp_path/"should_not_write")
