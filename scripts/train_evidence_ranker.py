"""Fit answer-conditioned source attribution on HotpotQA TRAIN only.

Question-group train/internal-validation split, never use DEV labels in the
optimizer or choose a confidence threshold on the user-facing pilot set.
The trained model is a small NumPy logistic ranker, saved as JSON; serving
imports no NumPy and inspects only question, proposed answer, actual READ text.

The internal validation's 2 support + 1 distractor is an *offline labelled
candidate-availability control*, NOT real Agent retrieval or heldout external QA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from sidra.evidence_ranker import EvidenceRanker,FEATURE_NAMES,features
from sidra.io import load_dataset
from sidra.retrieval import BM25Index


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda:f.read(1<<20),b""):
            h.update(data)
    return h.hexdigest()


def training_rows(data:Path,*,seed:int=712):
    docs,qs=load_dataset(data)
    if not qs or len({q.question_id for q in qs})!=len(qs):
        raise ValueError("empty/duplicate original TRAIN dataset")
    if any(q.split!="train" for q in qs):
        raise ValueError("evidence-ranker supervision must use TRAIN ONLY")
    scopes={}
    for d in docs:
        scopes.setdefault(d.scope,[]).append({
            "doc_id":d.doc_id,"title":d.title,"text":d.text,
        })
    train_x,train_y,val_x,val_y=[],[],[],[]
    groups={"train":[],"internal_val":[]}
    questions={"train":[],"internal_val":[]}
    for q in qs:
        candidates=scopes.get(q.scope,[])
        if len(candidates)<3:
            raise ValueError("TRAIN question has insufficient scoped passages")
        gold=set(q.supporting_doc_ids)
        if not gold or not gold.issubset({d["doc_id"] for d in candidates}):
            raise ValueError("missing labelled TRAIN supporting passage")
        n=int(hashlib.sha256(f"ranker-val:{seed}:{q.question_id}".encode()).hexdigest()[:8],16)
        phase="internal_val" if n%5==0 else "train"
        x=train_x if phase=="train" else val_x
        y=train_y if phase=="train" else val_y
        for d in candidates:
            x.append(features(q.question,q.gold_answer,d,candidates))
            y.append(int(d["doc_id"] in gold))
        groups[phase].append(q.question_id)
        questions[phase].append((q,candidates))
    if not train_x or not val_x:
        raise ValueError("question-group split created empty internal validation")
    if set(groups["train"])&set(groups["internal_val"]):
        raise ValueError("TRAIN/internal validation group leakage")
    return train_x,train_y,val_x,val_y,questions,groups


def train_ranker(data:Path,out:Path,*,seed:int=712,steps:int=400,
                 learning_rate:float=0.07)->dict:
    if out.exists():raise FileExistsError(out)
    if data.name!="converted" or not data.parent.name.startswith("train"):
        raise ValueError("only original TRAIN converted source is allowed")
    import numpy as np
    if steps<20 or not 0<learning_rate<0.5:
        raise ValueError("bad optimizer configuration")
    tx,ty,vx,vy,questions,groups=training_rows(data,seed=seed)
    x=np.asarray(tx,dtype=np.float64)
    y=np.asarray(ty,dtype=np.float64)
    means=x.mean(axis=0)
    scale=np.maximum(x.std(axis=0),0.02)
    z=(x-means)/scale
    w=np.zeros(z.shape[1],dtype=np.float64)
    b=0.0
    weights=np.where(y>0.5,4.0,1.0)
    norm=weights.sum()
    for it in range(steps):
        logits=np.clip(z@w+b,-35,35)
        prob=1/(1+np.exp(-logits))
        errors=(prob-y)*weights
        gradient=(z.T@errors)/norm+0.002*w
        bgrad=errors.sum()/norm
        w-=learning_rate*gradient
        b-=learning_rate*bgrad
    if not np.isfinite(w).all() or not math.isfinite(b):
        raise FloatingPointError("evidence logistic optimizer diverged")

    prototype=dict(weights=w.tolist(),bias=float(b),
                   means=means.tolist(),scales=scale.tolist(),
                   train_data_sha256=sha(data/"questions.jsonl"))
    # Use original TRAIN-only internal validation to preselect abstain cutoff.
    # Condition: model sees two TRAIN labelled sources and the first extraneous
    # scoped distractor; scorer sees only question+answer+read text, no labels.
    candidates=[]
    for q,docs in questions["internal_val"]:
        gold=set(q.supporting_doc_ids)
        actual=[d for d in docs if d["doc_id"] in gold]
        distractors=[d for d in docs if d["doc_id"] not in gold]
        if len(actual)<2 or not distractors:continue
        actual=actual[:2]+[distractors[0]]
        candidates.append((q,actual,gold))
    if not candidates:
        raise ValueError("no original TRAIN validation group with 2 supports")
    grid=(0.2,0.3,0.4,0.5,0.6,0.7)
    trials=[]
    for threshold in grid:
        model=EvidenceRanker(threshold=threshold,**prototype)
        correct=wrong=abstain=0
        for q,observed,gold in candidates:
            pred=model.select(q.question,q.gold_answer,observed)
            ids=set(pred["citations"])
            if not ids:abstain+=1
            elif ids==gold:correct+=1
            else:wrong+=1
        # Penalize false 2-doc citations more than an abstention.
        score=correct-0.5*wrong
        trials.append({"threshold":threshold,"correct_pair":correct,
                       "wrong_pair":wrong,"abstained":abstain,
                       "net_utility":score})
    chosen=max(trials,key=lambda x:(x["net_utility"],-x["wrong_pair"],
                                     x["correct_pair"],x["threshold"]))
    trained=EvidenceRanker(threshold=chosen["threshold"],**prototype)
    meta={
        "source":"HotpotQA TRAIN-only converted scoped passages",
        "train_questions_sha256":sha(data/"questions.jsonl"),
        "train_documents_sha256":sha(data/"documents.jsonl"),
        "train_questions":len(groups["train"]),
        "internal_validation_questions":len(groups["internal_val"]),
        "train_doc_examples":len(tx),"internal_validation_doc_examples":len(vx),
        "optimizer":"weighted NumPy logistic regression + L2",
        "steps":steps,"learning_rate":learning_rate,
        "seed":seed,"features":list(FEATURE_NAMES),
        "threshold_from_train_internal_validation":chosen["threshold"],
        "train_internal_validation_results":trials,
        "train_val_labels_not_available_to_runtime":True,
        "model_weights_trained":True,
        "model_semantic_entailment_capability":False,
        "oracle_val_condition":"2 real labelled supports + 1 distractor from TRAIN, not live Search Agent",
        "independent_external_test_not_run":True,
        "DEV_final_B160_used":False,
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"MODEL.json").write_text(json.dumps(trained.to_dict(),indent=2)+"\n")
    (out/"TRAIN_REPORT.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(json.dumps({k:v for k,v in meta.items() if k!="train_internal_validation_results"},
                     indent=2))
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--seed",type=int,default=712)
    a=p.parse_args()
    train_ranker(a.train_data,a.out,seed=a.seed)


if __name__=="__main__":
    main()
