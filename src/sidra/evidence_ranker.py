"""Train-only supervised, gold-blind-at-inference evidence attribution ranker.

Fit labels ONLY on original HotpotQA TRAIN supporting document IDs. At
inference it takes (question, candidate answer, actually READ documents),
then ranks these documents. There is no access to gold support annotations,
unread search hits or external corpus data at serving time.

This selects and quotes *observed* supporting context, but it does not prove
semantic entailment. It is a separate post-generation citation module, not
a reward/GRPO trainer. Scores are deterministic given the saved TRAIN model.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sidra.retrieval import tokenize


STOP={
    "a","an","and","the","in","of","on","is","was","are","were","be",
    "to","at","for","by","or","with","as","which","who","what","when",
    "where","how","why","did","does","do","can","could","would","has",
    "have","had","he","she","his","her","it","its","from","than",
    "that","this","these","those","both","other","same","their",
    "into","about","known","one","two","more","most","also",
}
BOOLEAN={"yes","no"}


def tokens(text:str)->set[str]:
    return {t for t in tokenize(text) if t not in STOP and len(t)>1}


def phrase_in(needle:str,haystack:str)->bool:
    pat=" ".join(tokenize(needle))
    corpus=" ".join(tokenize(haystack))
    return bool(pat and f" {pat} " in f" {corpus} ")


def segments(text:str)->list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+",text)
            if len(s.strip())>=15][:80]


def _fraction(a:set[str],b:set[str])->float:
    return len(a&b)/max(1,len(a))


FEATURE_NAMES=(
    "title_terms_in_question",
    "question_terms_in_title",
    "question_exact_title_mention",
    "question_terms_in_document",
    "best_sentence_question_overlap",
    "answer_in_document",
    "answer_in_title",
    "answer_sentence_question_overlap",
    "document_title_mentioned_elsewhere",
    "document_mentions_other_title",
    "log_document_length",
    "document_title_token_count",
)


def features(question:str,answer:str,doc:dict,others:list[dict])->list[float]:
    """Only current question/answer and truly READ texts: no hidden labels."""
    text=str(doc["text"])[:3500]
    title=str(doc["title"])[:160]
    qt=tokens(question)
    tt=tokens(title)
    dt=tokens(text)
    ans=answer.strip()
    sents=segments(text)
    overlaps=[_fraction(qt,tokens(s)) for s in sents]
    answer_sents=[s for s in sents if phrase_in(ans,s)] if ans.casefold() not in BOOLEAN else []
    answer_overlap=max((_fraction(qt,tokens(s)) for s in answer_sents),default=0.0)
    linked_elsewhere=any(phrase_in(title,str(o.get("text",""))[:3500])
                         for o in others if o.get("doc_id")!=doc["doc_id"]
                         and len(tokens(title))>=1)
    other_title_in_doc=any(phrase_in(str(o.get("title","")),text)
                           for o in others if o.get("doc_id")!=doc["doc_id"]
                           and len(tokens(str(o.get("title",""))))>=1)
    return [
        _fraction(tt,qt),
        _fraction(qt,tt),
        float(phrase_in(title,question)),
        _fraction(qt,dt),
        max(overlaps,default=0.0),
        float(bool(ans and ans.casefold() not in BOOLEAN and phrase_in(ans,text))),
        float(bool(ans and ans.casefold() not in BOOLEAN and phrase_in(ans,title))),
        answer_overlap,
        float(linked_elsewhere),
        float(other_title_in_doc),
        math.log1p(min(len(text),3500))/10.0,
        min(len(tt),12)/12.0,
    ]


def best_quote(question:str,answer:str,doc:dict)->str:
    """Extract real literal sentence, never generate text not in READ."""
    sentences=segments(str(doc["text"])[:3500])
    if not sentences:
        return str(doc["text"])[:160]
    qt=tokens(question)
    ranked=sorted(
        enumerate(sentences),
        key=lambda x:(-(
            _fraction(qt,tokens(x[1]))+
            0.85*int(answer.casefold() not in BOOLEAN
                     and phrase_in(answer,x[1]))),
            x[0]),
    )
    return ranked[0][1][:350]


@dataclass(frozen=True)
class EvidenceChoice:
    doc_id:str
    score:float
    quote:str


class EvidenceRanker:
    def __init__(self,*,weights:list[float],bias:float,
                 means:list[float],scales:list[float],threshold:float,
                 train_data_sha256:str):
        n=len(FEATURE_NAMES)
        if not all(len(x)==n for x in (weights,means,scales)):
            raise ValueError("evidence ranker feature dimension mismatch")
        if any(not math.isfinite(v) for a in (weights,means,scales) for v in a):
            raise ValueError("nonfinite evidence model coefficients")
        if any(x<=0 for x in scales) or not math.isfinite(bias):
            raise ValueError("invalid ranker scale or bias")
        if not 0<=threshold<=1:
            raise ValueError("invalid train-only decision threshold")
        self.weights=weights
        self.bias=bias
        self.means=means
        self.scales=scales
        self.threshold=threshold
        self.train_data_sha256=train_data_sha256

    def probability(self,vec:list[float])->float:
        if len(vec)!=len(self.weights):
            raise ValueError("unexpected evidence feature dimension")
        z=self.bias+sum(w*(v-m)/sd
                        for w,v,m,sd in zip(self.weights,vec,self.means,self.scales))
        z=max(-45.,min(45.,z))
        return 1/(1+math.exp(-z))

    def rank(self,question:str,answer:str,read_sources:list[dict])->list[EvidenceChoice]:
        if len({str(x["doc_id"]) for x in read_sources})!=len(read_sources):
            raise ValueError("duplicate READ sources")
        return sorted((
            EvidenceChoice(str(d["doc_id"]),
                           round(self.probability(features(question,answer,d,read_sources)),7),
                           best_quote(question,answer,d))
            for d in read_sources if d.get("text")
        ),key=lambda x:(-x.score,x.doc_id))

    def select(self,question:str,answer:str,read_sources:list[dict])->dict:
        """No fabricated citations: outputs at most two ACTUALLY READ sources."""
        answer=answer.strip()
        if not answer or len(read_sources)<2:
            return {"answer":"","citations":[],"evidence":[],"reason":"missing_answer_or_two_reads"}
        read_text=" ".join(str(d.get("text",""))+" "+str(d.get("title",""))
                           for d in read_sources)
        if answer.casefold() not in BOOLEAN and not phrase_in(answer,read_text):
            return {"answer":"","citations":[],"evidence":[],"reason":"answer_not_in_read_sources"}
        scores=self.rank(question,answer,read_sources)
        if len(scores)<2 or scores[1].score<self.threshold:
            return {"answer":"","citations":[],"evidence":[],
                    "reason":"insufficient_two_relevant_sources"}
        picked=scores[:2]
        if any(not x.quote for x in picked):
            return {"answer":"","citations":[],"evidence":[],"reason":"missing_real_quote"}
        return {
            "answer":answer,
            "citations":[x.doc_id for x in picked],
            "evidence":[{"doc_id":x.doc_id,"score":x.score,"quote":x.quote}
                        for x in picked],
            "reason":"two_predicted_READ_sources",
        }

    def to_dict(self)->dict:
        return {"features":list(FEATURE_NAMES),"weights":self.weights,
                "bias":self.bias,"means":self.means,"scales":self.scales,
                "threshold":self.threshold,
                "train_data_sha256":self.train_data_sha256,
                "training_scope":"HotpotQA TRAIN-only supporting-doc supervision",
                "semantic_entailment_verified":False}

    @classmethod
    def from_file(cls,path:Path)->"EvidenceRanker":
        d=json.loads(path.read_text())
        if d["features"]!=list(FEATURE_NAMES):
            raise ValueError("evidence features/version changed")
        return cls(weights=d["weights"],bias=d["bias"],
                   means=d["means"],scales=d["scales"],
                   threshold=d["threshold"],
                   train_data_sha256=d["train_data_sha256"])
