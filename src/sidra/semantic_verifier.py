"""CPU-only, gold-blind, calibrated-later NLI screening of actual cited READs.

This is an *opt-in* research verifier. It MUST NOT be confused with semantic
ground-truth: MNLI 3-way probabilities are model predictions, not proof; the
question-to-claim translator intentionally ABSTAINS on unsupported forms.
Training gold and original HotpotQA (title, sentence index) annotations are
strictly offline, and NEVER arguments to the scorer.

Important invariants:
- Only successful FINISH actions with citations to *actual* READs are checked.
- The response answer and question, NOT a reference answer, form hypothesis.
- Premises consist ONLY of previously read, cited texts. No hidden corpus.
- Every prediction names source IDs and model labels; no silent "verified".
- Running this module does NOT update the original SearchAgent nor RL reward.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Protocol

from sidra.evidence_diagnostics import source_only_verdict
from sidra.retrieval import tokenize


@dataclass(frozen=True)
class SemanticReview:
    status: str
    reason: str
    hypothesis: str
    cited_read_ids: tuple[str, ...]
    evidence_sentence_count: int
    entailment: float | None
    neutral: float | None
    contradiction: float | None
    model_prediction: str
    semantic_ground_truth: str = "not_established"

    def as_dict(self)->dict:
        return asdict(self)


class NLIScorer(Protocol):
    def score(self,premises:list[str],hypotheses:list[str])->list[dict[str,float]]: ...


def question_answer_hypothesis(question:str, answer:str)->str|None:
    """Conservative, deterministic relation-preserving WH-to-statement.

    Only supports a few grammars. Inventing an affirmative assertion from
    ambiguous multi-hop syntax produces false NLI approvals; no fallback
    generic "the answer is" claim is permitted.
    """
    q=" ".join(question.strip().split()).rstrip(" ?.")
    a=" ".join(answer.strip().split())
    if not q or not a or len(q)>550 or len(a)>130:
        return None
    q=re.sub(r"^Q\s*:\s*","",q,flags=re.I)
    patterns=(
        # "When was the director of X born?" / "Where was X born?"
        (r"^(when|where) was (.+?) born$", lambda m: m[2]+" was born "+("in " if m[1].lower()=="where" else "on ")+a),
        (r"^in (?:what|which) year was (.+?) (born|founded|established|opened|created|released)$",
         lambda m:m[1]+" was "+m[2]+" in "+a),
        (r"^in which (city|country|state|region) was (.+?) born$",lambda m:m[2]+" was born in "+a),
        # "What author created [work]?" and "Who wrote [book]?"
        (r"^what (author|writer|person) (wrote|created|directed|composed|invented|founded) (.+)$",
         lambda m:a+" "+m[2]+" "+m[3]),
        (r"^who (wrote|created|directed|composed|invented|founded|produced) (.+)$",
         lambda m:a+" "+m[1]+" "+m[2]),
        # "The original fifth anchor ... was founded in what year?"
        (r"^(.+?) was (founded|established|born|opened|created|released) in (what|which) year$",
         lambda m:m[1]+" was "+m[2]+" in "+a),
        (r"^(.+?) was (founded|established|born|opened|created|released) in (what|which) (city|country|state|region)$",
         lambda m:m[1]+" was "+m[2]+" in "+a),
        # "What year was [named event] founded?" (not free relinking)
        (r"^what year was (.+?) (founded|established|born|opened|created|released)$",
         lambda m:m[1]+" was "+m[2]+" in "+a),
        # "Where is [named object] located?" / "Where was [object] founded?"
        (r"^where (is|was) (.+?) (located|based|headquartered|founded)$",
         lambda m:m[2]+" "+m[1]+" "+m[3]+" in "+a),
    )
    for pattern,build in patterns:
        match=re.fullmatch(pattern,q,flags=re.I)
        if match is None:continue
        claim=" ".join(build(match).split())
        if len(claim)>700 or "?" in claim:return None
        return claim[0].upper()+claim[1:]+"."

    # Conservative WH noun-phrase substitution keeps the ENTIRE original
    # relation instead of inventing one ("of what company since 2006?" ->
    # "of <answer> since 2006."). This never uses a gold reference, and
    # deliberately declines competing WH clauses / unsupported grammar.
    interrogatives=re.findall(r"\b(?:what|which|who|where|when|whose)\b",q,flags=re.I)
    if len(interrogatives)!=1 or interrogatives[0].casefold() not in {"what","which"}:
        return None
    noun_wh=re.search(
        r"\b(?:what|which)\s+(?:[a-z0-9-]+\s+){0,4}"
        r"(?:company|year|region|country|city|state|adventure|composer|parents|team)\b",
        q,flags=re.I)
    if noun_wh is None:
        return None
    # Basic subject/verb gate blocks malformed fragments and "which of ...".
    if not re.search(
        r"\b(?:was|were|is|are|has|have|had|plays|played|ran|features|"
        r"made|led|founded|born|owned|released|created|located|won|wrote|"
        r"received|became|directed|chaired|contains|includes)\b",
        q,flags=re.I):
        return None
    statement=(q[:noun_wh.start()]+a+q[noun_wh.end():]).strip()
    if ("?" in statement or len(statement)>700 or
        re.search(r"\b(?:what|which|who|where|when|whose)\b",statement,flags=re.I)):
        return None
    return statement[0].upper()+statement[1:]+"."



def actual_cited_reads(trace:dict)->dict[str,dict]:
    """Last matched real READ observation per ID, only if action matches."""
    found={}
    for step in trace.get("steps",[]):
        action=step.get("action") or {}
        obs=step.get("observation") or {}
        if (action.get("kind")=="read" and obs.get("kind")=="read"
            and isinstance(action.get("doc_id"),str)
            and action["doc_id"]==obs.get("doc_id")):
            found[action["doc_id"]]={
                "title":str(obs.get("title","")),
                "text":str(obs.get("text","")),
            }
    return found


def sentence_windows(question:str,answer:str,reads:dict[str,dict],citations:tuple[str,...],
                     *, max_per_doc:int=2, max_chars:int=1450)->tuple[str,int]:
    """Gold-blind question+answer overlap selection from actually READ sources.

    No information outside the cited READ observations is available. If an
    answer-bearing sentence is missing, NLI is still allowed on question
    overlap (to reveal contradictions), but NOT assumed to entail.
    """
    if max_per_doc<1 or max_chars<100:raise ValueError("invalid evidence limits")
    words=set(tokenize(question))
    answer_words=set(tokenize(answer))
    chunks=[]
    total=0
    for doc_id in citations:
        obs=reads[doc_id]
        text=obs["text"].strip()
        candidates=[
            " ".join(x.split()) for x in re.split(r"(?<=[.!?])\s+|\n+",text)
            if x.strip()
        ]
        ranked=[]
        for i,sentence in enumerate(candidates):
            terms=set(tokenize(sentence))
            score=(3*len(answer_words&terms)+len(words&terms))
            ranked.append((-score,i,sentence))
        for _,_,sentence in sorted(ranked)[:max_per_doc]:
            if not sentence:continue
            prefix=(obs["title"].strip()+": ") if obs["title"].strip() else ""
            chunk=prefix+sentence
            space=max_chars-total
            if space<=0:break
            # Preserve model input length bound but mark truncation via caller.
            if len(chunk)>space:chunk=chunk[:space]
            chunks.append(chunk)
            total+=len(chunk)
    return "\n".join(chunks),len(chunks)


def semantic_review(trace:dict, question:str, scorer:NLIScorer, *,
                    min_citations:int=2,
                    entailment_threshold:float=0.85,
                    contradiction_threshold:float=0.80)->SemanticReview:
    """Screen a completed answer without any gold labels.

    A positive verdict is labelled *predicted*, not semantically verified.
    A model cannot be allowed to rescue illegal or fabricated citations.
    """
    if not (0.5<=entailment_threshold<=1 and 0.5<=contradiction_threshold<=1):
        raise ValueError("invalid NLI confidence thresholds")
    source=source_only_verdict(trace,min_citations=min_citations)
    citations=tuple(map(str,trace.get("citations") or []))
    reads=actual_cited_reads(trace)
    valid=tuple(i for i in citations if i in reads)
    def result(status,reason,hypothesis="",n=0,scores=None,pred="not_evaluated"):
        scores=scores or {}
        return SemanticReview(
            status,reason,hypothesis,valid,n,
            scores.get("entailment"),scores.get("neutral"),
            scores.get("contradiction"),pred)
    if source.reason!="lexical_only_entailment_unverified":
        return result("not_scored",source.reason)
    answer=str(trace.get("answer",""))
    hypothesis=question_answer_hypothesis(question,answer)
    if hypothesis is None:
        return result("not_scored","unsupported_question_form")
    premise,n=sentence_windows(question,answer,reads,citations)
    if not premise:
        return result("not_scored","empty_cited_read")
    scores_list=scorer.score([premise],[hypothesis])
    if len(scores_list)!=1:
        raise ValueError("NLI backend returned wrong batch length")
    s=scores_list[0]
    if set(s)!={"entailment","neutral","contradiction"}:
        raise ValueError("invalid NLI label map")
    import math
    if not all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1
               for x in s.values()) or abs(sum(s.values())-1)>0.01:
        raise ValueError("invalid NLI posterior")
    pred=max(("entailment","neutral","contradiction"),key=lambda k:s[k])
    if (pred=="entailment" and s["entailment"]>=entailment_threshold):
        return result("model_predicted_entailment","nli_entailment_not_proof",
                      hypothesis,n,s,pred)
    if (pred=="contradiction" and s["contradiction"]>=contradiction_threshold):
        return result("model_predicted_contradiction","nli_contradiction_not_proof",
                      hypothesis,n,s,pred)
    return result("model_uncertain","nli_neutral_or_below_threshold",
                  hypothesis,n,s,pred)


class LocalCPUMNLI:
    """Explicitly CPU-only / offline. Fails closed for unknown id2label."""

    def __init__(self,model_path:str, *,max_length:int=384,torch_threads:int=4):
        if not 64<=max_length<=512:raise ValueError("invalid NLI context bound")
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        torch.set_num_threads(max(1,min(torch_threads,8)))
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(
            model_path,local_files_only=True,dtype=torch.float32).to("cpu").eval()
        labels={i:str(n).casefold() for i,n in self.model.config.id2label.items()}
        if set(labels.values())!={"entailment","neutral","contradiction"}:
            raise ValueError("MNLI model must have entailment,neutral,contradiction labels")
        self.label_map=labels
        self.max_length=max_length
        self.calls=0

    def score(self,premises:list[str],hypotheses:list[str])->list[dict[str,float]]:
        if len(premises)!=len(hypotheses) or not premises:
            raise ValueError("NLI requires equal nonempty paired batches")
        tok=self.tokenizer(
            premises,hypotheses,return_tensors="pt",padding=True,
            truncation="only_first",max_length=self.max_length)
        with self.torch.inference_mode():
            prob=self.model(**tok).logits.softmax(dim=-1).cpu().tolist()
        self.calls+=len(premises)
        return [{self.label_map[i]:float(x) for i,x in enumerate(row)} for row in prob]
