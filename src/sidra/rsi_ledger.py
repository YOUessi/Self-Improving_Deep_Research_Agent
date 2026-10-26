"""RSI generation ledger and cross-role trust gates, TRAIN-only.

This module provides auditable handoffs, NOT an already-running RSI loop.

Trust split:
    Challenger: TRAIN question text + observed source summaries, no gold.
    Solver: task prompt + ordinary search/read tools, no verifier gold.
    Verifier: offline TRAIN-only gold and completed model/tool trace.
    Trainer: approved train-only action trajectories + rewards.
    Evaluator: separate frozen DEV; NEVER returned to Challenger/Trainer.

Promotions require *real different checkpoint SHA*, an independent task
comparison and an explicit improvement; negative generations stay recorded.
One challenge does not confer evidence that a model has learned anything.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict,dataclass
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.rsi_trace_audit import audit_tool_trace
from sidra.retrieval import BM25Index
from sidra.schema import AgentTrace,Document,Question
from sidra.train_reward import score_train_rollout


DENIED_FIELDS={"gold_answer","supporting_doc_ids","evaluation_answer",
               "hidden_tests","dev_label","test_label","supporting_facts"}


def _text(s:str)->str:
    return re.sub(r"\s+"," ",s.casefold()).strip()


def _hash(s:str)->str:
    return hashlib.sha256(_text(s).encode()).hexdigest()


@dataclass(frozen=True)
class Challenge:
    challenge_id:str
    source_train_question_id:str
    challenge_text:str
    source_commit:str
    challenger_model:str
    difficulty_hint:str
    text_sha256:str

    def solver_view(self)->dict:
        # Do not expose provenance labels / underlying source gold to solver.
        return {"challenge_id":self.challenge_id,"question":self.challenge_text}


def freeze_train_challenges(
    train_questions:list[Question],dev_ids:set[str],
    proposals:list[dict],out:Path, *,
    challenger_model:str,source_commit:str)->dict:
    if out.exists():raise FileExistsError(out)
    if not challenger_model.strip() or len(source_commit)!=40:
        raise ValueError("missing model / pinned GitHub source commit")
    qmap={q.question_id:q for q in train_questions}
    if len(qmap)!=len(train_questions) or any(q.split!="train" for q in train_questions):
        raise ValueError("Challenger input MUST be unique TRAIN questions")
    if set(qmap)&dev_ids:
        raise ValueError("TRAIN question set intersects DEV")
    frozen=[]
    seen_hashes=set()
    for index,p in enumerate(proposals):
        if set(p)!={"source_train_question_id","challenge_text","difficulty_hint"}:
            raise ValueError("unsupported Challenger fields (possible evaluator leakage)")
        qid=str(p["source_train_question_id"])
        text=str(p["challenge_text"]).strip()
        if qid not in qmap or qid in dev_ids:
            raise ValueError("Challenge does not originate from TRAIN")
        if any(x in _text(text) for x in DENIED_FIELDS):
            raise ValueError("challenge text contains restricted evaluator label names")
        if not 20<=len(text)<=500:
            raise ValueError("Challenge text length outside bounds")
        digest=_hash(text)
        if digest in seen_hashes:continue
        seen_hashes.add(digest)
        frozen.append(Challenge(
            challenge_id=f"challenge-{digest[:18]}",
            source_train_question_id=qid,
            challenge_text=text,
            source_commit=source_commit,
            challenger_model=challenger_model,
            difficulty_hint=str(p["difficulty_hint"])[:70],
            text_sha256=digest,
        ))
    if not frozen:raise ValueError("no unique TRAIN-safe challenges")
    out.mkdir(parents=True,exist_ok=False)
    (out/"CHALLENGER_PROPOSALS.jsonl").write_text(
        "\n".join(json.dumps(asdict(x),ensure_ascii=False) for x in frozen)+"\n")
    (out/"SOLVER_INPUT.jsonl").write_text(
        "\n".join(json.dumps(x.solver_view(),ensure_ascii=False) for x in frozen)+"\n")
    summary={
        "split":"train","frozen_count":len(frozen),
        "deduplicated_count":len(proposals)-len(frozen),
        "source_commit":source_commit,
        "challenger_model":challenger_model,
        "dev_gold_visible_to_challenger":False,
        "solver_receives_hidden_gold":False,
        "model_updates":0,
        "meaning":"frozen candidates, NOT an RSI generation or validated tasks",
    }
    (out/"MANIFEST.json").write_text(json.dumps(summary,indent=2)+"\n")
    return summary


def verify_train_traces(
    challenges:list[Challenge],traces:list[AgentTrace],
    train_qmap:dict[str,Question],documents:dict[str,Document],
    *,config:AgentConfig|None=None)->dict:
    by_challenge={c.challenge_id:c for c in challenges}
    if len(by_challenge)!=len(challenges):
        raise ValueError("duplicate challenge IDs")
    evaluated=[]
    shared_docs=list(documents.values())
    cfg=config or AgentConfig()
    shared_idx=BM25Index(shared_docs,title_boost=cfg.title_boost)
    for tr in traces:
        c=by_challenge.get(tr.question_id)
        if c is None:raise ValueError("unsolicited or unknown challenge ID")
        q=train_qmap.get(c.source_train_question_id)
        if q is None or q.split!="train":
            raise ValueError("verifier cannot inspect DEV/TEST gold")
        if tr.question!=c.challenge_text:
            raise ValueError("solver trace does not match frozen challenge")
        # Only valid if the challenge is identical (normalized) to original
        # TRAIN question. Novel challenger tasks require *new, independent*
        # TRAIN-only labels/verification; never inherit source gold casually.
        if _text(c.challenge_text)!=_text(q.question):
            evaluated.append({
                "challenge_id":c.challenge_id,"verdict":"unlabelled_challenge",
                "eligible_for_training":False,
                "reason":"novel prompt has no proven ground truth; source answer cannot be reused",
            })
            continue
        comparison=Question(
            question_id=tr.question_id, question=tr.question,
            gold_answer=q.gold_answer,
            supporting_doc_ids=q.supporting_doc_ids,
            split="train",scope=q.scope,
        )
        # SECURITY: evaluator metrics trust the supplied trace; replay the
        # entire gold-blind SearchAgent tool timeline against the immutable
        # source corpus before considering TRAIN-only reward labels.
        audit=audit_tool_trace(comparison,tr,shared_docs,config=cfg,
                               shared_index=shared_idx)
        if not audit.structurally_valid or not audit.complete_legal_finish:
            evaluated.append({
                "challenge_id":c.challenge_id,
                "verdict":"invalid_or_incomplete_tool_trace",
                "eligible_for_training":False,
                "tool_integrity":audit.as_dict(),
            })
            continue
        reward=score_train_rollout(comparison,tr,documents,
                                  max_tool_calls=(config or AgentConfig()).max_tool_calls)
        evaluated.append({
            "challenge_id":c.challenge_id,
            "verdict":"verified_train" if reward.strict_outcome else "not_verified",
            "eligible_for_training":bool(reward.strict_outcome),
            "reward":reward.as_dict(),
            "tool_integrity":audit.as_dict(),
        })
    return {
        "n":len(evaluated),
        "train_only":True,
        "approved_count":sum(int(x["eligible_for_training"]) for x in evaluated),
        "evaluated":evaluated,
        "critical_caveat":"New challenger questions cannot inherit source gold; must be independently verified on TRAIN before training.",
    }


def record_generation(
    *,generation:int,parent_adapter_sha256:str,new_adapter_sha256:str,
    source_train_rollouts_sha256:str,
    evaluator_report:dict,out:Path)->dict:
    if out.exists():raise FileExistsError(out)
    if generation<1:raise ValueError("generation must be positive")
    for name,value in (
        ("parent_adapter_sha256",parent_adapter_sha256),
        ("new_adapter_sha256",new_adapter_sha256),
        ("source_train_rollouts_sha256",source_train_rollouts_sha256),
    ):
        if not re.fullmatch("[0-9a-f]{64}",value):
            raise ValueError(f"invalid {name}")
    if new_adapter_sha256==parent_adapter_sha256:
        raise ValueError("no real model update: refusal to promote RSI generation")
    if set(evaluator_report)!={"n","before_verified","after_verified",
                              "before_answer_em","after_answer_em","frozen_dev_sha256"}:
        raise ValueError("evaluation report missing required frozen metrics")
    if not re.fullmatch("[0-9a-f]{64}",str(evaluator_report["frozen_dev_sha256"])):
        raise ValueError("invalid independent evaluation checksum")
    n=int(evaluator_report["n"])
    if n<32:raise ValueError("insufficient frozen evaluation size")
    before=float(evaluator_report["before_verified"])
    after=float(evaluator_report["after_verified"])
    if not (0<=before<=1 and 0<=after<=1):
        raise ValueError("bad verifier fractions")
    promoted=after>before
    record={
        "generation":generation,
        "parent_sha256":parent_adapter_sha256,
        "checkpoint_sha256":new_adapter_sha256,
        "train_rollout_sha256":source_train_rollouts_sha256,
        "evaluation":evaluator_report,
        "promoted":promoted,
        "status":"promoted" if promoted else "kept_for_negative_analysis",
        "actual_model_update":True,
        "single_generation_improvement_observed":promoted,
        "multi_generation_RSI_completed":False,
        "caution":"A single generation on public DEV is NOT a demonstrated multi-generation RSI loop; independent repeated validation required.",
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(record,indent=2)+"\n")
    return record
