"""TRAIN-only difficulty-stratified, deduplicated RSI Challenger input queue.

This is *not* a model-driven Challenger. The offline curriculum selector may
read TRAIN supporting_doc_ids to stratify task difficulty, but the emitted
Challenger input contains only question text and a fresh opaque task ID. No
answers or supporting IDs are sent to a model. Frozen DEV/final-B contents
are never loaded. Optional forbidden hashes/IDs must be supplied externally.

Difficulty categories are TRAIN-only *retrieval proxy* labels, not proof of
reasoning difficulty: 'direct' means all supports initially retrieved;
'bridge' means some, and 'hard' means none.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter,defaultdict
from dataclasses import asdict,dataclass
from difflib import SequenceMatcher
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.retrieval import BM25Index
from sidra.schema import Document,Question


def _norm(text:str)->str:
    return " ".join(text.casefold().split())


def _sha(text:str)->str:
    return hashlib.sha256(_norm(text).encode()).hexdigest()


@dataclass(frozen=True)
class TrainingChallengeSeed:
    opaque_challenge_id:str
    source_train_id:str
    difficulty_tier:str
    normalized_text_hash:str
    question:str

    def challenger_view(self)->dict:
        # STRICT: no label, support IDs, scope, difficulty proxy, source ID.
        return {"challenge_id":self.opaque_challenge_id,"question":self.question}

    def verifier_map(self)->dict:
        # This mapping stays in the trusted offline Verifier/curator domain.
        return {"challenge_id":self.opaque_challenge_id,
                "source_train_question_id":self.source_train_id,
                "text_sha256":self.normalized_text_hash,
                "tier":self.difficulty_tier}


def stratify_train_questions(
    questions:list[Question],docs:list[Document],config:AgentConfig|None=None,
    *,forbidden_ids:set[str]|None=None,
    forbidden_normalized_hashes:set[str]|None=None)->dict[str,list[TrainingChallengeSeed]]:
    config=config or AgentConfig()
    blocked=forbidden_ids or set()
    blocked_hashes=forbidden_normalized_hashes or set()
    if not questions or not docs:raise ValueError("empty TRAIN dataset/corpus")
    ids=[q.question_id for q in questions]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate TRAIN question IDs")
    if any(q.split!="train" for q in questions):
        raise ValueError("Challenger must receive TRAIN-only questions")
    if any(q.question_id in blocked for q in questions):
        raise ValueError("overlap with reserved frozen evaluation ID set")
    idx=BM25Index(docs,title_boost=config.title_boost)
    stratified=defaultdict(list)
    for q in questions:
        if not q.question.strip():
            raise ValueError("empty TRAIN question")
        fp=_sha(q.question)
        if fp in blocked_hashes:
            raise ValueError("TRAIN question duplicates reserved eval text hash")
        support=set(q.supporting_doc_ids)
        if not support:
            raise ValueError("cannot compute TRAIN-only difficulty without labels")
        hits=idx.search(q.question,k=config.search_top_k,scope=q.scope)
        exposed={h.doc_id for h in hits}
        count=len(exposed & support)
        tier="direct" if count==len(support) else ("bridge" if count else "hard")
        key=hashlib.sha256(
            f"rsi-train-challenge:{q.question_id}:{fp}".encode()).hexdigest()[:24]
        stratified[tier].append(
            TrainingChallengeSeed(f"c-{key}",q.question_id,tier,fp,q.question))
    return dict(stratified)


def select_generation_challenges(
    stratified:dict[str,list[TrainingChallengeSeed]],*,seed:int,
    generation:int,desired:int,
    prior_source_ids:set[str]|None=None,
    prior_text_hashes:set[str]|None=None,
    near_duplicate_threshold:float=0.93)->list[TrainingChallengeSeed]:
    if generation<1 or desired<1:
        raise ValueError("generation and desired size must be positive")
    if not 0.8<near_duplicate_threshold<=1.0:
        raise ValueError("near duplicate threshold must be (0.8,1]")
    blocked_ids=prior_source_ids or set()
    blocked_hashes=prior_text_hashes or set()
    pool={}
    for tier in ("direct","bridge","hard"):
        items=[x for x in stratified.get(tier,[])
               if x.source_train_id not in blocked_ids
               and x.normalized_text_hash not in blocked_hashes]
        items.sort(key=lambda x:hashlib.sha256(
            f"{generation}|{seed}|{x.source_train_id}".encode()).hexdigest())
        pool[tier]=items
    selected=[]
    seen_ids=set()
    seen_hashes=set()
    def append_unique(candidate:TrainingChallengeSeed)->bool:
        if candidate.source_train_id in seen_ids or candidate.normalized_text_hash in seen_hashes:
            return False
        for other in selected:
            # Cheap conservative proxy; truly semantic paraphrases still
            # need a separate model/human near-duplicate audit.
            similarity=SequenceMatcher(
                None,_norm(other.question),_norm(candidate.question)).ratio()
            if similarity>=near_duplicate_threshold:
                return False
        selected.append(candidate)
        seen_ids.add(candidate.source_train_id)
        seen_hashes.add(candidate.normalized_text_hash)
        return True
    # Round-robin across tiers prevents the easy category from silently
    # monopolizing the next generation, but sparse tiers are allowed.
    cursors={tier:0 for tier in pool}
    while len(selected)<desired:
        progress=False
        for tier in ("hard","bridge","direct"):
            values=pool[tier]
            while cursors[tier]<len(values):
                item=values[cursors[tier]]
                cursors[tier]+=1
                if append_unique(item):
                    progress=True
                    break
            if len(selected)>=desired:break
        if not progress:break
    if len(selected)<desired:
        raise ValueError(f"insufficient non-overlapping TRAIN candidates: "
                         f"requested {desired}, found {len(selected)}")
    return selected


def freeze_challenger_queue(
    selected:list[TrainingChallengeSeed],out:Path,
    *,source_git_sha:str,seed:int,generation:int)->dict:
    if out.exists():raise FileExistsError(out)
    if not re.fullmatch("[0-9a-f]{40}",source_git_sha):
        raise ValueError("invalid pinned source commit")
    if not selected:raise ValueError("empty challenger queue")
    ids=[x.opaque_challenge_id for x in selected]
    if len(set(ids))!=len(ids):
        raise ValueError("opaque IDs are not unique")
    out.mkdir(parents=True,exist_ok=False)
    def dump_lines(path:Path,rows:list[dict]):
        with path.open("x",encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    challenge_path=out/"CHALLENGER_INPUT.jsonl"
    verifier_path=out/"VERIFIER_ONLY_MAP.jsonl"
    dump_lines(challenge_path,[x.challenger_view() for x in selected])
    dump_lines(verifier_path,[x.verifier_map() for x in selected])
    hash_file=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result={
        "status":"frozen_train_only_inputs",
        "model_driven_challenger_executed":False,
        "real_solver_rollouts_executed":False,
        "training_or_checkpoint_updates":0,
        "source_git_sha":source_git_sha,"seed":seed,"generation":generation,
        "split":"train",
        "n":len(selected),
        "difficulty_retrieval_proxy_counts":dict(Counter(
            x.difficulty_tier for x in selected)),
        "challenger_input_sha256":hash_file(challenge_path),
        "verifier_only_map_sha256":hash_file(verifier_path),
        "gold_fields_in_challenger_input":False,
        "caveat":"Retrieval proxy uses TRAIN labels offline; novel challenger "
                 "questions need independently verified TRAIN truth and cannot inherit source answer.",
    }
    (out/"QUEUE_MANIFEST.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result
