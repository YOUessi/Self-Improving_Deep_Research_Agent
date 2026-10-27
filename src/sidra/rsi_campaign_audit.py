"""Offline, fail-closed cross-generation RSI Challenger campaign auditor.

Reads only TRAIN-only queue files and optionally model-generated *UNLABELLED*
candidate outputs. No gold answers, DEV/eval-A, final-B or model weights are
loaded; outputs contain only aggregate metrics and cryptographic file hashes.

This is a provenance/integrity guard, NOT a novel-question truth verifier
and NOT evidence of RSI learning. The verifier map is opened solely inside
this trusted CPU audit, never forwarded to Challenger/Trainer.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from sidra.rsi_challenge_scheduler import _norm
from sidra.rsi_model_challenger import parse_challenge


_HEX40=re.compile(r"^[0-9a-f]{40}$")
_HEX64=re.compile(r"^[0-9a-f]{64}$")
_TIERS={"hard","bridge","direct"}
_ACCEPTED="unlabelled_candidate_pending_independent_verification"


def _sha_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path:Path)->str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return _sha_bytes(path.read_bytes())


def _read_object(path:Path)->dict:
    x=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x,dict):raise ValueError(f"expected JSON object: {path.name}")
    return x


def _jsonl(path:Path)->list[dict]:
    if not path.is_file():raise FileNotFoundError(path)
    lines=path.read_text(encoding="utf-8").splitlines()
    if not lines or len(lines)>10000:
        raise ValueError("empty or oversized TRAIN campaign file")
    rows=[]
    for raw in lines:
        if not raw.strip():raise ValueError("blank JSONL record")
        x=json.loads(raw)
        if not isinstance(x,dict):
            raise ValueError("non-object campaign record")
        rows.append(x)
    return rows


def _require_sha(value:Any,*,n:int=64)->str:
    pattern=_HEX40 if n==40 else _HEX64
    if not isinstance(value,str) or not pattern.fullmatch(value):
        raise ValueError("invalid campaign SHA")
    return value


def _similar(a:str,b:str)->float:
    return SequenceMatcher(None,_norm(a),_norm(b)).ratio()


def audit_train_campaign(
    queue_dirs:list[Path],
    *,candidate_dirs:list[Path]|None=None,
    near_duplicate_threshold:float=0.92,
)->dict:
    """Validate one contiguous TRAIN-only challenge campaign (>=1 generation).

    candidate_dirs, if present, must match queue_dirs by generation number,
    and valid model suggestions are independently re-parsed from saved
    question/reason text. This still cannot prove which model produced them.

    Returns aggregate counts ONLY. A valid *novel question* has ZERO training
    gold authority and must be independently labelled from TRAIN evidence.
    """
    if not queue_dirs or len({str(x.resolve()) for x in queue_dirs})!=len(queue_dirs):
        raise ValueError("missing or repeated generation queue")
    if not 0.8<near_duplicate_threshold<=1:
        raise ValueError("invalid near-duplicate threshold")
    candidate_dirs=candidate_dirs or []
    if candidate_dirs and len(candidate_dirs)!=len(queue_dirs):
        raise ValueError("candidate generation count must match queue generation count")

    campaigns=[]
    for path in queue_dirs:
        manifest=_read_object(path/"QUEUE_MANIFEST.json")
        if manifest.get("status")!="frozen_train_only_inputs" or (
            manifest.get("split")!="train" or
            manifest.get("gold_fields_in_challenger_input") is not False or
            manifest.get("model_driven_challenger_executed") is not False or
            manifest.get("real_solver_rollouts_executed") is not False or
            manifest.get("training_or_checkpoint_updates")!=0):
            raise ValueError("queue has untrusted, unfrozen or non-TRAIN origin")
        generation=manifest.get("generation")
        seed=manifest.get("seed")
        n=manifest.get("n")
        if (type(generation) is not int or generation<1 or
            type(seed) is not int or type(n) is not int or n<1):
            raise ValueError("invalid generation/seed/count in queue manifest")
        _require_sha(manifest.get("source_git_sha"),n=40)
        _require_sha(manifest.get("challenger_input_sha256"))
        _require_sha(manifest.get("verifier_only_map_sha256"))
        if _sha_file(path/"CHALLENGER_INPUT.jsonl")!=manifest["challenger_input_sha256"]:
            raise ValueError("challenger queue checksum mismatch")
        if _sha_file(path/"VERIFIER_ONLY_MAP.jsonl")!=manifest["verifier_only_map_sha256"]:
            raise ValueError("verifier map checksum mismatch")
        inputs=_jsonl(path/"CHALLENGER_INPUT.jsonl")
        maps=_jsonl(path/"VERIFIER_ONLY_MAP.jsonl")
        if len(inputs)!=n or len(maps)!=n:
            raise ValueError("queue/map count mismatch")
        by_id={}
        for row in inputs:
            if (set(row)!={"challenge_id","question"} or
                not isinstance(row["challenge_id"],str) or
                not isinstance(row["question"],str) or
                not 15<=len(row["question"].strip())<=800):
                raise ValueError("Challenger input contains unexpected fields or content")
            key=row["challenge_id"]
            if not key.startswith("c-") or key in by_id:
                raise ValueError("duplicate/nonopaque Challenger ID")
            by_id[key]=row["question"]
        seen_map=set()
        sources=set()
        tiers=Counter()
        for row in maps:
            if set(row)!={"challenge_id","source_train_question_id",
                          "text_sha256","tier"}:
                raise ValueError("verifier map has inconsistent trust-domain schema")
            key=row["challenge_id"]
            source=row["source_train_question_id"]
            if (key not in by_id or key in seen_map or
                not isinstance(source,str) or not source or source in sources or
                row["tier"] not in _TIERS):
                raise ValueError("bad queue ↔ trusted verifier identity mapping")
            if _sha_bytes(_norm(by_id[key]).encode("utf-8"))!=row["text_sha256"]:
                raise ValueError("verifier text hash disagrees with isolated prompt")
            seen_map.add(key);sources.add(source);tiers[row["tier"]]+=1
        if seen_map!=set(by_id):
            raise ValueError("verifier map lacks some Challenger IDs")
        if manifest.get("difficulty_retrieval_proxy_counts")!=dict(tiers):
            raise ValueError("difficulty proxy tier counts changed")
        campaigns.append({
            "generation":generation,"seed":seed,"path":path,
            "manifest":manifest,"inputs":by_id,
            "sources":sources,"tiers":tiers,
        })

    campaigns.sort(key=lambda v:v["generation"])
    numbers=[v["generation"] for v in campaigns]
    if numbers!=list(range(1,len(numbers)+1)):
        raise ValueError("non-contiguous generation campaign")
    if len({v["seed"] for v in campaigns})!=len(campaigns):
        raise ValueError("generation sampling seeds must differ")
    seen_ids=set()
    seen_sources=set()
    prior_questions=[]
    for campaign in campaigns:
        for opaque,question in campaign["inputs"].items():
            if opaque in seen_ids:
                raise ValueError("opaque Challenger ID reused across generations")
            if any(_similar(question,previous)>=near_duplicate_threshold
                   for previous in prior_questions):
                raise ValueError("near-duplicate TRAIN questions across generations")
            seen_ids.add(opaque);prior_questions.append(question)
        if seen_sources & campaign["sources"]:
            raise ValueError("TRAIN source question reused in later generation")
        seen_sources.update(campaign["sources"])

    checked_candidate_count=0
    valid_novel_count=0
    counts=Counter()
    accepted_questions=[]
    verified_candidate_generations=0
    source_queue_shas={v["generation"]:v["manifest"]["challenger_input_sha256"]
                       for v in campaigns}
    if candidate_dirs:
        by_generation={}
        for path in candidate_dirs:
            summary=_read_object(path/"CHALLENGER_SUMMARY.json")
            generation=summary.get("source_queue_generation")
            if type(generation) is not int or generation in by_generation:
                raise ValueError("missing or repeated candidate generation")
            by_generation[generation]=(path,summary)
        if set(by_generation)!=set(numbers):
            raise ValueError("candidate and queue generation numbers differ")
        for campaign in campaigns:
            generation=campaign["generation"]
            path,summary=by_generation[generation]
            if (summary.get("source_split")!="train" or
                summary.get("source_queue_sha256")!=source_queue_shas[generation] or
                summary.get("backend_local_only") is not True or
                summary.get("challenger_gold_label_access") is not False or
                summary.get("independent_verifier_approved_novel_tasks")!=0 or
                summary.get("trainable_new_gold_labels_produced")!=0 or
                summary.get("optimizer_updates")!=0):
                raise ValueError("candidate manifest breaks TRAIN isolation")
            original=campaign["inputs"]
            candidates=_jsonl(path/"UNLABELLED_CHALLENGER_CANDIDATES.jsonl")
            if len(candidates)!=summary.get("n_attempted") or len(candidates)>len(original):
                raise ValueError("model Challenger candidate count mismatch")
            these=Counter()
            for idx,row in enumerate(candidates):
                if set(row)!={"challenge_id","proposal_question","reason",
                             "verdict","model_completion_sha256"}:
                    raise ValueError("model output has unsupported fields")
                key=row["challenge_id"]
                if key not in original or key in these:
                    raise ValueError("candidate identity missing or duplicated")
                these[key]+=1  # one response per unique frozen TRAIN task
                if not isinstance(row["verdict"],str):
                    raise ValueError("invalid candidate verdict type")
                if row["verdict"]==_ACCEPTED:
                    if (not isinstance(row["proposal_question"],str) or
                        not isinstance(row["reason"],str) or
                        not isinstance(row["model_completion_sha256"],str)):
                        raise ValueError("accepted candidate has malformed fields")
                    _require_sha(row["model_completion_sha256"])
                    # Cross-generation novelty and structural validity must
                    # be recomputed offline, not trusted from model output.
                    rechecked=parse_challenge(
                        json.dumps({"question":row["proposal_question"],
                                    "reason":row["reason"]},ensure_ascii=False),
                        challenge_id=key,source_question=original[key],
                        prior_proposals=accepted_questions,
                    )
                    if rechecked.verdict!=_ACCEPTED:
                        raise ValueError("model candidate independently rejected: "
                                         +rechecked.verdict)
                    accepted_questions.append(row["proposal_question"])
                    valid_novel_count+=1
                elif row["proposal_question"] or row["reason"]:
                    raise ValueError("rejected candidate must contain no candidate question")
                counts[row["verdict"]]+=1
                checked_candidate_count+=1
            actual=Counter(x["verdict"] for x in candidates)
            if (summary.get("verdict_histogram")!=dict(actual) or
                summary.get("n_structurally_valid_novel_candidates")!=actual.get(_ACCEPTED,0) or
                summary.get("n_attempted")!=len(candidates)):
                raise ValueError("reported candidate totals disagree with saved records")
            verified_candidate_generations+=1

    return {
        "status":"completed_gold_blind_multi_generation_campaign_audit",
        "queue_generation_count":len(campaigns),
        "queue_generations":numbers,
        "queue_seeds":[v["seed"] for v in campaigns],
        "frozen_train_questions_total":len(seen_ids),
        "unique_train_sources_total":len(seen_sources),
        "difficulty_proxy_totals":dict(sum((v["tiers"] for v in campaigns),Counter())),
        "cross_generation_near_duplicate_threshold":near_duplicate_threshold,
        "queue_manifests_sha256_by_generation":{
            str(v["generation"]):_sha_file(v["path"]/"QUEUE_MANIFEST.json")
            for v in campaigns},
        "candidate_generation_count_audited":verified_candidate_generations,
        "candidate_responses_checked":checked_candidate_count,
        "novel_structurally_valid_unlabelled_candidates":valid_novel_count,
        "candidate_verdict_totals":dict(counts),
        "novel_tasks_independently_fact_labelled":0,
        "new_trainable_tasks_granted":0,
        "actual_model_updates_verified_by_this_audit":0,
        "evaluation_gold_used":False,
        "model_origin_proven_by_this_audit":False,
        "multiple_optimizer_seed_replicates_verified":False,
        "multi_generation_RSI_completed":False,
        "caveat":"Checking three frozen TRAIN question queues is NOT three "
                 "RSI training generations. Candidate question syntactic "
                 "validity does not supply any independent ground truth; "
                 "must separately audit model origin, tool traces, evidence "
                 "entailment, actual optimizer updates and eval-A.",
    }
