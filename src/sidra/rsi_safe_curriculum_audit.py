"""E-026 independent, fail-closed audit of a frozen original-TRAIN curriculum.

The curriculum creator's word is not sufficient: replay file sha checks,
join learner questions back to canonical TRAIN-only originals through the
private Verifier map, recompute retrieval-tier proxy using TRAIN gold only,
and reject tampering, global near-duplicates, holdout collisions and DEV ID
collisions. Results are aggregate only, never model-facing labels.

This audit is NOT online Solver inference, Trainer optimizer activity or RSI.
"""
from __future__ import annotations

import json
import hashlib
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.experiment import sha256_file
from sidra.io import load_dataset
from sidra.rsi_challenge_scheduler import stratify_train_questions
from sidra.rsi_public_split_ids import frozen_public_dev_ids
from sidra.rsi_safe_curriculum import replay_history_for_exclusions


def _read_obj(path:Path)->dict:
    x=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(x,dict):raise ValueError("expected JSON object")
    return x


def _read_rows(path:Path)->list[dict]:
    if not path.is_file():raise FileNotFoundError(path)
    raw=path.read_text(encoding="utf-8").splitlines()
    if not raw:raise ValueError("empty immutable curriculum partition")
    result=[]
    for line in raw:
        if not line.strip():raise ValueError("blank curriculum line")
        value=json.loads(line)
        if not isinstance(value,dict):raise ValueError("non-object curriculum row")
        result.append(value)
    return result


def _same_digest(path:Path,expected:str)->None:
    if (not isinstance(expected,str) or len(expected)!=64
        or sha256_file(path)!=expected):
        raise ValueError("immutable artifact SHA-256 mismatch")


def _near(x:str,y:str)->float:
    return SequenceMatcher(None," ".join(x.casefold().split()),
                           " ".join(y.casefold().split())).ratio()


def audit_safe_original_train_campaign(
    train_data:Path,public_dev_id_manifest:Path,history_paths:list[Path],
    campaign_dir:Path,*,source_sha:str,
    config:AgentConfig|None=None,near_duplicate_threshold:float=0.93)->dict:
    if not 0.8<near_duplicate_threshold<=1:raise ValueError("bad duplicate threshold")
    config=config or AgentConfig(max_steps=6,max_tool_calls=5,search_top_k=3)
    manifest=_read_obj(campaign_dir/"CAMPAIGN_MANIFEST.json")
    if manifest.get("source_git_sha")!=source_sha or (
        manifest.get("stage")!="original_TRAIN_labelled_curriculum_only" or
        manifest.get("challenger_model_generated_new_tasks_this_run")!=0 or
        manifest.get("actual_optimizer_updates_this_run")!=0 or
        manifest.get("actual_RSI_checkpoint_generations_this_run")!=0 or
        manifest.get("no_dev_evalA_or_finalB_gold_accessed") is not True):
        raise ValueError("curriculum origin or stage/role claim corrupted")
    docs,questions=load_dataset(train_data)
    ids={q.question_id:q for q in questions}
    if (len(ids)!=len(questions)
        or not questions or any(q.split!="train" for q in questions)):
        raise ValueError("invalid original TRAIN-only source")
    if manifest.get("original_train_total")!=len(ids):
        raise ValueError("TRAIN source size mismatch")
    _same_digest(train_data/"questions.jsonl",
                 manifest.get("original_train_questions_sha256"))
    _same_digest(train_data/"documents.jsonl",
                 manifest.get("original_train_documents_sha256"))
    dev_ids,dev_sha=frozen_public_dev_ids(public_dev_id_manifest)
    if (manifest.get("public_dev_ID_only_manifest_sha256")!=dev_sha
        or manifest.get("public_dev_ID_count")!=len(dev_ids)
        or bool(set(ids)&dev_ids)):
        raise ValueError("frozen DEV ID-only manifest mismatch or leakage")
    excluded,reports=replay_history_for_exclusions(
        train_data,history_paths,config=config,source_sha=source_sha)
    if (len(excluded)!=manifest.get("prior_real_rollout_questions_excluded")
        or reports!=manifest.get("prior_real_rollout_audits")):
        raise ValueError("replayed TRAIN exposure history disagrees with manifest")
    tiers=stratify_train_questions(questions,docs,config,forbidden_ids=dev_ids)
    by_source={x.source_train_id:x for xs in tiers.values() for x in xs}
    if len(by_source)!=len(ids):raise ValueError("retrieval-tier source cardinality")

    holdout_file=campaign_dir/"VERIFIER_ONLY_ORIGINAL_TRAIN_HOLDOUT_IDS.jsonl"
    _same_digest(holdout_file,manifest.get(
        "independent_original_TRAIN_holdout_verifier_only_sha256"))
    holdout=_read_rows(holdout_file)
    if len(holdout)!=manifest.get("independent_original_TRAIN_holdout_n"):
        raise ValueError("frozen TRAIN holdout size mismatch")
    global_ids=set(excluded)
    all_text=[]
    holdout_ids=set()
    for item in holdout:
        if set(item)!={"original_train_question_id","normalized_question_sha256"}:
            raise ValueError("holdout contains unexpected label fields")
        qid=item["original_train_question_id"]
        if qid in global_ids or qid in dev_ids or qid not in ids:
            raise ValueError("TRAIN holdout overlaps exposure, DEV or unknown source")
        canonical=by_source[qid]
        if item["normalized_question_sha256"]!=canonical.normalized_text_hash:
            raise ValueError("original TRAIN holdout text SHA disagrees")
        if any(_near(canonical.question,x)>=near_duplicate_threshold
               for x in all_text):
            raise ValueError("TRAIN holdout contains near duplicate")
        holdout_ids.add(qid);global_ids.add(qid)
        all_text.append(canonical.question)
    if len(holdout_ids)!=len(holdout):
        raise ValueError("duplicate TRAIN holdout ID")
    cohorts=manifest.get("cohorts")
    if not isinstance(cohorts,list) or not cohorts:
        raise ValueError("empty curriculum campaign")
    all_cohort_ids=set()
    count=0
    for n,cohort in enumerate(cohorts,1):
        if (cohort.get("planned_curriculum_cohort")!=n or
            cohort.get("selection_seed")!=43+n-1 and
             cohort.get("selection_seed")!=manifest.get("cohorts",[])[0].get("selection_seed",-100)+n-1):
            raise ValueError("nonconsecutive curriculum cohort or seed mismatch")
        if any(cohort.get(x)!=0 for x in (
            "model_solver_rollouts_in_this_campaign",
            "real_optimizer_updates_in_this_campaign",
            "checkpoint_generations_completed",
            "semantic_verifier_admitted_novel_labels")):
            raise ValueError("unverified training/new labels embedded in curriculum")
        if cohort.get("actual_model_challenger_executed") is not False:
            raise ValueError("false model-driven challenger claim")
        directory=campaign_dir/f"cohort_{n:02d}"
        learner_path=directory/"SOLVER_INPUT.jsonl"
        map_path=directory/"VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl"
        _same_digest(learner_path,cohort.get("solver_input_sha256"))
        _same_digest(map_path,cohort.get("verifier_only_map_sha256"))
        learner=_read_rows(learner_path)
        trusted=_read_rows(map_path)
        if len(learner)!=len(trusted) or len(learner)!=cohort.get("n_original_train_questions"):
            raise ValueError("model-to-verifier row count disagreement")
        map_by_id={}
        for item in trusted:
            if set(item)!={"challenge_id","source_train_question_id",
                           "text_sha256","tier"}:
                raise ValueError("verifier map has illegal extra label fields")
            if item["challenge_id"] in map_by_id:
                raise ValueError("duplicate Verifier opaque ID")
            map_by_id[item["challenge_id"]]=item
        seen_ids=set()
        counter=Counter()
        for item in learner:
            if set(item)!={"challenge_id","question"}:
                raise ValueError("Solver received an unexpected label/field")
            opaque=item["challenge_id"]
            if opaque not in map_by_id or opaque in seen_ids:
                raise ValueError("Solver ↔ Verifier opaque ID mismatch")
            seen_ids.add(opaque)
            trusted_row=map_by_id[opaque]
            qid=trusted_row["source_train_question_id"]
            canonical=by_source.get(qid)
            if (canonical is None or qid in global_ids
                or qid in dev_ids or qid in excluded
                or opaque!=canonical.opaque_challenge_id
                or trusted_row["text_sha256"]!=canonical.normalized_text_hash
                or trusted_row["tier"]!=canonical.difficulty_tier
                or item["question"]!=canonical.question):
                raise ValueError("original TRAIN identity, question or tier forgery")
            if any(_near(item["question"],x)>=near_duplicate_threshold
                   for x in all_text):
                raise ValueError("cross-cohort/holdout near-duplicate question")
            all_text.append(item["question"])
            global_ids.add(qid);all_cohort_ids.add(qid)
            counter[canonical.difficulty_tier]+=1
            count+=1
        if seen_ids!=set(map_by_id):
            raise ValueError("missing Solver question / Verifier mapping")
        if dict(counter)!=cohort.get("difficulty_proxy"):
            raise ValueError("TRAIN difficulty proxy count mismatch")
    if count!=manifest.get("total_original_train_tasks_prepared"):
        raise ValueError("campaign total count mismatch")
    return {
        "stage":"independent_original_TRAIN_curriculum_audit",
        "status":"PASS_SAFE_ORIGINAL_TRAIN_PROMPT_BANK_NOT_TRAINING",
        "source_git_sha":source_sha,
        "manifest_sha256":sha256_file(campaign_dir/"CAMPAIGN_MANIFEST.json"),
        "original_train_n":len(ids),
        "public_DEV_exclusion_n":len(dev_ids),
        "historically_exposed_original_TRAIN_n":len(excluded),
        "independent_TRAIN_holdout_n":len(holdout_ids),
        "distinct_original_TRAIN_curriculum_tasks":count,
        "candidate_cohorts":len(cohorts),
        "corpus_and_BM25_replay_checks_passed":True,
        "all_original_answers_and_support_ids_kept_verifier_only":True,
        "novel_challenger_labels_admitted":0,
        "optimizer_updates_proven":0,
        "multi_generation_RSI_proven":False,
        "no_finalB_file_read":True,
        "critical_caveat":(
            "Source/role integrity gate, not a model-origin attestation, "
            "verified semantic entailment, independent Solver rollout or RSI proof."
        ),
    }
