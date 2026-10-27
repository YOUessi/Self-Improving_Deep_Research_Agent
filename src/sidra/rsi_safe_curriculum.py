"""E-026: original-TRAIN-labelled curriculum while novel Challenger labels remain quarantined.

This is a *safe way to begin actual Solver/Trainer work*, NOT a novel-label
Challenger, optimizer run, or validated RSI generation. Old real TRAIN
trajectories are replay-audited and used only to exclude already exposed
question IDs. The train answers never enter model-facing JSONL. The separate
frozen original-TRAIN holdout must NEVER be used for policy updates.

Uses only the original HotpotQA TRAIN dataset and M4 public DEV **ID-only**
manifest. In particular NO sealed final-B160 file is opened.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from difflib import SequenceMatcher
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.experiment import sha256_file
from sidra.io import iter_jsonl,load_dataset
from sidra.rsi_challenge_scheduler import (
    TrainingChallengeSeed,select_generation_challenges,
    stratify_train_questions,
)
from sidra.rsi_trainer_admission import frozen_public_dev_ids
from sidra.schema import Question
from sidra.rsi_trace_audit import audit_tool_trace
from scripts.audit_rsi_train_rollouts import verify_train_rollout_file


def _norm(text:str)->str:
    return " ".join(text.casefold().split())


def _key(text:str)->str:
    return hashlib.sha256(_norm(text).encode()).hexdigest()


def _priority(seed:int,key:str)->str:
    return hashlib.sha256(f"train:holdout:{seed}:{key}".encode()).hexdigest()


def _similar(first:str,second:str)->float:
    return SequenceMatcher(None,_norm(first),_norm(second)).ratio()


def replay_history_for_exclusions(
    train_data:Path, history_paths:list[Path], *,
    config:AgentConfig,source_sha:str)->tuple[set[str],list[dict]]:
    """Fail if even one imported TRAIN trajectory is tampered or non-TRAIN.

    A correct *replay* cannot certify the source policy was the claimed model.
    Consequently historical traces only exclude contaminated/exposed tasks;
    they are never admitted as optimizer samples by this function.
    """
    if not history_paths or len(set(history_paths))!=len(history_paths):
        raise ValueError("at least one unique real TRAIN history is required")
    if not re.fullmatch("[0-9a-f]{40}",source_sha):
        raise ValueError("source SHA must pin a real repository commit")
    exposed=set()
    audits=[]
    for path in history_paths:
        if not path.is_file():raise FileNotFoundError(path)
        report=verify_train_rollout_file(
            train_data,path,config=config,source_git_sha=source_sha)
        if (report["n_episodes"]!=report["structurally_valid_tool_traces"]
            or report["reported_reward_mismatches"]
            or report["original_train_reward_exact_matches"]!=report["n_episodes"]):
            raise ValueError("historical TRAIN tool or reward audit did not pass")
        for envelope in iter_jsonl(path):
            if not isinstance(envelope,dict) or (
                envelope.get("question_id",envelope.get("qid"))!=
                envelope["trace"]["question_id"]):
                raise ValueError("historical question identity mismatch")
            exposed.add(envelope["trace"]["question_id"])
        audits.append({
            "rollout_file_sha256":report["rollout_jsonl_sha256"],
            "n_episodes":report["n_episodes"],
            "structurally_valid_tool_traces":report["structurally_valid_tool_traces"],
            "strict_training_successes":report["independently_validated_strict_successes"],
            "used_as_optimizer_examples":False,
            "model_origin_independently_attested":False,
        })
    return exposed,audits


def build_safe_original_train_campaign(
    train_data:Path,public_dev_id_manifest:Path,history_paths:list[Path],
    out:Path,*,source_sha:str,seed:int=43,
    generations:int=3,per_generation:int=24,holdout_n:int=32,
    near_duplicate_threshold:float=0.93,
    config:AgentConfig|None=None)->dict:
    """Produce isolated learner prompts and *trusted* original-TRAIN identity maps.

    A generation here is a *planned curriculum cohort*, NOT an observed
    successive checkpoint generation; mandatory names reflect that distinction.
    """
    if out.exists():raise FileExistsError("immutable campaign output already exists")
    if (min(generations,per_generation,holdout_n)<1
        or not 0.8<near_duplicate_threshold<=1):
        raise ValueError("invalid curriculum, holdout, or duplicate constraints")
    config=config or AgentConfig(max_steps=6,max_tool_calls=5,search_top_k=3)
    # The dev function opens ONLY the public frozen ID selection manifest.
    dev_ids,dev_id_manifest_sha=frozen_public_dev_ids(public_dev_id_manifest)
    docs,questions=load_dataset(train_data)
    if any(q.split!="train" for q in questions):
        raise ValueError("all curriculum sources must be TRAIN")
    id_map={q.question_id:q for q in questions}
    if len(id_map)!=len(questions) or set(id_map)&dev_ids:
        raise ValueError("duplicate TRAIN ID or collision with frozen DEV")
    if len({_key(q.question) for q in questions})!=len(questions):
        raise ValueError("duplicate TRAIN question wording")
    exposed,history_reports=replay_history_for_exclusions(
        train_data,history_paths,config=config,source_sha=source_sha)
    if not exposed<=set(id_map):
        raise ValueError("history contains out-of-TRAIN IDs")
    stratified=stratify_train_questions(
        questions,docs,config,forbidden_ids=dev_ids)
    all_seeds={x.source_train_id:x for bucket in stratified.values() for x in bucket}
    if len(all_seeds)!=len(questions):
        raise ValueError("not exactly one retrieval-tier classification per TRAIN ID")
    # Holdout is selected independently of the model; its IDs are kept entirely
    # in a verifier-only file, *never* in a model prompt or optimizer manifest.
    eligible=[x for x in all_seeds.values() if x.source_train_id not in exposed]
    eligible.sort(key=lambda x:(_priority(seed,x.source_train_id),x.source_train_id))
    holdout=[]
    for x in eligible:
        if all(_similar(x.question,y.question)<near_duplicate_threshold for y in holdout):
            holdout.append(x)
        if len(holdout)==holdout_n:break
    if len(holdout)!=holdout_n:raise ValueError("insufficient disjoint TRAIN holdout")
    private_holdout_ids={x.source_train_id for x in holdout}
    holdout_hashes={x.normalized_text_hash for x in holdout}
    prior_ids=set(exposed)|private_holdout_ids
    prior_hashes=holdout_hashes.copy()
    forbidden_texts=[x.question for x in holdout]
    cohorts=[]
    for generation in range(1,generations+1):
        # Globally exclude near-duplicates with the independent holdout and
        # every previously frozen cohort, not only duplicates within a cohort.
        filtered={}
        for tier,items in stratified.items():
            filtered[tier]=[
                x for x in items
                if x.source_train_id not in prior_ids
                and x.normalized_text_hash not in prior_hashes
                and all(_similar(x.question,text)<near_duplicate_threshold
                        for text in forbidden_texts)
            ]
        selected=select_generation_challenges(
            filtered,seed=seed+generation-1,generation=generation,
            desired=per_generation,prior_source_ids=prior_ids,
            prior_text_hashes=prior_hashes,
            near_duplicate_threshold=near_duplicate_threshold)
        prior_ids.update(x.source_train_id for x in selected)
        prior_hashes.update(x.normalized_text_hash for x in selected)
        forbidden_texts.extend(x.question for x in selected)
        cohorts.append({
            "curriculum_cohort":generation,
            "seed":seed+generation-1,
            "selected":selected,
            "difficulty_proxy":dict(Counter(x.difficulty_tier for x in selected)),
        })
    # Final independence check BEFORE writing any output.
    if any(c["selected"] is None for c in cohorts):
        raise ValueError("invalid cohort")
    if not re.fullmatch("[0-9a-f]{40}",source_sha):
        raise ValueError("invalid pinned GitHub SHA")
    all_ids=[x.source_train_id for c in cohorts for x in c["selected"]]
    if (len(set(all_ids))!=len(all_ids) or
        set(all_ids)&(exposed|private_holdout_ids|dev_ids)):
        raise ValueError("cross-cohort/train-holdout/DEV identity leakage")
    out.mkdir(parents=True,exist_ok=False)

    def jsonl(path:Path,rows:list[dict])->str:
        with path.open("x",encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row,sort_keys=True,ensure_ascii=False)+"\n")
        return sha256_file(path)

    holdout_file=out/"VERIFIER_ONLY_ORIGINAL_TRAIN_HOLDOUT_IDS.jsonl"
    holdout_digest=jsonl(holdout_file,[
        {"original_train_question_id":x.source_train_id,
         "normalized_question_sha256":x.normalized_text_hash}
        for x in holdout])
    model_paths=set()
    cohort_report=[]
    for cohort in cohorts:
        g=cohort["curriculum_cohort"]
        directory=out/f"cohort_{g:02d}"
        directory.mkdir()
        selected=cohort["selected"]
        # Learner gets opaque ID and original TRAIN question text only:
        # no source ID, answer, supporting doc IDs, reward, tier or holdout.
        solver=directory/"SOLVER_INPUT.jsonl"
        solver_digest=jsonl(solver,[x.challenger_view() for x in selected])
        model_paths.add(solver)
        trusted=directory/"VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl"
        trusted_digest=jsonl(trusted,[x.verifier_map() for x in selected])
        cohort_report.append({
            "planned_curriculum_cohort":g,
            "selection_seed":cohort["seed"],
            "n_original_train_questions":len(selected),
            "difficulty_proxy":cohort["difficulty_proxy"],
            "solver_input_sha256":solver_digest,
            "verifier_only_map_sha256":trusted_digest,
            "actual_model_challenger_executed":False,
            "model_solver_rollouts_in_this_campaign":0,
            "real_optimizer_updates_in_this_campaign":0,
            "checkpoint_generations_completed":0,
            "semantic_verifier_admitted_novel_labels":0,
        })
    if holdout_file in model_paths:raise AssertionError("holdout output leak")
    result={
        "stage":"original_TRAIN_labelled_curriculum_only",
        "source_git_sha":source_sha,
        "original_train_questions_sha256":sha256_file(train_data/"questions.jsonl"),
        "original_train_documents_sha256":sha256_file(train_data/"documents.jsonl"),
        "public_dev_ID_only_manifest_sha256":dev_id_manifest_sha,
        "public_dev_ID_count":len(dev_ids),
        "original_train_total":len(questions),
        "prior_real_rollout_questions_excluded":len(exposed),
        "prior_real_rollout_audits":history_reports,
        "independent_original_TRAIN_holdout_n":holdout_n,
        "independent_original_TRAIN_holdout_verifier_only_sha256":holdout_digest,
        "cohorts":cohort_report,
        "total_original_train_tasks_prepared":len(all_ids),
        "train_original_label_provenance":"canonical HotpotQA TRAIN only; no newly invented labels",
        "no_dev_evalA_or_finalB_gold_accessed":True,
        "challenger_model_generated_new_tasks_this_run":0,
        "actual_optimizer_updates_this_run":0,
        "actual_RSI_checkpoint_generations_this_run":0,
        "independent_optimizer_seeds_run_this_campaign":0,
        "safe_training_claim":"prompt-only curriculum; run independent solver and verifier before training",
        "critical_caveat":(
            "Curriculum cohorts are NOT independently trained RSI generations. "
            "No invented Qwen3-8B challenge receives an inherited original answer. "
            "Real model-origin attestation and supported answer-sentence verification "
            "remain mandatory before training; no sealed final-B files opened."
        ),
    }
    (out/"CAMPAIGN_MANIFEST.json").write_text(
        json.dumps(result,sort_keys=True,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return result
