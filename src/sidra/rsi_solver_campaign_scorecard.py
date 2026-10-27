"""E-028: independent cross-cohort original-TRAIN Solver scorecard.

The Verifier reruns E-027's full saved transcript/BM25/token/seed checks for
EVERY cohort. Only aggregate training-stage counts leave the verifier. This
is scouting, not agent policy-gradient training and not RSI.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from sidra.experiment import sha256_file
from sidra.io import iter_jsonl
from sidra.rsi_solver_cohort import _check_path, _json, _source_map, audit_solver_run


def combine_verified_scouts(items: list[tuple[dict, dict]], *, planned_cohorts: int) -> dict:
    """Pure post-verifier reducer; fail closed on missing/invalid evidence.

    Each (run_manifest, independent_verifier_report) pair must already have
    passed audit_solver_run. This helper independently checks public counters
    and prevents cohort/seed/checkpoint mismatches across separate runs.
    """
    if not items or len(items) != planned_cohorts or planned_cohorts < 2:
        raise ValueError("all pre-registered independent cohorts are required")
    expected = list(range(1, planned_cohorts + 1))
    cohorts = [run.get("cohort") for run, _ in items]
    if cohorts != expected:
        raise ValueError("missing, duplicated or reordered cohort")
    seeds = [run.get("seed") for run, _ in items]
    if (not all(type(seed) is int for seed in seeds)
        or len(set(seeds)) != len(seeds)):
        raise ValueError("missing or repeated Solver sampling seed")
    consistency_fields = ("model_id", "model_checkpoint_sha256_user_supplied",
                          "decoder", "temperature", "top_p", "config")
    reference = {field: items[0][0].get(field) for field in consistency_fields}
    checkpoint = reference["model_checkpoint_sha256_user_supplied"]
    if not isinstance(checkpoint, str) or not re.fullmatch("[0-9a-f]{64}", checkpoint):
        raise ValueError("malformed checkpoint SHA")
    if not isinstance(reference["model_id"], str) or not reference["model_id"]:
        raise ValueError("missing model identifier")
    guard_enabled = items[0][0].get("observed_read_before_finish_guard", False)
    if type(guard_enabled) is not bool:
        raise ValueError("invalid rule guard role flag")
    total = Counter()
    tiers: dict[str, Counter] = {}
    per_cohort = []
    digests = set()
    for run, audit in items:
        if any(run.get(k) != v for k, v in reference.items()):
            raise ValueError("decoder/model/checkpoint/tool policy changed between cohorts")
        if (type(run.get("observed_read_before_finish_guard",False)) is not bool
            or run.get("observed_read_before_finish_guard",False) != guard_enabled
            or audit.get("observed_read_guard_declared",False) != guard_enabled):
            raise ValueError("observed READ guard arm changed across cohorts")
        if (run.get("stage") != "actual_model_solver_tool_rollouts_NOT_training"
            or run.get("TRAIN_gold_loaded_or_graded_by_solver") is not False
            or run.get("on_policy_training_distribution_proven") is not False
            or run.get("optimizer_updates") != 0
            or run.get("multi_generation_RSI_proven") is not False):
            raise ValueError("model role boundary or training claim invalid")
        n = audit.get("n")
        if type(n) is not int or n < 1 or n != run.get("n_episodes"):
            raise ValueError("unmatched independently verified number of episodes")
        if (audit.get("tool_replay_valid") != n
            or audit.get("gold_read_only_in_separate_verifier") is not True
            or audit.get("model_provenance_independently_attested") is not False
            or audit.get("training_updates_this_run") != 0
            or audit.get("optimizer_eligible") is not False
            or audit.get("multi_generation_RSI_proven") is not False
            or audit.get("solver_traces_sha256") != run.get("traces_sha256")
            or audit.get("model_actions_sha256") != run.get("model_action_io_sha256")):
            raise ValueError("independent verifier rejected a required integrity bound")
        io = audit.get("model_action_io_replay")
        if (not isinstance(io, dict)
            or io.get("verified_episode_IO_count") != n
            or io.get("verified_response_count") != run.get("model_calls")
            or io.get("all_model_contexts_and_seeds_match_tool_replay") is not True):
            raise ValueError("model-action transcript was not independently verified")
        digest = (run.get("traces_sha256"), run.get("model_action_io_sha256"))
        if digest in digests:
            raise ValueError("duplicate Solver replay across cohorts")
        digests.add(digest)
        by_tier = audit.get("difficulty_tier_aggregate")
        if not isinstance(by_tier, dict) or set(by_tier) != {"direct", "bridge", "hard"}:
            raise ValueError("missing pre-registered three-tier TRAIN breakdown")
        for tier in ("direct", "bridge", "hard"):
            bucket = by_tier[tier]
            for name in ("n", "strict", "answer_correct"):
                count = bucket.get(name)
                if type(count) is not int or count < 0:
                    raise ValueError("invalid aggregate QA counter")
                tiers.setdefault(tier, Counter())[name] += count
            if not 0 <= bucket["strict"] <= bucket["answer_correct"] <= bucket["n"]:
                raise ValueError("inconsistent answer/evidence counts")
        if sum(by_tier[t]["n"] for t in by_tier) != n:
            raise ValueError("tier totals do not match Solver episodes")
        strict = audit.get("strict_original_TRAIN_successes")
        if (type(strict) is not int
            or strict != sum(by_tier[t]["strict"] for t in by_tier)):
            raise ValueError("contradictory strict verified successes")
        finish = audit.get("legal_finished_answers")
        nonerror = audit.get("nonerror_terminations")
        if (type(finish) is not int or type(nonerror) is not int
            or not 0 <= finish <= nonerror <= n):
            raise ValueError("invalid finish/non-error totals")
        recall = audit.get("mean_gold_support_read_recall")
        if type(recall) not in (float, int) or not 0 <= recall <= 1:
            raise ValueError("invalid original TRAIN support exposure")
        total.update({"episodes": n, "answer_correct": sum(
            by_tier[t]["answer_correct"] for t in by_tier),
            "strict": strict, "finish": finish, "nonerror": nonerror,
            "model_calls": run["model_calls"],
            "prompt_tokens": run["prompt_tokens"],
            "completion_tokens": run["completion_tokens"]})
        total["weighted_read_support_recall"] += recall * n
        per_cohort.append({
            "cohort": run["cohort"], "sampling_seed": run["seed"],
            "episodes": n, "answer_correct": sum(
                by_tier[t]["answer_correct"] for t in by_tier),
            "strict": strict, "tool_legal_replayed": audit["tool_replay_valid"],
            "model_calls": run["model_calls"], "finish": finish,
            "mean_gold_support_read_recall": recall,
        })
    n = total["episodes"]
    correct = total["answer_correct"]
    strict = total["strict"]
    return {
        "stage": "independent_cross_cohort_original_TRAIN_scout_NOT_RSI",
        "cohorts": per_cohort, "pre_registered_cohort_count": planned_cohorts,
        "actual_distinct_original_TRAIN_questions": n,
        "model_identifier": reference["model_id"],
        "claimed_checkpoint_sha256_NOT_serving_attestation": checkpoint,
        "independent_model_origin_attestation": False,
        "observed_READ_before_FINISH_guard_enabled": guard_enabled,
        "unique_solver_sampling_seeds": seeds,
        "model_calls": total["model_calls"],
        "prompt_tokens": total["prompt_tokens"],
        "completion_tokens": total["completion_tokens"],
        "answer_correct_count": correct,
        "answer_exact_match": round(correct / n, 6),
        "strict_answer_plus_gold_passage_evidence_count": strict,
        "strict_answer_plus_evidence_rate": round(strict / n, 6),
        "answered_correctly_but_not_strictly_verified": correct - strict,
        "finish_count": total["finish"],
        "nonerror_count": total["nonerror"],
        "mean_support_read_recall": round(total["weighted_read_support_recall"] / n, 6),
        "difficulty_proxy_aggregate": {name: dict(tiers[name])
                                       for name in ("direct", "bridge", "hard")},
        "tool_and_model_transcripts_independently_replayed": True,
        "train_gold_used_only_by_offline_verifier": True,
        "dev_evalA_finalB_gold_used": False,
        "novel_Challenger_labels_approved": 0,
        "real_optimizer_updates": 0,
        "real_model_generations": 0,
        "independent_training_seeds_completed": 0,
        "eligible_for_online_GRPO_training_from_these_files": False,
        "why_not_train": (
            "One guided-vLLM episode per TRAIN task provides no group-wise "
            "same-question reward variance. The constrained behavior distribution "
            "has no verified matching policy logprob; reported checkpoint/model "
            "serving origin is not cryptographically attested. Saved trajectories "
            "are audit/scouting artifacts, not on-policy optimizer inputs."
        ),
        "conclusion": (
            "Measured real TRAIN solver/citation bottleneck; NOT improved QA, "
            "model post-training, multi-seed RL or self-improving RSI."
        ),
    }


def audit_cross_cohort_solver_campaign(
    *, train_data: Path, campaign: Path, packs: list[Path],
    runs: list[Path], source_sha: str, out: Path
) -> dict:
    """Create-only aggregate report; re-run every independent offline verifier.

    Never expose original TRAIN identity-to-label mapping in the output.
    """
    for path in (train_data, campaign, out, *packs, *runs):
        _check_path(path)
    if out.exists():
        raise FileExistsError(out)
    manifest = _json(campaign / "CAMPAIGN_MANIFEST.json")
    if manifest.get("stage") != "original_TRAIN_labelled_curriculum_only":
        raise ValueError("not a frozen original TRAIN curriculum")
    if manifest.get("source_git_sha") != source_sha:
        raise ValueError("campaign source commit mismatch")
    entries = manifest.get("cohorts")
    if (not isinstance(entries, list) or len(entries) < 2
        or len(packs) != len(entries) or len(runs) != len(entries)):
        raise ValueError("all frozen cohorts must be audited, never a subset")
    holdout_path = campaign / "VERIFIER_ONLY_ORIGINAL_TRAIN_HOLDOUT_IDS.jsonl"
    if (sha256_file(holdout_path)
        != manifest.get("independent_original_TRAIN_holdout_verifier_only_sha256")):
        raise ValueError("frozen original TRAIN holdout altered")
    holdout_ids = {x["original_train_question_id"]
                   for x in iter_jsonl(holdout_path)}
    if len(holdout_ids) != manifest.get("independent_original_TRAIN_holdout_n"):
        raise ValueError("duplicate/missing heldout IDs")
    seen_ids = set()
    paired = []
    for cohort, (pack, run, entry) in enumerate(zip(packs, runs, entries, strict=True), start=1):
        _, input_rows, private = _source_map(campaign, cohort)
        originals = {x["source_train_question_id"] for x in private.values()}
        if (len(originals) != len(input_rows)
            or originals & (seen_ids | holdout_ids)
            or entry.get("planned_curriculum_cohort") != cohort
            or entry.get("n_original_train_questions") != len(originals)):
            raise ValueError("cross-cohort duplicate or frozen TRAIN holdout leakage")
        seen_ids.update(originals)
        run_meta = _json(run / "SOLVER_RUN_MANIFEST.json")
        if run_meta.get("cohort") != cohort or run_meta.get("seed") != entry.get("selection_seed"):
            raise ValueError("Solver run's seed/cohort disagrees with frozen TRAIN campaign")
        report = audit_solver_run(train_data, campaign, pack, run, source_sha=source_sha)
        paired.append((run_meta, report))
    if len(seen_ids) != manifest["total_original_train_tasks_prepared"]:
        raise ValueError("original TRAIN campaign task-count mismatch")
    result = combine_verified_scouts(paired, planned_cohorts=len(entries))
    result["source_git_sha"] = source_sha
    result["campaign_manifest_sha256"] = sha256_file(campaign / "CAMPAIGN_MANIFEST.json")
    result["original_train_documents_sha256"] = sha256_file(train_data / "documents.jsonl")
    result["original_train_questions_sha256"] = sha256_file(train_data / "questions.jsonl")
    if (result["original_train_questions_sha256"] != manifest["original_train_questions_sha256"]
        or result["original_train_documents_sha256"] != manifest["original_train_documents_sha256"]):
        raise ValueError("canonical TRAIN source checksum disagreement")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fp:
        json.dump(result, fp, ensure_ascii=False, sort_keys=True, indent=2)
        fp.write("\n")
    return result
