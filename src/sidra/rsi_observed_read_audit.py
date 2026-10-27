"""E-029 two-phase original-TRAIN observed-READ shadow audit.

Phase 1 receives only model-visible packs and real Solver traces. The
independent offline verifier in phase 2 replays original BM25/model I/O and
only THEN opens canonical TRAIN answers/support IDs. Never output per-task
gold labels or pretend a shadow READ created a new model trajectory.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from sidra.experiment import sha256_file
from sidra.io import iter_jsonl, load_dataset
from sidra.rsi_observed_read import choose_observed_second_read, _validated_frontier
from sidra.rsi_solver_cohort import _check_path, _json, _source_map, _validate_public_pack, doc_alias
from sidra.rsi_solver_campaign_scorecard import audit_cross_cohort_solver_campaign


def _digest(observations: list[dict]) -> str:
    data = json.dumps(observations, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def shadow_slot(question_id: str, question: str, raw_trace: dict) -> dict:
    """Model-only: at most one replacement of an ACTUAL already-budgeted READ."""
    if (raw_trace.get("question_id") != question_id
        or raw_trace.get("question") != question):
        raise ValueError("model-only question/trace identity mismatch")
    observations = []
    for idx, event in enumerate(raw_trace["steps"]):
        if not isinstance(event, dict) or event.get("step") != idx:
            raise ValueError("invalid step indexing")
        act = event.get("action")
        if isinstance(act, dict) and act.get("kind") == "read" and any(
            o.get("kind") == "read" for o in observations
        ):
            selected = choose_observed_second_read(
                question, observations, max_steps=6, max_tool_calls=5
            )
            if selected:
                hits, used, _ = _validated_frontier(observations)
                if act.get("doc_id") not in hits or act["doc_id"] in used:
                    raise ValueError("actual read not in observed unread frontier")
                return {
                    "challenge_id": question_id,
                    "eligible": True,
                    "step": idx,
                    "actual_read_doc_id": act["doc_id"],
                    "shadow_read_doc_id": selected.doc_id,
                    "frontier_size": selected.visible_candidates,
                    "previous_read_ids": sorted(used),
                    "observation_prefix_sha256": _digest(observations),
                }
        if "observation" in event:
            observations.append(event["observation"])
    return {"challenge_id": question_id, "eligible": False}


def _rows(path: Path) -> list[dict]:
    _check_path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return list(iter_jsonl(path))


def _sha(sha: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{40}", sha):
        raise ValueError("must pin GitHub source SHA")
    return sha


def materialize_model_only_shadow(
    *, packs: list[Path], runs: list[Path], out: Path, source_sha: str,
    shadow_code_sha: str
) -> dict:
    """Untrusted/model-only role: forbidden to accept TRAIN corpus or labels."""
    _sha(source_sha)
    _sha(shadow_code_sha)
    if len(packs) != 3 or len(runs) != 3:
        raise ValueError("requires all three frozen cohort model-only runs")
    for p in [*packs, *runs, out]:
        _check_path(p)
    if out.exists():
        raise FileExistsError(out)
    rows: list[dict] = []
    inputs = []
    seen = set()
    for expected_cohort, (pack, run) in enumerate(zip(packs, runs, strict=True), 1):
        pm, prompts, _ = _validate_public_pack(pack)
        rm = _json(run / "SOLVER_RUN_MANIFEST.json")
        traces_file = run / "SOLVER_TRACES.jsonl"
        if (pm.get("cohort") != expected_cohort or pm.get("source_git_sha") != source_sha
            or rm.get("cohort") != expected_cohort or rm.get("source_git_sha") != source_sha
            or rm.get("public_solver_pack_sha256") != sha256_file(pack / "SOLVER_PACK_MANIFEST.json")
            or rm.get("traces_sha256") != sha256_file(traces_file)
            or rm.get("optimizer_updates") != 0
            or rm.get("TRAIN_gold_loaded_or_graded_by_solver") is not False):
            raise ValueError("untrusted model-only cohort/run hash or role mismatch")
        traces = _rows(traces_file)
        by_id = {t.get("question_id"): t for t in traces}
        if (len(traces) != len(prompts) or len(by_id) != len(traces)
            or any(x["challenge_id"] in seen for x in prompts)):
            raise ValueError("missing/duplicate model-only task")
        seen.update(p["challenge_id"] for p in prompts)
        for p in prompts:
            raw = by_id.get(p["challenge_id"])
            if raw is None:
                raise ValueError("missing Solver trace")
            rows.append(shadow_slot(p["challenge_id"], p["question"], raw))
        inputs.append({
            "cohort": expected_cohort,
            "pack_manifest_sha256": sha256_file(pack / "SOLVER_PACK_MANIFEST.json"),
            "solver_trace_sha256": sha256_file(traces_file),
            "solver_run_manifest_sha256": sha256_file(run / "SOLVER_RUN_MANIFEST.json"),
        })
    # All input checks completed before creating any output.
    out.mkdir(parents=True, exist_ok=False)
    p = out / "MODEL_ONLY_SHADOW_SLOTS.jsonl"
    with p.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    manifest = {
        "stage": "model_only_matched_read_shadow_NO_LABELS",
        "original_campaign_source_git_sha": source_sha,
        "shadow_code_git_sha": shadow_code_sha,
        "n_original_model_episodes": len(rows),
        "eligible_second_read_slots": sum(x["eligible"] for x in rows),
        "shadow_slots_sha256": sha256_file(p),
        "inputs": inputs,
        "original_train_gold_or_private_map_opened": False,
        "DEV_evalA_finalB_opened": False,
        "optimizer_updates": 0,
        "model_generations": 0,
    }
    (out / "MODEL_ONLY_SHADOW_MANIFEST.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def paired_read_delta(rows: list[tuple[int, int]], *, seed: int = 829,
                      draws: int = 5000) -> dict:
    """Paired question-level resampling, no duplicated-seed pseudo-replication."""
    if not rows:
        return {"mean": None, "ci95": None, "n_questions": 0}
    if not all(a in (0, 1) and b in (0, 1) for a, b in rows):
        raise ValueError("unsupported nonbinary exposure outcome")
    deltas = [b - a for a, b in rows]
    n = len(rows)
    rng = random.Random(seed)
    boots = sorted(
        sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(draws)
    )
    return {
        "mean": round(sum(deltas) / n, 6),
        "ci95": [round(boots[int(draws * .025)], 6),
                 round(boots[min(draws - 1, int(draws * .975))], 6)],
        "n_questions": n, "resamples": draws, "seed": seed,
    }


def verify_matched_read_shadow(
    *, train_data: Path, campaign: Path, packs: list[Path], runs: list[Path],
    model_only: Path, out: Path, source_sha: str, shadow_code_sha: str
) -> dict:
    """Trusted verifier: replay all original real traces before scoring shadow."""
    _sha(source_sha)
    _sha(shadow_code_sha)
    if len(packs) != 3 or len(runs) != 3:
        raise ValueError("incomplete original TRAIN cohort campaign")
    for p in [train_data, campaign, *packs, *runs, model_only, out]:
        _check_path(p)
    if out.exists():
        raise FileExistsError(out)
    sm = _json(model_only / "MODEL_ONLY_SHADOW_MANIFEST.json")
    shadow_file = model_only / "MODEL_ONLY_SHADOW_SLOTS.jsonl"
    if (sm.get("stage") != "model_only_matched_read_shadow_NO_LABELS"
        or sm.get("original_campaign_source_git_sha") != source_sha
        or sm.get("shadow_code_git_sha") != shadow_code_sha
        or sm.get("original_train_gold_or_private_map_opened") is not False
        or sm.get("DEV_evalA_finalB_opened") is not False
        or sm.get("optimizer_updates") != 0
        or sm.get("model_generations") != 0
        or sha256_file(shadow_file) != sm.get("shadow_slots_sha256")
        or len(sm.get("inputs", [])) != 3):
        raise ValueError("model-only shadow provenance mismatch")
    shadow_rows = _rows(shadow_file)
    if len(shadow_rows) != sm.get("n_original_model_episodes"):
        raise ValueError("shadow row count mismatch")
    # Gold-blind reproducibility check: proposals must exactly equal a clean,
    # deterministic replay from pinned *model-only* observations.
    expected = []
    for cohort, (pack, run, pins) in enumerate(zip(
        packs, runs, sm["inputs"], strict=True
    ), 1):
        pm, prompts, _ = _validate_public_pack(pack)
        rm = _json(run / "SOLVER_RUN_MANIFEST.json")
        if (pins.get("cohort") != cohort
            or pm.get("cohort") != cohort or rm.get("cohort") != cohort
            or pins.get("pack_manifest_sha256") != sha256_file(pack / "SOLVER_PACK_MANIFEST.json")
            or pins.get("solver_run_manifest_sha256") != sha256_file(run / "SOLVER_RUN_MANIFEST.json")
            or pins.get("solver_trace_sha256") != sha256_file(run / "SOLVER_TRACES.jsonl")):
            raise ValueError("model-only input re-pinning mismatch")
        source_traces = {x["question_id"]: x for x in _rows(run / "SOLVER_TRACES.jsonl")}
        if len(source_traces) != len(prompts):
            raise ValueError("duplicate Solver questions")
        for p in prompts:
            expected.append(shadow_slot(
                p["challenge_id"], p["question"], source_traces[p["challenge_id"]]
            ))
    if shadow_rows != expected or sum(x["eligible"] for x in expected) != sm["eligible_second_read_slots"]:
        raise ValueError("forged or altered model-only shadow proposal")
    # Full cross-cohort independent E-028 validation: original TRAIN source,
    # hashes, Tool I/O, model replies, heldout overlap, all 72 traces.
    baseline = audit_cross_cohort_solver_campaign(
        train_data=train_data, campaign=campaign, packs=packs,
        runs=runs, source_sha=source_sha, out=out / "INDEPENDENT_E028_REPLAY.json"
    )
    if (baseline["actual_distinct_original_TRAIN_questions"] != 72
        or baseline["answer_correct_count"] != 17
        or baseline["strict_answer_plus_gold_passage_evidence_count"] != 0
        or baseline["tool_and_model_transcripts_independently_replayed"] is not True):
        raise ValueError("known frozen E028 72-question baseline mismatch")

    docs, questions = load_dataset(train_data)  # TRUSTED VERIFIER ONLY
    by_original = {q.question_id: q for q in questions}
    if len(by_original) != len(questions) or any(q.split != "train" for q in questions):
        raise ValueError("untrusted original TRAIN labels")
    by_shadow = {x["challenge_id"]: x for x in expected}
    per_tier = {tier: Counter() for tier in ("direct", "bridge", "hard")}
    paired = []
    outcomes = Counter()
    for cohort, (pack, run) in enumerate(zip(packs, runs, strict=True), 1):
        _, _, mapping = _source_map(campaign, cohort)
        public, prompts, _ = _validate_public_pack(pack)
        traces = {x["question_id"]: x for x in _rows(run / "SOLVER_TRACES.jsonl")}
        for p in prompts:
            cid = p["challenge_id"]
            row = by_shadow[cid]
            if not row["eligible"]:
                continue
            step = row["step"]
            before = [
                s["observation"] for s in traces[cid]["steps"][:step]
                if "observation" in s
            ]
            hits, used, _ = _validated_frontier(before)
            frontier = set(hits) - used
            if (row["actual_read_doc_id"] not in frontier
                or row["shadow_read_doc_id"] not in frontier
                or len(frontier) != row["frontier_size"]):
                raise ValueError("shadow choice not a matched legal READ")
            original = by_original[mapping[cid]["source_train_question_id"]]
            gold_support = {doc_alias(cid, x) for x in original.supporting_doc_ids}
            missing = gold_support - used
            real_exposure = int(row["actual_read_doc_id"] in missing)
            proposed_exposure = int(row["shadow_read_doc_id"] in missing)
            tier = mapping[cid]["tier"]
            if tier not in per_tier:
                raise ValueError("unexpected frozen TRAIN tier")
            bucket = per_tier[tier]
            bucket["eligible"] += 1
            bucket["visible_missing_support"] += int(bool(frontier & missing))
            bucket["real_marginal_support"] += real_exposure
            bucket["shadow_marginal_support"] += proposed_exposure
            bucket["choice_changes"] += int(row["actual_read_doc_id"] != row["shadow_read_doc_id"])
            outcomes["paired_question_count"] += 1
            outcomes["visible_missing_support"] += int(bool(frontier & missing))
            outcomes["real_marginal_support"] += real_exposure
            outcomes["shadow_marginal_support"] += proposed_exposure
            outcomes["choice_changes"] += int(row["actual_read_doc_id"] != row["shadow_read_doc_id"])
            outcomes["shadow_only_correct"] += int(proposed_exposure == 1 and real_exposure == 0)
            outcomes["real_only_correct"] += int(real_exposure == 1 and proposed_exposure == 0)
            paired.append((real_exposure, proposed_exposure))

    report = {
        "stage": "E029_GOLD_BLIND_MATCHED_READ_SHADOW_VERIFIER_DIAGNOSTIC_NOT_RSI",
        "source_campaign_sha": source_sha,
        "shadow_code_sha": shadow_code_sha,
        "frozen_original_train_episode_count": 72,
        "eligible_matched_read_slots": outcomes["paired_question_count"],
        "aggregate": dict(outcomes),
        "tiers": {tier: dict(per_tier[tier]) for tier in ("direct", "bridge", "hard")},
        "paired_shadow_minus_real_marginal_support_exposure":
            paired_read_delta(paired),
        "model_only_shadow_sha256": sha256_file(shadow_file),
        "baseline_independent_replay_sha256": sha256_file(out / "INDEPENDENT_E028_REPLAY.json"),
        "same_search_read_and_model_decision_budget_for_compared_slot": True,
        "full_trajectory_continuation_or_model_effect_measured": False,
        "gold_answer_or_support_read_by_planner": False,
        "all_private_train_labels_opened_only_by_offline_verifier": True,
        "original_train_episodes_previously_used_for_E028_diagnostics": True,
        "dev_evalA_finalB_gold_accessed": False,
        "optimizer_updates": 0,
        "checkpoint_generations": 0,
        "multi_generation_RSI_proven": False,
        "decision": "DIAGNOSTIC_ONLY; never promote on these historical labels",
    }
    with (out / "E029_AGGREGATE_ONLY_REPORT.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")
    return report
