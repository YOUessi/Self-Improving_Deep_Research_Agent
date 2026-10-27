"""E-030: fail-closed cross-workstream TRAIN exposure and frozen-review audits.

Conservative scope: labels/answers read by an oracle teacher, sampled optimizer
questions, prior SFT actions, real RSI Solver questions, reserved RSI TRAIN
holdout IDs, and *both* prior E-029 blind card registries. Earlier exposure is
not identical to neural training, so preserve the source types separately.

This module can certify disjointness ONLY from the exact source files whose
SHA256 hashes were pinned in the audit. It cannot prevent future branches from
training on newly frozen questions. IDs and review packets stay on Tang.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from sidra.semantic_contrast import file_sha256


def _digest(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{no}: expected object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: empty source")
    return rows


def _source_ids(path: Path, kind: str) -> set[str]:
    if kind == "optimizer_summary":
        record = json.loads(path.read_text(encoding="utf-8"))
        ids = record.get("selected_training_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{path.name}: missing selected_training_ids")
        if any(not isinstance(x, str) or not x for x in ids):
            raise ValueError(f"{path.name}: malformed optimizer IDs")
        return set(ids)
    if kind == "registry":
        rows = _jsonl(path)
        ids = [r.get("source_train_question_id") for r in rows]
        claims = [r.get("claim_id") for r in rows]
        if (any(not isinstance(x, str) or not x for x in ids + claims)
                or len(ids) != len(set(ids)) or len(claims) != len(set(claims))):
            raise ValueError("untrusted blind registry")
        return set(ids)
    name = {
        "prior_train_jsonl": ("question_id", "qid", "trace.question_id"),
        "teacher_actions": ("question_id",),
        "rsi_solver": ("source_train_question_id",),
        "rsi_reserved": ("original_train_question_id",),
    }.get(kind)
    if name is None:
        raise ValueError(f"unsupported source kind: {kind}")
    ids = set()
    for row in _jsonl(path):
        choices = [
            (row.get("trace") or {}).get("question_id") if k=="trace.question_id"
            else row.get(k) for k in name
        ]
        choices = [x for x in choices if x is not None]
        if len(set(choices)) != 1 or not isinstance(choices[0], str) or not choices[0]:
            raise ValueError(f"{path.name}: unknown or conflicting source QID")
        ids.add(choices[0])
    return ids


def read_registry_verified(directory: Path, valid_train_ids: set[str]) -> set[str]:
    """Pin A/B blind packet, private registry and manifest before auditing."""
    manifest_path = directory/"MANIFEST.json"
    meta = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (meta.get("split") != "train"
        or meta.get("independent_human_adjudication_completed") is not False
        or meta.get("allow_as_verifier_threshold_calibration") is not False):
        raise ValueError("E-029 cohort must be unadjudicated TRAIN-only")
    registry = directory/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl"
    if file_sha256(registry) != meta.get("private_registry_sha256"):
        raise ValueError("blind private registry digest differs from manifest")
    ids = _source_ids(registry, "registry")
    cards = _jsonl(registry)
    if (len(cards) != meta.get("question_count") or
        not ids <= valid_train_ids or
        _digest({x["claim_id"] for x in cards}) != meta.get("frozen_claim_ids_sha256")):
        raise ValueError("blind cohort IDs disagree with pinned TRAIN source")
    for reviewer in ("A", "B"):
        f = directory/f"REVIEWER_{reviewer}_BLIND.jsonl"
        if file_sha256(f) != meta.get("reviewer_packet_sha256", {}).get(reviewer):
            raise ValueError("blind reviewer packet SHA256 changed")
        cr = _jsonl(f)
        if (len(cr) != len(cards) or
            {x.get("claim_id") for x in cr} != {x["claim_id"] for x in cards}):
            raise ValueError("blind reviewer cards differ from registry")
        if any(
            set(x) != {"claim_id", "question", "proposed_answer",
                       "provided_evidence", "rater_instruction", "allowed_labels"}
            for x in cr
        ):
            raise ValueError("reviewer packet has unapproved keys")
    return ids


def scan_exposures(
    *, valid_train_ids: set[str],
    inputs: dict[str, list[Path]],
) -> tuple[set[str], dict[str, set[str]], dict[str, dict]]:
    """Load *all* explicitly provided sources, hash raw bytes, reject bad IDs."""
    required = {"prior_train_jsonl", "teacher_actions", "optimizer_summary",
                "rsi_solver", "rsi_reserved", "registry"}
    if set(inputs) != required or any(not items for items in inputs.values()):
        raise ValueError("every source class required, including obsolete registries")
    digests = {}
    by_kind: dict[str, set[str]] = {}
    paths_seen = set()
    for kind in sorted(required):
        found = set()
        for path in inputs[kind]:
            path = Path(path)
            absolute = str(path.resolve(strict=True))
            if absolute in paths_seen:
                raise ValueError("duplicate source path across history classes")
            paths_seen.add(absolute)
            sha = file_sha256(path)
            ids = _source_ids(path, kind)
            if not ids <= valid_train_ids:
                raise ValueError(f"{kind}: source contains unknown/non-TRAIN question")
            found.update(ids)
            digests[absolute] = {"sha256": sha, "kind": kind,
                                 "distinct_qids": len(ids)}
        by_kind[kind] = found
    return set().union(*by_kind.values()), by_kind, digests


def exposure_report(
    *, frozen_directory: Path, valid_train_ids: set[str],
    inputs: dict[str, list[Path]],
) -> tuple[dict, set[str]]:
    old = read_registry_verified(frozen_directory, valid_train_ids)
    all_used, by_kind, digests = scan_exposures(
        valid_train_ids=valid_train_ids, inputs=inputs)
    # Do not publish individual QA identifiers, just counts & digests.
    old_overlaps = old & all_used
    # Distinguish prior optimizer/teacher/real-Solver exposure from a
    # separately discarded candidate packet, which never trained a model.
    workstream_overlaps = old & set().union(
        *(ids for kind, ids in by_kind.items() if kind != "registry"))
    obsolete_only = old_overlaps - workstream_overlaps
    report = {
        "study": "M7_E030_cross_workstream_TRAIN_contamination_audit",
        "frozen_cohort_size": len(old),
        "known_prior_exposure_count": len(all_used),
        "frozen_cohort_any_exposure_count": len(old_overlaps),
        "frozen_cohort_any_exposure_rate": round(len(old_overlaps)/len(old), 6),
        "workstream_train_or_teacher_or_solver_exposure_count": len(workstream_overlaps),
        "obsolete_card_packet_only_overlap_count": len(obsolete_only),
        "actual_optimizer_selected_overlap_count": len(
            old & by_kind["optimizer_summary"]),
        "workstream_exposed_distinct_questions": len(set().union(
            *(ids for kind, ids in by_kind.items() if kind != "registry"))),
        "frozen_cohort_overlap_by_source_kind": {
            k: len(old & v) for k, v in sorted(by_kind.items())},
        "historical_source_sha256": digests,
        "frozen_registry_sha256": file_sha256(
            frozen_directory/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl"),
        "historical_known_qid_set_sha256": _digest(all_used),
        "frozen_question_set_sha256": _digest(old),
        "audit_pass_disjointness": not old_overlaps,
        "old_frozen_cohort_invalid_for_independent_verifier_accuracy": bool(old_overlaps),
        "human_semantic_annotations_received": 0,
        "no_dev_or_final_labels_accessed": True,
        "scope_limit": (
            "Disjoint only from listed pinned TRAIN exposure sources, not all "
            "unseen histories or future workstream updates; oracle/teacher "
            "exposure is not the same as optimizer training."
        ),
    }
    return report, all_used


def make_refreeze(
    *, old: Path, valid_train_ids: set[str], inputs: dict[str, list[Path]],
    dataset: Path, original_train: Path, e026_rows: Path,
    out: Path, seed: int = 2030,
) -> dict:
    """Create a NEW independent packet after excluding all observed histories.

    Never modify E-029, GitHub's other workstreams, or existing packets. A
    revalidation gate is required before any future model/annotation scoring.
    """
    from sidra.blind_adjudication import freeze_blind_packet
    if out.exists():
        raise FileExistsError(out)
    report, exposed = exposure_report(
        frozen_directory=old, valid_train_ids=valid_train_ids, inputs=inputs)
    # The target E029 v2 packet has ALREADY BEEN EXAMINED, so it too must
    # be completely excluded from the replacement even if particular cards
    # were not used by training. The prior E029 v1 registry is supplied
    # separately as inputs["registry"]. Never recycle frozen study cards.
    old_questions = read_registry_verified(old, valid_train_ids)
    exposed.update(old_questions)
    # Recheck source identity: train documents and the 1024 originals agree
    # inside freeze_blind_packet, and all external exposure IDs are TRAIN.
    exposed_path = out/"PRIVATE_ALL_PRIOR_EXPOSED_TRAIN_QIDS.jsonl"
    cohort_dir = out/"REPLACEMENT_BLIND_64_TRAIN_ONLY"
    # Source-type selection feasibility is checked by freeze_blind_packet.
    # Never reuse an obsolete packet directory or ID-selective tuning.
    out.mkdir(parents=True, exist_ok=False)
    exposed_path.write_text(
        "".join(json.dumps({"question_id": q}, sort_keys=True)+"\n"
                for q in sorted(exposed)), encoding="utf-8")
    frozen = freeze_blind_packet(
        dataset=dataset, original_train=original_train,
        prior_e026_rows=e026_rows, out=cohort_dir, seed=seed,
        prior_train_action_files=(exposed_path,))
    final_ids = read_registry_verified(cohort_dir, valid_train_ids)
    if final_ids & exposed or len(final_ids)!=64:
        raise AssertionError("new holdout is contaminated or incomplete")
    # Read-only race check: an independent workstream might have written to
    # one of these history sources during this CPU-only freeze operation.
    for path, source in report["historical_source_sha256"].items():
        if file_sha256(Path(path)) != source["sha256"]:
            raise RuntimeError("history source changed during frozen cohort selection")
    if file_sha256(old/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl") != report["frozen_registry_sha256"]:
        raise RuntimeError("old blind packet changed during refreeze")
    aggregate = {
        **report,
        "previous_cohort_usable": False,
        "replacement_cohort_created": True,
        "replacement_questions": len(final_ids),
        "replacement_unique_questions_sha256": _digest(final_ids),
        "replacement_exposure_overlap_count": len(final_ids & exposed),
        "replacement_any_previous_E029_v1_or_v2_card_overlap_count": 0,
        "both_prior_E029_candidate_cohorts_held_out_from_replacement": True,
        "replacement_train_provenance_manifest_sha256": file_sha256(
            cohort_dir/"MANIFEST.json"),
        "private_exposure_ids_sha256": file_sha256(exposed_path),
        "replacement_rater_annotations": 0,
        "replacement_model_eval_completed": False,
        "replacement_is_prospectively_disjoint_at_current_history_snapshot": True,
        "future_exposure_must_be_rechecked_before_scoring": True,
    }
    (out/"AUDIT_AGGREGATES_ONLY.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    return aggregate
