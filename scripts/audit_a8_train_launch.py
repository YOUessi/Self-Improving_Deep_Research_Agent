"""CPU-only, fail-closed M7-A8 launch provenance and resource readout.

This preflight NEVER starts CUDA, serves a model, changes a checkpoint, or
changes the pre-registered experiment. It protects the *future* A8 attempt
from a silently changed TRAIN curriculum, parent LoRA, dev freeze or TRAIN64
cross-contamination, then reports competing GPU jobs (non-preemptively).

The previous A9 historical source replay is a prerequisite, NOT replaced by
this independent check. All checks happen before loading a model.

Run only on Tang with a PINNED GitHub source snapshot; see M7A10 report.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

# Support both `python -m scripts.audit_a8_train_launch` and direct script
# invocation from the pinned checkout (PYTHONPATH=src).
if __package__:
    from scripts.audit_a8_frozen_integrity import A8, validate_a8_freeze
    from scripts.train_finite_action_grpo import select_train_only
else:
    from audit_a8_frozen_integrity import A8, validate_a8_freeze
    from train_finite_action_grpo import select_train_only
from sidra.gpu_coordination import find_external_gpu_jobs
from sidra.schema import Question


PARENT_STOP_SFT_SHA256 = (
    "44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771"
)
DEV_MANIFEST_SHA256 = (
    "3f0c494cca8b44cfef90c78d4d5b1fbb4158dc7387ed0f24316a0a265471d947"
)
# Pre-registered TRAIN64+source membership yields an immutable optimizer
# question order under original V4 teacher seed153 (20 groups x 2 updates).
# Don't silently select a new training subset when source code changes.
EXPECTED_A8_TRAIN_SELECTION_SHA256 = (
    "449d09935e9dee7f22e7fb1bafb77a94346c759c81600bed6c23e0358034184c"
)
A8_TRAIN_SEED = 153
A8_GROUPS = 20
A8_GENERATIONS = 4
A8_UPDATES = 2
A8_MAX_STEPS = 6
A8_MAX_TOOLS = 5
A8_MAX_CONTEXT = 2048
A8_LR = 3e-6
A8_FINISH_AUX_WEIGHT = 0.2
A8_READ_AUX_WEIGHT = 0.2
A8_ANSWER_PROPOSER = "v4"
A8_CREDIT_MODE = "episode_total"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("TRAIN row must be a JSON object")
                out.append(row)
    return out


def _validated_ids(rows: Iterable[dict], *, name: str) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"{name} contains missing or malformed question IDs")
        ids.add(qid)
    return ids


def validate_a8_selection(
    *, train_rows: list[dict], teacher_train_rows: list[dict],
    teacher_heldout_rows: list[dict], frozen_rows: list[dict],
    dev_ids: set[str], seed: int = A8_TRAIN_SEED,
    groups: int = A8_GROUPS, updates: int = A8_UPDATES,
) -> dict:
    """Prove deterministic TRAIN-only optimizer membership without GPU.

    The teacher may have more than one action row for a question. Do not
    interpret the number of action rows as the number of train questions.
    No labels or question text appear in the returned report.
    """
    if min(groups, updates) < 1:
        raise ValueError("invalid preregistered group count / updates")
    qids = _validated_ids(train_rows, name="source TRAIN")
    if len(qids) != len(train_rows):
        raise ValueError("source TRAIN has duplicate question IDs")
    if any(x.get("split") != "train" for x in train_rows):
        raise ValueError("source contains non-TRAIN examples")
    if any(not x.get("question") or not isinstance(x.get("gold_answer"), str)
           or not isinstance(x.get("supporting_doc_ids"), list) for x in train_rows):
        raise ValueError("TRAIN source has invalid labels or questions")
    frozen_ids = _validated_ids(frozen_rows, name="frozen internal TRAIN64")
    if len(frozen_ids) != len(frozen_rows):
        raise ValueError("frozen TRAIN64 has duplicate question IDs")
    if not frozen_ids <= qids or any(x.get("split") != "train" for x in frozen_rows):
        raise ValueError("frozen TRAIN64 not a subset of source TRAIN")
    train_teacher_ids = _validated_ids(teacher_train_rows, name="teacher train actions")
    heldout_teacher_ids = _validated_ids(teacher_heldout_rows, name="teacher heldout actions")
    if not train_teacher_ids or not heldout_teacher_ids:
        raise ValueError("teacher source or heldout source is empty")
    if not (train_teacher_ids | heldout_teacher_ids) <= qids:
        raise ValueError("TRAIN teacher actions contain unknown question IDs")
    if train_teacher_ids & heldout_teacher_ids:
        raise ValueError("teacher training and teacher heldout IDs overlap")
    if train_teacher_ids & frozen_ids:
        raise ValueError("teacher TRAIN touched frozen TRAIN64")
    if heldout_teacher_ids & frozen_ids:
        raise ValueError("teacher HELDOUT touched frozen TRAIN64")
    if qids & dev_ids:
        raise ValueError("source TRAIN intersects frozen DEV")
    if any(x.get("student_model_gold_injected") is not False
           for x in teacher_train_rows + teacher_heldout_rows):
        raise ValueError("teacher action row violates model-gold isolation")
    qs = [Question.from_dict(q) for q in train_rows]
    selected = select_train_only(qs, dev_ids, seed=seed, n=groups * updates,
                                 allowed_train_ids=train_teacher_ids)
    selected_ids = [q.question_id for q in selected]
    if (len(selected_ids) != groups * updates
        or len(set(selected_ids)) != len(selected_ids)
        or set(selected_ids) & (frozen_ids | heldout_teacher_ids | dev_ids)):
        raise ValueError("selected TRAIN optimizer examples contaminated")
    return {
        "source_train_questions": len(qids),
        "teacher_train_unique_questions": len(train_teacher_ids),
        "teacher_heldout_unique_questions": len(heldout_teacher_ids),
        "frozen_internal_train_questions": len(frozen_ids),
        "selected_optimizer_train_questions": len(selected_ids),
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(selected_ids).encode("utf-8")).hexdigest(),
        "selected_frozen_overlap": 0,
        "selected_teacher_heldout_overlap": 0,
        "selected_public_dev_overlap": 0,
        "student_model_gold_injected": False,
        "question_gold_or_text_returned": False,
        "selected_counts_by_update": [groups] * updates,
    }


def assert_exact_preregistered_optimizer_selection(selection: dict) -> None:
    """Reject even TRAIN-only substitutions in the A8 optimizer sample."""
    if selection.get("selected_optimizer_train_questions") != A8_GROUPS * A8_UPDATES:
        raise ValueError("A8 pre-registered optimizer count drift")
    if selection.get("selected_ids_sha256") != EXPECTED_A8_TRAIN_SELECTION_SHA256:
        raise ValueError("A8 pre-registered TRAIN optimizer question order drift")


def validate_parent_adapter(path: Path) -> str:
    weights = path / "adapter_model.safetensors"
    if not weights.is_file() or not (path / "adapter_config.json").is_file():
        raise ValueError("missing original parent LoRA weights/config")
    digest = sha256(weights)
    if digest != PARENT_STOP_SFT_SHA256:
        raise ValueError("original parent stop-SFT adapter SHA drift")
    return digest


def validate_teacher_manifest(teacher: Path, *, train_sha: str, dev_sha: str) -> dict:
    summary = json.loads((teacher / "TEACHER_SUMMARY.json").read_text())
    expected = {
        "stage": "TRAIN_ONLY_GOLD_SUPERVISED_TEACHER_NO_RL",
        "teacher_uses_train_gold": True,
        "student_model_sees_gold_answer": False,
        "includes_only_real_strict_successes": True,
        "train_question_sha256": train_sha,
        "dev_manifest_sha256": dev_sha,
        "answer_proposer": A8_ANSWER_PROPOSER,
        "step_budget": A8_MAX_STEPS,
        "tool_budget": A8_MAX_TOOLS,
    }
    wrong = [key for key, value in expected.items() if summary.get(key) != value]
    if wrong:
        raise ValueError(f"TRAIN-only teacher preregistration drift: {wrong}")
    return {
        "teacher_summary_sha256": sha256(teacher / "TEACHER_SUMMARY.json"),
        "teacher_gold_used_only_for_curriculum_selection": True,
        "model_policy_receives_gold": False,
    }


def audit_preflight(
    *, root: Path, source: Path, frozen: Path, teacher: Path,
    dev_manifest: Path, parent_adapter: Path, source_checkout: Path,
    planned_output: Path, out: Path, gpu_jobs: list[dict] | None = None,
) -> dict:
    if out.exists():
        raise FileExistsError("refuse to overwrite an earlier launch preflight")
    if planned_output.exists():
        raise FileExistsError("planned new A8 result folder already exists")
    code_sha = (source_checkout / "GITHUB_COMMIT.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}", code_sha):
        raise ValueError("source checkout not pinned to an exact GitHub SHA")
    if dev_manifest.resolve() == (source_checkout / "GITHUB_COMMIT.txt").resolve():
        raise ValueError("invalid dev manifest reference")
    dev_sha = sha256(dev_manifest)
    if dev_sha != DEV_MANIFEST_SHA256:
        raise ValueError("frozen public DEV manifest SHA drift")
    if (sha256(source / "questions.jsonl") != A8.source_questions_sha256
        or sha256(source / "documents.jsonl") != A8.source_documents_sha256):
        raise ValueError("original TRAIN1024 source bytes changed")
    old_adapter_sha = validate_parent_adapter(parent_adapter)
    historical = validate_a8_freeze(m7root=root, source=source, frozen=frozen)
    if historical.get("status") != "PASS":
        raise ValueError("A9 independent pinned TRAIN64 source replay failed")
    teacher_manifest = validate_teacher_manifest(
        teacher, train_sha=A8.source_questions_sha256, dev_sha=dev_sha)
    dev = json.loads(dev_manifest.read_text())
    dev_ids = set(dev["pilot"]["selected_ids"]) | set(dev["heldout"]["selected_ids"])
    selection = validate_a8_selection(
        train_rows=jsonl(source / "questions.jsonl"),
        teacher_train_rows=jsonl(teacher / "TEACHER_ACTIONS_TRAIN.jsonl"),
        teacher_heldout_rows=jsonl(teacher / "TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl"),
        frozen_rows=jsonl(frozen / "questions.jsonl"),
        dev_ids=dev_ids)
    if selection["frozen_internal_train_questions"] != A8.count:
        raise ValueError("A8 frozen internal TRAIN64 size drift")
    assert_exact_preregistered_optimizer_selection(selection)
    jobs = find_external_gpu_jobs() if gpu_jobs is None else gpu_jobs
    # Names/kinds only: no foreign process argv, credentials, payloads.
    if any(not isinstance(j.get("pid"), int) or not isinstance(j.get("kind"), str)
           for j in jobs):
        raise ValueError("malformed GPU process metadata")
    report = {
        "status": ("CPU_INPUT_GATE_PASS_GPU_BUSY" if jobs
                   else "CPU_INPUT_GATE_PASS_GPU_NOT_RESERVED"),
        "stage": "M7A10_CPU_PRETRAIN_A8_PRECHECK",
        "pinned_source_commit": code_sha,
        "original_train_sha256": A8.source_questions_sha256,
        "frozen_train64_source_replay": historical["status"],
        "frozen_train64_ordered_id_sha256": A8.ordered_qid_sha256,
        "public_dev_manifest_sha256": dev_sha,
        "original_stop_sft_adapter_sha256": old_adapter_sha,
        **teacher_manifest,
        **selection,
        "pre_registered_train": {
            "seed": A8_TRAIN_SEED,
            "groups": A8_GROUPS,
            "generations": A8_GENERATIONS,
            "updates": A8_UPDATES,
            "max_steps": A8_MAX_STEPS,
            "max_tool_calls": A8_MAX_TOOLS,
            "max_context_tokens": A8_MAX_CONTEXT,
            "lr": A8_LR,
            "answer_proposer": A8_ANSWER_PROPOSER,
            "credit_mode": A8_CREDIT_MODE,
            "finish_aux_weight": A8_FINISH_AUX_WEIGHT,
            "read_aux_weight": A8_READ_AUX_WEIGHT,
            "start_from_original_stop_sft": True,
        },
        "external_gpu_job_types": sorted(
            {str(j["kind"]) for j in jobs}),
        "external_gpu_job_count": len(jobs),
        "gpu_training_launched": False,
        "gpu_launch_authorized_by_this_preflight": False,
        "model_weights_updated": False,
        "separate_gpu_flock_and_ownership_gate_still_required": True,
        "frozen_final_B160_accessed": False,
        "public_dev_gold_accessed": False,
        "warning": ("Internal TRAIN64 only; not external validation. "
                    "A8 hybrid supervision is not pure GRPO. "
                    "This CPU report cannot reserve GPU or prove QA gain."),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("root", "source", "frozen", "teacher", "dev-manifest",
                 "parent-adapter", "source-checkout", "planned-output", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    a = parser.parse_args()
    result = audit_preflight(
        root=a.root, source=a.source, frozen=a.frozen, teacher=a.teacher,
        dev_manifest=a.dev_manifest, parent_adapter=a.parent_adapter,
        source_checkout=a.source_checkout,
        planned_output=a.planned_output, out=a.out)
    # Print only safe aggregate metadata, never TRAIN question IDs/answer text.
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
