"""Materialize an exact-retrieval TRAIN-only *teacher-heldout* evaluation slice.

This is a carefully labelled TRAIN subset, NOT public DEV or sealed final.
Copying the ENTIRE corpus preserves global BM25 term frequencies and IDFs.
Only TEST/EVAL code sees original TRAIN answer/support fields; the policy
continues to receive only question text and actual tool observations.

Provenance: source TRAIN and teacher manifest hashes must match, and teacher
success paths come from genuine tool calls. All question IDs are disjoint
from the TRAIN teacher-action fit partition and GRPO curriculum.
"""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from pathlib import Path

from sidra.experiment import sha256_file
from sidra.io import iter_jsonl,load_dataset,save_jsonl


def materialize(source:Path,teacher_dir:Path,dev_manifest:Path,
                out:Path)->dict:
    if out.exists():raise FileExistsError(out)
    src_manifest=json.loads((teacher_dir/"TEACHER_SUMMARY.json").read_text())
    if (src_manifest.get("stage")!="TRAIN_ONLY_GOLD_SUPERVISED_TEACHER_NO_RL"
        or src_manifest.get("teacher_uses_train_gold") is not True
        or src_manifest.get("student_model_sees_gold_answer") is not False
        or src_manifest.get("includes_only_real_strict_successes") is not True
        or src_manifest.get("train_question_sha256")!=sha256_file(source/"questions.jsonl")
        or src_manifest.get("train_document_sha256")!=sha256_file(source/"documents.jsonl")
        or src_manifest.get("dev_manifest_sha256")!=sha256_file(dev_manifest)):
        raise ValueError("untrusted TRAIN-only teacher corpus provenance")
    original_dev=json.loads(dev_manifest.read_text())
    dev_ids=(set(original_dev["pilot"]["selected_ids"])
             | set(original_dev["heldout"]["selected_ids"]))
    fit_ids={str(row["question_id"]) for row in
             iter_jsonl(teacher_dir/"TEACHER_ACTIONS_TRAIN.jsonl")}
    held_ids={str(row["question_id"]) for row in
              iter_jsonl(teacher_dir/"TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl")}
    if not held_ids or not fit_ids or held_ids&fit_ids or held_ids&dev_ids:
        raise ValueError("teacher-heldout IDs overlap fit/DEV or are empty")
    documents,qs=load_dataset(source)
    qmap={q.question_id:q for q in qs}
    if len(qmap)!=len(qs) or not all(q.split=="train" for q in qs):
        raise ValueError("training corpus is not unique TRAIN-only")
    if not held_ids.issubset(qmap) or not fit_ids.issubset(qmap):
        raise ValueError("unknown TRAIN question ID in teacher records")
    if len(held_ids)!=int(src_manifest["heldout_train_trajectories"]):
        raise ValueError("heldout TRAIN count disagrees with teacher manifest")
    chosen=[qmap[qid] for qid in sorted(held_ids)]
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(source/"documents.jsonl",out/"documents.jsonl")
    if sha256_file(out/"documents.jsonl")!=src_manifest["train_document_sha256"]:
        raise RuntimeError("BM25 source corpus changed in holdout materialization")
    save_jsonl(out/"questions.jsonl",[asdict(q) for q in chosen])
    (out/"README.md").write_text(
        "# TRAIN-only teacher-reachable heldout (not DEV, not final-B160)\n"
        "This partition consists of heldout TRAIN question IDs that the offline "
        "TRAIN-only strict oracle teacher could solve. The model has NO gold "
        "access. The entire original TRAIN document corpus is duplicated "
        "byte-for-byte to preserve BM25 IDF. This selection is biased toward "
        "teacher-reachable examples and is not external generalization.\n")
    record={
        "evaluation_label":"TRAIN_teacher_success_reachable_holdout_NOT_DEV",
        "split":"train","n":len(chosen),
        "ids":sorted(held_ids),
        "none_in_teacher_fit_partition":True,
        "dev_question_ids_used":False,
        "sealed_final_B160_used":False,
        "original_documents_sha256":src_manifest["train_document_sha256"],
        "original_questions_sha256":src_manifest["train_question_sha256"],
        "teacher_manifest_sha256":sha256_file(teacher_dir/"TEACHER_SUMMARY.json"),
        "frozen_public_dev_manifest_sha256":sha256_file(dev_manifest),
        "holdout_questions_sha256":sha256_file(out/"questions.jsonl"),
    }
    (out/"TRAIN_HOLDOUT_MANIFEST.json").write_text(json.dumps(
        record,indent=2,ensure_ascii=False)+"\n")
    return record


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--teacher-dir",type=Path,required=True)
    p.add_argument("--dev-manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(materialize(a.source,a.teacher_dir,
                                  a.dev_manifest,a.out),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
