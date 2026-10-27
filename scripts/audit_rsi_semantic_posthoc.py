"""Independent CPU-only, TRAIN-only posthoc negative safety audit.

This NEVER asks the model again and NEVER upgrades a task. It replays the
trusted BM25 quote proof and restricts existing model verdicts using new
conservative negative rules. Previous reviewer output hashes are retained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sidra.rsi_grounded_audit import audit_saved_grounded_candidates
from sidra.rsi_semantic_negative_gate import negative_flags
from sidra.rsi_semantic_triage import PASS_VERDICT,build_review_packet


def _sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl(path:Path)->list[dict]:
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        obj=json.loads(line)
        if not isinstance(obj,dict):raise ValueError("expected JSON object")
        rows.append(obj)
    return rows


def classify_conservatively(packet,original:dict)->dict:
    """Only downgrade; never approve a label regardless of machine verdict."""
    if (original.get("challenge_id")!=packet.challenge_id
        or original.get("review_packet_sha256")!=packet.sha256
        or original.get("authorized_train_label_count")!=0
        or original.get("trainer_access_granted") is not False
        or not isinstance(original.get("conclusion"),str)
        or len(original.get("model_verdicts",[]))!=2
        or len(original.get("reviewer_output_sha256",[]))!=2):
        raise ValueError("untrusted previous reviewer result identity/policy")
    former=original["conclusion"]
    flags=()
    if former=="needs_independent_semantic_label_review":
        if original["model_verdicts"]!=["supported","supported"]:
            raise ValueError("positive model review lacks two matching verdicts")
        if len(set(x.casefold() for x in original.get("model_bridge_entities",[])))!=1:
            raise ValueError("positive model review has no common bridge")
        bridge=original["model_bridge_entities"][0]
        flags=negative_flags(
            packet.question,packet.answer,packet.evidence,bridge,
        )
        revised=("flagged_deterministic_premise_gap" if flags else former)
    else:
        revised=former
    return {
        "challenge_id":packet.challenge_id,
        "review_packet_sha256":packet.sha256,
        "original_model_review_sha256":original["reviewer_output_sha256"],
        "original_conclusion":former,
        "revised_conclusion":revised,
        "negative_flags":list(flags),
        "machine_reviewer_opinion_is_gold":False,
        "approved_train_labels":0,
        "promoted_to_trainer":False,
    }


def run(*,train_data:Path,queue:Path,manifest:Path,
        verifier_map:Path,generated:Path,semantic_dir:Path,out_dir:Path)->dict:
    if out_dir.exists():raise FileExistsError(out_dir)
    source_audit=audit_saved_grounded_candidates(
        train_data,queue,manifest,verifier_map,generated,
    )
    original_candidates=generated/"UNLABELLED_GROUNDED_CANDIDATES.jsonl"
    original_reviews=semantic_dir/"SEMANTIC_TRIAGE.jsonl"
    original_report=json.loads(
        (semantic_dir/"SEMANTIC_TRIAGE_SUMMARY.json").read_text())
    if (
        original_report.get("status")!="model_semantic_triage_unlabelled_never_gold"
        or original_report.get("source_candidates_sha256")!=_sha(original_candidates)
        or original_report.get("novel_train_labels_approved")!=0
        or original_report.get("model_optimizer_updates")!=0
    ):
        raise ValueError("original semantic triage report is untrusted")
    candidates={
        p["challenge_id"]:build_review_packet(p)
        for p in _load_jsonl(original_candidates)
        if p["verdict"]==PASS_VERDICT
    }
    reviews=_load_jsonl(original_reviews)
    if len(candidates)!=len(reviews) or len(reviews)!=source_audit[
        "literal_two_document_quote_passes"
    ]:raise ValueError("model reviews missing relative to original trusted quotes")
    processed=[]
    for old in reviews:
        packet=candidates.get(old.get("challenge_id"))
        if packet is None:
            raise ValueError("unknown challenge in saved model reviewer results")
        processed.append(classify_conservatively(packet,old))
    if len({x["challenge_id"] for x in processed})!=len(processed):
        raise ValueError("duplicate reviewed challenge")
    # Defend against any unexplained upward reclassification.
    changed=[x for x in processed if x["original_conclusion"]!=x["revised_conclusion"]]
    assert all(x["original_conclusion"]=="needs_independent_semantic_label_review"
               and x["revised_conclusion"]=="flagged_deterministic_premise_gap"
               for x in changed)
    result={
        "status":"CPU_posthoc_negative_only_reclassification",
        "split":"train","n":len(processed),
        "trusted_quote_replay_passes":source_audit["literal_two_document_quote_passes"],
        "original_saved_model_reviews_sha256":_sha(original_reviews),
        "original_saved_model_summary_sha256":_sha(
            semantic_dir/"SEMANTIC_TRIAGE_SUMMARY.json"),
        "original_frozen_challenger_sha256":_sha(original_candidates),
        "train_questions_sha256":_sha(train_data/"questions.jsonl"),
        "train_documents_sha256":_sha(train_data/"documents.jsonl"),
        "original_status_counts":dict(Counter(
            x["original_conclusion"] for x in processed)),
        "reclassified_counts":dict(Counter(
            x["revised_conclusion"] for x in processed)),
        "safety_downgrades":len(changed),
        "upgrades":0,"approved_novel_train_labels":0,
        "optimizer_updates":0,"multi_generation_RSI_completed":False,
        "requires_independent_semantic_truth_for_training":True,
        "caution":"Two source quotes, lexical event keywords and matching model bridge "
                  "strings are not full semantic entailment. These rules only reject. "
                  "No model-judged positive may enter TRAIN without independent labels.",
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    with (out_dir/"POSTHOC_NEGATIVE_REVIEW.jsonl").open("x") as f:
        for row in processed:
            f.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
    (out_dir/"POSTHOC_NEGATIVE_REVIEW_SUMMARY.json").write_text(
        json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    for key in (
        "--train-data","--queue","--manifest","--verifier-map",
        "--generated","--semantic-dir","--out-dir",
    ):
        p.add_argument(key,required=True,type=Path)
    args=p.parse_args()
    print(json.dumps(run(
        train_data=args.train_data,queue=args.queue,manifest=args.manifest,
        verifier_map=args.verifier_map,generated=args.generated,
        semantic_dir=args.semantic_dir,out_dir=args.out_dir,
    ),ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
