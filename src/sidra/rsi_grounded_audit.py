"""Independent TRAIN-only re-audit of saved quote-grounded Challenger proposals.

Never trusts candidate self-reported structural validity. Recomputes BM25
search packets from the original TRAIN corpus and checks literal evidence.
Quote validity is not semantic entailment and never grants new TRAIN labels.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from sidra.io import load_dataset
from sidra.retrieval import BM25Index
from sidra.rsi_challenge_scheduler import _sha
from sidra.rsi_grounded_challenger import (
    make_grounding_packet,verify_grounded_response,
)


def _hash(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _objects(path:Path)->list[dict]:
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise ValueError("blank verified artifact record")
        row=json.loads(line)
        if not isinstance(row,dict):
            raise ValueError("expected object in TRAIN-only artifact")
        rows.append(row)
    if not rows or len(rows)>10000:
        raise ValueError("missing or oversized TRAIN-only audit data")
    return rows


def audit_saved_grounded_candidates(
    train_data:Path,queue_file:Path,manifest_file:Path,verifier_map:Path,
    generated_dir:Path,
)->dict:
    manifest=json.loads(manifest_file.read_text())
    summary=json.loads((generated_dir/"GROUNDED_CANDIDATE_SUMMARY.json").read_text())
    if (manifest.get("status")!="frozen_train_only_inputs" or
        manifest.get("split")!="train" or
        manifest.get("gold_fields_in_challenger_input") is not False or
        summary.get("status")!="model_generated_train_only_quote_gate_not_training" or
        summary.get("local_only") is not True or
        summary.get("semantic_entailment_verified") is not False or
        summary.get("new_trainable_challenges_approved")!=0 or
        summary.get("optimizer_updates")!=0):
        raise ValueError("untrusted campaign or unsafe model label claim")
    if (_hash(queue_file)!=manifest.get("challenger_input_sha256") or
        _hash(verifier_map)!=manifest.get("verifier_only_map_sha256") or
        _hash(queue_file)!=summary.get("question_input_sha256")):
        raise ValueError("frozen trusted queue checksum mismatch")
    files={
        "questions.jsonl":"train_dataset_questions_sha256",
        "documents.jsonl":"train_dataset_documents_sha256",
    }
    for basename,key in files.items():
        if _hash(train_data/basename)!=summary.get(key):
            raise ValueError("TRAIN corpus changed after quote-generation")
    proposals_path=generated_dir/"UNLABELLED_GROUNDED_CANDIDATES.jsonl"
    if _hash(proposals_path)!=summary.get("grounded_candidates_output_sha256"):
        raise ValueError("generated candidate artifact checksum mismatch")
    docs,questions=load_dataset(train_data)
    if any(q.split!="train" for q in questions):
        raise ValueError("a DEV/TEST question entered the TRAIN verifier")
    qmap={q.question_id:q for q in questions}
    in_rows=_objects(queue_file)
    map_rows=_objects(verifier_map)
    model_rows=_objects(proposals_path)
    if (len(in_rows)!=manifest.get("n") or
        len(map_rows)!=len(in_rows) or
        len(model_rows)!=summary.get("n_attempted") or
        len(model_rows)>len(in_rows) or
        summary.get("generation")!=manifest.get("generation")):
        raise ValueError("queue/map/generation count mismatch")
    protected={}
    for item in map_rows:
        if (set(item)!={"challenge_id","source_train_question_id",
                       "text_sha256","tier"} or item["challenge_id"] in protected):
            raise ValueError("bad verifier-only map schema or repeated identity")
        protected[item["challenge_id"]]=item
    index=BM25Index(docs,title_boost=0.0)
    passed=0
    reasons=Counter()
    for position,row in enumerate(model_rows):
        source=in_rows[position]
        identity=source.get("challenge_id")
        if set(source)!={"challenge_id","question"} or (
            identity not in protected or row.get("challenge_id")!=identity):
            raise ValueError("model output reordered or forge-mapped to another task")
        source_ref=protected[identity]
        q=qmap.get(source_ref["source_train_question_id"])
        if (q is None or q.question!=source["question"] or
            _sha(q.question)!=source_ref["text_sha256"]):
            raise ValueError("model challenge not from trusted TRAIN question")
        packet=make_grounding_packet(
            identity,source["question"],scope=q.scope,index=index,top_k=4)
        if row.get("packet_sha256")!=packet.packet_sha256:
            raise ValueError("model-provided evidence packet checksum mismatch")
        if set(row)!={"challenge_id","question","proposed_answer","reason",
                      "evidence","verdict","packet_sha256","completion_sha256"}:
            raise ValueError("candidate attempted to inject additional fields")
        verdict=row["verdict"]
        if verdict=="quote_extracted_unlabelled_pending_semantic_verification":
            # Re-run independent exact quote check, never using the original
            # model completion text or any TRAIN q.gold_answer as the label.
            raw=json.dumps({
                "question":row["question"],"proposed_answer":row["proposed_answer"],
                "reason":row["reason"],"evidence":row["evidence"],
            },ensure_ascii=False)
            check=verify_grounded_response(raw,packet)
            if (check.verdict!=verdict or
                [dict(x) for x in check.evidence]!=row["evidence"]):
                raise ValueError("grounded model quote proof invalid on trusted corpus")
            passed+=1
        elif row["question"] or row["proposed_answer"] or row["reason"] or row["evidence"]:
            raise ValueError("failed proposal contains undeclared training content")
        reasons[verdict]+=1
    if (summary.get("verdict_histogram")!=dict(reasons) or
        summary.get("observed_quote_structural_passes")!=passed):
        raise ValueError("candidate declared counts inconsistent with actual evidence")
    return {
        "status":"independently_replayed_train_only_grounded_quote_audit",
        "generation":manifest["generation"],
        "candidate_count":len(model_rows),
        "literal_two_document_quote_passes":passed,
        "quote_verdict_counts":dict(reasons),
        "train_questions_sha256":_hash(train_data/"questions.jsonl"),
        "train_documents_sha256":_hash(train_data/"documents.jsonl"),
        "challenger_results_sha256":_hash(proposals_path),
        "grounding_packets_recomputed":len(model_rows),
        "label_or_supporting_ids_sent_to_challenger":False,
        "semantic_entailment_approved":0,
        "independent_answer_truth_approved":0,
        "new_trainable_challenges_approved":0,
        "model_origin_cryptographically_verified":False,
        "model_training_generations":0,
        "multi_generation_RSI_completed":False,
        "caveat":"Exact quote replay is deterministic source integrity, not "
                 "sufficient semantic evidence for newly invented claims. "
                 "The candidate answer may be mentioned in a source but "
                 "still not answer the question. Requires independent "
                 "entailment/answer verification before optimizer use.",
    }
