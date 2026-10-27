"""E-025: independently replayed, TRAIN-only Challenger -> Trainer quarantine.

This is a *read-only* source-integrity and trust-domain transfer audit. It
replays the E-022/023/024 evidence against original TRAIN documents and checks
the saved reports byte-for-byte. No source-trained LLM's reviewer opinion can
turn an invented question into a gold label: new tasks remain unlabelled until
independent, separately authenticated adjudication exists.

The 3 Challenger *cohorts* are NOT 3 model-training generations. This module
does not claim any optimizer step or evaluate public DEV/final-B.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from sidra.io import load_dataset
from sidra.rsi_challenge_scheduler import _norm
from sidra.rsi_grounded_audit import audit_saved_grounded_candidates
from sidra.rsi_semantic_triage import PASS_VERDICT,build_review_packet
from scripts.audit_rsi_semantic_posthoc import classify_conservatively


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def load_object(path:Path)->dict:
    row=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(row,dict):raise ValueError(f"not a JSON object: {path.name}")
    return row


def load_rows(path:Path)->list[dict]:
    data=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():raise ValueError("blank row in protected artifact")
        item=json.loads(line)
        if not isinstance(item,dict):raise ValueError("non-object protected record")
        data.append(item)
    if not data:raise ValueError("empty protected input")
    return data


def frozen_public_dev_ids(manifest_path:Path)->tuple[set[str],str]:
    """Read ONLY public DEV split IDs, never answer labels or any final-B file.

    Existing frozen M4 manifest has disjoint 'pilot' and 'heldout' ID lists.
    Reject malformed splits so accidental or adversarial TRAIN/DEV reuse
    fails before any cohort is processed.
    """
    p=str(manifest_path.resolve()).casefold()
    if any(x in p for x in ("final-b","final_b","b160")):
        raise ValueError("never inspect the sealed final-B ID/label manifest")
    manifest=load_object(manifest_path)
    joined=set()
    for split in ("pilot","heldout"):
        row=manifest.get(split)
        if not isinstance(row,dict):
            raise ValueError("missing frozen public DEV ID split")
        ids=row.get("selected_ids")
        if (not isinstance(ids,list) or not ids
            or not all(isinstance(q,str) and q for q in ids)
            or type(row.get("questions")) is not int
            or len(ids)!=row["questions"] or len(set(ids))!=len(ids)
            or joined.intersection(ids)):
            raise ValueError("malformed or overlapping frozen DEV ID split")
        joined.update(ids)
    return joined,sha(manifest_path)


@dataclass(frozen=True)
class Cohort:
    queue_dir:Path
    grounded_dir:Path
    semantic_dir:Path
    posthoc_dir:Path


def review_cohort(train_data:Path,cohort:Cohort)->tuple[dict,list[dict]]:
    qdir=cohort.queue_dir
    queue=qdir/"CHALLENGER_INPUT.jsonl"
    manifest=qdir/"QUEUE_MANIFEST.json"
    protected=qdir/"VERIFIER_ONLY_MAP.jsonl"
    grounded=cohort.grounded_dir/"UNLABELLED_GROUNDED_CANDIDATES.jsonl"
    original_reviews=cohort.semantic_dir/"SEMANTIC_TRIAGE.jsonl"
    original_summary=cohort.semantic_dir/"SEMANTIC_TRIAGE_SUMMARY.json"
    posthoc=cohort.posthoc_dir/"POSTHOC_NEGATIVE_REVIEW.jsonl"
    posthoc_summary=cohort.posthoc_dir/"POSTHOC_NEGATIVE_REVIEW_SUMMARY.json"
    provenance=load_object(manifest)
    if (provenance.get("split")!="train"
        or provenance.get("gold_fields_in_challenger_input") is not False
        or provenance.get("training_or_checkpoint_updates")!=0):
        raise ValueError("unsafe challenger queue provenance")
    replay=audit_saved_grounded_candidates(
        train_data,queue,manifest,protected,cohort.grounded_dir)
    src=load_object(original_summary)
    gate=load_object(posthoc_summary)
    if (
        src.get("status")!="model_semantic_triage_unlabelled_never_gold"
        or src.get("source_candidates_sha256")!=sha(grounded)
        or src.get("novel_train_labels_approved")!=0
        or src.get("model_optimizer_updates")!=0
        or gate.get("status")!="CPU_posthoc_negative_only_reclassification"
        or gate.get("split")!="train"
        or gate.get("upgrades")!=0
        or gate.get("approved_novel_train_labels")!=0
        or gate.get("optimizer_updates")!=0
        or gate.get("multi_generation_RSI_completed") is not False
        or gate.get("original_saved_model_reviews_sha256")!=sha(original_reviews)
        or gate.get("original_saved_model_summary_sha256")!=sha(original_summary)
        or gate.get("original_frozen_challenger_sha256")!=sha(grounded)
        or gate.get("train_questions_sha256")!=sha(train_data/"questions.jsonl")
        or gate.get("train_documents_sha256")!=sha(train_data/"documents.jsonl")
    ):
        raise ValueError("candidate/reviewer/posthoc artifact chain is untrusted")

    questions=load_rows(queue)
    maps=load_rows(protected)
    generated=load_rows(grounded)
    original=load_rows(original_reviews)
    final=load_rows(posthoc)
    by_map={x["challenge_id"]:x for x in maps}
    if len(by_map)!=len(maps) or len(by_map)!=len(questions):
        raise ValueError("malformed original train identity map")
    by_generated={x["challenge_id"]:x for x in generated}
    by_original={x["challenge_id"]:x for x in original}
    by_final={x["challenge_id"]:x for x in final}
    if (len(by_generated)!=len(generated)
        or len(by_original)!=len(original)
        or len(by_final)!=len(final)):
        raise ValueError("duplicate candidate/review/posthoc identity")
    qids={x["challenge_id"] for x in questions}
    if not set(by_generated).issubset(qids):
        raise ValueError("generated challenge not in frozen queue")
    if set(by_original)!=set(by_final):
        raise ValueError("posthoc outcome missing original model-review decision")

    pending=set()
    counts=Counter()
    prepared=[]
    for r in generated:
        challenge_id=r["challenge_id"]
        if challenge_id not in by_map:raise ValueError("missing independent TRAIN map")
        original_question=next(x["question"] for x in questions
                               if x["challenge_id"]==challenge_id)
        item=by_map[challenge_id]
        if not item.get("source_train_question_id") or not item.get("tier") in (
                "direct","bridge","hard"):
            raise ValueError("missing TRAIN source provenance or difficulty tier")
        if r["verdict"]!=PASS_VERDICT:
            if challenge_id in by_original:
                raise ValueError("failed literal quote candidate reached semantic reviewer")
            status="quote_rejected"
        else:
            packet=build_review_packet(r)
            old=by_original.get(challenge_id)
            saved=by_final.get(challenge_id)
            if old is None or saved is None:
                raise ValueError("missing independently reviewed quote-valid candidate")
            expected=classify_conservatively(packet,old)
            if expected!=saved:
                raise ValueError("semantic posthoc audit was tampered or not reproducible")
            status=saved["revised_conclusion"]
            if status=="needs_independent_semantic_label_review":
                pending.add(challenge_id)
            elif status not in (
                "flagged_unsupported","invalid_reviewer_output",
                "flagged_deterministic_premise_gap",
            ):
                raise ValueError("unknown semantic decision may confer label authority")
        counts[status]+=1
        prepared.append({
            "challenge_id":challenge_id,
            "source_train_id":item["source_train_question_id"],
            "source_question_text":original_question,
            "tier":item["tier"],
            "status":status,
            "candidate_question_text":r["question"] if r["verdict"]==PASS_VERDICT else "",
        })
    if (
        len(original)!=replay["literal_two_document_quote_passes"]
        or len(final)!=gate["n"]
        or len(final)!=gate["trusted_quote_replay_passes"]
        or gate.get("reclassified_counts")!=dict(Counter(x["revised_conclusion"] for x in final))
        or gate.get("original_status_counts")!=dict(Counter(x["original_conclusion"] for x in final))
        or gate.get("safety_downgrades")!=sum(
            x["original_conclusion"]!=x["revised_conclusion"] for x in final)
    ):
        raise ValueError("reported semantic audit counts differ from re-evaluated facts")
    return ({
        "cohort_index":provenance["generation"],
        "sampling_seed":provenance["seed"],
        "queue_file_sha256":sha(queue),
        "verifier_mapping_sha256":sha(protected),
        "grounded_candidates_sha256":sha(grounded),
        "semantic_reviews_sha256":sha(original_reviews),
        "posthoc_reviews_sha256":sha(posthoc),
        "original_train_sources":len(prepared),
        "quote_valid_n":len(original),
        "status_counts":dict(counts),
        "pending_independent_labels_n":len(pending),
    },prepared)


def audit_training_admission(
    train_data:Path,cohorts:list[Cohort],out_dir:Path, *,
    public_dev_id_manifest:Path,
    near_duplicate_cutoff:float=0.93)->dict:
    """Repeat all source checks before permitting any Solver->Trainer handoff.

    No novel label authority is implemented. Therefore export ONLY a zero-byte
    trainer input file (SHA-bound), never a machine-reviewed 'positive'.
    """
    if out_dir.exists():raise FileExistsError(out_dir)
    if len(cohorts)<2 or not 0.8<near_duplicate_cutoff<=1.0:
        raise ValueError("need >=2 cohorts, sensible near duplicate bound")
    path=str(train_data.resolve()).casefold()
    if any(part in path for part in ("final-b","final_b","b160","/dev/","/test/","/heldout/")):
        raise ValueError("evaluator/sealed corpus path prohibited")
    _docs,questions=load_dataset(train_data)
    if not questions or any(q.split!="train" for q in questions):
        raise ValueError("only TRAIN questions may enter RSI curator")
    qmap={q.question_id:q for q in questions}
    if len(qmap)!=len(questions):raise ValueError("duplicated TRAIN source IDs")
    reserved_public_dev_ids,dev_id_sha=frozen_public_dev_ids(public_dev_id_manifest)
    if set(qmap)&reserved_public_dev_ids:
        raise ValueError("TRAIN question IDs overlap frozen public DEV IDs")
    manifests=[]
    all_rows=[]
    for c in cohorts:
        m,records=review_cohort(train_data,c)
        manifests.append(m)
        all_rows+=records
    ordered=sorted(manifests,key=lambda x:x["cohort_index"])
    if ([x["cohort_index"] for x in ordered]!=list(range(1,len(cohorts)+1))
        or len({x["sampling_seed"] for x in manifests})!=len(cohorts)):
        raise ValueError("reused seed or noncontiguous Challenger cohorts")
    ids=[x["challenge_id"] for x in all_rows]
    source_ids=[x["source_train_id"] for x in all_rows]
    if len(ids)!=len(set(ids)) or len(source_ids)!=len(set(source_ids)):
        raise ValueError("cross-cohort duplicate challenge/source identity")
    for x in all_rows:
        source=qmap.get(x["source_train_id"])
        if source is None or source.question!=x["source_question_text"]:
            raise ValueError("Challenger is not bound to original TRAIN question")
    # Cross-cohort novelty check applies to *proposed novel question text*;
    # does NOT grant factual truth or authority to use those propositions.
    novel_texts=[_norm(x["candidate_question_text"]) for x in all_rows
                 if x["candidate_question_text"]]
    for i,text in enumerate(novel_texts):
        for other in novel_texts[:i]:
            if SequenceMatcher(None,text,other).ratio()>=near_duplicate_cutoff:
                raise ValueError("cross-cohort near-duplicate proposed new questions")

    result={
        "status":"TRAIN_ONLY_REPLAYED_QUARANTINE_NO_TRAINER_AUTHORITY",
        "source_domain":"original HotpotQA TRAIN converted corpus",
        "train_questions_sha256":sha(train_data/"questions.jsonl"),
        "train_documents_sha256":sha(train_data/"documents.jsonl"),
        "frozen_public_dev_id_manifest_sha256":dev_id_sha,
        "frozen_public_dev_id_count":len(reserved_public_dev_ids),
        "train_ids_disjoint_from_public_dev":True,
        "cohorts":ordered,
        "n_distinct_train_origins":len(all_rows),
        "n_generated_quote_audited":sum(x["quote_valid_n"] for x in manifests),
        "n_semantic_pending":sum(x["pending_independent_labels_n"] for x in manifests),
        "aggregate_status_counts":dict(Counter(x["status"] for x in all_rows)),
        "independent_verified_novel_labels":0,
        "eligible_novel_training_rows":0,
        "trainer_input_rows":0,
        "trainer_input_shasum_sha256":hashlib.sha256(b"").hexdigest(),
        "distinct_optimizer_training_generations":0,
        "trainer_checkpoint_sha256":None,
        "model_weights_updated_by_this_stage":False,
        "multi_generation_RSI_completed":False,
        "final_B160_accessed":False,
        "dev_gold_exported_to_Challenger_or_Trainer":False,
        "scientific_limit":(
            "Three Qwen Challenger candidate cohorts are NOT three optimiser "
            "generations. Same-model semantic reviews, literal quotes and "
            "CPU rejection rules cannot grant ground truth. Pending cases "
            "require an external independently validated TRAIN source proof."
        ),
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    (out_dir/"TRAINER_INPUT.jsonl").open("xb").close()
    (out_dir/"ADMISSION_AUDIT.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    return result
