"""M7 E-029: freeze a disjoint, DOUBLE-BLIND HUMAN-REVIEW REQUEST cohort.

Only original HotpotQA TRAIN question/answer/supporting sentence keys are used
for construction; no DEV labels and no model predictions enter the packets.
One question produces ONE task (full or ablated), preventing a rater from
seeing two variants of the same question. Two shuffled packets contain the
same tasks WITHOUT the full/ablated flag, question ID, editorial label, model
prediction, or evaluator answer key. Raw packets stay on Tang, not GitHub.

IMPORTANT: these are UNSCORED human-annotation tasks, not independently
adjudicated ground truth. The source TRAIN gold answer becomes a *candidate*
answer to examine, never a certified entailment label.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from sidra.io import load_dataset
from sidra.metrics import normalize_answer
from sidra.semantic_contrast import file_sha256

_REVIEW_LABELS=("supported","contradicted","insufficient_evidence","unclear")
_REVIEW_PROMPT=(
    "Using ONLY the quoted source sentences, decide whether they logically "
    "support the proposed answer to this exact question. Do not use outside "
    "world knowledge; missing evidence means insufficient, not contradiction. "
    "Record uncertain/ambiguous cases as unclear. Do not infer which evidence "
    "was removed or discuss cases with the other reviewer."
)


def _digest(s:str)->str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def freeze_blind_packet(*,dataset:Path,original_train:Path,
                        prior_e026_rows:Path,out:Path,seed:int=2029,
                        bridge_count:int=48,comparison_count:int=16)->dict:
    if out.exists():raise FileExistsError(out)
    if bridge_count<=0 or comparison_count<=0 or any(
        n%2 for n in (bridge_count,comparison_count)):
        raise ValueError("positive, even target per question type required")
    _,questions=load_dataset(dataset)
    if not questions or any(x.split!="train" for x in questions):
        raise ValueError("blind benchmark requires TRAIN-only converted questions")
    qmap={x.question_id:x for x in questions}
    if len(qmap)!=len(questions):
        raise ValueError("duplicate converted TRAIN questions")
    originals=json.loads(original_train.read_text(encoding="utf-8"))
    original_map={str(r["_id"]):r for r in originals}
    if not originals or len(original_map)!=len(originals) or set(qmap)!=set(original_map):
        raise ValueError("original/converted TRAIN cohort IDs disagree")
    for qid,q in qmap.items():
        source=original_map[qid]
        if (q.question!=source.get("question") or
            normalize_answer(q.gold_answer)!=normalize_answer(str(source.get("answer","")))):
            raise ValueError("TRAIN original/converted answers disagree")
    e026=[json.loads(line) for line in prior_e026_rows.read_text().splitlines()]
    if len(e026)!=36 or len({x["case_id"] for x in e026})!=36:
        raise ValueError("prior E-026 exclusion must be complete and immutable")
    exclude={str(x["question_id"]) for x in e026}
    if len(exclude)!=12 or not exclude<=set(qmap):
        raise ValueError("unexpected original E-026 question IDs")
    candidates={"bridge":[],"comparison":[]}
    for qid,src in original_map.items():
        if qid in exclude:continue
        kind=src.get("type")
        if kind not in candidates:continue
        facts=src.get("supporting_facts")
        if (not isinstance(facts,list) or len(facts)<2
            or len(set((str(t),i) for t,i in facts))!=len(facts)):
            continue
        contexts={str(title):sentences for title,sentences in src["context"]}
        titles={str(t) for t,i in facts}
        if len(titles)!=2:continue
        if any(t not in contexts or not isinstance(i,int) or i<0
               or i>=len(contexts[t]) for t,i in facts):
            continue
        candidates[kind].append((qid,src,contexts))
    targets={"bridge":bridge_count,"comparison":comparison_count}
    registry=[];packets=[];selection_ids=[]
    for kind,needed in targets.items():
        eligible=sorted(candidates[kind],key=lambda r:(_digest(
            f"{seed}:m7-e029:{r[0]}"),r[0]))
        if len(eligible)<needed:
            raise ValueError(f"insufficient new TRAIN {kind} samples")
        for ix,(qid,src,contexts) in enumerate(eligible[:needed]):
            # Pre-balance *within each source type*; independent of model,
            # answer or editorial labels. Missing half is NOT assumed neutral.
            variant="full_sources" if ix%2==0 else "one_support_title_omitted"
            facts=[(str(t),i) for t,i in src["supporting_facts"]]
            dropped_title=None
            if variant!="full_sources":
                names=sorted({t for t,i in facts})
                dropped_title=names[int(_digest(f"{seed}:{qid}:drop")[:2],16)%2]
                selected=[(t,i) for t,i in facts if t!=dropped_title]
            else:
                selected=facts
            if not selected:raise ValueError("missing source facts after omission")
            clue=_digest(f"m7-e029:{seed}:{qid}:{variant}")[:22]
            claim_id=f"blind-{clue}"
            evidence=[
                {"title":t,"source_sentence_index":i,"quoted_sentence":str(contexts[t][i])}
                for t,i in selected]
            if not all(x["quoted_sentence"].strip() for x in evidence):
                raise ValueError("empty cited TRAIN original supporting sentence")
            raw_candidate={
                "claim_id":claim_id,
                "question":str(src["question"]),
                "proposed_answer":str(src["answer"]),
                "provided_evidence":evidence,
                "rater_instruction":_REVIEW_PROMPT,
                "allowed_labels":_REVIEW_LABELS,
            }
            if any(k in raw_candidate for k in ("question_id","gold_answer",
                "expected_label","model_posterior","variant")):
                raise AssertionError("reviewer sees evaluator-only fields")
            packets.append(raw_candidate)
            registry.append({
                "claim_id":claim_id,"source_train_question_id":qid,
                "source_type":kind,"variant_private":variant,
                "omitted_title_private":dropped_title,
                "all_original_support_fact_keys":[[t,i] for t,i in facts],
                "shown_original_support_fact_keys":[[t,i] for t,i in selected],
                "independent_human_labels_available":False,
                "question_answer_from_TRAIN_not_semantically_verified":True,
            })
            selection_ids.append(qid)
    if len(set(selection_ids))!=sum(targets.values()):
        raise AssertionError("duplicate TRAIN questions in blind packet")
    if set(selection_ids)&exclude:raise AssertionError("E-026 leakage")
    out.mkdir(parents=True,exist_ok=False)
    for label,nonce in (("A","reviewer-A"),("B","reviewer-B")):
        shuffled=sorted(packets,key=lambda x:_digest(
            f"{seed}:{nonce}:{x['claim_id']}"))
        if len(set(x["claim_id"] for x in shuffled))!=len(shuffled):
            raise AssertionError("duplicate blind annotation card")
        (out/f"REVIEWER_{label}_BLIND.jsonl").write_text(
            "\n".join(json.dumps(x,ensure_ascii=False,sort_keys=True)
                      for x in shuffled)+"\n",encoding="utf-8")
    (out/"REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl").write_text(
        "\n".join(json.dumps(x,ensure_ascii=False,sort_keys=True)
                  for x in sorted(registry,key=lambda x:x["claim_id"]))+"\n",
        encoding="utf-8")
    manifest={
        "study":"M7_E029_prospective_TRAIN_human_annotation_packet_NOT_ADJUDICATED",
        "cohort":"independent_from_E026_and_disjoint_by_question",
        "split":"train","question_count":len(packets),"unique_question_count":len(set(selection_ids)),
        "excluded_prior_E026_question_count":len(exclude),
        "source_type_balance":{"bridge":bridge_count,"comparison":comparison_count},
        "variant_balance_private":{
            "full_sources":sum(x["variant_private"]=="full_sources" for x in registry),
            "one_support_title_omitted":sum(x["variant_private"]!="full_sources" for x in registry),
        },
        "one_card_per_question_per_reviewer":True,
        "reviewer_A_B_have_same_claims_but_different_order":True,
        "reviewer_packets_exclude_gold_expected_labels_model_predictions":True,
        "human_labels_received":0,
        "independent_human_adjudication_completed":False,
        "model_semantic_accuracy_known":False,
        "allow_as_verifier_threshold_calibration":False,
        "excluded_question_ids_sha256":_digest("\n".join(sorted(exclude))),
        "source_sha256":{
            "converted_questions":file_sha256(dataset/"questions.jsonl"),
            "converted_documents":file_sha256(dataset/"documents.jsonl"),
            "original_train":file_sha256(original_train),
            "prior_E026_frozen_rows":file_sha256(prior_e026_rows),
        },
        "seed":seed,
        "reviewer_instructions":_REVIEW_PROMPT,
        "required_follow_up":(
            "Two truly independent human reviewers must separately annotate "
            "all claim IDs and resolve disagreements by third-party review. "
            "Blind machine-generated agreement is NOT independent evidence. "
            "Holdout questions cannot be fed to training while used for evaluation."),
        "dev_gold_in_policy_or_scorer":False,
        "eval_A64_or_final_B160_accessed":False,
        "no_agent_calls_or_model_weight_updates":True,
    }
    (out/"MANIFEST.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8")
    return manifest


def compare_independent_reviews(*,review_a:Path,review_b:Path,manifest:Path)->dict:
    """Audit machine-readable ratings, NOT a proof ratings were made by humans."""
    meta=json.loads(manifest.read_text())
    if meta.get("human_labels_received")!=0 or meta.get("independent_human_adjudication_completed") is not False:
        raise ValueError("unexpectedly pre-adjudicated source packet")
    def read(file:Path):
        rows=[json.loads(x) for x in file.read_text(encoding="utf-8").splitlines()]
        if len(rows)!=meta["question_count"]:raise ValueError("incomplete review")
        ids=set();out={}
        for r in rows:
            if set(r)!={"claim_id","rating","reviewer_id","human_attestation"}:
                raise ValueError("unsupported review metadata")
            if r["rating"] not in _REVIEW_LABELS or r["human_attestation"] is not True:
                raise ValueError("invalid rating / missing reviewer attestation")
            if not isinstance(r["claim_id"],str) or r["claim_id"] in ids:
                raise ValueError("bad/duplicate claim ID")
            ids.add(r["claim_id"]);out[r["claim_id"]]=r
        if len({x["reviewer_id"] for x in rows})!=1:
            raise ValueError("each review must come from one declared reviewer")
        return out,next(iter({x["reviewer_id"] for x in rows}))
    a,aid=read(review_a);b,bid=read(review_b)
    if aid==bid or not aid or not bid or set(a)!=set(b):
        raise ValueError("must have independent declared reviewers for same cases")
    disagreements=[
        cid for cid in sorted(a)
        if a[cid]["rating"]!=b[cid]["rating"]
    ]
    return {
        "n":len(a),"rater_identifiers_distinct":True,
        "independently_human_annotated_confirmed_by_software":False,
        "raw_agreement_rate":round((len(a)-len(disagreements))/len(a),6),
        "disagreements_require_third_party_resolution":disagreements,
        "all_cases_gold_ready":False,
        "warning":"Human independence and adjudication must be verified outside code; declared reviewer IDs alone do not prove it.",
    }
