"""Only submit TRAIN-only, independently quote-audited rows to a local reviewer.

The machine's semantic answer is a *triage opinion* and can never approve
training or inherited gold labels. Uses two source-order-independent passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from sidra.rsi_grounded_audit import audit_saved_grounded_candidates
from sidra.rsi_semantic_triage import (
    PASS_VERDICT,LocalSemanticReviewer,build_review_packet,consensus,
)


def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,train_data:Path,queue:Path,manifest:Path,verifier_map:Path,
    generated:Path,out_dir:Path,endpoint:str,model:str,seed:int=43,
)->dict:
    if out_dir.exists():raise FileExistsError(out_dir)
    # Fail before API access if frozen input, quote proofs or train-only
    # provenance have been altered since original Challenger generation.
    independent=audit_saved_grounded_candidates(
        train_data,queue,manifest,verifier_map,generated)
    reviewer=LocalSemanticReviewer(endpoint,model)
    src=generated/"UNLABELLED_GROUNDED_CANDIDATES.jsonl"
    rows=[json.loads(x) for x in src.read_text(encoding="utf-8").splitlines()]
    passed=[x for x in rows if x["verdict"]==PASS_VERDICT]
    if len(passed)!=independent["literal_two_document_quote_passes"]:
        raise RuntimeError("independent source quote count mismatch")
    seen=set()
    results=[]
    for index,candidate in enumerate(passed):
        packet=build_review_packet(candidate)
        if packet.challenge_id in seen:
            raise ValueError("duplicated verifier challenge")
        seen.add(packet.challenge_id)
        reviews=[
            reviewer.review(packet,reverse=False,seed=seed+2*index),
            reviewer.review(packet,reverse=True,seed=seed+2*index+1),
        ]
        result=consensus(packet,reviews)
        results.append(result)
        print("SEMANTIC_TRIAGE",len(results),"/",len(passed),
              result["challenge_id"],result["conclusion"],flush=True)
    statuses=dict(Counter(x["conclusion"] for x in results))
    summary={
        "status":"model_semantic_triage_unlabelled_never_gold",
        "trusted_quote_verifier_report":independent,
        "train_dataset_questions_sha256":sha(train_data/"questions.jsonl"),
        "train_dataset_documents_sha256":sha(train_data/"documents.jsonl"),
        "source_candidates_sha256":sha(src),
        "model_reviewed_candidates":len(results),
        "source_order_reviewer_calls":len(results)*2,
        "reviewer_model":model,
        "endpoint_is_local_loopback":True,
        "seed":seed,
        "verdict_histogram":statuses,
        "novel_train_labels_approved":0,
        "model_optimizer_updates":0,
        "multi_generation_RSI_completed":False,
        "no_dev_eval_a_or_final_b_access":True,
        "caution":"Qwen model judgment, even with two consistent exact quote witnesses, "
                  "cannot establish ground truth, cross-entity uniqueness, or RSI improvement. "
                  "This is a candidate rejection/independent review prioritization filter.",
    }
    out_dir.mkdir(parents=True,exist_ok=False)
    with (out_dir/"SEMANTIC_TRIAGE.jsonl").open("x",encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n")
    (out_dir/"SEMANTIC_TRIAGE_SUMMARY.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return summary


def main():
    ap=argparse.ArgumentParser()
    for key in (
        "--train-data","--queue","--manifest","--verifier-map",
        "--generated","--out-dir",
    ):
        ap.add_argument(key,required=True,type=Path)
    ap.add_argument("--endpoint",default="http://127.0.0.1:18986/v1/chat/completions")
    ap.add_argument("--model",default="sidra-8b-semantic-reviewer")
    ap.add_argument("--seed",type=int,default=4300)
    x=ap.parse_args()
    summary=run(
        train_data=x.train_data,queue=x.queue,manifest=x.manifest,
        verifier_map=x.verifier_map,generated=x.generated,
        out_dir=x.out_dir,endpoint=x.endpoint,model=x.model,seed=x.seed)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
