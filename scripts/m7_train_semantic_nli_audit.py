"""M7 E-024: TRAIN-only real-rollout semantic NLI screening; CPU ONLY.

Evaluates existing immutable TRAIN-only actual SearchAgent episodes. Does not
modify agents, tool budgets, training labels, rollout policies or weights.
NLI scores on the actual cited READ snippets and conservative wh->claim only.
Outputs TRAIN per-episode diagnostics and a question-cluster bootstrap.
Never accept DEV/TEST, eval-A64 or final-B160; no network or GPU execution.

Example:
 CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
 PYTHONPATH=src:. python scripts/m7_train_semantic_nli_audit.py \\
  --dataset ~/experiments/sidra_m4/data/train_oracle1024_seed43/converted \\
  --original-train ~/experiments/sidra_m4/data/train_oracle1024_seed43/original_train_subset.json \\
  --rollouts ~/experiments/sidra_m6/results/real_train_rollouts12x4_stop/rollouts_train.jsonl \\
  --rollouts ~/experiments/sidra_m6/results/train_rollout_io8x4/rollouts_train.jsonl \\
  --model <local_cached_MNLI_snapshot> --out <NEW_OUT> --bootstrap 2000

Limited coverage is expected and counted as abstention, NOT as a proof that
model-verifier semantic reasoning has been solved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

from sidra.io import load_dataset,iter_jsonl,save_jsonl
from sidra.metrics import normalize_answer
from sidra.semantic_verifier import LocalCPUMNLI,semantic_review


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def question_cluster_bootstrap(
    rows:list[dict],metrics:tuple[str,...],*,repeats:int=2000,seed:int=2027,
)->dict:
    if not rows or repeats<100:raise ValueError("insufficient rows or bootstrap repeats")
    groups:dict[str,list[dict]]=defaultdict(list)
    for row in rows:
        groups[str(row["question_id"])].append(row)
    group_ids=sorted(groups)
    rng=random.Random(seed)
    out={}
    for metric in metrics:
        real=sum(float(x[metric]) for x in rows)/len(rows)
        vals=[]
        for _ in range(repeats):
            sample=[groups[group_ids[rng.randrange(len(group_ids))]]
                    for _ in group_ids]
            num=sum(float(row[metric]) for subset in sample for row in subset)
            denom=sum(len(subset) for subset in sample)
            vals.append(num/denom)
        vals.sort()
        out[metric]={
            "mean":round(real,6),
            "lower95":round(vals[int(0.025*(repeats-1))],6),
            "upper95":round(vals[int(0.975*(repeats-1))],6),
            "question_clusters":len(group_ids),
            "episodes":len(rows),
        }
    return out


def audit_train(
    *,dataset:Path,original_train:Path,rollouts:list[Path],
    scorer,out:Path,bootstrap:int=2000,model_manifest:dict|None=None,
    require_bridge_link:bool=False,
)->dict:
    if out.exists():raise FileExistsError(out)
    if not rollouts:raise ValueError("need TRAIN-only rollout files")
    docs,questions=load_dataset(dataset)
    if any(q.split!="train" for q in questions):
        raise ValueError("TRAIN-only audit rejects DEV/TEST labels")
    lookup={q.question_id:q for q in questions}
    if len(lookup)!=len(questions):raise ValueError("duplicate TRAIN IDs")
    originals=json.loads(original_train.read_text())
    by_original={str(x["_id"]):x for x in originals}
    if len(by_original)!=len(originals) or not originals:
        raise ValueError("original TRAIN IDs duplicate or empty")
    # The original TRAIN labels stay inside the offline evaluator and are
    # NEVER passed to the online semantic_review scorer.
    from sidra.evidence_diagnostics import offline_train_diagnosis
    started=time.monotonic()
    records=[]
    for path in rollouts:
        episodes=list(iter_jsonl(path))
        if not episodes:raise ValueError("empty rollout file")
        for sample in episodes:
            qid=str(sample["question_id"])
            if qid not in lookup or qid not in by_original:
                raise ValueError("rollout is not in original TRAIN split")
            q=lookup[qid]
            tr=sample["trace"]
            if (tr.get("question_id")!=qid or tr.get("question")!=q.question):
                raise ValueError("rollout question or ID differs from TRAIN")
            # Source-only NLI gets only user question, assistant answer, actual
            # SEARCH/READ/FINISH trace — neither gold nor supporting indices.
            start=time.monotonic()
            review=semantic_review(tr,q.question,scorer,
                                   require_bridge_link=require_bridge_link)
            cpu_s=time.monotonic()-start
            offline=offline_train_diagnosis(q,tr,by_original[qid])
            records.append({
                "question_id":qid,"source_rollout":path.name,
                "source_experiment":path.parent.name,
                "rollout_index":sample.get("rollout_index"),
                "model_answer":str(tr.get("answer",""))[:200],
                "model_semantic_status":review.status,
                "model_semantic_reason":review.reason,
                "model_entailment_probability":review.entailment,
                "model_neutral_probability":review.neutral,
                "model_contradiction_probability":review.contradiction,
                "hypothesis":review.hypothesis,
                "model_evidence_snippets":review.evidence_sentence_count,
                "observed_bridge_link":review.observed_bridge_link,
                "bridge_link_rejected":int(
                    review.status=="model_entailment_blocked_no_observed_bridge"),
                "raw_nli_entailment_above_threshold":int(
                    review.model_prediction=="entailment"
                    and review.entailment is not None
                    and review.entailment>=0.85),
                "nli_screen_elapsed_ms":round(cpu_s*1000,3),
                "model_predicted_entailment":int(review.status=="model_predicted_entailment"),
                "model_predicted_contradiction":int(review.status=="model_predicted_contradiction"),
                "nli_scored":int(review.entailment is not None),
                "answer_em":offline["answer_em"],
                "support_retrieval_complete":offline["support_retrieval_complete"],
                "support_read_complete":offline["support_read_complete"],
                "original_sentence_read_complete":offline["original_sentence_read_complete"],
                "original_sentence_cited_complete":offline["original_sentence_cited_complete"],
                "strict_sentence_and_em_but_not_entailment":offline[
                    "strict_sentence_and_em_but_not_entailment"],
                "tool_calls":offline["tool_calls"],
                "model_calls":offline["model_calls"],
                "prompt_tokens":offline["prompt_tokens"],
                "completion_tokens":offline["completion_tokens"],
                "elapsed_ms":offline["elapsed_ms"],
            })
    aggregates=question_cluster_bootstrap(records,(
        "nli_scored","model_predicted_entailment","model_predicted_contradiction",
        "answer_em","support_retrieval_complete","support_read_complete",
        "original_sentence_cited_complete","strict_sentence_and_em_but_not_entailment",
        "tool_calls","model_calls","prompt_tokens","completion_tokens","elapsed_ms",
    ),repeats=bootstrap)
    status=Counter(r["model_semantic_status"] for r in records)
    positives=[x for x in records if x["model_predicted_entailment"]]
    reason=Counter(r["model_semantic_reason"] for r in records)
    report={
        "stage":"TRAIN-only offline CPU MNLI experimental screening",
        "n_episodes":len(records),
        "n_question_clusters":len(set(x["question_id"] for x in records)),
        "nli_scored_count":sum(x["nli_scored"] for x in records),
        "cross_citation_title_link_required":require_bridge_link,
        "cross_citation_link_observed_count":sum(
            x["observed_bridge_link"] is True for x in records),
        "raw_nli_confident_entailment_count":sum(
            x["raw_nli_entailment_above_threshold"] for x in records),
        "raw_nli_confident_entailment_blocked_count":sum(
            x["bridge_link_rejected"] for x in records),
        "nli_status_counts":dict(sorted(status.items())),
        "nli_reason_counts":dict(sorted(reason.items())),
        "predicted_entailment_count":len(positives),
        "predicted_entailment_gold_wrong_count":sum(
            1-x["answer_em"] for x in positives),
        "predicted_entailment_missing_original_sentence_count":sum(
            1-x["original_sentence_cited_complete"] for x in positives),
        "question_cluster_bootstrap":aggregates,
        "rollout_tool_averages":{
            k:aggregates[k]["mean"] for k in
            ("tool_calls","model_calls","prompt_tokens","completion_tokens","elapsed_ms")
        },
        "nli_wall_total_seconds":round(time.monotonic()-started,3),
        "nli_calls":getattr(scorer,"calls",None),
        "source_sha256":{
            "documents.jsonl":sha256_file(dataset/"documents.jsonl"),
            "questions.jsonl":sha256_file(dataset/"questions.jsonl"),
            "original_train":sha256_file(original_train),
            **{str(p):sha256_file(p) for p in rollouts},
        },
        "NLI_model":model_manifest or {"kind":"injected_test_only"},
        "no_dev_correct_answer_available_to_verifier":True,
        "no_eval_A64_or_final_B160_processed":True,
        "semantic_entailment_ground_truth_measured":False,
        "model_weights_changed":False,
        "meaning":(
            "An abstaining CPU MNLI *model prediction* for question-to-claim "
            "on already-READ citations. No independent entailment ground truth, "
            "no model task gain, no calibrated threshold and no policy update."
        ),
    }
    out.mkdir(parents=True,exist_ok=False)
    save_jsonl(out/"train_only_per_episode.jsonl",records)
    (out/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--original-train",type=Path,required=True)
    p.add_argument("--rollouts",type=Path,action="append",required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--max-length",type=int,default=384)
    p.add_argument("--require-bridge-link",action="store_true",
                   help="Optional gold-blind cited-title cross-document linkage gate")
    args=p.parse_args()
    weights=args.model/"model.safetensors"
    if not weights.is_file():raise FileNotFoundError(
        "Local offline safetensors NLI checkpoint is required")
    cpu=LocalCPUMNLI(str(args.model),max_length=args.max_length)
    manifest={"name":args.model.name,
              "architecture":type(cpu.model).__name__,
              "checkpoint_sha256":sha256_file(weights),
              "label_map":cpu.label_map,
              "max_input_tokens":cpu.max_length,
              "device":"cpu",
              "web_download":False}
    result=audit_train(
        dataset=args.dataset,original_train=args.original_train,
        rollouts=args.rollouts,scorer=cpu,
        out=args.out,bootstrap=args.bootstrap,model_manifest=manifest,
        require_bridge_link=args.require_bridge_link)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
