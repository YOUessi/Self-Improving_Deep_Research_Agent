"""M7 E-025: CPU TRAIN-only counterfactual citation NLI audit on REAL rollouts.

No live model SearchAgent run or RL optimization. This opt-in OFFLINE
verifier uses exactly the already-recorded SEARCH/READ/FINISH observations,
no reference answer in the semantic scorer. Original HotpotQA labels are
joined ONLY AFTER scoring for outcome diagnostics.

Compares on identical episodes:
  raw confident MNLI, source title-link gate, individual citation necessity,
  and intersection; all are MODEL/HEURISTIC screens, NOT human entailment truth.
A model confidently approving a wrong-answer example is a false-accept proxy,
not a labelled NLI false positive. Independent experimental semantics labels
are still needed before deployment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

from sidra.citation_counterfactual import citation_leave_one_out
from sidra.evidence_diagnostics import offline_train_diagnosis
from sidra.io import iter_jsonl,load_dataset,save_jsonl
from sidra.metrics import normalize_answer
from sidra.semantic_verifier import LocalCPUMNLI,semantic_review
from scripts.m7_train_semantic_nli_audit import question_cluster_bootstrap


METRICS=(
    "nli_scored","raw_nli_entailment","title_link_accept",
    "loco_accept","title_and_loco_accept",
    "raw_nli_wrong_answer_accept","title_link_wrong_answer_accept",
    "loco_wrong_answer_accept","title_and_loco_wrong_answer_accept",
    "raw_nli_missing_original_sentence_accept",
    "title_link_missing_original_sentence_accept",
    "loco_missing_original_sentence_accept",
    "title_and_loco_missing_original_sentence_accept",
    "answer_em","support_retrieval_complete","support_read_complete",
    "original_sentence_cited_complete","tool_calls","model_calls",
    "prompt_tokens","completion_tokens","elapsed_ms",
    "loco_additional_nli_calls","loco_additional_cpu_ms",
    "loco_minus_title_accept","loco_minus_title_wrong",
)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def audit_counterfactual_train(*,dataset:Path,original_train:Path,
                               rollouts:list[Path],scorer,out:Path,
                               bootstrap:int=2000,model_manifest:dict|None=None)->dict:
    if out.exists():raise FileExistsError(out)
    if not rollouts:raise ValueError("require real TRAIN-only rollout files")
    _,qs=load_dataset(dataset)
    if not qs or any(q.split!="train" for q in qs):
        raise ValueError("TRAIN-only verifier refuses DEV/TEST gold")
    qmap={q.question_id:q for q in qs}
    if len(qmap)!=len(qs):raise ValueError("duplicate training questions")
    original_rows=json.loads(original_train.read_text(encoding="utf-8"))
    orig={str(row["_id"]):row for row in original_rows}
    if len(orig)!=len(original_rows) or set(qmap)!=set(orig):
        raise ValueError("original/converted TRAIN cohort IDs mismatch")
    # Invariants are validated before ANY model scoring, no original labels
    # or gold fields are ever arguments to semantic_review/LOCO.
    for qid,q in qmap.items():
        row=orig[qid]
        if (row.get("question")!=q.question or
            normalize_answer(row.get("answer",""))!=normalize_answer(q.gold_answer)):
            raise ValueError("original/converted TRAIN labels mismatch")

    processed=[]
    seen=set()
    start=time.monotonic()
    for file in rollouts:
        if not file.is_file():raise FileNotFoundError(file)
        source=file.parent.name
        stream=list(iter_jsonl(file))
        if not stream:raise ValueError("empty real TRAIN rollouts")
        for sample in stream:
            qid=str(sample.get("question_id",""))
            trace=sample["trace"]
            index=sample.get("rollout_index")
            key=(source,qid,index)
            if key in seen:raise ValueError("duplicate trained rollout episode")
            seen.add(key)
            q=qmap.get(qid)
            if q is None or (trace.get("question_id")!=qid
                             or trace.get("question")!=q.question):
                raise ValueError("rollout ID/question does not match TRAIN")
            review=semantic_review(trace,q.question,scorer)
            tick=time.monotonic()
            gate=citation_leave_one_out(
                trace,q.question,scorer,base_review=review,
                entailment_threshold=0.85,removal_ceiling=0.65,
                min_entailment_drop=0.20)
            gate_ms=(time.monotonic()-tick)*1000
            offline=offline_train_diagnosis(q,trace,orig[qid])

            raw=int(review.status=="model_predicted_entailment")
            title=int(raw and review.observed_bridge_link is True)
            loco=int(gate.status=="counterfactual_necessary_model_prediction")
            both=int(bool(title and loco))
            wrong=int(not offline["answer_em"])
            missing=int(not offline["original_sentence_cited_complete"])
            row={
                "question_id":qid,"source_experiment":source,
                "rollout_index":index,
                "model_semantic_status":review.status,
                "model_semantic_reason":review.reason,
                "model_predicted_entailment_probability":review.entailment,
                "observed_cross_citation_title_link":review.observed_bridge_link,
                "loco_status":gate.status,
                "loco_reason":gate.reason,
                "loco_per_citation_entailment_if_removed":gate.ablated_entailments,
                "loco_weakest_entailment_drop":gate.weakest_entailment_drop,
                "loco_model_input_tokens_max":gate.model_input_tokens_max,
                "loco_additional_nli_calls":gate.additional_nli_calls,
                "loco_additional_cpu_ms":round(gate_ms,4),
                "nli_scored":int(review.status!="not_scored"),
                "raw_nli_entailment":raw,
                "title_link_accept":title,"loco_accept":loco,
                "title_and_loco_accept":both,
                "raw_nli_wrong_answer_accept":raw*wrong,
                "title_link_wrong_answer_accept":title*wrong,
                "loco_wrong_answer_accept":loco*wrong,
                "title_and_loco_wrong_answer_accept":both*wrong,
                "raw_nli_missing_original_sentence_accept":raw*missing,
                "title_link_missing_original_sentence_accept":title*missing,
                "loco_missing_original_sentence_accept":loco*missing,
                "title_and_loco_missing_original_sentence_accept":both*missing,
                "loco_minus_title_accept":loco-title,
                "loco_minus_title_wrong":loco*wrong-title*wrong,
                **{m:offline[m] for m in ("answer_em",
                   "support_retrieval_complete","support_read_complete",
                   "original_sentence_cited_complete","tool_calls",
                   "model_calls","prompt_tokens","completion_tokens",
                   "elapsed_ms")},
            }
            processed.append(row)

    if not processed:raise ValueError("no real episode diagnostics")
    assert all(r["nli_scored"]>=r["raw_nli_entailment"]>=
               r["title_link_accept"] for r in processed)
    boots=question_cluster_bootstrap(processed,METRICS,repeats=bootstrap)
    result={
        "stage":"TRAIN-only real-trace citation-necessity LOCO NLI falsification pilot",
        "implementation":"M7 E-025 fixed 0.85/0.65/.20 model-probability thresholds",
        "n_episodes":len(processed),
        "n_question_clusters":len(set(x["question_id"] for x in processed)),
        "cohort_rollouts":[str(x) for x in rollouts],
        "source_sha256":{
            "train_questions":sha256(dataset/"questions.jsonl"),
            "train_documents":sha256(dataset/"documents.jsonl"),
            "original_train":sha256(original_train),
            **{"rollout_"+str(i):sha256(file) for i,file in enumerate(rollouts)},
        },
        "NLI_model":model_manifest or {"name":"test_injected_mock_scorer"},
        "counts":{metric:sum(row[metric] for row in processed)
                  for metric in METRICS if metric in (
                    "nli_scored","raw_nli_entailment","title_link_accept",
                    "loco_accept","title_and_loco_accept",
                    "raw_nli_wrong_answer_accept",
                    "title_link_wrong_answer_accept",
                    "loco_wrong_answer_accept",
                    "title_and_loco_wrong_answer_accept",
                    "raw_nli_missing_original_sentence_accept",
                    "title_link_missing_original_sentence_accept",
                    "loco_missing_original_sentence_accept",
                    "title_and_loco_missing_original_sentence_accept",
                    "loco_additional_nli_calls",
                  )},
        "loco_status":dict(Counter(row["loco_status"] for row in processed)),
        "loco_reason":dict(Counter(row["loco_reason"] for row in processed)),
        "question_cluster_bootstrap":boots,
        "bootstrap_question_cluster_not_episode":True,
        "run_wall_s":round(time.monotonic()-start,4),
        "no_dev_correct_answer_available_to_verifier":True,
        "no_eval_A64_or_final_B160_processed":True,
        "model_weights_changed":False,
        "semantic_ground_truth_established":False,
        "critical_caution":(
            "LOCO only tests model dependence on cited READs; a lower "
            "MNLI score is not factual necessity or semantic proof. "
            "Small same-question rollout clusters, no independent NLI "
            "annotations; not a deployable/verifiably calibrated gate. "
            "This replay DOES NOT improve search/answer policy."
        ),
    }
    out.mkdir(parents=True,exist_ok=False)
    save_jsonl(out/"train_only_per_episode.jsonl",processed)
    (out/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",type=Path,required=True)
    p.add_argument("--original-train",type=Path,required=True)
    p.add_argument("--rollouts",type=Path,action="append",required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--bootstrap",type=int,default=2000)
    p.add_argument("--max-length",type=int,default=384)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    cpu=LocalCPUMNLI(str(a.model),max_length=a.max_length)
    weights=next((f for f in (a.model/"model.safetensors",
                              a.model/"pytorch_model.bin") if f.is_file()),None)
    if weights is None:raise FileNotFoundError("No local saved MNLI weights")
    manifest={"name":a.model.name,
              "architecture":type(cpu.model).__name__,
              "checkpoint_sha256":sha256(weights),
              "label_map":cpu.label_map,
              "max_input_tokens":cpu.max_length,
              "device":"cpu","web_download":False}
    report=audit_counterfactual_train(
        dataset=a.dataset,original_train=a.original_train,
        rollouts=a.rollouts,scorer=cpu,out=a.out,
        bootstrap=a.bootstrap,model_manifest=manifest)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
