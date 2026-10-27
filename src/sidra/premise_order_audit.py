"""M7 E-028: semantic-preserving source-order falsification of CPU NLI.

For a hypothesis and SAME verified original HotpotQA TRAIN supporting sentence
multiset, reversing the independent source order cannot change its logical
support status. Model predictions/entailment scores, however, may change.

This is an INPUT-ORDER ROBUSTNESS DIAGNOSTIC, not independent semantic labels,
a trainable Verifier, online QA evaluation, or evidence that a source was read
by the SearchAgent. It never accesses DEV/TEST gold or calls the agent.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

_LABELS=("entailment","neutral","contradiction")
_STATUS="researcher_curated_not_independent_human_adjudication"


def sha256_file(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1048576),b""):
            digest.update(part)
    return digest.hexdigest()


def validate_inputs(rows:list[dict],summary:dict)->list[dict]:
    """Only provenance-pinned TRAIN semantic cases, with no gold used in scorer.

    The original E-026 results are read-only and not treated as ground truth.
    We check their source premise against official support title+index order,
    then vary ONLY line order; same bytes in every original premise line.
    """
    if (summary.get("annotation_status")!=_STATUS
        or summary.get("n_cases")!=36
        or summary.get("independent_question_clusters")!=12
        or summary.get("dev_gold_in_scorer_or_policy") is not False
        or summary.get("eval_A64_or_final_B160_accessed") is not False):
        raise ValueError("requires original TRAIN-only E-026 36/12 provenance manifest")
    if not isinstance(summary.get("input_sha256"),dict):
        raise ValueError("missing immutable TRAIN source digests")
    if set(summary["input_sha256"])!={"train_questions","train_documents",
                                      "original_train","controlled_cases"}:
        raise ValueError("unexpected cohort sources")
    if any(not isinstance(s,str) or len(s)!=64 or
           any(c not in "0123456789abcdef" for c in s)
           for s in summary["input_sha256"].values()):
        raise ValueError("invalid original TRAIN source digest")
    if len(rows)!=36:raise ValueError("require exactly 36 frozen original rows")
    unique=set();qids=set();eligible=[];cases_per_q=Counter()
    for r in rows:
        cid=r.get("case_id")
        qid=r.get("question_id")
        if not isinstance(cid,str) or not cid or cid in unique:
            raise ValueError("duplicate/missing E-026 case ID")
        unique.add(cid)
        if not isinstance(qid,str) or not qid:raise ValueError("missing question cluster")
        qids.add(qid);cases_per_q[qid]+=1
        if r.get("expected_label") not in _LABELS:
            raise ValueError("invalid editorial label")
        premise=r.get("source_premise")
        hypothesis=r.get("hypothesis")
        source_keys=r.get("support_fact_keys")
        if not isinstance(premise,str) or not isinstance(hypothesis,str) or not hypothesis:
            raise ValueError("missing source premise/hypothesis")
        lines=premise.split("\n")
        if (not isinstance(source_keys,list) or
            not 1<=len(source_keys)==len(lines) or
            any(not isinstance(k,list) or len(k)!=2 or
                not isinstance(k[0],str) or
                not isinstance(k[1],int) or k[1]<0 for k in source_keys)):
            raise ValueError("original supporting sentence index/line mismatch")
        if any(not line.startswith(title+": ") or not line[len(title)+2:].strip()
               for line,(title,_) in zip(lines,source_keys,strict=True)):
            raise ValueError("source lines must be exact labelled TRAIN sentence order")
        if len(set((title,index) for title,index in source_keys))!=len(source_keys):
            raise ValueError("duplicate original supporting sentence provenance")
        original=r.get("model_posterior")
        if (not isinstance(original,dict) or set(original)!=set(_LABELS) or
            not all(isinstance(v,(float,int)) and math.isfinite(v) and 0<=v<=1
                    for v in original.values()) or
            abs(sum(original.values())-1)>0.01 or
            r.get("model_predicted_label")!=max(_LABELS,key=lambda x:original[x])):
            raise ValueError("bad original frozen NLI prediction")
        if len(lines)>1:
            reverse="\n".join(reversed(lines))
            if reverse==premise:raise ValueError("reversing sources changed nothing")
            eligible.append({"case_id":cid,"question_id":qid,
                             "phenomenon":r.get("phenomenon"),
                             "expected_label":r["expected_label"],
                             "hypothesis":hypothesis,
                             "original_premise":premise,
                             "permuted_premise":reverse,
                             "source_keys":source_keys,
                             "original_model_posterior":original})
    if len(qids)!=12 or set(cases_per_q.values())!={3}:
        raise ValueError("invalid TRAIN independent question clusters")
    if not eligible:raise ValueError("no multi-source cases")
    return eligible


def _quantile(samples:list[float],p:float)->float:
    return round(sorted(samples)[int(p*(len(samples)-1))],6)


def cluster_bootstrap(rows:list[dict],numerator:str,denominator:str, *,
                      n:int=5000,seed:int=2027)->dict:
    """Question-cluster percentile bootstrap; case rows are not independent."""
    if n<100:raise ValueError("bootstrap needs >=100 draws")
    by_q=defaultdict(list)
    for row in rows:by_q[row["question_id"]].append(row)
    if not by_q:raise ValueError("empty cohort")
    qids=sorted(by_q)
    num=sum(float(x[numerator]) for x in rows)
    den=sum(float(x[denominator]) for x in rows)
    if den<=0:return {"value":None,"count":int(num),"denominator":0,
                      "ci95":[None,None],"eligible_bootstrap_draws":0}
    rng=random.Random(seed);draws=[]
    for _ in range(n):
        groups=[by_q[qids[rng.randrange(len(qids))]] for _ in qids]
        d=sum(float(x[denominator]) for g in groups for x in g)
        if d:
            draws.append(sum(float(x[numerator]) for g in groups for x in g)/d)
    if not draws:raise ValueError("no valid clustered draws")
    return {"value":round(num/den,6),
            "count":round(num,6),"denominator":int(den),
            "ci95":[_quantile(draws,.025),_quantile(draws,.975)],
            "eligible_bootstrap_draws":len(draws),
            "question_clusters":len(qids)}


def run_order_audit(rows:list[dict],summary:dict,scorer, *,
                    out:Path,bootstrap:int=5000,seed:int=2027,
                    threshold:float=0.85)->dict:
    if out.exists():raise FileExistsError(out)
    if not .5<=threshold<1:raise ValueError("fixed high-confidence threshold invalid")
    valid=validate_inputs(rows,summary)
    if scorer is None:raise ValueError("real NLI backend required")
    tok=getattr(scorer,"tokenizer",None)
    max_len=getattr(scorer,"max_length",None)
    reports=[];started=time.monotonic()
    for case in valid:
        permuted=case["permuted_premise"]
        hypothesis=case["hypothesis"]
        pair_tokens=None
        if tok is not None:
            ids=tok(permuted,hypothesis,truncation=False)["input_ids"]
            if ids and isinstance(ids[0],list):
                if len(ids)!=1:raise ValueError("unexpected tokenizer pair batch")
                ids=ids[0]
            pair_tokens=len(ids)
            if max_len is not None and pair_tokens>max_len:
                raise ValueError("reordered pair would truncate; refusing NLI call")
        tick=time.monotonic()
        result=scorer.score([permuted],[hypothesis])
        wall_ms=round(1000*(time.monotonic()-tick),4)
        if len(result)!=1:raise ValueError("NLI returned wrong batch size")
        new=result[0]
        if (set(new)!=set(_LABELS) or
            any(not isinstance(x,(int,float)) or not math.isfinite(x)
                or x<0 or x>1 for x in new.values()) or
            abs(sum(new.values())-1)>0.01):
            raise ValueError("malformed NLI posterior")
        old=case["original_model_posterior"]
        before=max(_LABELS,key=lambda x:old[x])
        after=max(_LABELS,key=lambda x:new[x])
        b_accept=bool(before=="entailment" and old["entailment"]>=threshold)
        a_accept=bool(after=="entailment" and new["entailment"]>=threshold)
        delta=float(new["entailment"]-old["entailment"])
        reports.append({
            "case_id":case["case_id"],"question_id":case["question_id"],
            "phenomenon":case["phenomenon"],
            "expected_label_editorial_not_independent_truth":case["expected_label"],
            "unchanged_hypothesis":hypothesis,
            "same_original_source_keys":case["source_keys"],
            "original_premise":case["original_premise"],
            "reversed_premise":permuted,
            "original_posterior":old,"reversed_posterior":new,
            "original_predicted_label":before,"reversed_predicted_label":after,
            "original_accept_at_fixed_threshold":b_accept,
            "reversed_accept_at_fixed_threshold":a_accept,
            "label_flip":int(before!=after),
            "acceptance_flip":int(b_accept!=a_accept),
            "orig_accept":int(b_accept),"new_accept":int(a_accept),
            "abs_entailment_probability_delta":abs(delta),
            "directional_entailment_probability_delta":delta,
            "one":1,"pair_tokens":pair_tokens,
            "nli_wall_ms":wall_ms,
        })
    # No model calls beyond reorder-candidate scores. All original probabilities
    # are reused from checksum-proven E-026; no retuning or answer labels used.
    flip=cluster_bootstrap(reports,"acceptance_flip","one",n=bootstrap,seed=seed)
    label=cluster_bootstrap(reports,"label_flip","one",n=bootstrap,seed=seed+1)
    absdelta=cluster_bootstrap(
        reports,"abs_entailment_probability_delta","one",n=bootstrap,seed=seed+2)
    summary_out={
        "stage":"M7_E028_train_only_semantic_preserving_premise_order_falsification",
        "status":"diagnostic_only_do_not_promote_default_verifier",
        "case_rows_original":len(rows),
        "eligible_multisource_rows":len(reports),
        "one_source_rows_excluded":len(rows)-len(reports),
        "unique_question_clusters":len(set(x["question_id"] for x in reports)),
        "permutations_per_eligible_case":1,
        "real_new_CPU_NLI_calls":len(reports),
        "unchanged_evidence_multiset":True,
        "unchanged_hypothesis_per_case":True,
        "model_weights_changed":False,
        "threshold_fixed_prior":threshold,
        "label_flip_rate":label,
        "acceptance_flip_rate":flip,
        "mean_abs_entailment_probability_delta":absdelta,
        "accept_before":sum(x["orig_accept"] for x in reports),
        "accept_after":sum(x["new_accept"] for x in reports),
        "acceptance_flips_by_editorial_class":{
            k:{"eligible":sum(r["expected_label_editorial_not_independent_truth"]==k
                               for r in reports),
               "flips":sum(r["expected_label_editorial_not_independent_truth"]==k
                           and r["acceptance_flip"] for r in reports)}
            for k in _LABELS
        },
        "max_abs_p_entail_delta":round(max(
            x["abs_entailment_probability_delta"] for x in reports),6),
        "nli_new_cpu_wall_s":round(time.monotonic()-started,5),
        "source_E026_input_sha256":summary["input_sha256"],
        "source_E026_model_weights_sha256":summary["model_manifest"]["weights_sha256"],
        "original_E026_editorial_labels_human_adjudicated":False,
        "dev_gold_used":False,
        "eval_A64_or_final_B160_accessed":False,
        "agent_tool_budget_changed":False,
        "agent_tokens_and_latency_measured":False,
        "claim":(
            "Order-only changes preserve the available source sentences and "
            "hypothesis. Any changed NLI verdict falsifies order invariance of "
            "this model input, not external truth of a claim. The original "
            "three-way editorial labels are NOT independently adjudicated."),
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"per_case.jsonl").write_text(
        "\n".join(json.dumps(x,ensure_ascii=False,sort_keys=True)
                   for x in reports)+"\n",encoding="utf-8")
    (out/"summary.json").write_text(
        json.dumps(summary_out,ensure_ascii=False,indent=2,sort_keys=True)+"\n",
        encoding="utf-8")
    return summary_out
