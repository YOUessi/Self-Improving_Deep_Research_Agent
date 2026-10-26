"""Fail-closed paired DEV audit for on-policy action-policies.

This is a *research evaluation gate* for the same frozen HotpotQA DEV slice.
It refuses comparisons with different dataset SHA, code SHA, decoding-policy,
sampling mode, seed, tool budgets, model or nonidentical question IDs.

Never runs a model. It reads already-finished immutable per-example results,
recalculates paired improvements and computes exact discordant-pair binomial
tests plus seeded bootstrap confidence bounds. No DEV labels enter training.

Usage:
 PYTHONPATH=src python scripts/audit_paired_onpolicy.py \\
   --baseline /.../dev_pilot32_SFT \\
   --candidate /.../dev_pilot32_NEW \\
   --out /.../M7_PAIR_AUDIT.json

Only validated independent improvements qualify as *candidate* promotion;
this does NOT turn a public DEV gain into independent heldout evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

METRICS=(
    "exact_match","verified_success","evidence_recall","token_f1",
    "tool_calls","model_calls","completion_tokens","elapsed_ms",
)
BINARY_METRICS=("exact_match","verified_success")


def _hash(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda:file.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path:Path)->dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_examples(path:Path)->dict[str,dict]:
    rows={}
    for i,line in enumerate(path.read_text(encoding="utf-8").splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        qid=str(row.get("question_id",""))
        if not qid or qid in rows:
            raise ValueError(f"missing or duplicate question ID at {path}:{i}")
        rows[qid]=row
    if not rows:raise ValueError("empty DEV evaluation")
    return rows


def _check_finite_companion(b_dir:Path,c_dir:Path)->dict:
    """Runtime decoder intervention cannot change between paired arms."""
    p="FINITE_ACTION_POLICY.json"
    a=b_dir/p
    b=c_dir/p
    if a.exists()!=b.exists():
        raise ValueError("finite-action decoder companion exists in only one arm")
    if not a.exists():
        return {"kind":"legacy_unconstrained","distribution_documented":False}
    x,y=_json(a),_json(b)
    unchanged=("policy_class","selection_mode","categorical_distribution",
               "random_seed_keying","trained_on_eval_gold","test_scope",
               "tool_budget_max","step_budget_max")
    for k in unchanged:
        if x.get(k)!=y.get(k):
            raise ValueError(f"decoder/selection mismatch at {k}")
    if (x.get("selection_mode") not in {"sample","greedy"}
            or x.get("trained_on_eval_gold") is not False):
        raise ValueError("unsafe on-policy test contamination")
    if (not x.get("source_adapter_sha256")
            or x["source_adapter_sha256"]==y.get("source_adapter_sha256")):
        raise ValueError("same underlying adapter weights in two evaluation arms")
    return {
        "kind":"finite_constrained_categorical_agent",
        "distribution_documented":True,
        "selection_mode":x["selection_mode"],
        "old_adapter_sha256":x["source_adapter_sha256"],
        "new_adapter_sha256":y["source_adapter_sha256"],
    }


def _validate_manifests(b_dir:Path,c_dir:Path)->tuple[dict,dict,dict]:
    m0,m1=_json(b_dir/"manifest.json"),_json(c_dir/"manifest.json")
    if m0.get("split")!="dev" or m1.get("split")!="dev":
        raise ValueError("DEV-only paired comparison (never sealed TEST/final)")
    for m in (m0,m1):
        if any(x in str(m.get("dataset_dir","")).casefold()
               for x in ("final-b160","sealed_final","final_b160")):
            raise ValueError("sealed final set must never enter pilot audits")
        if m.get("synthetic_fixture") or m.get("no_weight_updates_in_this_evaluation") is not True:
            raise ValueError("expected frozen DEV evaluation without training")
    for field in (
        "data_sha256","split","agent_config","seed","sample_count",
        "evaluation_offset","generation_max_new_tokens","model_id",
        "model_or_policy","agent_roles","git_commit",
    ):
        if m0.get(field)!=m1.get(field):
            raise ValueError(f"unmatched paired experiment: {field}")
    datasets=m0.get("data_sha256") or {}
    if set(datasets)!={"documents.jsonl","questions.jsonl"} or not all(
        isinstance(v,str) and len(v)==64 for v in datasets.values()):
        raise ValueError("missing immutable dataset SHA256 in manifest")
    if not isinstance(m0.get("sample_count"),int) or m0["sample_count"]<1:
        raise ValueError("bad sample count")
    decoder=_check_finite_companion(b_dir,c_dir)
    if decoder["kind"]=="legacy_unconstrained":
        if (not m0.get("adapter_path")
                or m0["adapter_path"]==m1.get("adapter_path")):
            raise ValueError("cannot identify two different model adapters")
    return m0,m1,decoder


def _sign_p_improvement(positive:int,negative:int)->float:
    """One-sided exact sign-test p under paired null; ties excluded."""
    n=positive+negative
    if not n:return 1.0
    return sum(math.comb(n,k) for k in range(positive,n+1))/(2**n)


def _bootstrap(deltas:list[float],*,rng:random.Random,draws:int)->list[float]:
    n=len(deltas)
    means=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n
                 for __ in range(draws))
    return [round(means[int(draws*0.025)],6),
            round(means[min(draws-1,int(draws*0.975))],6)]


def audit_pair(baseline:Path,candidate:Path,out:Path, *,
               seed:int=20261027,bootstrap:int=5000)->dict:
    if bootstrap<200:
        raise ValueError("at least 200 bootstrap draws required")
    if out.exists():
        raise FileExistsError(out)
    b_manifest,c_manifest,policy=_validate_manifests(baseline,candidate)
    b_rows=_read_examples(baseline/"per_example.jsonl")
    c_rows=_read_examples(candidate/"per_example.jsonl")
    if b_rows.keys()!=c_rows.keys():
        raise ValueError("nonidentical question IDs: paired comparison invalid")
    if len(b_rows)!=b_manifest["sample_count"]:
        raise ValueError("per-example count disagrees with frozen manifest")
    ids=sorted(b_rows)
    for qid in ids:
        for row in (b_rows[qid],c_rows[qid]):
            if row.get("split")!="dev":
                raise ValueError("contaminated per-example split")
    rng=random.Random(seed)
    measured={}
    for name in METRICS:
        try:
            before=[float(b_rows[q][name]) for q in ids]
            after=[float(c_rows[q][name]) for q in ids]
        except (KeyError,ValueError,TypeError) as exc:
            raise ValueError(f"non-numeric/missing metric {name}") from exc
        if any(not math.isfinite(v) for v in before+after):
            raise ValueError(f"nonfinite metric {name}")
        if name in BINARY_METRICS and any(v not in (0.0,1.0) for v in before+after):
            raise ValueError(f"non-binary success flag {name}")
        diff=[a-b for a,b in zip(after,before)]
        res={
            "before_mean":round(sum(before)/len(ids),6),
            "after_mean":round(sum(after)/len(ids),6),
            "paired_delta_mean":round(sum(diff)/len(ids),6),
            "paired_delta_ci95":_bootstrap(diff,rng=rng,draws=bootstrap),
            "improved_cases":sum(x>0 for x in diff),
            "worsened_cases":sum(x<0 for x in diff),
        }
        if name in BINARY_METRICS:
            res.update({
                "before_count":int(sum(before)),
                "after_count":int(sum(after)),
                "discordant_exact_sign_p_one_sided":
                    round(_sign_p_improvement(res["improved_cases"],
                                              res["worsened_cases"]),7),
            })
        measured[name]=res
    answer=measured["exact_match"]
    evidence=measured["verified_success"]
    # A public DEV pilot improvement is merely suggestive, and even that
    # requires >0 gains and intervals excluding zero on BOTH primary metrics.
    both_positive=all(
        m["paired_delta_mean"]>0 and m["paired_delta_ci95"][0]>0
        for m in (answer,evidence))
    result={
        "stage":"onpolicy_paired_dev_audit","n":len(ids),
        "sealed_final_B160_used":False,"eval_gold_used_for_training":False,
        "same_frozen_data_sha256":b_manifest["data_sha256"],
        "same_eval_source_commit":b_manifest["git_commit"],
        "same_budget":b_manifest["agent_config"],
        "same_model_id":b_manifest["model_id"],
        "eval_selection_seed":b_manifest["seed"],
        "rng_seed":seed,"bootstrap_draws":bootstrap,
        "policy_comparison":policy,
        "baseline_adapter_path":b_manifest.get("adapter_path"),
        "candidate_adapter_path":c_manifest.get("adapter_path"),
        "baseline_metrics_sha256":_hash(baseline/"per_example.jsonl"),
        "candidate_metrics_sha256":_hash(candidate/"per_example.jsonl"),
        "metrics":measured,
        "pilot_positive_gate":both_positive,
        "promote_to_production":False,
        "independent_generalization_established":False,
        "conclusion":(
            "Positive matched public DEV pilot; requires independent multi-seed "
            "evaluation before research benefit can be claimed."
            if both_positive else
            "No statistically supported joint answer-and-evidence improvement "
            "on matched public DEV pilot; keep baseline."
        ),
        "caution":(
            "Small paired bootstrap/one-sided sign p are descriptive. "
            "Do not tune on or open sealed final-B160. "
            "This report is an evaluator artifact, never GRPO reward."
        ),
    }
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",
                   encoding="utf-8")
    return result


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--candidate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=20261027)
    p.add_argument("--bootstrap",type=int,default=5000)
    args=p.parse_args()
    result=audit_pair(
        args.baseline,args.candidate,args.out,
        seed=args.seed,bootstrap=args.bootstrap)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
