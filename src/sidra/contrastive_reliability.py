"""M7 E-027: OFFLINE TRAIN-only paired support-ablation NLI reliability audit.

This is NOT a new SearchAgent rollout or a Verifier policy. It consumes
the immutable E-026 research-curated TRAIN sentence-contrast NLI outputs.
It checks the *same fixed hypothesis* before and after an actual original
support sentence was removed, without calling a model again. A source may
be necessary for the current NLI prediction, but necessity is not semantic
truth and must not be used as an oracle for retrieval/LLM optimization.

All expected labels are EDITORIAL; no independent human NLI ground truth.
Never use DEV / eval-A / final-B rows, or send labels to a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter,defaultdict
from pathlib import Path


_EXPECTED_LABELS={"entailment","contradiction","neutral"}
_EXPECTED_TYPES={
    "entailment":{"bridge_entailment","direct_entailment"},
    "contradiction":{"explicit_negation"},
    "neutral":{"missing_bridge_source","unsupported_bridge"},
}


def digest_file(p:Path)->str:
    with p.open("rb") as fh:
        h=hashlib.sha256()
        for b in iter(lambda:fh.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def _rate_ci_by_question(rows:list[dict],key:str,denominator:str,
                         *,bootstrap:int,seed:int)->dict:
    by_question=defaultdict(list)
    for row in rows:by_question[row["question_id"]].append(row)
    ids=sorted(by_question)
    if not ids:raise ValueError("empty question cohort")
    if bootstrap<100:raise ValueError("bootstrap must be >=100")
    num=sum(float(r[key]) for r in rows)
    den=sum(float(r[denominator]) for r in rows)
    if den<=0:
        # Some legal TRAIN cohorts have zero missing-support contrasts.
        # Explicit null/zero-denominator metrics are more honest than
        # raising or inventing zero model-necessity performance.
        return {"value":None,"numerator":round(num,6),
                "denominator":0,"question_clusters":len(ids),
                "question_cluster_ci95":[None,None],
                "valid_bootstrap_draws":0,"reason":"no_eligible_paired_cases"}
    rng=random.Random(seed)
    samples=[]
    for _ in range(bootstrap):
        chosen=[by_question[ids[rng.randrange(len(ids))]] for _ in ids]
        bd=sum(float(r[denominator]) for g in chosen for r in g)
        if bd>0:
            samples.append(
                sum(float(r[key]) for g in chosen for r in g)/bd)
    samples.sort()
    if not samples:raise ValueError("all cluster resamples have zero denominator")
    return {"value":round(num/den,6),"numerator":round(num,6),
            "denominator":round(den,6),"question_clusters":len(ids),
            "question_cluster_ci95":[
                round(samples[int(.025*(len(samples)-1))],6),
                round(samples[int(.975*(len(samples)-1))],6)],
            "valid_bootstrap_draws":len(samples)}


def compute_diagnostics(
    rows:list[dict], *,entail_cutoff:float=0.85,
    ablation_ceiling:float=0.65,min_drop:float=0.20,
    bootstrap:int=5000,seed:int=2027,
)->dict:
    """Fixed thresholds inherited from E-025; do not fit to E-026 outputs.

    Group by original question. Every question has one positive, one explicit
    negation, and one neutral. The neutral case may be the SAME hypothesis
    with an original support sentence removed (paired ablation), or a
    different unsupported near-miss claim (unpaired).
    """
    if not (0.5<=entail_cutoff<1 and
            0<=ablation_ceiling<entail_cutoff and
            0<min_drop<1):
        raise ValueError("invalid predeclared thresholds")
    if not rows:raise ValueError("empty analysis")
    by_question=defaultdict(list);ids=set()
    for r in rows:
        cid=r.get("case_id");qid=r.get("question_id")
        if not isinstance(cid,str) or not cid or cid in ids:
            raise ValueError("duplicate or invalid case ID")
        ids.add(cid)
        if not isinstance(qid,str) or not qid:
            raise ValueError("invalid question cluster")
        label=r.get("expected_label");phen=r.get("phenomenon")
        if label not in _EXPECTED_LABELS or phen not in _EXPECTED_TYPES[label]:
            raise ValueError("invalid editorial label/phenomenon pair")
        p=r.get("model_posterior")
        if (not isinstance(p,dict) or set(p)!=_EXPECTED_LABELS or
            not all(isinstance(v,(float,int)) and math.isfinite(v)
                    and 0<=v<=1 for v in p.values()) or
            abs(sum(p.values())-1)>0.01):
            raise ValueError("invalid NLI posterior")
        fact_keys=r.get("support_fact_keys")
        if (not isinstance(fact_keys,list) or
            any(not isinstance(x,list) or len(x)!=2
                or not isinstance(x[0],str) or not isinstance(x[1],int)
                or x[1]<0 for x in fact_keys)
            or not fact_keys or
            len({(x[0],x[1]) for x in fact_keys})!=len(fact_keys)):
            raise ValueError("missing/duplicate source-provenance keys")
        if not isinstance(r.get("hypothesis"),str) or not r["hypothesis"]:
            raise ValueError("missing hypothesis")
        if r.get("model_predicted_label") not in _EXPECTED_LABELS:
            raise ValueError("unscored/truncated row; fail closed")
        by_question[qid].append(r)
    if len(rows)!=3*len(by_question):
        raise ValueError("require three editorial cases per question")
    ablations=[];diagnostics=[];sampled=[]
    for qid,group in sorted(by_question.items()):
        labels=Counter(x["expected_label"] for x in group)
        if labels!=Counter({"entailment":1,"contradiction":1,"neutral":1}):
            raise ValueError("each question needs entailment+contradiction+neutral")
        positive=next(x for x in group if x["expected_label"]=="entailment")
        negation=next(x for x in group if x["expected_label"]=="contradiction")
        neutral=next(x for x in group if x["expected_label"]=="neutral")
        p_full=positive["model_posterior"]["entailment"]
        p_neutral=neutral["model_posterior"]["entailment"]
        p_neg=negation["model_posterior"]["entailment"]
        ablation=None
        if neutral["phenomenon"]=="missing_bridge_source":
            # The ONLY legal paired contrast is the exact same hypothesis
            # with an actual strict subset of official support sentence IDs.
            a={(str(title),int(i)) for title,i in positive["support_fact_keys"]}
            b={(str(title),int(i)) for title,i in neutral["support_fact_keys"]}
            if (neutral["hypothesis"]!=positive["hypothesis"] or
                not b<a):
                raise ValueError("invalid missing-bridge support pair")
            delta=float(p_full-p_neutral)
            ablation={
                "question_id":qid,
                "positive_case_id":positive["case_id"],
                "ablated_case_id":neutral["case_id"],
                "support_fact_count_full":len(a),
                "support_fact_count_after_removal":len(b),
                "model_p_full":round(p_full,6),
                "model_p_missing":round(p_neutral,6),
                "entailment_delta":round(delta,6),
                "passes_model_necessity":bool(p_full>=entail_cutoff and
                    p_neutral<=ablation_ceiling and delta>=min_drop),
                "editorial_truth_unchanged":False,
            }
            ablations.append(ablation)
        diagnostics.append({
            "question_id":qid,"n_cases":3,
            "full_confident":int(p_full>=entail_cutoff),
            "negated_confident_false_accept":int(p_neg>=entail_cutoff),
            "neutral_confident_false_accept":int(p_neutral>=entail_cutoff),
            "neutral_case_type":neutral["phenomenon"],
            "negation_model_p":round(p_neg,6),
            "neutral_model_p":round(p_neutral,6),
            "full_model_p":round(p_full,6),
            "has_removal_pair":ablation is not None,
            "support_removal_necessity_pass":(
                ablation["passes_model_necessity"] if ablation else None),
        })
        sampled.append({
            "question_id":qid,"has_pair":int(ablation is not None),
            "delta":(ablation["entailment_delta"] if ablation else 0),
            "necessity":int(bool(ablation and
                                ablation["passes_model_necessity"])),
            "positive_confident":int(p_full>=entail_cutoff),
            "negation_false_accept":int(p_neg>=entail_cutoff),
            "neutral_false_accept":int(p_neutral>=entail_cutoff),
            "case_one":1,
        })
    out={
        "n_cases":len(rows),"n_question_clusters":len(by_question),
        "n_support_removal_pairs":len(ablations),
        "thresholds_predeclared_from_E025":{
            "entailment":entail_cutoff,"ablation_ceiling":ablation_ceiling,
            "minimum_probability_drop":min_drop},
        "metrics":{
            "full_support_confident_accept":_rate_ci_by_question(
                sampled,"positive_confident","case_one",
                bootstrap=bootstrap,seed=seed),
            "explicit_negation_confident_false_accept":_rate_ci_by_question(
                sampled,"negation_false_accept","case_one",
                bootstrap=bootstrap,seed=seed+1),
            "neutral_confident_false_accept":_rate_ci_by_question(
                sampled,"neutral_false_accept","case_one",
                bootstrap=bootstrap,seed=seed+2),
            "paired_support_removal_model_necessity_pass":_rate_ci_by_question(
                sampled,"necessity","has_pair",
                bootstrap=bootstrap,seed=seed+3),
            "paired_support_removal_mean_entailment_delta":_rate_ci_by_question(
                sampled,"delta","has_pair",
                bootstrap=bootstrap,seed=seed+4),
        },
        "neutral_case_false_accept_by_type":{
            type_name:{
                "confident_false_accepts":sum(int(r["neutral_case_type"]==type_name
                    and r["neutral_confident_false_accept"]) for r in diagnostics),
                "count":sum(int(r["neutral_case_type"]==type_name)
                            for r in diagnostics),
            }
            for type_name in ("missing_bridge_source","unsupported_bridge")
        },
        "per_question":diagnostics,
        "per_ablation":ablations,
        "no_new_NLI_calls":True,
        "real_SearchAgent_policy_rollout":False,
        "gold_answers_used_for_policy":False,
        "NLI_ground_truth_established":False,
        "label_quality":"researcher_curated_editorial_not_independently_adjudicated",
        "decision":"diagnostic_only_do_not_promote_to_Verifier",
        "warning":(
            "MODEL necessity of a cited source is not sufficiency or "
            "semantic truth. The editorial neutral near-misses are small "
            "and 2 unsupported_bridge cases are not independent validation. "
            "The paired p-deltas are observational on controlled TRAIN "
            "text; no QA EM, original sentence citations or token/latency "
            "changes in an actual new SearchAgent policy are measured."),
    }
    return out


def audit_file(input_rows:Path,source_summary:Path,out:Path, *,
               bootstrap:int=5000)->dict:
    if out.exists():raise FileExistsError(out)
    source=json.loads(source_summary.read_text(encoding="utf-8"))
    if (source.get("annotation_status")!=
        "researcher_curated_not_independent_human_adjudication"
        or source.get("n_cases")!=36
        or source.get("independent_question_clusters")!=12
        or source.get("eval_A64_or_final_B160_accessed") is not False
        or source.get("dev_gold_in_scorer_or_policy") is not False):
        raise ValueError("untrusted E026 TRAIN-only source manifest")
    # Pin this E-026 original TRAIN fixture; do not analyze arbitrary
    # user-supplied DEV results in this dedicated post-hoc script.
    sha=source.get("input_sha256") or {}
    required={
        "train_questions":"53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8",
        "original_train":"83f03106f12d2c22f36a6957527889fd80f3d13100b03255fc42028ad218f417",
        "controlled_cases":"0034bdaa1644e14c5c8df53502158f6efedfa7fb9860a99c1c721bdcc460a803",
    }
    # Reference fixture pins are deliberately hardcoded to prevent
    # accidentally mixing a DEV cohort with this TRAIN-only audit.
    if any(sha.get(k)!=v for k,v in required.items()):
        raise ValueError("E026 source is not the frozen TRAIN-only cohort")
    rows=[json.loads(x) for x in input_rows.read_text(encoding="utf-8").splitlines()
          if x.strip()]
    if len(rows)!=36:raise ValueError("E026 paired input must have 36 cases")
    report=compute_diagnostics(rows,bootstrap=bootstrap)
    report["source_sha256"]={
        "e026_rows":digest_file(input_rows),
        "e026_summary":digest_file(source_summary),
        **required,
    }
    report["NLI_model"]=source.get("model_manifest")
    out.mkdir(parents=True,exist_ok=False)
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report
