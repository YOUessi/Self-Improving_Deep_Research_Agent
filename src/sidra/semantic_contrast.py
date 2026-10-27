"""M7 E-026: provenance-checked TRAIN-only NLI controlled contrast suite.

Unlike the E-025 80 real model rollouts, this module evaluates short
researcher-curated textual *hypotheses* on official original HotpotQA TRAIN
supporting sentences. It DOES NOT train the agent, fake SearchAgent traces or
establish independent human-annotated semantic ground truth.

The three-way 'expected_label' is an editorial controlled-case annotation
from exact TRAIN supporting sentences (not the HotpotQA answer). It is kept
out of model calls. Each source question is a single bootstrap cluster.
Hard neutral cases explicitly include one unsupported/incorrect bridge.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter,defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sidra.io import load_dataset
from sidra.metrics import normalize_answer

LABELS=("entailment","contradiction","neutral")


def file_sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


@dataclass(frozen=True)
class ContrastCase:
    case_id:str
    question_id:str
    premise:str
    hypothesis:str
    expected_label:str
    phenomenon:str
    fact_ids:tuple[tuple[str,int],...]
    source_type:str
    editorial_rationale:str


def resolve_controlled_cases(
    *,dataset:Path,original_train:Path,cases_path:Path,
)->tuple[list[ContrastCase],dict]:
    """Reject non-TRAIN, nonsupporting or inconsistent original annotations."""
    _,questions=load_dataset(dataset)
    if not questions or any(q.split!="train" for q in questions):
        raise ValueError("TRAIN-only suite refuses DEV/TEST labels")
    qmap={q.question_id:q for q in questions}
    if len(qmap)!=len(questions):raise ValueError("duplicate TRAIN question ID")
    original=json.loads(original_train.read_text(encoding="utf-8"))
    omap={str(row["_id"]):row for row in original}
    if not original or len(omap)!=len(original) or set(qmap)!=set(omap):
        raise ValueError("converted/original TRAIN cohort IDs do not match")
    for qid,q in qmap.items():
        row=omap[qid]
        if (row.get("question")!=q.question or
            normalize_answer(str(row.get("answer","")))!=
            normalize_answer(q.gold_answer)):
            raise ValueError("original TRAIN labels differ from converted TRAIN")
    suite=json.loads(cases_path.read_text(encoding="utf-8"))
    if suite.get("version")!="m7-e026-v1":
        raise ValueError("unreviewed contrast suite version")
    if suite.get("annotation_status")!="researcher_curated_not_independent_human_adjudication":
        raise ValueError("suite cannot claim independent human ground truth")
    if not isinstance(suite.get("cases"),list) or not suite["cases"]:
        raise ValueError("empty controlled cases")
    allowed={"case_id","question_id","fact_ids","drop_fact_ids","hypothesis",
             "expected_label","phenomenon","rationale"}
    result=[];seen=set();question_ids=set();label_counter=Counter()
    for item in suite["cases"]:
        if set(item)!=allowed:
            raise ValueError("unsupported field in semantic fixture, possible gold exposure")
        cid=item["case_id"];qid=item["question_id"]
        if (not isinstance(cid,str) or not cid.strip() or cid in seen or
            qid not in omap):
            raise ValueError("invalid/duplicate case ID or non-TRAIN question")
        seen.add(cid);question_ids.add(qid)
        row=omap[qid]
        original_facts={(str(t),int(i)) for t,i in row.get("supporting_facts",[])}
        fact_ids=tuple((str(t),int(i)) for t,i in item["fact_ids"])
        dropped={(str(t),int(i)) for t,i in item["drop_fact_ids"]}
        if (len(fact_ids)<1 or len(fact_ids)!=len(set(fact_ids))
            or not set(fact_ids)<=original_facts or not dropped<set(fact_ids)):
            raise ValueError("fact selection not a valid subset of original TRAIN support")
        selected=tuple(f for f in fact_ids if f not in dropped)
        context={title:sentences for title,sentences in row["context"]}
        for title,idx in selected:
            if title not in context or idx<0 or idx>=len(context[title]):
                raise ValueError("invalid original supporting sentence index")
        hypothesis=str(item["hypothesis"]).strip()
        if not 15<=len(hypothesis)<=320 or not hypothesis.endswith("."):
            raise ValueError("expected well-formed fixed semantic hypothesis")
        label=str(item["expected_label"])
        if label not in LABELS:raise ValueError("invalid controlled-case label")
        premise="\n".join(f"{title}: {context[title][idx]}" for title,idx in selected)
        if not premise.strip():raise ValueError("empty original TRAIN premise")
        result.append(ContrastCase(
            case_id=cid,question_id=qid,premise=premise,hypothesis=hypothesis,
            expected_label=label,phenomenon=str(item["phenomenon"]),
            fact_ids=selected,source_type=str(row.get("type","")),
            editorial_rationale=str(item["rationale"])))
        label_counter[label]+=1
    if len(question_ids)<8 or any(label_counter[x]<8 for x in LABELS):
        raise ValueError("insufficient independent TRAIN groups or label coverage")
    meta={
        "source_sha256":{
            "train_questions":file_sha256(dataset/"questions.jsonl"),
            "train_documents":file_sha256(dataset/"documents.jsonl"),
            "original_train":file_sha256(original_train),
            "controlled_cases":file_sha256(cases_path),
        },
        "unique_train_question_clusters":len(question_ids),
        "label_counts":dict(label_counter),
        "editorial_labels_not_human_adjudicated":True,
    }
    return result,meta


def _paired_token_length(scorer,premise:str,hypothesis:str)->int|None:
    tok=getattr(scorer,"tokenizer",None)
    if tok is None:return None
    ids=tok(premise,hypothesis,truncation=False)["input_ids"]
    if ids and isinstance(ids[0],list):
        if len(ids)!=1:raise ValueError("unexpected tokenization batch")
        ids=ids[0]
    return len(ids)


def _check_row(probs:dict[str,float])->None:
    if (set(probs)!=set(LABELS)
        or any(not isinstance(x,(float,int)) or not math.isfinite(x) or x<0 or x>1
               for x in probs.values())
        or abs(sum(probs.values())-1)>0.01):
        raise ValueError("invalid MNLI posterior in controlled suite")


def _cluster_rate(
    rows:list[dict],clusters:list[str],metric:str,denominator:str,
    *,repeats:int,seed:int,
)->dict:
    if repeats<100:raise ValueError("bootstrap requires >=100 draws")
    grouped=defaultdict(list)
    for row in rows:grouped[row["question_id"]].append(row)
    rng=random.Random(seed)
    top=sum(float(x[metric]) for x in rows)
    bottom=sum(float(x[denominator]) for x in rows)
    observed=top/bottom if bottom else None
    boots=[]
    for _ in range(repeats):
        sample=[grouped[clusters[rng.randrange(len(clusters))]] for _ in clusters]
        total=sum(float(x[denominator]) for group in sample for x in group)
        if total:
            boots.append(sum(float(x[metric]) for group in sample
                             for x in group)/total)
    if not boots:raise ValueError("no bootstrap resample contains denominator")
    boots.sort()
    return {
        "rate":round(observed,6) if observed is not None else None,
        "numerator":int(top),"denominator":int(bottom),
        "question_cluster_ci95":[
            round(boots[int(.025*(len(boots)-1))],6),
            round(boots[int(.975*(len(boots)-1))],6),
        ],
        "usable_cluster_resamples":len(boots),
    }


def run_controlled_suite(cases:list[ContrastCase],scorer,*,meta:dict,
                         out:Path,bootstrap:int=2000,seed:int=2027)->dict:
    """Never use DEV data. NLI scorer only receives literal premise+hypothesis.

    No SearchAgent calls, no policy tuning, no model weight writes. Labels
    appear only in post-score confusion and editorial audit outputs.
    """
    if out.exists():raise FileExistsError(out)
    if bootstrap<100:raise ValueError("bootstrap below preregistered minimum")
    rows=[];started=time.monotonic()
    for case in cases:
        token_len=_paired_token_length(scorer,case.premise,case.hypothesis)
        max_len=getattr(scorer,"max_length",None)
        # Critical: reject silently truncated NLI. Existing LocalCPUMNLI uses
        # truncation="only_first"; raw token gate must run before model.score.
        if token_len is not None and max_len is not None and token_len>max_len:
            pred="not_scored_truncation"
            probs={x:None for x in LABELS}
            time_ms=0.0
        else:
            tick=time.monotonic()
            scores=scorer.score([case.premise],[case.hypothesis])
            time_ms=round(1000*(time.monotonic()-tick),4)
            if len(scores)!=1:raise ValueError("NLI backend returned wrong batch")
            probs=scores[0]
            _check_row(probs)
            pred=max(LABELS,key=lambda x:probs[x])
        accepted=int(pred=="entailment" and probs["entailment"] is not None
                     and probs["entailment"]>=.85)
        valid=int(pred in LABELS)
        row={
            "case_id":case.case_id,"question_id":case.question_id,
            "phenomenon":case.phenomenon,
            "source_type":case.source_type,"expected_label":case.expected_label,
            "model_predicted_label":pred,"model_posterior":probs,
            "model_entailment_confident_085":accepted,
            "false_accept_non_entail":int(accepted and case.expected_label!="entailment"),
            "positive_accept":int(accepted and case.expected_label=="entailment"),
            "positive_rejected":int(not accepted and case.expected_label=="entailment"),
            "accuracy_3way":int(pred==case.expected_label),
            "not_entail_label":int(case.expected_label!="entailment"),
            "entail_label":int(case.expected_label=="entailment"),
            "all_cases":1,"nli_scored":valid,
            "nli_cpu_inference_ms":time_ms,
            "input_pair_token_length":token_len,
            "max_model_tokens":max_len,
            # Editorial rationale and source sentence text are not sent to
            # scorer; they are retained for human inspection only.
            "editorial_rationale":case.editorial_rationale,
            "support_fact_keys":[[t,i] for t,i in case.fact_ids],
            "hypothesis":case.hypothesis,
            "source_premise":case.premise,
        }
        rows.append(row)
    clusters=sorted({x.question_id for x in cases})
    confusion=defaultdict(Counter)
    for r in rows:confusion[r["expected_label"]][r["model_predicted_label"]]+=1
    comparisons={
        "three_way_accuracy":_cluster_rate(
            rows,clusters,"accuracy_3way","all_cases",repeats=bootstrap,seed=seed),
        "non_entail_confident_false_accept_rate":_cluster_rate(
            rows,clusters,"false_accept_non_entail","not_entail_label",
            repeats=bootstrap,seed=seed+1),
        "entailment_confident_accept_recall":_cluster_rate(
            rows,clusters,"positive_accept","entail_label",
            repeats=bootstrap,seed=seed+2),
        "nli_coverage":_cluster_rate(
            rows,clusters,"nli_scored","all_cases",
            repeats=bootstrap,seed=seed+3),
    }
    report={
        "stage":"M7 E-026 TRAIN-only researcher-curated controlled NLI stress, NOT live agent",
        "annotation_status":"researcher_curated_not_independent_human_adjudication",
        "n_cases":len(rows),
        "independent_question_clusters":len(clusters),
        "case_types":dict(Counter(x.phenomenon for x in cases)),
        "confusion_matrix":{label:dict(confusion[label]) for label in LABELS},
        "paired_group_metrics":comparisons,
        "model_manifest":meta.get("NLI_model"),
        "input_sha256":meta["source_sha256"],
        "bootstrap_repeats":bootstrap,"bootstrap_seed":seed,
        "nli_calls":getattr(scorer,"calls",None),
        "inference_wall_s":round(time.monotonic()-started,3),
        "NLI_model_weights_changed":False,
        "SearchAgent_modified":False,
        "agent_tool_calls_per_case":None,
        "agent_prompt_tokens_per_case":None,
        "agent_answer_EM":None,
        "agent_sentence_citation_success":None,
        "dev_gold_in_scorer_or_policy":False,
        "eval_A64_or_final_B160_accessed":False,
        "interpretation":(
            "Controlled TRAIN textual hypotheses and official supporting sentences, "
            "not externally adjudicated semantics or real SearchAgent QA. "
            "A negative case is a researcher-chosen contrast, and NLI confidence "
            "is not evidence truth; no promotion or deployment from this suite."
        ),
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"controlled_rows.jsonl").write_text(
        "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),
        encoding="utf-8")
    (out/"summary.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",
                                    encoding="utf-8")
    return report
