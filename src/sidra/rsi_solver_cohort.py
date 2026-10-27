"""E-027: three-stage original-TRAIN cohort model rollout with distinct trust domains.

PREPARER (trusted, CPU) sees the private TRAIN source/opaque map and creates
a sanitized model-only retrieval corpus. The SOLVER process receives only
the sanitized documents + questions (NO original questions.jsonl, private
map, gold answer, supporting IDs or DEV labels). The VERIFIER (CPU, offline)
independently remakes that projection from canonical TRAIN source, replays
every BM25 Search/READ/FINISH and computes TRAIN-only rewards. Model output
is NOT a cryptographic attestation of model origin. Constrained vLLM
episodes here are NOT optimizer-ready on-policy training trajectories.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter,defaultdict
from dataclasses import asdict
from pathlib import Path

from sidra.agent import AgentConfig,SearchAgent
from sidra.experiment import sha256_file
from sidra.io import iter_jsonl,save_jsonl,load_dataset
from sidra.retrieval import BM25Index
from sidra.rsi_trace_audit import audit_tool_trace
from sidra.schema import AgentTrace,Document,Question
from sidra.train_reward import score_train_rollout


def _check_path(path:Path)->None:
    forbidden=("final-b","final_b","b160","/test/","/heldout/")
    if any(x in str(path.resolve()).casefold() for x in forbidden):
        raise ValueError("refuse heldout/final-B path; no permission to open")


def _objects(path:Path)->list[dict]:
    _check_path(path)
    if not path.is_file():raise FileNotFoundError(path)
    rows=list(iter_jsonl(path))
    if not rows or any(not isinstance(x,dict) for x in rows):
        raise ValueError("empty/malformed JSONL")
    return rows


def _json(path:Path)->dict:
    _check_path(path)
    obj=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj,dict):raise ValueError("invalid manifest")
    return obj


def doc_alias(challenge_id:str,original_doc_id:str)->str:
    """Opaque ID has no source TRAIN question ID or answer encoded."""
    if not re.fullmatch(r"c-[0-9a-f]{24}",challenge_id):
        raise ValueError("malformed challenge ID")
    if not original_doc_id:raise ValueError("empty source doc")
    return challenge_id+"::d"+hashlib.sha256(original_doc_id.encode()).hexdigest()[:16]


def _source_map(campaign:Path,cohort:int):
    if cohort<1:raise ValueError("invalid cohort")
    campaign_manifest=_json(campaign/"CAMPAIGN_MANIFEST.json")
    cohorts=campaign_manifest["cohorts"]
    if cohort>len(cohorts):raise ValueError("unknown cohort")
    item=cohorts[cohort-1]
    if item["planned_curriculum_cohort"]!=cohort:
        raise ValueError("manifest cohort identity mismatch")
    cohort_dir=campaign/f"cohort_{cohort:02d}"
    inp=cohort_dir/"SOLVER_INPUT.jsonl"
    mapping=cohort_dir/"VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl"
    if (sha256_file(inp)!=item["solver_input_sha256"]
        or sha256_file(mapping)!=item["verifier_only_map_sha256"]):
        raise ValueError("immutable curriculum cohort input/map SHA mismatch")
    prompts=_objects(inp)
    rows=_objects(mapping)
    if len(prompts)!=len(rows) or len(prompts)!=item["n_original_train_questions"]:
        raise ValueError("cohort length mismatch")
    by_id={}
    for item in rows:
        if set(item)!={"challenge_id","source_train_question_id","text_sha256","tier"}:
            raise ValueError("untrusted private map fields")
        if item["challenge_id"] in by_id:raise ValueError("duplicate private mapping")
        by_id[item["challenge_id"]]=item
    if (len({x["challenge_id"] for x in prompts})!=len(prompts)
        or set(by_id)!={x["challenge_id"] for x in prompts}
        or any(set(x)!={"challenge_id","question"} for x in prompts)):
        raise ValueError("private mapping / question mismatch")
    return campaign_manifest,prompts,by_id


def _make_projection(documents:list[Document],prompts:list[dict],
                     by_id:dict[str,dict])->list[Document]:
    scoped=defaultdict(list)
    for d in documents:
        if d.scope=="global":
            raise ValueError("public/global cross-question TRAIN corpus not allowed")
        scoped[d.scope].append(d)
    result=[]
    seen=set()
    for item in prompts:
        cid=item["challenge_id"]
        qid=by_id[cid]["source_train_question_id"]
        if not scoped[qid]:
            raise ValueError("no canonical TRAIN documents for cohort question")
        for orig in scoped[qid]:
            alias=doc_alias(cid,orig.doc_id)
            if alias in seen:raise ValueError("opaque doc hash collision")
            seen.add(alias)
            result.append(Document(
                doc_id=alias,scope=cid,title=orig.title,text=orig.text))
    return result


def prepare_solver_pack(train_documents:Path,campaign:Path,cohort:int,
                        out:Path,*,source_sha:str)->dict:
    """Trusted CPU preparer: produces model-only docs without reading QA labels."""
    _check_path(train_documents)
    if out.exists():raise FileExistsError(out)
    if not re.fullmatch("[0-9a-f]{40}",source_sha):
        raise ValueError("invalid pinned source SHA")
    manifest,prompts,by_id=_source_map(campaign,cohort)
    if manifest["source_git_sha"]!=source_sha:
        raise ValueError("source commit changed since TRAIN campaign freeze")
    if sha256_file(train_documents)!=manifest["original_train_documents_sha256"]:
        raise ValueError("canonical TRAIN documents checksum changed")
    docs=[Document.from_dict(x) for x in _objects(train_documents)]
    projected=_make_projection(docs,prompts,by_id)
    if any(orig in json.dumps([asdict(d) for d in projected])
           for orig in by_id.values() if False):
        raise AssertionError("unreachable")
    out.mkdir(parents=True,exist_ok=False)
    save_jsonl(out/"SOLVER_INPUT.jsonl",prompts)
    save_jsonl(out/"SOLVER_DOCUMENTS.jsonl",[asdict(d) for d in projected])
    public_manifest={
        "stage":"model_only_original_TRAIN_solver_input",
        "source_git_sha":source_sha,
        "campaign_manifest_sha256":sha256_file(campaign/"CAMPAIGN_MANIFEST.json"),
        "cohort":cohort,"questions":len(prompts),
        "documents":len(projected),
        "question_sha256":sha256_file(out/"SOLVER_INPUT.jsonl"),
        "documents_sha256":sha256_file(out/"SOLVER_DOCUMENTS.jsonl"),
        "original_corpus_sha256":sha256_file(train_documents),
        "private_train_source_question_ID_map_emitted":False,
        "TRAIN_gold_labels_loaded_in_model_process":False,
        "optimizer_eligible":False,
        "sealed_eval_or_finalB_used":False,
    }
    (out/"SOLVER_PACK_MANIFEST.json").write_text(
        json.dumps(public_manifest,ensure_ascii=False,indent=2)+"\n")
    return public_manifest


def _validate_public_pack(pack:Path):
    manifest=_json(pack/"SOLVER_PACK_MANIFEST.json")
    if (manifest["stage"]!="model_only_original_TRAIN_solver_input"
        or manifest.get("TRAIN_gold_labels_loaded_in_model_process") is not False
        or manifest.get("optimizer_eligible") is not False):
        raise ValueError("incorrect role or model training claim")
    inp=pack/"SOLVER_INPUT.jsonl"
    docs=pack/"SOLVER_DOCUMENTS.jsonl"
    if (sha256_file(inp)!=manifest["question_sha256"]
        or sha256_file(docs)!=manifest["documents_sha256"]):
        raise ValueError("solver pack changed after freezing")
    prompts=_objects(inp)
    documents=[Document.from_dict(d) for d in _objects(docs)]
    ids=[x["challenge_id"] for x in prompts]
    if len(set(ids))!=len(ids) or len(ids)!=manifest["questions"]:
        raise ValueError("missing/duplicate prompt")
    if any(set(x)!={"challenge_id","question"} or not x["question"].strip()
           for x in prompts):
        raise ValueError("solver inputs contain restricted fields")
    if (len(documents)!=manifest["documents"]
        or any(d.scope not in set(ids)
               or not d.doc_id.startswith(d.scope+"::d")
               for d in documents)):
        raise ValueError("model-only corpus scope mismatch")
    return manifest,prompts,documents


def run_real_solver(pack:Path,out:Path,*,endpoint:str,model:str,
                    checkpoint_sha256:str,source_sha:str,
                    seed:int=43,max_tokens:int=240,
                    config:AgentConfig|None=None)->dict:
    """Solver process: receives ONLY public/sanitized model pack, no gold."""
    if out.exists():raise FileExistsError(out)
    if not endpoint.startswith(("http://127.0.0.1:","http://localhost:")):
        raise ValueError("solver may only call local inference service")
    if not re.fullmatch("[0-9a-f]{64}",checkpoint_sha256):
        raise ValueError("unverified model/adapter file SHA-256")
    public,prompts,docs=_validate_public_pack(pack)
    if public["source_git_sha"]!=source_sha:
        raise ValueError("source code and pack commit disagree")
    from sidra.policies import OpenAICompatiblePolicy
    cfg=config or AgentConfig(max_steps=6,max_tool_calls=5,search_top_k=3)
    policy=OpenAICompatiblePolicy(
        endpoint,model,temperature=0.75,top_p=0.9,seed=seed,
        max_tokens=max_tokens,constrain_actions=True,constrain_document_ids=True)
    agent=SearchAgent(docs,policy,cfg)
    traces=[]
    model_calls=[]
    for ix,item in enumerate(prompts):
        start=len(policy.action_io)
        q=Question(item["challenge_id"],item["question"],"",
                   (),split="train",scope=item["challenge_id"])
        trace=agent.run(q)
        traces.append(trace.as_dict())
        model_calls.append({"challenge_id":item["challenge_id"],
                            "calls":policy.action_io[start:]})
        print("SOLVER",ix+1,"/",len(prompts),"termination",trace.termination,
              "tool_calls",trace.tool_calls,flush=True)
    out.mkdir(parents=True,exist_ok=False)
    save_jsonl(out/"SOLVER_TRACES.jsonl",traces)
    save_jsonl(out/"MODEL_ACTION_IO.jsonl",model_calls)
    report={
        "stage":"actual_model_solver_tool_rollouts_NOT_training",
        "source_git_sha":source_sha,
        "public_solver_pack_sha256":sha256_file(pack/"SOLVER_PACK_MANIFEST.json"),
        "cohort":public["cohort"],
        "n_episodes":len(traces),
        "model_id":model,
        "model_checkpoint_sha256_user_supplied":checkpoint_sha256,
        "model_origin_independently_attested":False,
        "temperature":0.75,"top_p":0.9,"seed":seed,
        "decoder":"vLLM_dynamic_JSON_schema_observed_docIDs",
        "model_calls":policy.model_calls,
        "prompt_tokens":policy.prompt_tokens,
        "completion_tokens":policy.completion_tokens,
        "config":asdict(cfg),
        "terminations":dict(Counter(t["termination"] for t in traces)),
        "traces_sha256":sha256_file(out/"SOLVER_TRACES.jsonl"),
        "model_action_io_sha256":sha256_file(out/"MODEL_ACTION_IO.jsonl"),
        "TRAIN_gold_loaded_or_graded_by_solver":False,
        "on_policy_training_distribution_proven":False,
        "optimizer_updates":0,
        "novel_new_labels_admitted":0,
        "multi_generation_RSI_proven":False,
    }
    (out/"SOLVER_RUN_MANIFEST.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2)+"\n")
    return report


def audit_solver_run(train_data:Path,campaign:Path,pack:Path,run:Path,
                     *,source_sha:str,config:AgentConfig|None=None)->dict:
    """Independent offline canonical TRAIN verifier: returns aggregate only."""
    _check_path(train_data)
    r=_json(run/"SOLVER_RUN_MANIFEST.json")
    m,prompts,by_public=_validate_public_pack(pack)
    if (r["stage"]!="actual_model_solver_tool_rollouts_NOT_training"
        or r["source_git_sha"]!=source_sha
        or r["public_solver_pack_sha256"]!=sha256_file(pack/"SOLVER_PACK_MANIFEST.json")
        or r["cohort"]!=m["cohort"] or r.get("optimizer_updates")!=0
        or r.get("TRAIN_gold_loaded_or_graded_by_solver") is not False
        or sha256_file(run/"SOLVER_TRACES.jsonl")!=r["traces_sha256"]
        or sha256_file(run/"MODEL_ACTION_IO.jsonl")!=r["model_action_io_sha256"]):
        raise ValueError("Solver model-run artifact integrity mismatch")
    campaign_manifest,canonical_prompts,by_private=_source_map(
        campaign,m["cohort"])
    if (campaign_manifest["source_git_sha"]!=source_sha
        or canonical_prompts!=prompts
        or sha256_file(campaign/"CAMPAIGN_MANIFEST.json")!=m["campaign_manifest_sha256"]):
        raise ValueError("campaign/solver source discrepancy")
    docs,questions=load_dataset(train_data)
    if any(q.split!="train" for q in questions):
        raise ValueError("VERIFIER refuses non-TRAIN gold")
    by_train={q.question_id:q for q in questions}
    if len(by_train)!=len(questions) or (
        sha256_file(train_data/"documents.jsonl")!=m["original_corpus_sha256"]):
        raise ValueError("canonical source changed")
    projection=_make_projection(docs,prompts,by_private)
    if [asdict(x) for x in projection]!=[asdict(x) for x in by_public]:
        raise ValueError("SOLVER corpus differs from canonical projection")
    traces=_objects(run/"SOLVER_TRACES.jsonl")
    io=_objects(run/"MODEL_ACTION_IO.jsonl")
    if (len(traces)!=len(prompts) or len(io)!=len(prompts)
        or len({t["question_id"] for t in traces})!=len(traces)
        or [x["challenge_id"] for x in io]!=[x["challenge_id"] for x in prompts]
        or {t["question_id"] for t in traces}!={x["challenge_id"] for x in prompts}):
        raise ValueError("Solver trace cardinality or ID mismatch")
    cfg=config or AgentConfig(max_steps=6,max_tool_calls=5,search_top_k=3)
    if r["config"]!=asdict(cfg):raise ValueError("tool budget changed")
    docs_by_id={d.doc_id:d for d in projection}
    by_opaque={p["challenge_id"]:p for p in prompts}
    index=BM25Index(projection,title_boost=cfg.title_boost)
    trace_checks=[]
    totals=Counter()
    by_tier=defaultdict(Counter)
    for raw in traces:
        tr=AgentTrace(**raw)
        cid=tr.question_id
        p=by_opaque[cid]
        orig=by_train[by_private[cid]["source_train_question_id"]]
        if tr.question!=p["question"] or orig.question!=tr.question:
            raise ValueError("original QA question diverged")
        aliases={d.doc_id:doc_alias(cid,d.doc_id)
                 for d in docs if d.scope==orig.scope}
        gold_support=tuple(aliases[x] for x in orig.supporting_doc_ids)
        q=Question(cid,tr.question,orig.gold_answer,
                   gold_support,split="train",scope=cid)
        replay=audit_tool_trace(q,tr,projection,config=cfg,shared_index=index)
        if not replay.structurally_valid:
            raise ValueError("replayed tool trace invalid: "+replay.reason)
        reward=score_train_rollout(q,tr,docs_by_id,
                                  max_tool_calls=cfg.max_tool_calls)
        tier=by_private[cid]["tier"]
        totals["n"]+=1
        totals["strict"]+=int(replay.complete_legal_finish and reward.strict_outcome==1)
        totals["finish"]+=int(tr.termination=="finish")
        totals["nonerror"]+=int(tr.termination!="error")
        totals["read_support_sum"]+=reward.read_support_recall
        by_tier[tier]["n"]+=1
        by_tier[tier]["strict"]+=int(replay.complete_legal_finish and reward.strict_outcome==1)
        by_tier[tier]["answer_correct"]+=int(reward.answer_correct)
        trace_checks.append({"cid":cid,"reward":reward.as_dict()})
    # Verifier-only outcome file is optional for future *trusted* Trainer, but
    # the PUBLIC report below never releases gold-derived per-task labels.
    return {
        "stage":"independent_original_TRAIN_solver_verification",
        "n":totals["n"],
        "tool_replay_valid":totals["n"],
        "strict_original_TRAIN_successes":totals["strict"],
        "legal_finished_answers":totals["finish"],
        "nonerror_terminations":totals["nonerror"],
        "mean_gold_support_read_recall":round(totals["read_support_sum"]/totals["n"],6),
        "difficulty_tier_aggregate":{
            key:dict(by_tier[key]) for key in ("direct","bridge","hard")
        },
        "canonical_corpus_sha256":sha256_file(train_data/"documents.jsonl"),
        "solver_traces_sha256":r["traces_sha256"],
        "model_actions_sha256":r["model_action_io_sha256"],
        "model_checkpoint_sha256_declared":r["model_checkpoint_sha256_user_supplied"],
        "model_provenance_independently_attested":False,
        "gold_read_only_in_separate_verifier":True,
        "training_updates_this_run":0,
        "new_challenger_labels_admitted":0,
        "optimizer_eligible":False,
        "multi_generation_RSI_proven":False,
        "no_dev_evalA_or_finalB_gold_accessed":True,
        "limitation":"Tool trace/reward replay does not authenticate model origin or "
                     "justify an on-policy update under constrained decoding.",
    }
