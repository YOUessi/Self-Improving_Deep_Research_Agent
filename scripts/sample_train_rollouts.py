"""Sample REAL multi-turn TRAIN Agent rollouts; preflight for Agentic GRPO.

This script does NOT update weights and must not be misreported as GRPO.
Each generation is a fresh online SearchAgent episode: the *model* selects
SEARCH / READ / FINISH actions; the sandboxed BM25 tool returns observations;
reward is computed ONLY after completion from HotpotQA TRAIN gold.

Designed to verify actual group reward variance and failure cases BEFORE
expensive RL. No evaluation dev answers/supports ever reach this process.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

from sidra.agent import AgentConfig, SearchAgent
from sidra.grounded_policy import GroundedPolicy
from sidra.io import load_dataset, save_jsonl
from sidra.policies import OpenAICompatiblePolicy
from sidra.train_reward import score_train_rollout, group_normalized_advantages


def select_train_questions(questions:list,dev_ids:set[str], *,
                           limit:int,seed:int)->list:
    if limit<1:raise ValueError("limit must be >= 1")
    if any(q.split!="train" for q in questions):
        raise ValueError("non-TRAIN questions forbidden in RL preflight")
    ids=[q.question_id for q in questions]
    if len(ids)!=len(set(ids)) or (set(ids)&dev_ids):
        raise ValueError("training/dev identifiers overlap or duplicate")
    ordered=sorted(questions,key=lambda q:(
        hashlib.sha256(f"{seed}:{q.question_id}".encode()).hexdigest(),q.question_id))
    if len(ordered)<limit:raise ValueError("insufficient TRAIN questions")
    return ordered[:limit]


def summarize_groups(groups:list[list[dict]])->dict:
    if not groups or not all(g and len(g)==len(groups[0]) for g in groups):
        raise ValueError("no balanced rollout groups")
    flat=[x for g in groups for x in g]
    nonconstant=sum(len({x["reward"]["total"] for x in g})>1 for g in groups)
    nonzero=sum(float(x["reward"]["strict_outcome"])>0 for x in flat)
    obs=Counter(x["trace"]["termination"] for x in flat)
    avg=lambda key:round(sum(float(x["reward"][key]) for x in flat)/len(flat),6)
    return {
        "train_groups":len(groups), "generations_per_question":len(groups[0]),
        "sampled_episodes":len(flat),
        "reward_mean":avg("total"),
        "reward_min":min(x["reward"]["total"] for x in flat),
        "reward_max":max(x["reward"]["total"] for x in flat),
        "variable_reward_groups":nonconstant,
        "variable_reward_group_fraction":round(nonconstant/len(groups),6),
        "strictly_correct_training_rollouts":nonzero,
        "read_support_recall_mean":avg("read_support_recall"),
        "finish_reasons":dict(sorted(obs.items())),
        "meaning":"REAL MODEL train-only rollouts; NO optimizer or GRPO training yet",
    }


def collect(dataset:Path,dev_manifest:Path,out:Path, *,
            endpoint:str,model:str,limit:int=12,generations:int=4,
            seed:int=43,temperature:float=0.75,top_p:float=0.9,
            bridge:bool=False, max_tools:int=5,max_steps:int=6)->dict:
    if out.exists():raise FileExistsError(out)
    if generations<2:raise ValueError("GRPO preflight needs at least 2 generations")
    docs,questions=load_dataset(dataset)
    manifest=json.loads(dev_manifest.read_text())
    dev_ids=set(manifest["pilot"]["selected_ids"]) | set(manifest["heldout"]["selected_ids"])
    chosen=select_train_questions(questions,dev_ids,limit=limit,seed=seed)
    lookup={d.doc_id:d for d in docs}
    runner=SearchAgent(docs,
           OpenAICompatiblePolicy(endpoint,model),AgentConfig(
               max_steps=max_steps,max_tool_calls=max_tools,search_top_k=3))
    groups=[]
    started=time.perf_counter()
    for i,q in enumerate(chosen):
        trajectories=[]
        for sample in range(generations):
            sampled_seed=int(hashlib.sha256(
                f"rollout:{seed}:{q.question_id}:{sample}".encode()).hexdigest()[:8],16)
            base=OpenAICompatiblePolicy(
                endpoint,model,constrain_actions=True,
                constrain_document_ids=True,
                temperature=temperature,top_p=top_p,seed=sampled_seed)
            policy=GroundedPolicy(base,min_reads=2)
            if bridge:
                from sidra.bridge_policy import BridgeGroundedPolicy
                policy=BridgeGroundedPolicy(policy)
            runner.policy=policy
            trace=runner.run(q)
            reward=score_train_rollout(q,trace,lookup,
                                       max_tool_calls=max_tools)
            trajectories.append({
                "question_id":q.question_id,"rollout_index":sample,
                "seed":sampled_seed,"trace":trace.as_dict(),
                "reward":reward.as_dict()})
        advantages=group_normalized_advantages(
            [r["reward"]["total"] for r in trajectories])
        for row,adv in zip(trajectories,advantages):
            row["group_advantage"]=round(adv,8)
        groups.append(trajectories)
        print("GROUP",i+1,"/",len(chosen),"qa=",q.question_id,
              "reward_spread=",round(max(r["reward"]["total"] for r in trajectories)-
                                         min(r["reward"]["total"] for r in trajectories),5),
              "strict=",sum(x["reward"]["strict_outcome"] for x in trajectories),
              "wall_s=",round(time.perf_counter()-started,1),flush=True)
    out.mkdir(parents=True,exist_ok=False)
    save_jsonl(out/"rollouts_train.jsonl",[x for g in groups for x in g])
    summary=summarize_groups(groups)
    summary.update({
        "dataset":str(dataset), "split":"train",
        "dev_or_test_labels_loaded_into_policy":False,
        "model_id":model,
        "temperature":temperature, "top_p":top_p,
        "selection_seed":seed,"tool_limit":max_tools,"max_steps":max_steps,
        "bridge":bridge,
        "train_questions_sha256":hashlib.sha256(
            "\n".join(q.question_id for q in chosen).encode()).hexdigest(),
        "wall_seconds":round(time.perf_counter()-started,2),
        "policy_updates":0,
    })
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",required=True,type=Path)
    p.add_argument("--dev-manifest",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--endpoint",default="http://127.0.0.1:18981/v1/chat/completions")
    p.add_argument("--model",required=True)
    p.add_argument("--limit",type=int,default=12)
    p.add_argument("--generations",type=int,default=4)
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--temperature",type=float,default=0.75)
    p.add_argument("--bridge",action="store_true")
    a=p.parse_args()
    print(json.dumps(collect(a.dataset,a.dev_manifest,a.out,
                endpoint=a.endpoint,model=a.model,limit=a.limit,
                generations=a.generations,seed=a.seed,
                temperature=a.temperature,bridge=a.bridge),
                indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
