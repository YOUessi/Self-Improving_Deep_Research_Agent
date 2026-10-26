"""TRAIN-only action-curriculum variants for Qwen3 LoRA/SFT.

Fixed engineering problems from SFT-v1:
  1. Original SFT export hardcoded remaining_steps = 8-step, while the actual
     SearchAgent uses max_steps=6. Fix to exactly 6-step on TRAIN-only rows.
  2. Original teacher serialized every Action field, even when irrelevant.
     Use action-type minimal JSON matching the real serving schema.
  3. Train stop/abstain behavior using validated TRAIN observations only:
     finish when both evidence passages are already READ; abstain on a
     last-turn with no evidence; optionally abort with only one read when
     neither another tool + answer nor complete support is possible.

Two independently evaluated curricula:
  --mode format : changes 1+2 only (control).
  --mode stop   : changes 1+2, plus terminal-state upweight and constrained
                  no-evidence/one-read last-turn abstention examples.

Every row is from the TRAIN split; all sample prompts never include gold
answer/support fields. TRAIN labels are used solely for supervised targets.
No developer or heldout labels may appear in the optimizer inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from sidra.io import iter_jsonl, load_dataset, save_jsonl

KINDS={"search","read","finish"}


def canonical_action(obj:dict)->dict:
    kind=obj.get("kind")
    if kind not in KINDS:raise ValueError("invalid training action")
    if kind=="search":
        query=obj.get("query","")
        if not isinstance(query,str) or not query.strip():
            raise ValueError("empty search query")
        return {"kind":"search","query":query}
    if kind=="read":
        doc_id=obj.get("doc_id","")
        if not isinstance(doc_id,str) or not doc_id:
            raise ValueError("empty read doc ID")
        return {"kind":"read","doc_id":doc_id}
    answer=obj.get("answer","")
    citations=obj.get("citations",[])
    if not isinstance(answer,str) or not isinstance(citations,list):
        raise ValueError("malformed finish target")
    if not all(isinstance(c,str) for c in citations):
        raise ValueError("invalid cited document IDs")
    return {"kind":"finish","answer":answer,"citations":citations}


def mutate_prompt(source:list[dict], *, remaining:int|None=None,
                  extra_instruction:str="", history:list|None=None)->list[dict]:
    if len(source)!=2 or source[0].get("role")!="system" or source[1].get("role")!="user":
        raise ValueError("unexpected training chat prompt")
    prompt=copy.deepcopy(source)
    user=json.loads(prompt[1]["content"])
    if any(k in user for k in ("gold_answer","supporting_doc_ids",
                               "evaluation_gold_answer")):
        raise ValueError("gold leaked into agent user prompt")
    if remaining is not None:
        if remaining<1:raise ValueError("remaining_steps must be positive")
        user["remaining_steps"]=remaining
    if history is not None:user["tool_history"]=copy.deepcopy(history)
    prompt[1]["content"]=json.dumps(user,ensure_ascii=False)
    if extra_instruction:
        prompt[0]["content"] += "\n"+extra_instruction
    return prompt


def compact_row(source:dict,action:dict, *, prompt:list[dict],
                variant:str="ordinary")->dict:
    if set(source)!={"prompt","completion","question_id","step"}:
        raise ValueError("unexpected source training row fields")
    # Keep exactly the input schema required by the original train SFT script.
    return {"prompt":prompt,"completion":[{"role":"assistant","content":
            json.dumps(action,ensure_ascii=False,separators=(',',':'))}],
            "question_id":source["question_id"],"step":source["step"]}


def curriculum(source:Path,train_data:Path,dev_freeze:Path,out:Path,
               *, mode:str="stop",max_steps:int=6,
               seed:int=43)->dict:
    if mode not in {"format","stop"}:
        raise ValueError("mode must be format or stop")
    if out.exists():
        raise FileExistsError(out)
    _,questions=load_dataset(train_data)
    train_ids={q.question_id for q in questions if q.split=="train"}
    if len(train_ids)!=len(questions):raise ValueError("non-train question")
    dev=json.loads(dev_freeze.read_text())
    dev_ids=set(dev["pilot"]["selected_ids"])|set(dev["heldout"]["selected_ids"])
    if train_ids & dev_ids:raise ValueError("train/dev ID overlap")

    rows=list(iter_jsonl(source))
    grouped=defaultdict(list)
    original_kinds=Counter()
    for row in rows:
        qid=str(row["question_id"])
        if qid not in train_ids or qid in dev_ids:
            raise ValueError("action supervision is not train-only")
        if int(row["step"])>=max_steps:
            raise ValueError("TRAIN example step exceeds environment budget")
        if len(row["completion"])!=1:raise ValueError("one assistant completion required")
        raw=json.loads(row["completion"][0]["content"])
        action=canonical_action(raw)
        original_kinds[action["kind"]]+=1
        grouped[qid].append((row,action))
    built=[]
    extras=Counter()
    for qid in sorted(grouped):
        items=sorted(grouped[qid],key=lambda pair:int(pair[0]["step"]))
        if items[0][1]["kind"]!="search":
            raise ValueError("verified oracle trajectories must begin with search")
        finish=[pair for pair in items if pair[1]["kind"]=="finish"]
        if len(finish)!=1 or items[-1][1]["kind"]!="finish":
            raise ValueError("expected exactly one verified train final action")
        seen_reads=set()
        for row,action in items:
            prompt=mutate_prompt(row["prompt"],remaining=max_steps-int(row["step"]))
            history=json.loads(prompt[1]["content"])["tool_history"]
            ids={str(obs["doc_id"]) for obs in history if
                 obs.get("kind")=="read" and obs.get("doc_id")}
            if action["kind"]=="finish":
                if len(ids)<2 or not set(action["citations"]).issubset(ids):
                    raise ValueError("train teacher finish cites unread/insufficient source")
                if not action["answer"].strip():
                    raise ValueError("teacher final answer empty")
            built.append(compact_row(row,action,prompt=prompt))
            if mode=="stop" and action["kind"]=="finish":
                # Reweight STOP-on-complete-evidence and use the actual last
                # action budget. No new answer is invented; teacher target is
                # the same verified TRAIN-only answer and citations.
                built.append(compact_row(
                    row,action,prompt=mutate_prompt(
                        row["prompt"],remaining=1),variant="finish-last-budget"))
                extras["verified_finish_at_last_step"]+=1
        if mode!="stop":
            continue

        # For approximately one in three TRAIN questions, create a legitimate
        # last-turn-with-no-evidence state. Abstention never uses gold answer.
        k=int(hashlib.sha256(f"{seed}:abstain:{qid}".encode()).hexdigest()[:8],16)
        if k%3==0:
            first=items[0][0]
            prompt=mutate_prompt(first["prompt"],remaining=1,
                                 extra_instruction=(
                                   "A final answer requires actually READ sources. "
                                   "If no sources are read and no complete "
                                   "research step remains, finish with an empty "
                                   "answer and no citations."),
                                 history=[])
            action={"kind":"finish","answer":"","citations":[]}
            built.append(compact_row(first,action,prompt=prompt,
                                     variant="last-step-empty"))
            extras["last_step_no_evidence_abstention"]+=1

        # This is an impossible-to-fully-support final-turn state: there is a
        # single genuine TRAIN read but the task requires two supporting docs.
        if k%5==0:
            candidate=next((row for row,action in items
                            if action["kind"] in {"read","search"} and
                            len([o for o in json.loads(row["prompt"][1]["content"])["tool_history"]
                                 if o.get("kind")=="read"])==1),None)
            if candidate is not None:
                prompt=mutate_prompt(candidate["prompt"],remaining=1,
                    extra_instruction=(
                      "Evidence has only one READ document. A two-source "
                      "answer cannot be verified before the remaining "
                      "action limit. Abstain rather than fabricate citations."))
                action={"kind":"finish","answer":"","citations":[]}
                built.append(compact_row(candidate,action,prompt=prompt,
                                         variant="single-source-abstention"))
                extras["one_read_last_step_abstention"]+=1

    hist=Counter(json.loads(r["completion"][0]["content"])["kind"] for r in built)
    out.mkdir(parents=True,exist_ok=False)
    dest=out/"actions.jsonl"
    save_jsonl(dest,built)
    metadata={
        "mode":mode,"train_only":True,"source_file":str(source),
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_train_question_count":len(grouped),
        "total_source_actions":len(rows),
        "original_action_counts":dict(sorted(original_kinds.items())),
        "output_action_counts":dict(sorted(hist.items())),
        "output_total_actions":len(built),"extra_variants":dict(sorted(extras.items())),
        "max_steps_aligned_to_runtime":max_steps,
        "canonical_action_variants":True,"selection_seed":seed,
        "dev_gold_used":False,"dev_ids_overlap":False,
        "train_source_only":"HotpotQA TRAIN verified oracle trajectories",
        "interpretation":"SFT curriculum, not real Agent performance, RL or RSI"}
    (out/"CURRICULUM_MANIFEST.json").write_text(
        json.dumps(metadata,ensure_ascii=False,indent=2)+"\n")
    return metadata


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--train-data",type=Path,required=True)
    p.add_argument("--dev-freeze",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--mode",choices=("format","stop"),required=True)
    p.add_argument("--max-steps",type=int,default=6)
    p.add_argument("--seed",type=int,default=43)
    a=p.parse_args()
    print(json.dumps(curriculum(a.source,a.train_data,a.dev_freeze,a.out,
                  mode=a.mode,max_steps=a.max_steps,seed=a.seed),
                  indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
