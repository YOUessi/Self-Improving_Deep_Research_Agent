"""Strict TRAIN-only oracle *teacher* demonstrations of REAL Search/Read/Finish.

This is **SUPERVISED imitation data**, not GRPO/RSI and not an oracle-enabled
student policy. The offline teacher sees TRAIN answers and supporting doc IDs
only to choose among the EXACT nonoracle catalog actions available to a student
at that step. Search/Read observations come from the real SearchAgent BM25 and
cannot be fabricated. We save ONLY trajectories that actually finish with
correct answer, all gold citations, and answer in cited source text.

The student's action_prompt contains ONLY question + actual observed tool
text + legal action catalog. The oracle selector, source gold and support doc
IDs are never supplied to the student prompt. Heldout public DEV is excluded
from teacher inputs completely; the frozen manifest is inspected only for IDs.

Failure modes are auditable: if catalog cannot express the gold answer or the
budget cannot retrieve the supporting sources, skip the demonstration rather
than invent "successful" expert targets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from sidra.agent import AgentConfig,SearchAgent
from sidra.experiment import sha256_file
from sidra.finite_action import (
    FiniteActionConfig,action_prompt,build_legal_actions,
    choice_token_ids,logprob_of_action,
)
from sidra.io import load_dataset,save_jsonl,iter_jsonl
from sidra.metrics import normalize_answer
from sidra.retrieval import BM25Index
from sidra.schema import Action,Document,Question
from sidra.train_reward import score_train_rollout


class TrainOnlyOracleTeacher:
    """Gold controls only the offline target ID, never the model's observations."""
    name="TRAIN_ONLY_ORACLE_TEACHER_NOT_AN_AGENT_MODEL"

    def __init__(self, question:Question, index:BM25Index,
                 *,config:AgentConfig,finite_cfg:FiniteActionConfig):
        if question.split!="train":
            raise ValueError("Teacher may not access any non-TRAIN question")
        if not question.supporting_doc_ids or not question.gold_answer.strip():
            raise ValueError("Teacher requires valid TRAIN-only labels")
        self.question=question
        self.index=index
        self.config=config
        self.finite_cfg=finite_cfg
        self.steps=[]
        self.model_calls=0
        self.prompt_tokens=0
        self.completion_tokens=0

    def decide(self,question:str,observations:list[dict],remaining:int)->Action:
        if question!=self.question.question:
            raise ValueError("Teacher question identity mismatch")
        options=build_legal_actions(
            question,observations,cfg=self.finite_cfg,
            remaining=remaining,max_tool_calls=self.config.max_tool_calls)
        gold=set(self.question.supporting_doc_ids)
        read={str(o["doc_id"]):o for o in observations if o.get("kind")=="read"}
        search_hits={
            str(hit.get("doc_id")) for obs in observations
            if obs.get("kind")=="search" for hit in obs.get("hits",[])
        }
        target=normalize_answer(self.question.gold_answer)
        missing_read=gold-set(read)
        # FINISH must actually be available in the exact model-visible catalog,
        # after reading every gold source, never an invented direct oracle action.
        for i,a in enumerate(options):
            if (a.kind=="finish" and a.answer.strip()
                and normalize_answer(a.answer)==target
                and gold.issubset(set(a.citations))
                and set(a.citations).issubset(set(read))
                and target in normalize_answer(" ".join(
                    str(read[c]["text"]) for c in a.citations))):
                chosen=i
                break
        else:
            # When a gold paragraph is already returned by a real search, read
            # it rather than using an unseen oracle document.
            relevant_read=[
                (i,a) for i,a in enumerate(options)
                if a.kind=="read" and a.doc_id in missing_read
            ]
            if relevant_read:
                chosen=relevant_read[0][0]
            else:
                # An offline teacher may inspect TRAIN retrieval index to
                # score available queries. It does not transmit its hidden
                # labels, proposed hits or grades into the student prompt.
                ranked=[]
                for i,a in enumerate(options):
                    if a.kind!="search":continue
                    hits=self.index.search(
                        a.query,k=self.config.search_top_k,
                        scope=self.question.scope)
                    fresh={h.doc_id for h in hits}-search_hits
                    new_gold=len(fresh & missing_read)
                    ranked.append((new_gold,len(fresh),-i,i))
                if ranked:
                    chosen=max(ranked)[-1]
                else:
                    # Demonstration is deliberately rejected unless it
                    # actually reaches success after real tool execution.
                    chosen=next((i for i,a in enumerate(options)
                                 if a.kind=="finish" and not a.answer),
                                len(options)-1)
        self.steps.append({
            "question_id":self.question.question_id,
            "question":question,
            "remaining":remaining,
            "observations":copy.deepcopy(observations),
            "allowed_actions":[a.to_dict() for a in options],
            "target_index":chosen,
            "target_kind":options[chosen].kind,
            "student_model_gold_injected":False,
            "teacher_gold_used_for_target":True,
        })
        return options[chosen]


def require_train_only(questions:list[Question],dev_ids:set[str])->None:
    ids=[q.question_id for q in questions]
    if not questions or any(q.split!="train" for q in questions):
        raise ValueError("teacher DATA MUST be TRAIN-only")
    if len(ids)!=len(set(ids)) or (set(ids)&dev_ids):
        raise ValueError("TRAIN IDs duplicate or overlap with frozen DEV")


def make_teacher_data(data:Path,dev_manifest:Path,out:Path, *,
                      seed:int=43,question_limit:int=1024,
                      max_steps:int=6,max_tool_calls:int=5,
                      max_context:int=2048,
                      answer_proposer:str="v2")->dict:
    if out.exists():raise FileExistsError(out)
    if question_limit<1:raise ValueError("question_limit must be positive")
    documents,questions=load_dataset(data)
    manifest=json.loads(dev_manifest.read_text())
    dev_ids=set(manifest["pilot"]["selected_ids"]) | set(manifest["heldout"]["selected_ids"])
    require_train_only(questions,dev_ids)
    chosen=sorted(questions,key=lambda q:(
        hashlib.sha256(f"teacher:{seed}:{q.question_id}".encode()).hexdigest(),
        q.question_id))[:question_limit]
    config=AgentConfig(max_steps=max_steps,max_tool_calls=max_tool_calls,search_top_k=3)
    finite_cfg=FiniteActionConfig(max_prompt_tokens=max_context,
                                 answer_proposer=answer_proposer)
    index=BM25Index(documents)
    runner=SearchAgent(documents,policy=None,config=config)
    lookup={d.doc_id:d for d in documents}
    records=[]
    error_counts=Counter()
    success_counts=Counter()
    t0=time.monotonic()
    for n,q in enumerate(chosen):
        policy=TrainOnlyOracleTeacher(q,index,config=config,finite_cfg=finite_cfg)
        runner.policy=policy
        trace=runner.run(q)
        reward=score_train_rollout(q,trace,lookup,
                                  max_tool_calls=max_tool_calls)
        passed=bool(reward.strict_outcome and trace.termination=="finish")
        if passed and len(policy.steps)!=len(trace.steps):
            raise RuntimeError("teacher decision count differs from real tool trace")
        if passed:
            for k,step in enumerate(policy.steps):
                observed=trace.steps[k].get("action")
                if observed!=step["allowed_actions"][step["target_index"]]:
                    raise RuntimeError("teacher proposed a different action than real agent")
            records.extend(policy.steps)
            success_counts["questions"]+=1
            success_counts["actions"]+=len(policy.steps)
            success_counts["search"]+=trace.search_calls
            success_counts["read"]+=trace.read_calls
            success_counts["finish"]+=int(trace.termination=="finish")
        else:
            error_counts[trace.termination]+=1
            if reward.answer_correct:
                error_counts["correct_answer_but_not_strict_evidence"]+=1
            if all(step["target_kind"]!="finish" or not (
                step["allowed_actions"][step["target_index"]]["answer"])
                for step in policy.steps):
                error_counts["never_selected_nonempty_finish"]+=1
        if (n+1)%128==0:
            print("TEACHER_PROGRESS",n+1,"/",len(chosen),
                  "strict_successes",success_counts["questions"],
                  "elapsed_s",round(time.monotonic()-t0,1),flush=True)
    if not records:raise RuntimeError("No strict successful real teacher trajectories")
    # Group by question for frozen TRAIN-only expert-action holdout (nothing
    # from this partition enters optimizer). This is not DEV generalization.
    successes=sorted({row["question_id"] for row in records},key=lambda q:(
        hashlib.sha256(f"expert-holdout:{seed}:{q}".encode()).hexdigest(),q))
    if len(successes)<3:
        raise RuntimeError("insufficient independent solved TRAIN trajectories")
    held_count=max(1,math.ceil(len(successes)*0.2))
    held=set(successes[:held_count])
    train=[r for r in records if r["question_id"] not in held]
    val=[r for r in records if r["question_id"] in held]
    assert set(x["question_id"] for x in train).isdisjoint(
        x["question_id"] for x in val)
    out.mkdir(parents=True,exist_ok=False)
    save_jsonl(out/"TEACHER_ACTIONS_TRAIN.jsonl",train)
    save_jsonl(out/"TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl",val)
    summary={
        "stage":"TRAIN_ONLY_GOLD_SUPERVISED_TEACHER_NO_RL",
        "teacher_uses_train_gold":True,
        "student_model_sees_gold_answer":False,
        "teacher_uses_real_BM25_SearchAgent":True,
        "includes_only_real_strict_successes":True,
        "question_count_attempted":len(chosen),
        "strict_success_trajectories":len(successes),
        "strict_success_fraction":len(successes)/len(chosen),
        "verified_real_tool_actions":success_counts["actions"],
        "train_trajectories":len(successes)-held_count,
        "heldout_train_trajectories":held_count,
        "train_action_labels":len(train),
        "heldout_train_action_labels":len(val),
        "rejected_termination_counts":dict(error_counts),
        "answer_proposer":answer_proposer,
        "tool_budget":max_tool_calls,"step_budget":max_steps,
        "dev_manifest_sha256":sha256_file(dev_manifest),
        "train_question_sha256":sha256_file(data/"questions.jsonl"),
        "train_document_sha256":sha256_file(data/"documents.jsonl"),
        "seed":seed,"seconds":round(time.monotonic()-t0,3),
        "checkpoint_trained":False,
        "online_GRPO_executed":False,
        "critical_caveat":(
            "Offline TRAIN-gold teacher targets, not model-generated actions, "
            "are supervised labels only. Heldout expert-action targets are "
            "still TRAIN, NOT independent DEV or real policy multi-hop gains."
        ),
    }
    (out/"TEACHER_SUMMARY.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    return summary


def _encode_teacher_rows(tokenizer,rows:list[dict], *,
                         config:FiniteActionConfig)->tuple[list[dict],Counter]:
    usable=[]
    skipped=Counter()
    for r in rows:
        actions=[Action.from_dict(a) for a in r["allowed_actions"]]
        if not 0<=int(r["target_index"])<len(actions):
            raise ValueError("teacher target not in legal action catalog")
        replay=build_legal_actions(
            r["question"],r["observations"],
            cfg=config,remaining=int(r["remaining"]),
            max_tool_calls=5)
        if replay!=actions:
            raise ValueError("teacher catalog changed: cannot train different action policy")
        try:
            ids=action_prompt(tokenizer,r["question"],r["observations"],
                              int(r["remaining"]),actions,cfg=config)
            token_ids=choice_token_ids(tokenizer,len(actions))
        except ValueError as exc:
            if "context overflow" in str(exc):
                skipped["context_overflow"]+=1
                continue
            raise
        usable.append({
            "question_id":r["question_id"],
            "prompt_ids":ids,
            "legal_ids":token_ids,
            "target":int(r["target_index"]),
        })
    return usable,skipped


def evaluate_teacher_label_fit(model,examples:list[dict])->dict:
    import torch
    model.eval()
    exact=0
    loss_sum=0.0
    with torch.no_grad():
        for x in examples:
            ids=torch.tensor([x["prompt_ids"]],device=model.device,dtype=torch.long)
            logits=model(input_ids=ids).logits[0,-1,:].float()
            legal=logits[torch.tensor(x["legal_ids"],device=ids.device)]
            lp=torch.log_softmax(legal,dim=-1)
            exact+=int(int(torch.argmax(legal))==x["target"])
            loss_sum+=float(-lp[x["target"]])
    return {
        "n":len(examples),
        "teacher_action_accuracy":exact/max(1,len(examples)),
        "mean_masked_cross_entropy":loss_sum/max(1,len(examples)),
    }


def fit_supervised_teacher(*,data_dir:Path,model_path:Path,adapter_path:Path,
                           out:Path,seed:int=43,steps:int=96,
                           learning_rate:float=2e-5,
                           max_context:int=2048,
                           min_train_trajectories:int=10)->dict:
    """Imitate strict-success TRAIN-only actions; NOT on-policy GRPO."""
    if out.exists():raise FileExistsError(out)
    if steps<1 or min_train_trajectories<1:
        raise ValueError("need positive supervision budget")
    report=json.loads((data_dir/"TEACHER_SUMMARY.json").read_text())
    if (report.get("stage")!="TRAIN_ONLY_GOLD_SUPERVISED_TEACHER_NO_RL"
        or not report.get("includes_only_real_strict_successes")
        or report.get("student_model_sees_gold_answer") is not False):
        raise ValueError("missing trusted train-only supervised teacher manifest")
    if report["train_trajectories"]<min_train_trajectories:
        raise ValueError("not enough independently solved TRAIN teacher tasks")
    rows=list(iter_jsonl(data_dir/"TEACHER_ACTIONS_TRAIN.jsonl"))
    val=list(iter_jsonl(data_dir/"TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl"))
    if {r["question_id"] for r in rows}&{r["question_id"] for r in val}:
        raise ValueError("expert-action group leakage")
    import torch,transformers,peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError("Tang GPU is required")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    cfg=FiniteActionConfig(max_prompt_tokens=max_context,
                           answer_proposer=report["answer_proposer"])
    train,skip_train=_encode_teacher_rows(tokenizer,rows,config=cfg)
    validation,skip_val=_encode_teacher_rows(tokenizer,val,config=cfg)
    if len(train)<steps or not validation:
        raise ValueError("insufficient fit/validation teacher actions after tokenization")
    # Balanced deterministic per-Q task supervision so a single long teacher
    # trajectory does not overwhelm the adapter.
    from collections import defaultdict
    groups=defaultdict(list)
    for row in train:groups[row["question_id"]].append(row)
    group_ids=list(groups)
    random.shuffle(group_ids)
    order=[]
    for depth in range(max(map(len,groups.values()))):
        for qid in group_ids:
            if depth<len(groups[qid]):order.append(groups[qid][depth])
    selected=order[:steps]
    base=AutoModelForCausalLM.from_pretrained(
        str(model_path),local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda")
    model=PeftModel.from_pretrained(
        base,str(adapter_path),is_trainable=True,local_files_only=True)
    model.config.use_cache=False
    model.eval()  # no dropout, same normalized legal token distribution.
    optimizer=torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,weight_decay=0.)
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise RuntimeError("no LoRA parameters trainable")
    torch.cuda.reset_peak_memory_stats()
    pre=evaluate_teacher_label_fit(model,validation)
    t0=time.monotonic()
    losses=[]
    for ix,r in enumerate(selected):
        optimizer.zero_grad(set_to_none=True)
        lp=logprob_of_action(model,r["prompt_ids"],r["legal_ids"],r["target"])
        loss=-lp
        if not torch.isfinite(loss):raise FloatingPointError("NaN/Inf teacher loss")
        loss.backward()
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.0))
        if not math.isfinite(grad) or grad<=0:
            raise RuntimeError("vanishing/nonfinite supervised categorical gradient")
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if (ix+1)%16==0:
            print("TEACHER_SFT_UPDATE",ix+1,"/",len(selected),
                  "loss",round(losses[-1],4),
                  "elapsed_s",round(time.monotonic()-t0,1),flush=True)
    post=evaluate_teacher_label_fit(model,validation)
    out.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    old_sha=sha256_file(adapter_path/"adapter_model.safetensors")
    new_sha=sha256_file(out/"adapter_model.safetensors")
    if new_sha==old_sha:
        raise RuntimeError("teacher SFT did not modify LoRA")
    summary={
        "stage":"SUPERVISED_TEACHER_DISTILLATION_NOT_ONPOLICY_GRPO",
        "train_only_gold_in_teacher_offline":True,
        "model_policy_prompt_gold":False,
        "real_train_strict_success_trajectories":report["strict_success_trajectories"],
        "train_teacher_actions_available":len(train),
        "heldout_train_teacher_actions_available":len(validation),
        "optimizer_updates":len(losses),
        "optimizer_action_label_targets":len(losses),
        "tool_observation_token_targets":0,
        "before_heldout_train_teacher_action":pre,
        "after_heldout_train_teacher_action":post,
        "mean_supervised_loss":sum(losses)/len(losses),
        "peak_gpu_bytes":torch.cuda.max_memory_allocated(),
        "seconds":round(time.monotonic()-t0,3),
        "seed":seed,
        "answer_proposer":report["answer_proposer"],
        "teacher_manifest_sha256":sha256_file(data_dir/"TEACHER_SUMMARY.json"),
        "old_adapter_sha256":old_sha,"new_adapter_sha256":new_sha,
        "train_label_tokenization_skips":dict(skip_train),
        "heldout_label_tokenization_skips":dict(skip_val),
        "online_GRPO_executed":False,
        "reason":"Oracle TRAIN labels were used to select legal offline teacher actions. This is supervised training, not RL.",
        "caution":"Heldout teacher-fit changes alone are not evidence of real tool QA improvements; paired frozen public DEV and fresh online GRPO experiments are needed.",
    }
    (out/"TEACHER_SFT_SUMMARY.json").write_text(
        json.dumps(summary,indent=2,ensure_ascii=False)+"\n")
    print("TEACHER_SFT_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    subs=p.add_subparsers(dest="stage",required=True)
    x=subs.add_parser("make")
    x.add_argument("--data",type=Path,required=True)
    x.add_argument("--dev-manifest",type=Path,required=True)
    x.add_argument("--out",type=Path,required=True)
    x.add_argument("--seed",type=int,default=43)
    x.add_argument("--question-limit",type=int,default=1024)
    x.add_argument("--max-steps",type=int,default=6)
    x.add_argument("--max-tool-calls",type=int,default=5)
    x.add_argument("--max-context",type=int,default=2048)
    x.add_argument("--answer-proposer",choices=["legacy","v2"],default="v2")
    y=subs.add_parser("fit")
    y.add_argument("--teacher-data",type=Path,required=True)
    y.add_argument("--model",type=Path,required=True)
    y.add_argument("--adapter",type=Path,required=True)
    y.add_argument("--out",type=Path,required=True)
    y.add_argument("--seed",type=int,default=43)
    y.add_argument("--steps",type=int,default=96)
    y.add_argument("--learning-rate",type=float,default=2e-5)
    y.add_argument("--max-context",type=int,default=2048)
    y.add_argument("--min-train-trajectories",type=int,default=10)
    a=p.parse_args()
    if a.stage=="make":
        report=make_teacher_data(
            a.data,a.dev_manifest,a.out,seed=a.seed,
            question_limit=a.question_limit,max_steps=a.max_steps,
            max_tool_calls=a.max_tool_calls,max_context=a.max_context,
            answer_proposer=a.answer_proposer)
        print("TEACHER_MAKE_COMPLETE",json.dumps(report,ensure_ascii=False))
    else:
        fit_supervised_teacher(
            data_dir=a.teacher_data,model_path=a.model,adapter_path=a.adapter,
            out=a.out,seed=a.seed,steps=a.steps,
            learning_rate=a.learning_rate,max_context=a.max_context,
            min_train_trajectories=a.min_train_trajectories)

if __name__=="__main__":
    main()
