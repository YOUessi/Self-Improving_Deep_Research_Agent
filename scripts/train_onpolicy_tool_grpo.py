"""On-policy multi-turn TRAIN-only group-relative LoRA pilot for SIDRA.

Different from M6 replay:
- ACTIONS are sampled live from the CURRENT LoRA model (no vLLM / constrained
  JSON decoder); true SearchAgent tools execute after each model action;
- do_sample=True, temperature=1, top_p=1, top_k=0 and repetition penalty=1:
  the behavior action-token distribution is the SAME raw-softmax model whose
  token log-probs are used in the clipped GRPO surrogate;
- only *generated assistant action tokens* contribute to the loss; everything
  from user/system/SEARCH/READ is conditioning-only; every training question
  is from HotpotQA TRAIN, and dev gold never reaches the policy;
- the GROUP reward is computed only AFTER the complete multi-step rollout,
  using TRAIN labels. At least one group must have nonconstant rewards or
  training aborts with no saved fake adapter.

This is a small custom beta=0, single-optimizer-step-per-group-batch
ON-POLICY group-relative policy-gradient experiment; it does NOT reproduce
every TRL GRPO feature or claim convergence/RSI. Re-sampling and re-scoring
must occur after every update. The previous M6 constrained replay explicitly
DID NOT satisfy the distribution match this script enforces.

Usage (Tang, isolated CodeRL-Lab .venv):
 python scripts/train_onpolicy_tool_grpo.py --train-data <TRAIN/converted> \\
    --dev-manifest <DEV/FROZEN_MANIFEST.json> --model <Qwen3-0.6B-Base> \\
    --adapter <SFT-stop-LoRA> --out <NEW-dir> --seed 43 \\
    --max-updates 1 --groups-per-update 6 --num-generations 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

from sidra.agent import AgentConfig,SearchAgent
from sidra.io import load_dataset,save_jsonl
from sidra.policies import SYSTEM,_json_object
from sidra.schema import Action
from sidra.train_reward import group_normalized_advantages,score_train_rollout


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda:stream.read(8<<20),b""):
            h.update(part)
    return h.hexdigest()


def select_train(questions, dev_ids:set[str], *, seed:int, max_questions:int):
    if max_questions<1:raise ValueError("max_questions must be positive")
    ids=[q.question_id for q in questions]
    if (len(ids)!=len(set(ids)) or any(q.split!="train" for q in questions)
        or bool(set(ids)&dev_ids)):
        raise ValueError("TRAIN-only split required; duplicate or dev ID leakage")
    chosen=sorted(questions,key=lambda q:(
        hashlib.sha256(f"{seed}:onpolicy:{q.question_id}".encode()).hexdigest(),
        q.question_id))
    if len(chosen)<max_questions:
        raise ValueError("insufficient TRAIN questions")
    return chosen[:max_questions]


def masked_action_logprobs(model,tokens,context_length:int):
    """Exact unfiltered conditional log-probs for sampled assistant tokens only."""
    import torch
    if tokens.ndim!=2 or tokens.shape[0]!=1 or not 1<=context_length<tokens.shape[1]:
        raise ValueError("invalid action-token mask boundary")
    prediction=model(input_ids=tokens).logits
    # Token ids [prompt | assistant], label token at index p requires
    # logits from p-1. Never train on prompt/system/tool returned tokens.
    logits=prediction[:,context_length-1:-1,:].float()
    targets=tokens[:,context_length:]
    sampled=torch.log_softmax(logits,dim=-1).gather(
        -1,targets.unsqueeze(-1)).squeeze(-1).squeeze(0)
    assert sampled.numel()==tokens.shape[1]-context_length
    return sampled


class OnPolicyHFAction:
    """Model-only tool policy: does not receive Question.gold/support labels."""
    name="train_onpolicy_raw_fullsoftmax_actions"

    def __init__(self,model,tokenizer,*,max_new_tokens:int=120,
                 max_context_tokens:int=1536,seed:int=43):
        self.model=model
        self.tokenizer=tokenizer
        self.max_new_tokens=max_new_tokens
        self.max_context_tokens=max_context_tokens
        self.seed=seed
        self.model_calls=0
        self.prompt_tokens=0
        self.completion_tokens=0
        self.samples:list[dict]=[]

    def decide(self,question:str,observations:list[dict],remaining:int)->Action:
        import torch
        messages=[
            {"role":"system","content":SYSTEM},
            {"role":"user","content":json.dumps({
                "question":question,"remaining_steps":remaining,
                "tool_history":observations,
            },ensure_ascii=False)},
        ]
        if "gold_answer" in messages[1]["content"] or "supporting_doc_ids" in messages[1]["content"]:
            raise ValueError("gold label leakage into on-policy rollout")
        rendered=self.tokenizer.apply_chat_template(
            messages,add_generation_prompt=True,tokenize=False,
            enable_thinking=False)
        input_ids=self.tokenizer(rendered,add_special_tokens=False,
                                 return_tensors="pt")["input_ids"]
        prefix_len=input_ids.shape[-1]
        if prefix_len+self.max_new_tokens>self.max_context_tokens:
            raise ValueError("on-policy action context length exceeds cap")
        input_ids=input_ids.to(self.model.device)
        call_seed=self.seed+self.model_calls
        self.model_calls+=1
        self.prompt_tokens+=prefix_len
        self.model.eval()  # No LoRA dropout: behavior and optimizer probability agree.
        torch.manual_seed(call_seed)
        torch.cuda.manual_seed_all(call_seed)
        with torch.no_grad():
            # Exactly full-softmax unfiltered sampling, no JSON grammar masking.
            generated=self.model.generate(
                input_ids=input_ids,do_sample=True,
                temperature=1.0,top_p=1.0,top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        sampled=generated[0,prefix_len:].detach().cpu().tolist()
        self.completion_tokens+=len(sampled)
        raw=self.tokenizer.decode(sampled,skip_special_tokens=True)
        self.samples.append({
            "prompt_ids":input_ids[0].detach().cpu().tolist(),
            "generated_ids":sampled,
            "raw_completion":raw,
            "seed":call_seed,"context_len":prefix_len,
            "contains_tool_observation_targets":False,
        })
        return Action.from_dict(_json_object(raw))


def clipped_loss(new_lp,old_lp,adv:float,epsilon:float=0.2):
    import torch
    if not math.isfinite(adv) or not 0<epsilon<1:
        raise ValueError("invalid advantage or clip")
    ratio=torch.exp((new_lp-old_lp).clamp(-20,20))
    a=torch.tensor(adv,device=ratio.device,dtype=ratio.dtype)
    return -torch.minimum(ratio*a,torch.clamp(ratio,1-epsilon,1+epsilon)*a).mean()


def update_from_new_rollouts(model,episodes:list[dict],optimizer, *,
                             max_context_tokens:int,clip:float=0.2):
    """On-policy update: old log-probs from SAME model, immediately before step.

    Generates all episodes at unchanged parameters. No other model updates may
    happen between generation and this calculation. One optimizer step, then
    all old log-probabilities are discarded: next update MUST resample.
    """
    import torch
    active=[]
    for ep in episodes:
        adv=float(ep["advantage"])
        if abs(adv)<1e-7:continue
        for action in ep["samples"]:
            p=action["prompt_ids"]
            c=action["generated_ids"]
            if not p or not c or len(p)+len(c)>max_context_tokens:
                continue
            active.append((p,c,adv))
    if not active:
        return {"updated":False,"reason":"zero_advantages_or_no_valid_actions",
                "assistant_actions":0,"assistant_tokens":0}
    model.eval()
    refs=[]
    with torch.no_grad():
        for p,c,_ in active:
            ids=torch.tensor([p+c],device=model.device,dtype=torch.long)
            refs.append(masked_action_logprobs(model,ids,len(p)).detach())
    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise RuntimeError("No trainable LoRA parameters")
    weight_before=params[0].detach().float().clone()
    optimizer.zero_grad(set_to_none=True)
    losses=[]
    for ix,(p,c,adv) in enumerate(active):
        ids=torch.tensor([p+c],device=model.device,dtype=torch.long)
        new=masked_action_logprobs(model,ids,len(p))
        loss=clipped_loss(new,refs[ix],adv,epsilon=clip)
        if not torch.isfinite(loss):raise FloatingPointError("non-finite policy loss")
        (loss/len(active)).backward()
        losses.append(float(loss.detach().cpu()))
    grad=float(torch.nn.utils.clip_grad_norm_(params,1.0))
    if not math.isfinite(grad) or grad<=0:
        raise RuntimeError("Policy gradient is zero or nonfinite")
    optimizer.step()
    weight_delta=float((params[0].detach().float()-weight_before).abs().max())
    if weight_delta<=0:
        raise RuntimeError("optimizer changed no trainable weights")
    return {
        "updated":True,"assistant_actions":len(active),
        "assistant_tokens":sum(len(c) for _,c,_ in active),
        "tool_observation_tokens_in_loss":0,
        "grad_norm_before_clip":grad,"parameter_delta_max":weight_delta,
        "loss_mean":sum(losses)/len(losses),
        "loss_min":min(losses),"loss_max":max(losses),
    }


def train(*,data:Path,dev_manifest:Path,model_path:Path,adapter_path:Path,
          out:Path,seed:int=43,max_updates:int=1,
          groups_per_update:int=6,generations:int=4,
          max_new_tokens:int=120,max_context_tokens:int=1536,
          max_steps:int=6,max_tool_calls:int=5,
          learning_rate:float=1.0e-5)->dict:
    if out.exists():raise FileExistsError(out)
    if min(max_updates,groups_per_update,generations,max_new_tokens,max_steps,
           max_tool_calls)<1:raise ValueError("invalid training limits")
    if generations<2:raise ValueError("need >=2 sampled completions per task")
    import torch,transformers,trl,peft
    from peft import PeftModel
    from transformers import AutoTokenizer,AutoModelForCausalLM

    docs,qs=load_dataset(data)
    dev=json.loads(dev_manifest.read_text())
    dev_ids=set(dev["pilot"]["selected_ids"])|set(dev["heldout"]["selected_ids"])
    eligible=select_train(qs,dev_ids,seed=seed,
                          max_questions=max_updates*groups_per_update)
    lookup={d.doc_id:d for d in docs}
    cfg=AgentConfig(max_steps=max_steps,max_tool_calls=max_tool_calls,
                    search_top_k=3)
    tokens=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(
        str(model_path),local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda")
    model=PeftModel.from_pretrained(
        base,str(adapter_path),is_trainable=True,local_files_only=True)
    model.config.use_cache=False
    model.eval()
    optimizer=torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,weight_decay=0)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    runner=SearchAgent(docs,OnPolicyHFAction(model,tokens),cfg)
    started=time.monotonic()
    records=[]
    summary_updates=[]
    for iteration in range(max_updates):
        batch=eligible[iteration*groups_per_update:(iteration+1)*groups_per_update]
        collected=[]
        reward_groups=[]
        batch_seeds=[]
        for idx,q in enumerate(batch):
            group=[]
            for gen in range(generations):
                rollout_seed=int(hashlib.sha256(
                    f"grpo:{seed}:{iteration}:{q.question_id}:{gen}".encode()).hexdigest()[:8],16)
                policy=OnPolicyHFAction(
                    model,tokens,max_new_tokens=max_new_tokens,
                    max_context_tokens=max_context_tokens,seed=rollout_seed)
                runner.policy=policy
                trace=runner.run(q)
                reward=score_train_rollout(q,trace,lookup,
                                          max_tool_calls=max_tool_calls)
                group.append({
                    "qid":q.question_id,"episode_index":gen,
                    "reward":reward.as_dict(),"trace":trace.as_dict(),
                    "samples":policy.samples,"seed":rollout_seed,
                })
                batch_seeds.append(rollout_seed)
            advances=group_normalized_advantages(
                [x["reward"]["total"] for x in group])
            for ep,adv in zip(group,advances):
                ep["advantage"]=float(adv)
            reward_groups.append(group)
            collected.extend(group)
            print("ONLINE_GROUP",iteration,idx+1,"/",len(batch),
                  "scores",[x["reward"]["total"] for x in group],
                  "has_variance",any(abs(x)>1e-6 for x in advances),
                  "elapsed_s",round(time.monotonic()-started,1),flush=True)
        varying=sum(any(abs(x["advantage"])>1e-6 for x in g) for g in reward_groups)
        upd=update_from_new_rollouts(
            model,collected,optimizer,max_context_tokens=max_context_tokens)
        summary_updates.append({
            "iteration":iteration,"qids":[q.question_id for q in batch],
            "real_sampled_episodes":len(collected),
            "nonconstant_reward_groups":varying,
            "reward_mean":round(sum(x["reward"]["total"] for x in collected)/
                                len(collected),6),
            "strict_success_count":sum(x["reward"]["strict_outcome"] for x in collected),
            "termination_counts":dict(Counter(x["trace"]["termination"] for x in collected)),
            **upd,
        })
        # Trajectories contain TRAIN-only prompts and outputs, not DEV.
        for ep in collected:
            ep["update_iteration"]=iteration
            records.append(ep)
        print("ONLINE_UPDATE_RESULT",json.dumps(summary_updates[-1],
              ensure_ascii=False),flush=True)

    if not any(s["updated"] for s in summary_updates):
        raise RuntimeError("no nonconstant group policy update; refusing false success")
    out.mkdir(parents=True,exist_ok=False)
    save_jsonl(out/"train_onpolicy_rollouts.jsonl",records)
    model.save_pretrained(str(out))
    tokens.save_pretrained(str(out))
    p=out/"adapter_model.safetensors"
    if not p.exists():raise FileNotFoundError("optimizer saved no LoRA weights")
    origin=sha256(adapter_path/"adapter_model.safetensors")
    dest=sha256(p)
    if origin==dest:
        raise RuntimeError("optimizer did not alter adapter SHA")
    summary={
        "model_path":str(model_path),"source_adapter":str(adapter_path),
        "source_adapter_sha256":origin,"result_adapter_sha256":dest,
        "train_dataset":str(data),
        "train_questions_sha256":sha256(data/"questions.jsonl"),
        "dev_manifest_sha256":sha256(dev_manifest),
        "selected_train_questions":[q.question_id for q in eligible],
        "dev_questions_used_as_optimizer_gold":False,
        "split":"train","seed":seed,
        "on_policy_generation_and_loss_match":True,
        "generation_distribution":(
            "Unconstrained full softmax, temp=1 top_p=1 top_k=0, "
            "no repetition/logits adjustments; old/new token logprobs use same LM."
        ),
        "TRL_GRPOTrainer_used":False,
        "loss_type":"custom token-level clipped GRPO-style beta0 group-relative",
        "real_multi_turn_tool_rollouts":True,
        "tool_observation_label_tokens":0,
        "num_optimizer_updates":sum(s["updated"] for s in summary_updates),
        "update_reports":summary_updates,
        "max_context_tokens":max_context_tokens,
        "max_new_tokens":max_new_tokens,
        "versions":{"torch":torch.__version__,"transformers":transformers.__version__,
                    "trl":trl.__version__,"peft":peft.__version__},
        "training_seconds":round(time.monotonic()-started,2),
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
        "critical_caveat":(
            "Small on-policy research pilot, NOT TRL/verl GRPO convergence, "
            "multi-seed statistical proof or RSI. No constrained-decoding mismatch. "
            "Must compare model QA on fixed eval data and measure rewards again."
        ),
    }
    (out/"ONPOLICY_TRAIN_SUMMARY.json").write_text(
        json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    print("ONPOLICY_TRAIN_COMPLETE",json.dumps(summary,ensure_ascii=False),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",type=Path,required=True)
    p.add_argument("--dev-manifest",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--updates",type=int,default=1)
    p.add_argument("--groups-per-update",type=int,default=6)
    p.add_argument("--generations",type=int,default=4)
    p.add_argument("--max-new-tokens",type=int,default=120)
    p.add_argument("--max-context-tokens",type=int,default=1536)
    args=p.parse_args()
    train(data=args.train_data,dev_manifest=args.dev_manifest,model_path=args.model,
          adapter_path=args.adapter,out=args.out,seed=args.seed,
          max_updates=args.updates,groups_per_update=args.groups_per_update,
          generations=args.generations,max_new_tokens=args.max_new_tokens,
          max_context_tokens=args.max_context_tokens)


if __name__=="__main__":
    main()
