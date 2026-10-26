"""TRAIN-only masked multi-turn group-relative policy update (REPLAY PILOT).

The input is actual sampled SearchAgent tool-interaction rollouts produced by
scripts/sample_train_rollouts.py using Qwen3/vLLM. Every sampled assistant
ACTION string and its exact tool-context prompt is recorded. This optimizer:
  * scores reward GROUPS computed from actual TRAIN-only completed episodes;
  * applies group-normalized advantages to *assistant action tokens only*;
  * accumulates a one-step clipped policy-gradient update on LoRA weights;
  * skips tool observations as labels, and never reads DEV/test gold labels.

Important honesty: the vLLM samples used JSON-SCHEMA CONSTRAINED decoding,
whereas this replay computes *unconstrained base LM* log-probs. It is therefore
a biased REPLAY GRPO-STYLE pilot, NOT standard on-policy Agentic GRPO.
Do not call it full GRPO convergence or RSI. No DEV score is consulted by
the optimizer. Future iterations need matched guided-decoding log-probs or
native TRL/verl tool rollout integration.

Use:
 python scripts/replay_masked_grpo_pilot.py --rollouts <train_only.jsonl> \
    --dev-manifest <frozen_manifest.json> --model <Qwen3-0.6B-Base> \
    --adapter <SFT-LoRA> --out <new output directory> --max-events 28
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as source:
        for b in iter(lambda: source.read(8<<20),b""):
            h.update(b)
    return h.hexdigest()


def validate_train_replay(records:list[dict], dev_manifest:Path)->list[list[dict]]:
    m=json.loads(dev_manifest.read_text())
    banned=set(m["pilot"]["selected_ids"])|set(m["heldout"]["selected_ids"])
    byqid=defaultdict(list)
    for row in records:
        qid=str(row.get("question_id",""))
        if not qid or qid in banned:
            raise ValueError("TRAIN-only replay includes dev/invalid question ID")
        if row.get("trace",{}).get("question_id")!=qid:
            raise ValueError("trace question mismatch")
        if not isinstance(row.get("model_action_io"),list):
            raise ValueError("missing exact model action strings")
        if any("gold_answer" in json.dumps(action.get("messages",[])) or
               "supporting_doc_ids" in json.dumps(action.get("messages",[]))
               for action in row["model_action_io"]):
            raise ValueError("ground truth leaked into assistant prompt")
        if not math.isfinite(float(row["group_advantage"])):
            raise ValueError("invalid group-relative advantage")
        byqid[qid].append(row)
    groups=[sorted(group,key=lambda r:r["rollout_index"])
            for qid,group in sorted(byqid.items())]
    if not groups or len({len(g) for g in groups})!=1:
        raise ValueError("unequal/empty rollout groups")
    return groups


def validate_train_membership(groups:list[list[dict]],train_dataset:Path)->str:
    from sidra.io import load_dataset
    _docs,questions=load_dataset(train_dataset)
    if any(q.split!="train" for q in questions):
        raise ValueError("non-TRAIN split in policy update source")
    qmap={q.question_id:q for q in questions}
    if len(qmap)!=len(questions):
        raise ValueError("duplicate TRAIN question IDs")
    for g in groups:
        for record in g:
            q=qmap.get(record["question_id"])
            if q is None or record["trace"]["question"]!=q.question:
                raise ValueError("replay not matched to an original TRAIN question")
    return sha256_file(train_dataset/"questions.jsonl")


def clip_policy_loss(logp_new,logp_old,advantage:float,epsilon:float=0.2):
    """Token-level PPO/GRPO clipped surrogate; only assistant tokens input."""
    import torch
    if not (0<epsilon<1) or not math.isfinite(advantage):
        raise ValueError("invalid group policy loss configuration")
    ratio=torch.exp((logp_new-logp_old).clamp(-20,20))
    adv=torch.as_tensor(advantage,device=logp_new.device,dtype=ratio.dtype)
    return -torch.minimum(ratio*adv,torch.clamp(ratio,1-epsilon,1+epsilon)*adv).mean()


def assistant_token_logprobs(model,input_ids,context_length:int):
    """Return log P of generated assistant token ids only.

    Given tokens [PROMPT | ASSISTANT_ACTION], the labels are strictly
    input_ids[P:], conditional logits strictly logits[P-1 : -1].
    All prompt, tool output and question tokens are *conditioning only*.
    """
    import torch
    if input_ids.ndim!=2 or input_ids.shape[0]!=1:
        raise ValueError("expected a single tokenized action")
    total=input_ids.shape[1]
    if not 1<=context_length<total:
        raise ValueError("bad assistant token boundary")
    logits=model(input_ids=input_ids).logits
    next_logits=logits[:,context_length-1:total-1,:].float()
    target=input_ids[:,context_length:total]
    gathered=torch.log_softmax(next_logits,dim=-1).gather(
        -1,target.unsqueeze(-1)).squeeze(-1).squeeze(0)
    if gathered.shape[0]!=total-context_length:
        raise AssertionError("assistant mask / label alignment failed")
    return gathered


def tokenized_assistant_action(tokenizer,record:dict,*,max_length:int=2048):
    messages=record["messages"]
    if not isinstance(messages,list) or not messages:
        raise ValueError("missing logged model prompt")
    completion=record["raw_completion"]
    if not isinstance(completion,str) or not completion.strip():
        raise ValueError("empty assistant completion")
    prompt=tokenizer.apply_chat_template(
        messages,tokenize=False,add_generation_prompt=True,
        enable_thinking=False)
    # Independent tokenization is valid at Qwen3's explicit assistant chat
    # prefix separator; tested via exact round-trip in CPU tests/smoke.
    p=tokenizer(prompt,add_special_tokens=False)["input_ids"]
    c=tokenizer(completion,add_special_tokens=False)["input_ids"]
    if not c or not p or len(p)+len(c)>max_length:
        return None
    return p+c,len(p),len(c)


def run(rollouts:Path,dev_manifest:Path,train_dataset:Path,model_dir:Path,
        adapter_dir:Path,out:Path, *,
        seed:int=43,max_events:int=28,
        max_length:int=2048,lr:float=8e-6,
        clip_epsilon:float=0.2)->dict:
    if out.exists():
        raise FileExistsError(out)
    if max_events<2:raise ValueError("max_events must be at least two")
    if not (rollouts.is_file() and dev_manifest.is_file() and
            (adapter_dir/"adapter_model.safetensors").is_file()):
        raise FileNotFoundError("missing TRAIN rollouts/manifest/adapter")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM,AutoTokenizer

    records=[json.loads(line) for line in rollouts.read_text().splitlines() if line.strip()]
    groups=validate_train_replay(records,dev_manifest)
    train_dataset_sha256=validate_train_membership(groups,train_dataset)
    # Only groups with nonzero GROUP rewards can supply a relative gradient.
    varying=[g for g in groups if any(abs(float(r["group_advantage"]))>1e-6 for r in g)]
    if not varying:raise ValueError("all TRAIN groups have constant rewards: abort GRPO replay")
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(
        str(model_dir),local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda")
    model=PeftModel.from_pretrained(
        base,str(adapter_dir),is_trainable=True,local_files_only=True)
    model.config.use_cache=False
    # Disable LoRA dropout to compare training logits with frozen rollout
    # checkpoint without stochastic regularization.
    model.eval()
    random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    source_adapter_sha=sha256_file(adapter_dir/"adapter_model.safetensors")
    entries=[]
    skipped={"empty_completion_or_long_context":0,"zero_advantage":0}
    for group in varying:
        for record in group:
            adv=float(record["group_advantage"])
            if abs(adv)<=1e-6:
                skipped["zero_advantage"]+=len(record["model_action_io"])
                continue
            for step_idx,event in enumerate(record["model_action_io"]):
                packed=tokenized_assistant_action(tokenizer,event,max_length=max_length)
                if packed is None:
                    skipped["empty_completion_or_long_context"]+=1
                    continue
                entries.append((record["question_id"],record["rollout_index"],
                                step_idx,adv,packed))
    if not entries:
        raise ValueError("no nonconstant train model actions within max_length")
    # Keep all sampled actions from a chosen group where possible; limited
    # event count is deterministic in frozen qid/generation/turn order.
    entries=entries[:max_events]

    # Reference probabilities from EXACT initial unmodified LoRA checkpoint.
    old_probs=[]
    with torch.no_grad():
        for _,_,_,_,(all_ids,context_len,_) in entries:
            tokens=torch.tensor([all_ids],device="cuda",dtype=torch.long)
            old_probs.append(assistant_token_logprobs(
                model,tokens,context_len).detach())

    params=[p for p in model.parameters() if p.requires_grad]
    if not params:raise RuntimeError("PEFT did not enable trainable weights")
    previous_tensor=params[0].detach().float().clone()
    optimizer=torch.optim.AdamW(params,lr=lr,weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    beginning=time.perf_counter()
    scalar_losses=[]
    for index,(qid,roll_idx,turn,adv,(all_ids,context_len,actlen)) in enumerate(entries):
        tokens=torch.tensor([all_ids],device="cuda",dtype=torch.long)
        current=assistant_token_logprobs(model,tokens,context_len)
        loss=clip_policy_loss(current,old_probs[index],adv,epsilon=clip_epsilon)
        if not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite GRPO-style replay loss")
        (loss/len(entries)).backward()
        scalar_losses.append(float(loss.detach().cpu()))
        if (index+1)%5==0:
            print("BACKPROP",index+1,"/",len(entries),"loss",round(scalar_losses[-1],5),
                  "cuda_gib",round(torch.cuda.memory_allocated()/2**30,3),flush=True)

    gradnorm=float(torch.nn.utils.clip_grad_norm_(params,max_norm=1.0))
    if not math.isfinite(gradnorm) or gradnorm<=0:
        raise RuntimeError("no usable policy gradient: refusing to save fake update")
    optimizer.step()
    change=float((params[0].detach().float()-previous_tensor).abs().max().cpu())
    if change<=0:
        raise RuntimeError("policy weights unchanged after optimizer step")

    out.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    new_adapter=out/"adapter_model.safetensors"
    if not new_adapter.is_file():
        raise RuntimeError("PEFT optimizer did not save an adapter")
    aftersha=sha256_file(new_adapter)
    if aftersha==source_adapter_sha:
        raise RuntimeError("adapter SHA unchanged; training did not update model")
    result={
        "stage":"TRAIN-only multi-turn masked group-relative CLIPPED REPLAY PILOT",
        "standard_agentic_grpo":False,
        "on_policy_grpo":False,
        "genuine_model_parameter_update":True,
        "optimizer_steps":1,
        "train_only_grouped_real_model_rollouts":len(groups),
        "train_variable_reward_groups":len(varying),
        "backpropagated_assistant_actions":len(entries),
        "assistant_generated_tokens":sum(x[4][2] for x in entries),
        "tool_observation_tokens_as_labels":0,
        "max_length":max_length,
        "lr":lr,"clip_epsilon":clip_epsilon,
        "grad_norm_before_clip":gradnorm,
        "parameter_change_max_abs":change,
        "loss_mean":sum(scalar_losses)/len(scalar_losses),
        "loss_min":min(scalar_losses),"loss_max":max(scalar_losses),
        "source_rollout_sha256":sha256_file(rollouts),
        "verified_train_dataset_questions_sha256":train_dataset_sha256,
        "source_adapter_sha256":source_adapter_sha,
        "saved_adapter_sha256":aftersha,
        "time_seconds":round(time.perf_counter()-beginning,3),
        "peak_cuda_memory_bytes":torch.cuda.max_memory_allocated(),
        "skipped":skipped,
        "primary_caveat":"vLLM constrained decoding, but log-probs from unconstrained Transformers LM: biased replay objective, NOT standard on-policy GRPO.",
        "research_conclusion":"Only a one-step optimization proof; must re-sample from updated policy and evaluate independent QA for actual RL learning.",
    }
    (out/"GRPO_REPLAY_PILOT.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rollouts",type=Path,required=True)
    p.add_argument("--dev-manifest",type=Path,required=True)
    p.add_argument("--train-data",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--adapter",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--max-events",type=int,default=28)
    p.add_argument("--max-length",type=int,default=2048)
    p.add_argument("--lr",type=float,default=8e-6)
    p.add_argument("--seed",type=int,default=43)
    a=p.parse_args()
    run(a.rollouts,a.dev_manifest,a.train_data,a.model,a.adapter,a.out,
        seed=a.seed,max_events=a.max_events,max_length=a.max_length,lr=a.lr)


if __name__=="__main__":
    main()
