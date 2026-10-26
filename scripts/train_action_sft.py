"""REAL train-only Qwen3 LoRA/SFT experiment using CodeRL-Lab's TRL method.

Inputs: verified TRAIN-only action demonstrations from build_train_demos.py.
Neither public HotpotQA dev nor sealed heldout labels enter the optimizer.

Train and internal validation are partitioned by question_id (not randomly
by trajectory step). Report training loss, model versions, seed, and adapter
hashes; loss decrease is NOT evidence of heldout Agent improvement.

Run on Tang with CodeRL-Lab's pre-existing .venv:
 python scripts/train_action_sft.py --data <train_sft_actions.jsonl> \
   --model <Qwen3-0.6B-Base snapshot> --out <new output> --max-steps 40
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def split_training_rows(rows:list[dict], *, seed:int=43,
                        max_examples:int=1600,
                        validation_percentage:int=10) ->tuple[list[dict],list[dict]]:
    if not 0<validation_percentage<50:
        raise ValueError("validation_percentage must be 1..49")
    if max_examples < 16:
        raise ValueError("need at least 16 action examples")
    # Stable selection at QUESTION group level; no train/validation leakage.
    groups={}
    for r in rows:
        if set(r)!={"prompt","completion","question_id","step"}:
            raise ValueError("unexpected training row fields")
        qid=str(r["question_id"])
        groups.setdefault(qid,[]).append(r)
        if any("gold_answer" in json.dumps(m) or
               "supporting_doc_ids" in json.dumps(m) for m in r["prompt"]):
            raise ValueError("ground truth was injected into policy prompt")
    ordered=sorted(groups,key=lambda q:hashlib.sha256(f"{seed}:{q}".encode()).hexdigest())
    train, val=[],[]
    for qid in ordered:
        # Assign group by a DIFFERENT hash from selection to avoid correlation.
        bit=int(hashlib.sha256(f"validation:{seed}:{qid}".encode()).hexdigest()[:8],16)%100
        dest=val if bit<validation_percentage else train
        for row in groups[qid]:
            if len(train)+len(val) >= max_examples:
                break
            dest.append(row)
        if len(train)+len(val)>=max_examples:
            break
    if not train or not val:
        raise ValueError("group split has an empty partition")
    assert not {x["question_id"] for x in train} & {x["question_id"] for x in val}
    return train,val


def run(data:Path,model_dir:Path,out:Path, *, seed:int=43,
        max_steps:int=40, max_length:int=1536,
        max_examples:int=1600, grad_accum:int=8,
        learning_rate:float=0.00015,
        dry_run:bool=False) ->dict:
    if out.exists():
        raise FileExistsError(out)
    if not model_dir.is_dir() or max_steps<1 or max_length<256:
        raise ValueError("invalid model path or training limits")
    import torch
    import peft
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig,SFTTrainer

    rows=[json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    source_sha=sha256(data)
    train,val=split_training_rows(rows,seed=seed,max_examples=max_examples)
    tokenizer=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    def prepare(examples):
        result=[]
        dropped=0
        for row in examples:
            # Template is non-thinking Qwen3 chat; completion-only loss
            # prevents gradients from optimizing tool histories.
            prompt=tokenizer.apply_chat_template(
                row["prompt"],tokenize=False,add_generation_prompt=True,
                enable_thinking=False)
            completions=row["completion"]
            if len(completions)!=1 or completions[0]["role"]!="assistant":
                raise ValueError("expected one assistant action per sample")
            action=completions[0]["content"]
            suffix=action+tokenizer.eos_token
            if len(tokenizer.encode(prompt+suffix,add_special_tokens=False))>max_length:
                dropped+=1
                continue
            result.append({"prompt":prompt,"completion":suffix,
                           "question_id":str(row["question_id"])})
        return result,dropped
    train_ready,train_dropped=prepare(train)
    val_ready,val_dropped=prepare(val)
    if not train_ready or not val_ready:
        raise ValueError("no train/validation samples after context-length filtering")
    if {r["question_id"] for r in train_ready} & {r["question_id"] for r in val_ready}:
        raise AssertionError("group leakage after token filtering")
    meta={
        "stage":"real train-only LoRA/SFT" if not dry_run else "TRAIN-ONLY DRY-RUN (no optimizer)",
        "source_action_file":str(data),"source_sha256":source_sha,
        "model_snapshot":str(model_dir),"seed":seed,"max_steps":max_steps,
        "max_length":max_length,"gradient_accumulation_steps":grad_accum,
        "learning_rate":learning_rate,
        "train_action_examples":len(train_ready),
        "internal_validation_action_examples":len(val_ready),
        "distinct_train_questions":len({x["question_id"] for x in train_ready}),
        "distinct_validation_questions":len({x["question_id"] for x in val_ready}),
        "train_dropped_overlength":train_dropped,
        "validation_dropped_overlength":val_dropped,
        "dev_or_hidden_test_optimizations":False,
        "global_batch_effective":grad_accum,
        "versions":{"torch":torch.__version__,"transformers":transformers.__version__,
                    "trl":trl.__version__,"peft":peft.__version__},
        "dataset_generation":"label-conditioned oracle demos from TRAIN ONLY; not measured model trajectories",
        "interpretation":"Training loss alone is not proof of answer/evidence generalization",
    }
    out.mkdir(parents=True,exist_ok=False)
    (out/"DATA_SELECTION.json").write_text(
        json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
    if dry_run:
        print(json.dumps(meta,indent=2))
        return meta
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    model=AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache=False
    lora=LoraConfig(r=16,lora_alpha=32,lora_dropout=0.05,bias="none",
                    task_type="CAUSAL_LM",
                    target_modules=["q_proj","k_proj","v_proj","o_proj",
                                    "gate_proj","up_proj","down_proj"])
    args=SFTConfig(
        output_dir=str(out),
        max_steps=max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=max(1,max_steps//10),
        logging_steps=5,
        save_strategy="no",
        eval_strategy="steps",
        eval_steps=max(10,max_steps//2),
        seed=seed,data_seed=seed,
        bf16=True,fp16=False,
        gradient_checkpointing=True,
        max_length=max_length,
        packing=False,
        completion_only_loss=True,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer=SFTTrainer(model=model,args=args,
                      train_dataset=Dataset.from_list(
                          [{"prompt":x["prompt"],"completion":x["completion"]}
                           for x in train_ready]),
                      eval_dataset=Dataset.from_list(
                          [{"prompt":x["prompt"],"completion":x["completion"]}
                           for x in val_ready]),
                      processing_class=tokenizer,
                      peft_config=lora)
    started=time.perf_counter()
    result=trainer.train()
    elapsed=time.perf_counter()-started
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    adapter_files=sorted(out.glob("adapter_model.*"))
    meta.update({
        "training_elapsed_seconds":round(elapsed,2),
        "train_metrics":{k:float(v) for k,v in result.metrics.items()
                         if isinstance(v,(int,float))},
        "trainable_parameters":sum(p.numel() for p in trainer.model.parameters()
                                    if p.requires_grad),
        "peak_cuda_allocated_bytes":torch.cuda.max_memory_allocated(),
        "adapter_file_sha256":{p.name:sha256(p) for p in adapter_files},
        "trained":True,
    })
    (out/"RUN_SUMMARY.json").write_text(json.dumps(meta,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(meta,indent=2))
    return meta


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True,type=Path)
    p.add_argument("--model",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--max-steps",type=int,default=40)
    p.add_argument("--max-length",type=int,default=1536)
    p.add_argument("--max-examples",type=int,default=1600)
    p.add_argument("--grad-accum",type=int,default=8)
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--dry-run",action="store_true")
    a=p.parse_args()
    run(a.data,a.model,a.out,seed=a.seed,max_steps=a.max_steps,
        max_length=a.max_length,max_examples=a.max_examples,
        grad_accum=a.grad_accum,dry_run=a.dry_run)


if __name__=="__main__":
    main()
