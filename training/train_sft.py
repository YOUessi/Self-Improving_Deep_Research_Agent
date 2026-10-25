"""Real LoRA SFT training from verified train traces; run on Tang with GPU.

Requires `pip install -e '.[sft]'` and an appropriate training-capable GPU.
Generated fixture data is for pipeline smoke-testing ONLY, not research claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--model", required=True, help="e.g. Qwen/Qwen3-0.6B")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=8)
    args = parser.parse_args()
    # Lazy import so the entire CPU baseline works without torch/transformers.
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit("Missing training dependencies. Run: pip install -e '.[sft]' ") from exc
    if not torch.cuda.is_available():
        raise SystemExit("SFT requires a GPU; use Tang instead of claiming a CPU smoke run trained a model")
    if args.epochs <= 0 or args.lr <= 0:
        parser.error("epochs and lr must be positive")
    items = []
    for index, line in enumerate(args.train_jsonl.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not obj.get("prompt") or not obj.get("completion"):
            raise ValueError(f"Malformed prompt/completion at line {index}")
        # Extra QA metadata and gold labels must never be supplied to the model trainer.
        items.append({"prompt": obj["prompt"], "completion": obj["completion"]})
    if not items:
        raise SystemExit("No training examples. Export verified train traces first.")
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=False)
    config = SFTConfig(
        output_dir=str(args.output), num_train_epochs=args.epochs,
        learning_rate=args.lr, max_length=args.max_length,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        gradient_checkpointing=True, logging_steps=5,
        completion_only_loss=True, packing=False,
        save_strategy="epoch", seed=args.seed, report_to="none",
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
    )
    adapter = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r,
                         target_modules="all-linear", lora_dropout=0.05,
                         task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=args.model, args=config, train_dataset=Dataset.from_list(items),
                         peft_config=adapter)
    trainer.train()
    trainer.save_model(str(args.output / "final_adapter"))
    (args.output / "train_metadata.json").write_text(json.dumps({
        "model": args.model, "epochs": args.epochs, "lr": args.lr,
        "seed": args.seed, "examples": len(items), "lora_r": args.lora_r,
        "training_type": "LoRA SFT (not GRPO, not RSI)"}, indent=2) + "\n", encoding="utf-8")
    print(f"Saved LoRA adapter and metadata under {args.output}")


if __name__ == "__main__":
    main()
