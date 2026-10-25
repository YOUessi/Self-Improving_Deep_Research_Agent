# GPU training roadmap: SFT available, GRPO and RSI gated

**Do not describe smoke-test trace export as a trained model.**

## Stage T1: LoRA SFT on verified train-only trajectories

The exporter checks the question split and re-evaluates every candidate trace, rejects non-train examples and any trajectory without grounded, fully supported correct answers. It writes per-step `prompt` / `completion` messages so `TRL SFTTrainer` can train action prediction with *completion-only loss*.

```bash
sidra eval --dataset examples/fixture --split train --policy fixture-multihop --out results/train_smoke
sidra export-sft --dataset examples/fixture --traces results/train_smoke/traces.jsonl --out results/train_smoke_actions.jsonl
# Use real train split before claiming LLM ability improvement; on Tang:
python -m pip install -e '.[sft]'
python training/train_sft.py --train-jsonl /path/to/real_train_action_examples.jsonl \
  --model Qwen/Qwen3-0.6B --output training_runs/lora_sft_v01 \
  --epochs 1 --lr 0.0001 --seed 42
```

**Caution:** training on 40 synthetic fixture questions does not demonstrate generalization and is only valid for runtime checks. The training script is not yet GPU-validated; confirm pinned versions of transformers/trl/peft on Tang and do a tiny dry run before scaling. The model will learn agent *action formatting* and possibly behavior, not guaranteed Search/RSI improvement.

## Stage T2: Agentic GRPO (not yet implemented)

1. Define environment protocol mapping a rollout to `search/read/finish`, bounded 7 tool calls and deterministic tool observations.
2. Produce N sampled trajectories per prompt; score independent of policy transcript text via oracle/human or verifiable QA, citation support and reward hacking penalty. Limit reward to **train** labels; use no dev/test solutions in reward.
3. Plug into a pinned and tested version of verl multi-turn rollout (or another verified training harness); record exact code and model version.
4. Include equal-budget `base`, `SFT`, `RL without retrieval`, `SFT + GRPO`, `SFT + GRPO + verifier` controls. Run multiple seeds and a frozen external benchmark.
5. Do not claim GRPO training occurred until GPUs actually perform parameter updates and weights/checkpoints + logs exist.

## Stage T3: Recursive self-improvement (not yet implemented)

`Challenger` proposes harder multi-hop tasks based on failure taxonomy, *independent* `Verifier` checks answer/evidence, `Solver` generates trajectories, `Trainer` updates a model. Separate generated training tasks from frozen dev/test and report per-generation accepted-task count, checkpoint hash and external score. This is necessary to distinguish repeated prompting from genuine model-level self-improvement.

## Machine roles

GitHub: source, reproducible configs, docs, PR and error log. Mac: author code, CPU tests, data QA. Tang: serve GPU model, run SFT/RL, record logs and checkpoint metadata, push back only small run manifests and summaries (never checkpoints or private corpora).
