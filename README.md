# Self-Improving Deep Research Agent

Research-first implementation of a traceable, evidence-grounded, multi-hop search agent with reproducible evaluation and a later self-improvement / reinforcement-learning loop.

## Research questions

1. Does iterative search and document reading improve verified multi-hop answer quality over single-query retrieval under the same tool budget?
2. Can verifier-filtered execution trajectories improve a small model after SFT/GRPO without leaking evaluation answers?
3. Does Challenger → Solver → Verifier → Trainer self-improvement generalize to held-out tasks, or merely overfit the synthetic/task-generation distribution?

## Evaluation contract

- Separate **smoke tests** (deterministic fixtures), **offline baselines**, **real held-out benchmark**, and **GPU-trained models**. Never present fixture accuracy as a benchmark result.
- Freeze benchmark splits, prompts, models, seeds, retrieval corpus and budget before comparisons.
- Report exact-match/F1, supporting-evidence precision/recall, trajectory/tool cost, tokens and latency. Preserve raw JSONL traces; do not include test answers in policy prompts.
- Explicitly label every result as `not run`, `smoke only`, `baseline`, `trained`, or `external validation`.

## Development policy

GitHub is the single source of truth. Develop and review code in GitHub; only run model/GPU experiments on Tang when necessary. Do not commit secrets, checkpoints, private corpora, or large result artifacts.

Development starts in a feature branch with a PR. See `docs/PROBLEM_LOG.md` for encountered problems, mitigations and validation evidence, and `docs/EXPERIMENT_LOG.md` for every experiment.

> Status (2026-10-26): repository initialization. No GPU training or RSI gain is claimed.
