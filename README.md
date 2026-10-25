# Self-Improving Deep Research Agent (SIDRA)

**Status (2026-10-26): M1 traceable-search baseline and CPU smoke tests complete. Tang has run cached pretrained Qwen2.5-1.5B/3B **on synthetic fixtures only**. In the latest 3B 8-question dev smoke check, answer EM=1/8 and independently verified success=0/8. No real HotpotQA baseline, SFT, GRPO, or RSI improvement has been demonstrated.**

Research-oriented, evidence-grounded multi-hop agent with **search → read → finish**, explicit tool budgets, read-before-cite validation, JSONL trajectories, reproducible metrics, train-only SFT export, and an OpenAI-compatible model adapter (for vLLM, etc.). **GitHub is the single source of truth; Tang is reserved for experiments requiring GPU.**

## Fast start (no GPU, no API key, no network)

```bash
python -m pip install -e '.[test]'
python scripts/generate_fixture.py --out examples/fixture --n 40 --seed 13
pytest -q
sidra eval --dataset examples/fixture --split dev --policy fixture-onehop --out results/dev_onehop
sidra eval --dataset examples/fixture --split dev --policy fixture-multihop --out results/dev_multihop
sidra eval --dataset examples/fixture --split train --policy fixture-multihop --out results/train_multihop
sidra export-sft --dataset examples/fixture --traces results/train_multihop/traces.jsonl --out results/train_action_sft.jsonl
```

If pip cannot reach PyPI but pytest is already installed, use `PYTHONPATH=src python -m pytest -q` and `PYTHONPATH=src python -m sidra.cli ...` (documented in [problems](docs/PROBLEM_LOG.md)).

**DO NOT cite fixture results as genuine model performance:** the fixture policy is a *template parser*, not an LLM. The generated fake entity chains are only used to verify infrastructure and prevent leakage.

## Run actual multi-hop benchmark baseline

1. Download **HotpotQA distractor** data yourself from the original distribution source, respecting its license. Do not commit downloaded data (ignored by git). You may also supply your own `documents.jsonl` / `questions.jsonl` matching [schema](docs/ARCHITECTURE.md).
2. Convert existing *dev* data (this tool does not change split labels):

```bash
python scripts/convert_hotpotqa.py --input /path/to/hotpot_dev_distractor_v1.json --out benchmarks/hotpotqa_dev --split dev --limit 200
```

3. On Tang, if a compatible instruct model is already cached, the simplest **offline GPU** baseline avoids installing vLLM:

```bash
sidra eval --dataset benchmarks/hotpotqa_dev --split dev --policy transformers \
  --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 8 --max-tool-calls 7 \
  --limit 200 --out results/hotpot_dev_qwen2p5_pretrained
```

4. Alternatively start vLLM (if separately installed) or use an explicitly selected OpenAI-compatible endpoint. Evaluate with fixed budgets:

```bash
export OPENAI_API_KEY=""  # local vLLM usually does not need one
sidra eval --dataset benchmarks/hotpotqa_dev --split dev --policy openai \
  --endpoint http://127.0.0.1:8000/v1/chat/completions --model Qwen/Qwen3-0.6B \
  --max-steps 8 --max-tool-calls 7 --search-top-k 3 --limit 200 --out results/hotpot_dev_qwen3_s0
```

> **Scope caution:** HotpotQA converter uses *per-question candidate passages*, not open-web or full-Wikipedia retrieval. Report this in all comparisons. Raw answers remain evaluator-only. Do not use dev/test traces for SFT.

## Output of each evaluation

| File | Content |
|---|---|
| `traces.jsonl` | Step-by-step `search` / `read` / `finish` action, tool observation, termination, budget usage |
| `per_example.jsonl` | EM, token-F1, evidence precision/recall, verified success, calls and wall time |
| `summary.json` | Means over questions; never alone sufficient for model-level claims |
| `manifest.json` | SHA256 of datasets, split, policy, seed, limits, run stage and Git SHA |

Existing run directories cannot be overwritten by the runner. Training checkpoints and outputs must be stored outside GitHub (`results/`, `training_runs/` are ignored).

## Next milestones

- **M1:** CPU search/trace harness and HotpotQA adapter — implemented and fixture-tested.
- **M2:** Real model baseline on frozen non-synthetic held-out dev/test benchmark. Pretrained Qwen inference **on synthetic fixtures only** includes an 8-question 3B dev check (EM=1/8, verified=0/8); real benchmark **not yet run**.
- **M3:** Verifier-filtered *train-only* trajectory SFT. LoRA training entrypoint in `training/train_sft.py`; GPU run **not yet verified**.
- **M4:** Agentic RL / GRPO with reward/rollout integration and a strong SFT-only control. Design in [training notes](training/README.md), **not implemented**.
- **M5:** Challenger/Solver/Verifier self-improvement for multiple generations with separate frozen external test. **Not implemented**.

See [architecture](docs/ARCHITECTURE.md), [evaluation protocol](docs/EVALUATION_PROTOCOL.md), [experiment log](docs/EXPERIMENT_LOG.md) and [problem/solution log](docs/PROBLEM_LOG.md).

## Safety and data handling

Never commit keys, private user data, model weights, copyrighted benchmark dumps or confidential observations. Tools are bounded: `read` is only permitted on IDs returned by `search`, and `finish` can cite only read IDs. Web search is **not** included in M1; BM25 is over the provided local document corpus. The first version is not a general internet-browsing research agent.
