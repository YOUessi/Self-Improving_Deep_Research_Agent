# Self-Improving Deep Research Agent (SIDRA)

**Research status (2026-10-27, M6 Draft):** Qwen3-8B(-AWQ) weights are verified and real vLLM inference ran on Tang. HotpotQA distractor DEV has an immutable hash-frozen 256-QA set (pilot32, eval-A64, sealed final-B160). Qwen3-0.6B stop-curriculum SFT with dynamic online document-ID tool constraints achieved **17/64 answer EM** and **6/64 cited original supporting-sentence success** on eval-A64; this remains publicly sourced DEV, not an external hidden test. A gold-blind extra bridge search improved PILOT retrieval coverage but **reduced** real Agent answer EM. Real TRAIN sampled multi-turn rollouts showed sparse group rewards (5/12 and 2/8 variable groups). One actual masked group-relative replay LoRA update changed weights but **did not improve** pilot32 answer or evidence success; due constrained-inference vs unconstrained-logprob mismatch it is **NOT standard on-policy Agentic GRPO**. Genuine multi-turn on-policy GRPO and iterative RSI are **NOT implemented or validated**. Full [M6 results, limitations and checks](docs/M6_RESULTS_AND_RL_VALIDITY_20261027.md).

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

## Completed stages and remaining research gates

| Stage | Evidence-based status |
| --- | --- |
| **M1** Traceable local Search Agent | Search/read/finish tool sandbox, limits, JSONL traces, citation gate; deterministic CPU fixtures |
| **M2** Qwen3 and real multi-hop data | Checksum-verified Qwen3-8B BF16 & AWQ, vLLM API, genuine HotpotQA distractor runs |
| **M3** Multi-role Planner/Researcher/Solver/Verifier | Implemented and tested; multi-role overhead/negative results preserved; no general dominance |
| **M4** Frozen 256 DEV QA, evidence guard and train-only SFT | Pilot32/eval-A64/sealed final-B160, 8B evidence guard, genuine Qwen3-0.6B LoRA SFT |
| **M5** Stop/abstain SFT and dynamic DocID restrictions | Actual TRAIN-only 40-step LoRA ablation; same-vLLM eval-A64 Qwen3-0.6B result EM17/64, sentence-grounded6/64 |
| **M6** Second-hop query study and group-relative reward pilot | Pilot retrieval exposure improved, but Agent EM worsened; actual TRAIN 48+32 sampled tool episodes; one biased replay optimizer update with **zero** observed held-out pilot improvement |
| **M7 / open** True on-policy tool GRPO + multi-generation RSI | **Not done.** Needs reward-spread controls, matching sampling/logprobs, independent seeds, semantic attribution, Challenger–Solver–Verifier–Trainer updates and sealed final evaluation |

For review see [M6 execution report](docs/M6_RESULTS_AND_RL_VALIDITY_20261027.md), [experiment ledger](docs/EXPERIMENT_LOG.md), [problem/solution ledger](docs/PROBLEM_LOG.md), and [M5 preregistered protocol](docs/M5_PROTOCOL_20261027.md).

**Dataset caveat:** HotpotQA distractor tests use per-question candidate passages, **not full Wikipedia or actual web retrieval**. The original supporting-sentence audit checks that labeled sentences appeared in cited READ content, not full semantic entailment. All DEV samples are excluded from the RL/SFT optimizer; sealed final-B160 is still unused. Checkpoint binaries and raw train/dev data stay on Tang, not GitHub.


## Safety and data handling

Never commit keys, private user data, model weights, copyrighted benchmark dumps or confidential observations. Tools are bounded: `read` is only permitted on IDs returned by `search`, and `finish` can cite only read IDs. Web search is **not** included in M1; BM25 is over the provided local document corpus. The first version is not a general internet-browsing research agent.
