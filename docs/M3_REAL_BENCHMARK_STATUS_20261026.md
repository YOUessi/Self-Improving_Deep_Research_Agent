# M3 real HotpotQA interim results — 2026-10-26

This page is a factual checkpoint, **not an algorithmic claim of improvement**.

## Dataset and reproducibility
- HotpotQA distractor validation, downloaded from HF mirror (original CMU dataset).
- Original `hotpot_dev_distractor_v1.json` SHA-256:
  `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`.
- Original dev set: 7,405 QA examples; freeze first 24 via SHA256(seed=42, question ID). Each example has the provided 10 distractor candidate paragraphs, so do **not** compare as full-wikipedia/open-web retrieval.
- Retrieval-only E-003 on frozen 24 dev: BM25 Top3 support recall **64.58%**, both supports covered on **9/24**; Top5 recall **77.08%**, both supports **14/24**; Top10 covers all 10 candidate paragraphs by construction.
- Model: checksum-verified official Qwen3-8B BF16 on Tang RTX 4090 Laptop with GPU/CPU offload (12GiB cap), no fine-tuning. Qwen3 nonthinking chat template, greedy decoding, max 64 new tokens per action, max 6 steps, max 5 tool calls, BM25 top3.

## Four actual, previously frozen dev examples (two matched comparisons)

| Frozen positions | Single policy | Multi-role policy | Evidence verified | Mean model calls | Mean prompt tokens | Mean latency |
| --- | --- | --- | --- | --- | --- | --- |
| Positions 0–1, before multi-role repair | Single: 2/2 answer EM | Multi: 0/2 answer EM | 0/2 for both | Single 3.0, Multi 3.5 | Single 1719, Multi 2327.5 | Single 48.1s, Multi 63.7s |
| Positions 2–3, after bounded multi-role repair | Single: 0/2 answer EM | Multi: 0/2 answer EM | 0/2 for both | Single 4.0, Multi 5.0 | Single 3021, Multi 4315 | Single 72.4s, Multi 104.5s |

**Warning:** n=2 per condition is a **software/behavior diagnostic**, not meaningful statistical evidence. Position groups used **different GitHub code versions**; do not pool across them as if they shared a treatment implementation. The repaired multi-role change improved **neither** answer nor strict evidence success on its held-out pair, and increased inference cost.

### Technical observations
- Correct final answers can appear after one search and one read, without citing both gold supporting passages: answer EM ≠ verified evidence-complete success.
- Nonthinking Qwen3 BF16 emits some valid structured actions, but later turns frequently return an unparsable JSON object under max_new_tokens=64.
- Sequential Planner/Researcher/Solver/Verifier did not autonomously fix this. A bounded format retry itself incurs additional model calls and can still fail.
- Role count alone is not an improvement. Research design must track calls, tokens and latency alongside strict evidence metrics.

## Current implementations and pending checks
- `src/sidra/policies.py` now includes an **opt-in** strict action JSON Schema request for compatible OpenAI/vLLM endpoints, exercised in tests; **not yet validated with actual vLLM server**.
- The isolated M3 vLLM 0.31.0 setup contains official, SHA-verified vLLM and Torch wheels but may still be installing transitive dependencies (monitor `~/experiments/sidra_m3/logs/vllm_deps_after_torch_local.log`). Do not claim successful AWQ inference until import, CUDA and served request tests succeed.
- Raw evaluation outputs, manifests, failure traces remain on Tang under `~/experiments/sidra_m3/results` and are not uploaded to GitHub due dataset/reproduction boundaries.
- No SFT, GRPO or recursive self-improvement weight update has occurred in this project.

### Next decisive evaluation
Run Qwen3-8B-AWQ with vLLM JSON-schema constrained actions, frozen real HotpotQA dev split, compare both single and multi-role, fixed tool budget and model sampling protocol. Evaluate meaningful held-out sample counts rather than just 2 QA items. Keep a no-constraint control to isolate the action-format benefit.
