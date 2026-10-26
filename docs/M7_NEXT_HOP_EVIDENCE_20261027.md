# M7 E-021 — Frontier-first next-hop evidence retrieval (2026-10-27)

**Status: validated CPU/CI retrieval-only experiment, NOT a learned Search Agent gain, semantic verifier, Agentic GRPO, or RSI.** Dedicated branch: `feat/m7-next-hop-evidence`; parent research branch: `feat/m6-next-hop-search-grpo-rollouts`. This document records the *first fixed candidate*, developed on TRAIN64 and checked on the already opened DEV pilot32; it must not be edited to optimize on pilot results.

## Motivation, algorithm, and invariant

M6 E-016 showed retrieval exposure rising from 14/32 to 22/32 after a forced second search, but **real Qwen3-0.6B** bridge-wrapper answer EM fell from 6/32 to 3/32 under the same budget. Exposure is not reading, cited evidence, or a correct answer. The new *opt-in* frontier-first policy checks **only previously observed SEARCH hits, the question, and one actually READ passage** before a second query:

1. If an already-exposed, unread hit title matches the proposed bridge entity, or that full title is explicitly visible in READ text/question, issue a normal **READ** instead of a duplicate SEARCH.
2. Otherwise derive a bridge entity using the existing `next_bridge_query` from the observed first-hop content and perform the normal billed **SEARCH**, followed by a normal **READ**.
3. A new SEARCH requires >=3 remaining action steps (including FINISH) and >=2 remaining tool calls. Never use gold answer, supporting passage IDs, original sentence indices, corpus-wide invisible neighbors, or dev labels for query planning. This wrapper is not enabled as default.
4. Both controls and the candidate execute **actual `SearchAgent.search/read/finish` state transitions** using the same BM25 index and identical budget: max_steps=6, max_tool_calls=5, search_top_k=3, max_read_chars=3500. A scripted top-1 initial read and scripted second-read selection make this a **retrieval-only diagnostic**; the script does not generate LLM answers.

**Three fixed variants:** `baseline` = 1 initial SEARCH and 2 reads if hits exist; `forced_bridge` = E-016-style second SEARCH after first read, then second read; `frontier_first` = read visible linked frontier hit before considering a second SEARCH. All score support labels **only after the episode terminates**, in an offline evaluator.

## Data and reproducibility

- **TRAIN developer set:** uniformly seeded sample **64/1024** HotpotQA distractor converted TRAIN examples, `random.Random(43)` without checking outcomes. Training files on Tang: `~/experiments/sidra_m4/data/train_oracle1024_seed43/converted/`. SHA256 documents `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`, questions `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`.
- **One-time open pilot check:** exactly **32** frozen DEV pilot examples in `~/experiments/sidra_m4/data/frozen_seed42_32p_224h/pilot/converted/`. Documents SHA256 `5aabc048b344940ffdce2880a7a57a683d4009777a796fd0602ed683a500cc6c`; questions `94db9bdfff053555b9a8e0da9a850b5bee3de73e2406ec85c63461b5ae033e6e`. The CLI **fails closed** for arbitrary other DEV32 input using both hashes; does not load eval-A64 or final-B160.
- Reproduce locally from the branch: `PYTHONPATH=src python scripts/m7_retrieval_probe.py --dataset <TRAIN_CONVERTED_DIR> --mode train --limit 64 --seed 43 --out <UNUSED_OUTPUT_DIR>`; then (only for the frozen already-open pilot) use `--dataset <KNOWN_PILOT32_CONVERTED_DIR> --mode pilot --out <UNUSED_OUTPUT_DIR>`. Runs refuse to overwrite output directories.
- Recorded 2026-10-27 Tang outputs: `~/experiments/sidra_m7_evidence/results/train_seed43_n64/{summary.json,train_only_per_question.jsonl}`; `.../results/pilot32_sealed_plan_v1/summary.json`; hash-guard reproducibility rerun `.../results/pilot32_hashguard_rerun/summary.json`. PILOT runs export **aggregates only**, no per-question gold diagnostics. Bootstrap resamples *paired question units*, 5,000 repeats, seed 2027, percentile 95% intervals.
- Source: `src/sidra/next_hop_policy.py`, `scripts/m7_retrieval_probe.py`, `tests/test_next_hop_policy.py`. Local Tang **102 CPU tests passed** after fixture setup; GitHub Actions CPU tests are separate verification. Experiment is CPU-only; no GPU process was started/stopped.

## Observations: all scores are retrieval-only, NOT EM

| Dataset / policy | Both supports exposed | Both supports actually READ | Avg read-support recall | Avg SEARCH calls | Avg billed tool calls |
|---|---:|---:|---:|---:|---:|
| TRAIN64 baseline | 22/64 | 18/64 | 0.609375 | 1.00000 | 3.00000 |
| TRAIN64 forced bridge | 38/64 | 22/64 | 0.640625 | 1.96875 | 3.96875 |
| TRAIN64 frontier-first | **37/64** | **27/64** | **0.687500** | **1.62500** | **3.62500** |
| Open pilot32 baseline | 14/32 | 11/32 | 0.609375 | 1.00000 | 3.00000 |
| Open pilot32 forced bridge | 22/32 | 13/32 | 0.593750 | 2.00000 | 3.90625 |
| Open pilot32 frontier-first | **22/32** | **16/32** | **0.671875** | **1.62500** | **3.59375** |

### Frontier-first − forced-bridge paired bootstrap delta, 95% percentile CI

| Metric | TRAIN64 delta [CI] | Open pilot32 delta [CI] |
|---|---|---|
| Both supports exposed fraction | **−0.015625** [−0.046875, 0] | **0** [0, 0] |
| Both supports actually READ fraction | **+0.078125** [0, 0.15625] | **+0.093750** [0, 0.21875] |
| Mean READ-support recall | **+0.046875** [+0.007812, +0.093750] | **+0.078125** [+0.015625, +0.140625] |
| Mean SEARCH calls | **−0.343750** [−0.468750, −0.234375] | **−0.375000** [−0.562500, −0.218750] |
| Mean billed tool calls | **−0.343750** [−0.468750, −0.234375] | **−0.312500** [−0.468750, −0.156250] |

**Tradeoff and honest interpretation:** TRAIN frontier-first sacrifices *one* fully exposed evidence pair vs forced bridge (**37 vs 38**) while improving fully **READ** evidence by five cases (**27 vs 22**) and reducing search calls. On open pilot, exposure ties (**22 vs 22**) and actual read coverage improves **three cases** (**16 vs 13**), with fewer tools. Binary complete-READ improvement has a 95% bootstrap interval **touching zero** on both splits; no strong improvement claim yet. The experiment shows a useful **engineering hypothesis**, not sufficient statistical evidence for default promotion.

**Unmeasured / must stay null:** answer EM, strict citation success, original HotpotQA supporting *sentence*-level citations, calibrated semantic entailment, real LLM prompt/completion tokens, real LLM inference latency, and robustness across model seeds. The scripted policy issues **zero model calls**. `cpu_wall_ms` reflects tiny process-local BM25 timings rounded to milliseconds; it is **not a model latency comparison**. This prototype must be run with the **same real Qwen3 stop LoRA and tool schema** on the open pilot before a preregistered eval-A64 comparison; the sealed final-B160 remains untouched.

## Next gate

1. Train-only model rollout diagnostics: attribute *Search Agent* failures as not-retrieved / retrieved-not-read / read-not-cited / wrong-answer / verifier-rejected. Treat lexical answer-in-source separately from semantic entailment.
2. Add independent, original HotpotQA sentence-index support **audit** (already available in E-020) to the matched real model pilot; never show original dev sentence labels to the policy.
3. Freeze variant + NLI/semantic verifier specification from TRAIN-only data and compare real Qwen3 stop adapter on opened pilot32 under fixed tool/model/token budgets. Record verifier rejections and false approvals, cost and paired bootstrap. If a pilot merits promotion, use eval-A64 only once under a predeclared plan; no final-B160 until the larger M7 on-policy/RSI gates are met.
