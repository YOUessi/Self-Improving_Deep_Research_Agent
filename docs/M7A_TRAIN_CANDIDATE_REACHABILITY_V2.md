# M7-A: why categorical on-policy Search/Read/Finish had an answer-ceiling bottleneck

**2026-10-27 | CPU-only reproducible diagnostic, TRAIN labels used ONLY OFFLINE.**

## Problem

Earlier real on-policy finite-action GRPO pilots (seed53 and seed54) updated LoRA weights but observed **zero strict TRAIN successful answers**. Existing public DEV pilot32 paired evaluations also did **not** establish improved QA. The categorical action decoder fixes the mismatch between actual sampling and log-probability normalization, but **only** from a fixed, non-oracle finite catalog. If the gold answer is not present in any catalog answer choice, even a perfect RL optimizer cannot select it.

This report measures a *lexical proposer ceiling under artificially provided oracle supporting documents*, not any model rollout or real retrieval. The code driving the live policy has **no access to source gold, supporting IDs, DEV, or final-B160**; the isolated diagnostic is allowed to see TRAIN labels to calculate an optimistic ceiling.

## Implementation

A backward-compatible `FiniteActionConfig(answer_proposer="legacy"|"v2")` adds opt-in V2 candidates from model-visible question + read text only. No API, retriever ground truth, oracle, DEV labels or source-document preload enters the real action proposer.

V2 recognises Unicode names (`Déjà Vu`), compound person names (`Klemens von Metternich` and `Jim Bob and Michelle`), full calendar dates (`July 23, 1968`), qualified quantities (`at least 25`), explicit periodicity (`monthly`), headquarters composites and question-listed comparative choices. It ranks candidates from source sentences using question-surface overlap and coarse question type. The old candidate generator stays `legacy` by default to preserve historical comparability. Decoder and proposer versions are written to each evaluation companion and checked by a fail-closed paired evaluator.

The on-policy LM samples exactly **one action-ID token** from the masked allowed IDs, using `softmax(logits[legal_ids])`; the optimizer uses the *same subset and denominator* at every sampled state, and now audits **every action** for sampling-vs-replay probability mismatch before any optimizer step. No retrieved text or system/user tokens can become policy-gradient targets; only the selected action-ID token is the target. The legal catalogs still respect original SearchAgent read-before-cite rules, step and tool-call budgets.

## Reproducible TRAIN-only optimistic support experiment

Source GitHub snapshot: `0f6d41940a0b45175c8bb3a19f8c2629dcd29bdc`.

- `documents.jsonl` SHA256: `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`
- `questions.jsonl` SHA256: `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`
- 1,024 **TRAIN** questions, separate from frozen public DEV and sealed final-B160. All 1,024 TRAIN questions were processed. CPU test suite: **148 passed**.
- To compute the optimistic ceiling only, give a simulated completed source state access to each question's **known TRAIN supporting document IDs** and answer; do NOT feed those IDs or answers to the actual model. Measure whether any extracted answer choice exactly matches TRAIN gold under the scorer's answer normalization.

| Quantity (TRAIN, not policy success) | Legacy | V2 |
| --- | ---: | ---: |
| Gold string appears in at least one oracle-provided support text | 993 / 1024 (97.0%) | same |
| Exact gold answer available among up to 3 lexical choices per support | 290 / 1024 (**28.3%**) | 443 / 1024 (**43.3%**) |
| Gold answer in full legal catalog when 1 decision step remains | 290 / 1024 (28.3%) | **443 / 1024 (43.3%)** |
| Gold answer in full legal catalog when 2 decision steps remain | 243 / 1024 (23.7%) | **403 / 1024 (39.4%)** |

Paired TRAIN-only V2−legacy answer-availability delta: **+153/1024 = +14.94 percentage points**; question-level seeded bootstrap 95% descriptive interval **[+11.91pp, +18.07pp]**. V2 gains eligibility on 218 TRAIN questions but **regresses on 65**; this is not a monotonic improvement for every question. Exactly 3 text candidates per read paragraph and at most 9 total legal actions remain in force.

Real (not oracle) first-hop BM25 search for the *unmodified question* among 3 retrieved documents returns at least one TRAIN support in **951/1024 (92.9%)**, but **all** required support documents in only **389/1024 (38.0%)**; thus multi-hop retrieval remains essential. These are retrieval **reachability** diagnostics, not read/cite/answer success.

Breakdown of optimistic answer-choice availability:

| Question category | Legacy | V2 | Dataset count |
| --- | ---: | ---: | ---: |
| when | 62.2% | 81.1% | 90 |
| person | 21.2% | 43.2% | 222 |
| where | 17.1% | 41.5% | 41 |
| quantity | 0% | 30.4% | 23 |
| which | 33.2% | 43.3% | 307 |
| other | 22.9% | 34.1% | 340 |
| frequency | 0% | 100% | **1** (uninterpretable as reliable rate) |

## Explicit limits / strict future gates

1. All figures are measured on **TRAIN with oracle support supplied only to the offline diagnostic**. The model still must *actually search and read those paragraphs* before citing them, select the right answer, and produce a complete outcome; therefore **no QA model improvement is claimed here**.
2. V2 has a narrower/fixed answer-candidate class than open-ended decoding. It solves the categorical probability mismatch, **not** full natural-language generation or semantic answer checking. A perfect selector's reachable gold ceiling in these idealized states is 43.3% and may be lower with real imperfect retrieval and limited steps.
3. The change might overfit TRAIN linguistic patterns. Future runs must compare starting LoRA vs updated LoRA using **the same V2 proposer, frozen public DEV pilot32, fixed decoder selection mode and seed**, track answer EM and sentence-level verified citations, and eventually independent evaluation; otherwise **do not claim GRPO gain or RSI**.
4. Dataset text is not uploaded to GitHub. Persisted aggregate raw source on Tang: `~/experiments/sidra_m7/results/train1024_candidate_reachability_0f6d41940a0b45175c8bb3a19f8c2629dcd29bdc.json`.

Reproduce on Tang CPU, no GPU required:

```bash
PYTHONPATH=src python scripts/audit_train_candidate_reachability.py \
  --train-data /home/you/experiments/sidra_m4/data/train_oracle1024_seed43/converted \
  --out /home/you/experiments/sidra_m7/results/UNIQUE_TRAIN_REACHABILITY_REPORT.json
```

Companion strict paired-run evaluator `scripts/audit_paired_onpolicy.py` rejects mixed dataset/source/decoder/proposer/seed/budget and unchanged checkpoints, calculates paired bootstrap deltas and exact sign p-values, never opens final-B160, and refuses promotion unless evidence *and* answer success improve.
