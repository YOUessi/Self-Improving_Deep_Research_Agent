# M7 E-022 — Actual model-trajectory bottlenecks and official HotpotQA sentence audit

**Executed on Tang:** 2026-10-27 (Asia/Shanghai). Dedicated branch `feat/m7-next-hop-evidence`; *tested source commit* `6185e55051ac25e353623355633c1d0122d7bd6c`. `111 passed` on the existing Tang Python 3 CPU environment; GitHub Actions runs [40408976530](https://github.com/YOUessi/Self-Improving_Deep_Research_Agent/actions/runs/40408976530) and [40408970382](https://github.com/YOUessi/Self-Improving_Deep_Research_Agent/actions/runs/40408970382) **both SUCCESS on the identical SHA**. Tang GPU busy with unrelated online GRPO work during this CPU-only task; **no GPU jobs launched or interrupted**.

**Scope:** analyze *already executed real Qwen3-0.6B SearchAgent multi-turn TRAIN rollouts*, versus the genuine **original TRAIN HotpotQA `(title,sentence_index)` annotations** (NOT synthetic support guesses). Observe previous, frozen, open pilot32 model traces with a **label-free source-only verifier**. No new model inference or fine-tuning is claimed; no independent eval-A64 or sealed final-B160 accessed.

## Input datasets, hashes, split isolation

Two existing M6 groups:
- `real_train_rollouts12x4_stop`: **12 TRAIN questions × 4 completions = 48** separate real sampled multi-turn SearchAgent episodes, from stop-SFT Qwen3-0.6B. TRAIN rollout SHA256 `1f1129d8f94bc7a212590a63b14299e917a4f5e6b13b8ffab4a4079ebd492ffb`.
- `train_rollout_io8x4`: **8 TRAIN questions × 4 completions = 32** separate real model episodes, separate generation settings; SHA256 `5af71a10d57b15e91ca4a400126ce3c667c9784793caa4dd7f13709bacce4aad`.
- **TRAIN only** converted data: `~/experiments/sidra_m4/data/train_oracle1024_seed43/converted/`. Questions SHA256 `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`; documents `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`.
- **Original genuine TRAIN labels**: `~/experiments/sidra_m4/data/train_oracle1024_seed43/original_train_subset.json`, exactly 1,024 IDs matching converted TRAIN; SHA256 `83f03106f12d2c22f36a6957527889fd80f3d13100b03255fc42028ad218f417`. Original title+sentence-index supporting-fact labels are read by an **offline evaluator after the rollouts**, not the model or any strategy.
- Separate **opened DEV pilot32 label-free**: existing frozen `m6_pilot32_stop` and `bridge_stop_pilot32` trajectories. CLI checks pinned converted-documents SHA `5aabc048b344940ffdce2880a7a57a683d4009777a796fd0602ed683a500cc6c` and converted-questions SHA `94db9bdfff053555b9a8e0da9a850b5bee3de73e2406ec85c63461b5ae033e6e` in previously recorded manifests **before reading any trajectory**. It **never opens** the DEV questions, their hidden gold answers, or supporting facts; no per-DEV-example output.
- Immutable run outputs on Tang: `~/experiments/sidra_m7_evidence_run2/results/{train12x4,train8x4,pilot_stop32,pilot_bridge32}/summary.json`; TRAIN-only also has `train_only_per_episode.jsonl`. Reproduction commands below refuse to overwrite existing results.

## Actual TRAIN-only bottleneck results

Each episode is scored post-hoc on **support passage exposed in SEARCH**, **really READ**, **all original HotpotQA supporting sentences visible in READ**, **all sentences in cited READ text**, **correct normalized answer**, and **lexical answer string in any single cited READ**. These are different outcomes; none is the same as verified semantic entailment.

| Real model episodes | 12×4 (N=48) | 8×4 (N=32) |
| --- | ---: | ---: |
| Both required passages **exposed in SEARCH** | **11/48** (22.92%) | **10/32** (31.25%) |
| Both required passages **actually READ** | **3/48** (6.25%) | **2/32** (6.25%) |
| All original labelled support sentences in READ | **3/48** | **2/32** |
| All original labelled support sentences in **cited READ** | **3/48** | **2/32** |
| Answer exact match (EM) | **4/48** (8.33%) | **1/32** (3.13%) |
| Answer EM **and** original labelled sentences cited | **1/48** (2.08%) | **0/32** |
| Mean supporting sentence fraction READ | 42.71% | 42.19% |
| Mean supporting sentence fraction in cited READ | 28.13% | 23.44% |
| Mean mean tool calls | 2.8333 | 2.6875 |
| Mean prompt tokens | 2,945.04 | 2,776.41 |
| Mean generated tokens | 147.79 | 145.78 |
| Mean real previous-run model latency | 3.5344s | 3.4395s |

**Descriptive earliest-missing-gate attribution** (mutually exclusive count; NOT counterfactual causal proof):

| Bottleneck | 12×4 | 8×4 |
| --- | ---: | ---: |
| At least one gold support never SEARCH-exposed | **37** | **22** |
| Both support exposed, but >=1 not actually READ | **8** | **8** |
| Complete sentence coverage, answer wrong/absent | **2** | **2** |
| Complete sentence coverage, correct answer; semantics STILL unchecked | **1** | **0** |

In the **12×4** cohort: post-hoc observation-only source verifier sees **20 abstentions**, **2 illegal/malformed action errors**, **10 lexical-unanchored answers**, **16 lexical candidates**. In the **8×4** cohort: **15 abstentions**, **2 errors**, **5 lexical-unanchored**, **10 lexical candidates**. A lexical candidate says *the normalized answer text occurs somewhere inside one actually READ cited document*, **not** that the cited text entails the answer.

The dominant **observed** bottleneck in these TRAIN traces is *not reaching both required source passages at all*, followed by *failing to READ the exposed second source*. More extensive query strategy plus budget-aware frontier reading is better-motivated than simply tightening a Verifier on these traces, but this remains a research hypothesis: traces may fail on answer reasoning for multiple reasons simultaneously.

### Correct uncertainty unit: TRAIN question-cluster bootstrap

TRAIN episodes from the same question share difficulty, documents and question text. **Resample whole questions** (12 or 8 clusters), each with its four model rollouts; do **not** treat all 48/32 as independent Bernoulli examples. Fixed random seed 2027, 2,000 draws, percentile 95% intervals.

| Metric | 12×4 mean [cluster 95% CI] | 8×4 mean [cluster 95% CI] |
| --- | --- | --- |
| Both support passages exposed | .229167 [.062500, .458333] | .312500 [.062500, .593750] |
| Both support passages READ | .062500 [.000000, .187500] | .062500 [.000000, .187500] |
| All original supporting sentences cited | .062500 [.000000, .187500] | .062500 [.000000, .187500] |
| Answer EM | .083333 [.020833, .166667] | .031250 [.000000, .093750] |
| Answer EM + original sentence coverage in cited text | .020833 [.000000, .062500] | 0 [0,0] |

**Do not compare the two cohorts as a matched causal ablation:** they used different sampling temperatures and partly different question groups. Intervals here are uncertain due to **small 8/12-question cluster counts**, despite more episode rollouts. These are intervals for empirical means conditioned on the original task cohort; no model improvement is claimed.

## Open pilot32: source-only label-blind Verifier audit

Ran strictly on the *pre-existing* 32 open-pilot model traces and their manifest hashes. No DEV answer, original fact, answer-EM or gold-support ID was accessed by this M7 script.

| Observable source-only verdict | Stop LoRA policy, 32 traces | M6 bridge policy, 32 traces |
| --- | ---: | ---: |
| At least two distinct cited documents actually READ, answer text in **one same** cited READ | **28/32** | **28/32** |
| Normalized answer absent from each cited READ | **4/32** | **3/32** |
| Too few cited READ documents | 0/32 | **1/32** |
| Semantic entailment verified | **NOT RUN** | **NOT RUN** |
| Mean billed tool calls | 3.1875 | 4.0000 |
| Mean prompt tokens | 3,372.50 | 2,715.97 |
| Mean completion tokens | 175.28 | 136.47 |
| Mean actual model latency | 4.0374s | 3.1957s |

**Important distinction:** M6's *previously reported* gold-derived answer EM for this stop policy was only 6/32 (bridge: 3/32). The much larger **28/32 lexical matches** are therefore **not reliable entailment or QA correctness**, even when the policy actually cites two READ documents. Counts refer to the same stored run but this new label-blind script does **not** recompute DEV EM. Prompt/token/latency differences are from *different pre-existing policies/serving behavior*, not a calibrated verifier runtime experiment.

Additional adversarial unit tests cover an answer split across a document boundary (must NOT be accepted as a per-document match), hallucinated unread citations, duplicated citation IDs, malformed agent action, original support sentence truncated out of READ text, and a negated claim in which the answer string occurs but cannot be treated as entailment. **This is a source-presence diagnostic only**. There is not yet a trusted calibrated English NLI / relation verifier, and the default M6 `EvidenceAudit` still returns a `lexical_only` reason (not semantic proof).

## Reproduction (CPU-only)

On Tang, checkout the exact source commit above (or compare immutable Github contents with it), ensure `PYTHONPATH=src:.`, then:

```bash
python3 scripts/m7_evidence_diagnostics.py --mode train \
 --dataset ~/experiments/sidra_m4/data/train_oracle1024_seed43/converted \
 --original-train ~/experiments/sidra_m4/data/train_oracle1024_seed43/original_train_subset.json \
 --rollouts ~/experiments/sidra_m6/results/real_train_rollouts12x4_stop/rollouts_train.jsonl \
 --out <NEW_EMPTY_RESULTS_DIR> --bootstrap 2000

python3 scripts/m7_evidence_diagnostics.py --mode pilot \
 --traces ~/experiments/sidra_m6/results/m6_pilot32_stop/traces.jsonl \
 --pilot-manifest ~/experiments/sidra_m6/results/m6_pilot32_stop/manifest.json \
 --out <NEW_EMPTY_RESULTS_DIR> --bootstrap 2000
```

The second TRAIN cohort and opened bridge pilot reuse the same commands with respective archived paths. Other DEV manifests are **rejected before trajectory access**. The TRAIN original subset must match all 1,024 converted TRAIN IDs, original question text, gold answer and valid original sentence indices; duplicate episodes fail closed.

## Research next gates (still TODO)

1. Implement/compare **TRAIN-only** query rewrite variants that improve observed model **both-support retrieval** and **both-support READ** under exactly the 6-step/5-tool fixed budget. Do **not** use pilot labels as training targets.
2. Train a separately calibrated, answer-aware **semantic entailment verifier** on disjoint TRAIN data: require a faithful question→proposition representation (bridge joins often need multiple facts), explicit negative cases, hallucinated citation / negated claim controls, calibrated abstention, and false approval/false rejection on held-out TRAIN question groups. Until then mark `semantic_entailment=not_evaluated`; no implied NLI success from exact quote overlap.
3. Embed the predeclared verifier and policy in **matched real Qwen3 stop-adapter pilot32** with equal inference settings and budgets, including answer EM, original sentence coverage, tool/token cost and paired bootstrap. Avoid repeated pilot tuning.
4. Only after a frozen variant and pilot gate passes, pre-register independent **eval-A64**, run it once, and leave **final-B160 entirely sealed**.

This E-022 is an **evaluator-instrumentation and factual bottleneck finding**, not a new learned-model performance result or a replacement for M7 E-021.
