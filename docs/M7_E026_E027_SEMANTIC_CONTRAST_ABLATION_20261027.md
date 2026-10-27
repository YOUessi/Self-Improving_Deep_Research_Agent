# M7 E-026 / E-027 — controlled TRAIN semantic NLI and paired source-necessity falsification

**Executed 2026-10-27 (Tang CPU-only), no model training.** Status: **DIAGNOSTIC ONLY / NO VERIFIER PROMOTION**. E-026 NLI run pinned to GitHub source `86f61c5cffd3a76893a522b37490f456678caffc`. E-027 paired diagnostics and re-run of full CPU test suite pinned to GitHub source `650d595910db41e1af83d89033fdf8e1bf0b2895`; **154 tests passed**. On-device source checkouts were downloaded from immutable GitHub archives, with source SHA recorded in each snapshot. GitHub CI on these changes must be checked independently (local tests are not a CI substitute). No GPU process was spawned or interrupted.

## Why this follows E-025

E-025 found that an MNLI leave-one-citation-out test can pass an **incorrect** real Qwen3 answer: citation model-necessity does not establish answer truth. E-026/E-027 test this distinction on a smaller, controllable sample where the exact original HotpotQA **TRAIN** supporting sentences and curated counterexamples are fixed and inspectable.

The new controlled data consist of 36 researcher-curated claim/premise contrasts over **12 unique original HotpotQA TRAIN questions**, three per question. Each source sentence is keyed by the original `(document title, sentence index)` pair, resolved only from TRAIN originals. For each question there is one editorial entailment, one explicit negation, and one editorial neutral example. Ten neutral examples remove one supporting sentence from an otherwise identical hypothesis; the other two are challenging **unsupported bridge** near-misses. **Labels are editorial and have NOT received independent human adjudication.** These are not prospective SearchAgent actions and not frozen DEV QA labels.

## Reproducible source/model

- TRAIN converted questions SHA256: `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`.
- TRAIN converted documents SHA256: `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`.
- Original TRAIN full subset SHA256: `83f03106f12d2c22f36a6957527889fd80f3d13100b03255fc42028ad218f417`.
- Curated controlled cases SHA256: `0034bdaa1644e14c5c8df53502158f6efedfa7fb9860a99c1c721bdcc460a803`.
- Cached CPU mDeBERTa-v3 MNLI: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, pinned weight SHA256 `65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e`, pair token ceiling 384, offline mode.
- Genuine forward passes E-026: **36** (no truncation), approximately **3.423 s** CPU inference total for 36 controlled cases. E-027: **0 new NLI calls**; entirely offline paired analysis.
- E-026 model + source outputs: `~/experiments/sidra_m7_evidence_run5/results/e026_original_train36_mnli/`; E-027 immutable v2 output: `~/experiments/sidra_m7_evidence_run5/results/e027_train36_ablation_v2/`.
- E-026 `summary.json` SHA256 `2f4e5850aacf2e75c90a5370d43f0f2dcf9bc627d3ccee20a5490e2dd05be859`; E-027 v2 `summary.json` SHA256 `ae37ff5820b11e43b69fd4aaab7063ed74927c32fc7fccfd3d5f89c807211699`. Neither overwrites a previous run.

## E-026: three-way predictions versus editorial labels (N=36)

| Editorial label | Predicted entailment | Predicted contradiction | Predicted neutral | Total |
| --- | ---: | ---: | ---: | ---: |
| Entailment | 12 | 0 | 0 | 12 |
| Contradiction | 0 | 12 | 0 | 12 |
| Neutral | **2** | **2** | 8 | 12 |

Three-way editorial-label agreement **32/36 = 88.89%**, question-cluster bootstrap 95% percentile CI **[80.56%, 97.22%]** (5,000 draws, fixed seed2027). At the **fixed prior** `p_entail >= 0.85` threshold, correct-editorial-entailment acceptance is **11/12**, CI **[0.75,1.0]**, while **2 of 24 non-entail editorial cases are confidently accepted**, rate 8.33%, CI **[0, 0.208333]**. All uncertainty resamples the **12 original TRAIN question clusters**, never 36 examples as independent units. Tiny confidence intervals such as 0/12 with bootstrap CI [0,0] are descriptive and **do not** establish zero population risk.

The two **high-confidence, unsupported** editorial near-misses are:
1. `duggar-unsupported_bridge`: a source says Jill Dillard is a television personality; another says the program features the Duggar family. The fixed sampled sentences do not explicitly establish that Jill herself appears on that specific show. MNLI nonetheless assigns **p_entail = 0.994826**. A different human adjudicator could assess implicit context differently; this remains an editorial near-miss, not externally verified ground truth.
2. `facup-unsupported_bridge`: a source says Arsenal won the **2017 FA Community Shield**, while another merely describes the **FA Cup**. The unsupported claim says Arsenal won the **2017 FA Cup**. MNLI assigns **p_entail = 0.940611**. This is a clear cross-competition relation/entity conflation risk, not evidence that the stated FA Cup result follows from the given premises.

Two other editorial neutral omissions (`leiter`, `mall`) are predicted as **contradiction**, not entailment; this is a failure to distinguish insufficient evidence from false propositions but does not cause a high-confidence approval.

## E-027: exact-hypothesis paired source removal (10 TRAIN pairs)

For the ten `missing_bridge_source` cases, E-027 formally checks that the *hypothesis text is unchanged* and the ablated evidence corresponds to a **strict subset of the same exact original supporting sentence keys**. No arbitrary paraphrase or unobserved corpus content may be inserted into the ablation.

The E-025 prior thresholds are reused without fitting them to E-026: complete model `p_entail >= 0.85`, ablated evidence `p_entail <= 0.65`, and reduction `>= 0.20`. **9/10** source-removal pairs satisfy all three conditions (question-cluster bootstrap CI **[0.666667, 1.0]**). The one failure occurs because the *full-premise* NLI `p_entail` is only ~0.716 (the Astor Place Riot contrast), **not because the ablated evidence still suffices**. All ten ablated evidences receive low entailment scores and **0/10** are confidently accepted.

Mean paired model entailment probability reduction: **0.945874**, question-cluster bootstrap 95% CI **[0.885858, 0.981572]**. The ten pairs come from ten of the twelve original question clusters; do not bootstrap at the sentence/action level.

**Critical joint interpretation:** MNLI is highly sensitive to missing explicit original supporting sentences in clean, hand-constructed bridge contrasts. Yet it **confidently approves both** editorial unsupported near-miss relations (2/2), for which the labelled "neutral" hypothesis differs from the positive and no valid identical-hypothesis removal pair exists. Thus **large leave-one-source score drops are not a certificate of factual correctness or sound bridge linkage**. This aligns with the earlier E-025 real-model counterexample rather than overturning it.

Ablating a source sentence does **not** change the real-world truth of a proposition; it changes what is justified by the *remaining cited evidence*. The E-027 record explicitly preserves this separation. Every answer-EM / strict-support / tool-call / model-token / real agent-latency field is **unmeasured** in this controlled study.

## Engineering integrity and negative results

- Full local CPU test suite at E-026 source: **149 passed**.
- The initial E-027 test run exposed a genuine missing-case bug: a fixture containing **no legitimate support-removal pairs** caused a zero-denominator exception. Changed this to `null` metric + `reason=no_eligible_paired_cases` rather than misreporting 0% or crashing. Also corrected an ambiguous field that could be read as saying source removal changes the proposition's truth. Final E-027 source: **154 passed**.
- Input provenance gates require the pinned TRAIN question/original/curated-case SHA256 values, 12 question clusters, 36 rows, legal original supporting-sentence ID subsets, valid label maps and valid normalized NLI posteriors. Refuses overwritten result directory.
- No DEV answer/labels or DEV-derived threshold tuning, no new Pilot32 run, no eval-A64, no sealed final-B160, no SearchAgent tool budget change, no model weight update. Existing E-025/E-026 comparison is **offline and retrospective**.

## Reproduce E-026 and E-027 without GPU

From E-026 GitHub snapshot:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=src:. \
  ~/projects/CodeRL-Lab/.venv/bin/python scripts/m7_semantic_contrast_suite.py \
  --dataset ~/experiments/sidra_m4/data/train_oracle1024_seed43/converted \
  --original-train ~/experiments/sidra_m4/data/train_oracle1024_seed43/original_train_subset.json \
  --cases research/m7_semantic_contrast_v1.json \
  --model ~/.cache/huggingface/hub/models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli/snapshots/8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c \
  --out <NEW_E026_RESULT_DIRECTORY> --bootstrap 5000
```

From E-027 GitHub snapshot:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. ~/projects/CodeRL-Lab/.venv/bin/python \
  scripts/m7_contrastive_read_necessity_audit.py \
  --e026-rows ~/experiments/sidra_m7_evidence_run5/results/e026_original_train36_mnli/controlled_rows.jsonl \
  --e026-summary ~/experiments/sidra_m7_evidence_run5/results/e026_original_train36_mnli/summary.json \
  --out <NEW_E027_RESULT_DIRECTORY> --bootstrap 5000
```

## Promotion gate after E-027

**DO NOT treat 88.9% editorial-label agreement as real-Verifier accuracy. DO NOT apply a lexical 'FA Cup vs Shield' hardcoded fix.** The narrow suite of 12 TRAIN questions is a helpful failure reproducer, not an independent deployment benchmark. The next scientifically defensible step is to freeze an independently adjudicated, disjoint TRAIN-question semantic evaluation cohort (at least two blind reviewers for disputed neutral cases), with entity/type confusions, wrongly linked bridging relations and negation scope, then calibrate a non-degenerate *false-approval versus false-rejection* tradeoff using TRAIN-only data. Only then run a *new matched* real Qwen3 model pilot32 under the existing 6-step/5-tool budget and pre-specified metrics. Access independent eval-A64 only once after the policy is fixed and promotion justified; final-B160 stays sealed.
