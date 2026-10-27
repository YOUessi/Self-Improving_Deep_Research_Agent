# M7-C E-025 — Replayed TRAIN-to-Trainer admission quarantine (2026-10-27)

**Research classification: real CPU evidence audit + two injected negative controls; NO new labels and NO optimizer generation.** We intentionally did **not** start a GPU training run with unsupported invented answers. Three earlier Qwen3-8B Challenger **candidate cohorts** are not successive Trainer checkpoint generations.

## Task and evidence lineage

- Independent GitHub branch: `feat/m7-rsi-multi-generation`; exact execution code snapshot **`b3b23a87b02dc247953cc4bd8267cf71ca35b9fd`**, separate from the `feat/m7-onpolicy-agent-grpo` branch. Full Tang **173/173 CPU tests passed**, including 11 new E-025 checks. No GPU process or model serving needed.
- Original HotpotQA **TRAIN** questions SHA256 `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`, documents SHA256 `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`. Only `load_dataset(train_data)` may open corpus records, which must all have `split=train`.
- Existing frozen *public DEV ID-only* selection manifest SHA256 `3f0c494cca8b44cfef90c78d4d5b1fbb4158dc7387ed0f24316a0a265471d947`. The gate reads **only the `pilot.selected_ids` and `heldout.selected_ids` arrays and counts**, checks uniqueness and exactly **256** public DEV IDs disjoint from all original TRAIN IDs; does not ingest DEV answer labels/sentence gold. It rejects any final-B path without opening the file.
- The three independently sourced (TRAIN-only, seeds **43/44/45**) Qwen3-8B Challenger cohorts were already generated in E-022/023/024. From **72 distinct frozen original TRAIN question origins**, E-022 attempted **24 model-generated novel challenges** (8 per seed). Their original typed source / difficulty class mapping is held in a separate `VERIFIER_ONLY_MAP.jsonl`; model-facing `CHALLENGER_INPUT.jsonl` includes only opaque ID and TRAIN question text.
- New E-025 implementation (`src/sidra/rsi_trainer_admission.py`) independently recomputes the full frozen queue/source checksum and **BM25 quote packet replay** for every cohort, then verifies the original model-review SHA, the posthoc-report SHA linkage and the *exact deterministic downgrade decision* for each of the eight quote-valid cases. It then checks **cross-cohort challenge ID/source ID uniqueness, original TRAIN source text and candidate near-duplicates**. The trust boundary does not send reviewer output or TRAIN gold into a Solver/Trainer prompt.

## Executed end-to-end CPU audit

New CLI: `scripts/audit_rsi_training_admission.py` (repeatable `--cohort QUEUE GROUNDED SEMANTIC POSTHOC`), explicitly requiring a `--public-dev-id-manifest`; immutable no-overwrite report on Tang at:
`~/experiments/sidra_rsi_m7/trainer_admission/E025_devIDbound_b3b23a87/ADMISSION_AUDIT.json`.
Only aggregate counts, file digests, and task status are published; no private model traces or generated training data were pushed to GitHub.

| Cohort | Proposed new questions | Literal-quote valid | Unsupported model review | Invalid review | Additional verified negative pattern | Semantically pending new label |
|---|---:|---:|---:|---:|---:|---:|
| g1, seed 43 | 8 | 4 | 3 | 1 | 0 | 0 |
| g2, seed 44 | 8 | 3 | 1 | 0 | 2 | 0 |
| g3, seed 45 | 8 | 1 | 0 | 0 | 0 | 1 |
| **Total** | **24** | **8** | **4** | **1** | **2** | **1** |

Other **16/24** proposed novel questions did not even pass literal quote provenance. The one remaining semantically plausible *Desiderata* question remains **pending independent authority**, rather than automatically inheriting HotpotQA TRAIN source answer labels.

**Exact Trainer admission: 0 novel questions**. The write-once `TRAINER_INPUT.jsonl` file is actually zero bytes, SHA256 **`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`**. This is deliberate fail-closed behavior; it is not an assertion that self-improving training has happened. There were **0 checkpoint updates, 0 multi-seed Trainer replicates, 0 verified RSI generations** in this stage. Original TRAIN questions with their existing gold may be used by a **separate** verified original-TRAIN training path; novel model-generated questions are explicitly barred from inheriting those labels.

## Real mutation tests, not only unit tests

On a disposable **copy** of the original artifacts, we ran two end-to-end negative controls:

1. Forged one saved semantic-review record to assert `approved_train_labels=1`, leaving original source evidence untouched. E-025 **raised ValueError and produced no Trainer output directory**. The original E-024 files were neither changed nor overwritten.
2. Put one real original TRAIN ID into a disposable *copy* of the frozen public DEV pilot-ID list. E-025 **raised ValueError before processing the cohort artifacts and produced no output directory**. No DEV answer/label file was read.

Tamper outcomes are saved without task IDs at `~/experiments/sidra_rsi_m7/trainer_admission/E025_MUTATION_GATES.json`; **2/2 blocked**. These are localized integrity tests, not proofs of comprehensive adversarial security.

## What this proves and what it does NOT

**Proven:** existing E-022 → E-024 results can be independently reconstructed from original TRAIN corpus and protected manifest and cannot be promoted by changing a saved reviewer/posthoc field; a frozen DEV ID collision is rejected. The multi-cohort provenance and duplicate check passes on this real batch. No secret, model weights or final-B160 was accessed or committed.

**Not proven:** machine-generated answers are semantically true; same-model review can be considered independent factual annotation; actual model weights improved; two chained model-generations were trained; evidence from multiple independent optimizer seeds improved public DEV/eval-A QA. **NONE of those claims is made**.

The remaining bottleneck is *annotation authority*. Options for the next genuinely training-relevant stage:

1. Implement **narrow, testable two-hop relation proofs derived from original TRAIN source sentences** (allow zero automatic positive labels outside an explicitly proven template), and separately annotated negative controls. The candidate Desiderata case needs actual two-document relational necessity, not just two quotes.
2. Use original HotpotQA TRAIN-labelled tasks as the safe first Solver–Trainer curriculum, with adaptive difficulty based only on previous TRAIN-only legal tool-trace/reward audits. Keep unverified Challenger novelty quarantined; do not share oracle gold with the online policy, its candidate catalog, or its generation context.
3. Only after independently verified new tasks and actual **successive** distinct checkpoint/replay/evaluation records, run >=3 genuinely different training seeds, fixed disjoint TRAIN holdout, predeclared public DEV screen and independent eval-A. Keep final-B160 sealed.

## Reproducibility

Source:
- `src/sidra/rsi_trainer_admission.py`
- `scripts/audit_rsi_training_admission.py`
- `tests/test_rsi_trainer_admission.py`

CPU tests: `~/experiments/sidra_rsi_m7/logs/E025_tests_b3b23a87.log`.
Real audit: `~/experiments/sidra_rsi_m7/logs/E025_devIDbound_b3b23a87.log`.
The three original E-022/023/024 cohort inputs remain immutable in `~/experiments/sidra_rsi_m7/{queues,challenger,semantic_triage,semantic_triage_posthoc}/`.

**Publication / merge policy:** keep Draft PR #11. This stage demonstrates a safer data-lineage gate and a repeatable negative test, not a multi-generation RSI improvement result.
