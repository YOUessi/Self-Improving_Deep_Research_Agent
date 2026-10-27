# M7-C E-026 — Original-TRAIN curriculum / independent tamper audit (2026-10-27)

**State: COMPLETED, CPU evidence gate, not RSI training.** E-025 correctly quarantined all novel Qwen3-8B Challenger tasks (0 accepted new answer labels); to avoid training on false invented labels, this stage creates the first separately auditable **original HotpotQA TRAIN-labelled** Solver task bank and freezes an independent original-TRAIN holdout. No model-generated new answer label was admitted, no model checkpoint changed in this stage, and no GPU was used.

## Read-only preflight and context

- Exclusive branch: `feat/m7-rsi-multi-generation`, Draft PR #11. Previous head `226f1a79b4147dcc04ea03de537f8c7778724d66` had green GitHub CI (PR and push runs). Online GRPO branch was only read, not written.
- Tang online. Preflight GPU: single RTX 4090 Laptop GPU, **349 MiB used, 15,571 MiB free**, no SIDRA model-serving or training process detected. This E-026 workload is **CPU-only**, so no GPU reservation, preemption or kernel was needed.
- M6 evidence read: E-017 had **48 + 32** real, model-produced TRAIN tool episodes across two separate recorded runs, not optimizer generations. Earlier GRPO replay suffered a constrained-decoding/logprob mismatch and a null pilot gain. M6 DEV pilot and eval-A were **not** read or used as training labels here.
- E-025 previously replayed 24 novel Challenger proposals, found zero admissible machine-generated labels. These remain quarantined; there is no inherited answer from the original source QA.

## Actual E-026 source replay and curriculum result

The real CPU preflight of M6 history used the frozen original TRAIN conversion, with complete BM25 hit/score replay, actual READ text, SearchAgent state/budget, and independently recomputed TRAIN-only reward (using the canonical TRAIN labels *offline only*). We explicitly reject the full history as training samples if **any** trace or reward checksum disagrees; replay does not cryptographically authenticate which LLM produced a trace.

| Actual replay source | Episodes | Legal BM25/tool traces | Matching reported TRAIN reward | Strict original-TRAIN success |
|---|---:|---:|---:|---:|
| M6 real_train_rollouts12x4_stop | 48 | 48/48 | 48/48 | 1 |
| M6 train_rollout_io8x4 | 32 | 32/32 | 32/32 | 0 |
| **Total** | **80** | **80/80** | **80/80** | **1/80** |

The two M6 runs cover **12 distinct original TRAIN IDs** in total; these **12 were excluded** from the new curriculum. They are not treated as 80 independent problems. Their model-origin attestation remains **unverified** and thus they do not become a new Trainer-positive sample bank.

Canonical original HotpotQA **TRAIN** conversion contains **1,024 questions** and **10,229 passage documents**, with stable data checksums. The only existing frozen public DEV artifact read was its **ID-only** selection manifest: 32 pilot + 224 heldout = **256 excluded DEV IDs** (this does not entail reading DEV/final-B160 gold answers, task passages or labels). A new disjoint, private, verifier-only **32-question original TRAIN holdout** was sealed before task selection.

| Planned curriculum cohort | Selection seed (NOT optimizer seed) | Direct | Bridge | Hard | Original TRAIN tasks |
|---|---:|---:|---:|---:|---:|
| 1 | 43 | 8 | 8 | 8 | 24 |
| 2 | 44 | 8 | 8 | 8 | 24 |
| 3 | 45 | 8 | 8 | 8 | 24 |
| **Total** | — | **24** | **24** | **24** | **72** |

The tier is a **TRAIN-only retrieval difficulty proxy**: all supporting passages initially retrieved (direct), some (bridge), none (hard). This is NOT proof of logical task complexity. All 72 curriculum tasks have different source TRAIN IDs, no above-threshold near-duplicates between cohorts or the private TRAIN holdout, and no overlap with the 12 previously exposed original TRAIN questions or 256 frozen DEV IDs.

Every cohort has **two separate immutable artifacts**:
- `SOLVER_INPUT.jsonl`: opaque challenge ID + verbatim original TRAIN question; **never** emits gold answer, original ID, supporting IDs, tier or reward.
- `VERIFIER_ONLY_ORIGINAL_TRAIN_MAP.jsonl`: opaque ID → canonical original TRAIN ID, retrieval tier, normalized question digest. Canonical TRAIN answers/support remain in the **offline verifier corpus**; no original-label data is invented or inherited by new Qwen3-8B questions.

The independent audit of the *saved files* (not creator self-attestation) returned `PASS_SAFE_ORIGINAL_TRAIN_PROMPT_BANK_NOT_TRAINING`. Real immutable bank manifest SHA256: `2e18f86a6208a45522395ca5e31f9600858dee449157a11d53f2e9b0441abb83`.

## Independent attack-mutation controls (actual saved real corpus, not mocked unit results)

Three disposable copies of the **real** 72-task bank were attacked without modifying the immutable original, and the validator was run against the source TRAIN and M6 replay:
1. Modified a learner question and updated its saved SHA in the manifest to launder the rewrite → **REJECT**: canonical original TRAIN identity/text mismatch.
2. Swapped a private Verifier source TRAIN ID and updated its saved SHA → **REJECT**: source/opaque ID/tier mismatch.
3. Injected an original TRAIN task ID into a disposable copy of the DEV-ID-only selection list → **REJECT**: frozen DEV exclusion mismatch/leakage.

**3/3 injections rejected**, independent unmodified-campaign positive control passed. Every mutation test used a temporary directory; only aggregate rejection reasons were retained in `E026_MUTATION_GATES.json` and no modified campaign was promoted.

## GitHub implementation, tests and resolved engineering problems

New files:
- `src/sidra/rsi_safe_curriculum.py` — history replay, canonical original TRAIN task selection, difficulty tiers, global cross-cohort/holdout deduplication, disjoint frozen holdout and separate model/verifier prompt serialization.
- `src/sidra/rsi_train_history_audit.py` — installed package-level independent TRAIN rollout replay; reusable without a `scripts.*` import.
- `src/sidra/rsi_public_split_ids.py` — dependency-free DEV **ID-only** exclusion; sealed final-B paths blocked before file opening.
- `src/sidra/rsi_safe_curriculum_audit.py` — independent verification of all saved hashes, source text, role maps, tiers, historical exposure, near duplicates and heldout/DEV boundaries.
- `scripts/build_rsi_safe_curriculum.py`, `scripts/audit_rsi_safe_curriculum.py` — repeatable read-only CPU audit and create-only bank generation CLIs.
- `tests/test_rsi_safe_curriculum.py` — real executor-backed tiny corpus fixtures, tamper tests, source-map SHA laundering, leakage checks and **cross-CWD CLI import** tests.

**Engineering failure and repair:** the initial CLI run on Tang failed before touching data because `rsi_trainer_admission.py` pulls `scripts.audit_rsi_semantic_posthoc` and `scripts.*` is not importable with only `PYTHONPATH=src` from a directly launched script. In contrast, the pytest working directory disguised the defect. Moved public DEV-ID loading to a dependency-free `sidra.rsi_public_split_ids` helper and the history replay to a packaged `sidra.rsi_train_history_audit` helper; added subprocess regressions with CWD outside the repository.

**Latest executed CPU suite at code snapshot `f7aa1fc5881a0667af3ae1951a2da9c8392e0e61`: 184/184 tests passed.** Independent real audit and all three mutated-copy attacks also passed. GitHub PR CI must be rechecked after publication; only actual check status should be called green.

## Reproduce without reading any sealed final-B file

From GitHub snapshot `f7aa1fc5881a0667af3ae1951a2da9c8392e0e61` under Tang `PYTHONPATH=src`:

```bash
python scripts/build_rsi_safe_curriculum.py \
  --train-data ~/experiments/sidra_m4/data/train_oracle1024_seed43/converted \
  --public-dev-id-manifest ~/experiments/sidra_m4/data/frozen_seed42_32p_224h/FROZEN_MANIFEST.json \
  --history ~/experiments/sidra_m6/results/real_train_rollouts12x4_stop/rollouts_train.jsonl \
  --history ~/experiments/sidra_m6/results/train_rollout_io8x4/rollouts_train.jsonl \
  --source-sha a1c649c5793aeb371f2ce76a756d36aacc9f7c12 \
  --seed 43 --cohorts 3 --per-cohort 24 --train-holdout 32 \
  --out ~/experiments/sidra_rsi_m7/safe_original_train_curriculum/NEW_CREATE_ONLY
```

Then `scripts/audit_rsi_safe_curriculum.py` with the same TRAIN directory, DEV ID-only manifest, `--history` inputs, `--source-sha`, `--campaign` output, and an unused `--out` audit path. Neither CLI performs model inference or training.

Real immutable Tang artifacts:
- `~/experiments/sidra_rsi_m7/safe_original_train_curriculum/E026_3cohorts_24each_32holdout_a1c649c5/`
- `~/experiments/sidra_rsi_m7/safe_original_train_curriculum/E026_independent_audit_f7aa1fc5.json`
- `~/experiments/sidra_rsi_m7/safe_original_train_curriculum/E026_MUTATION_GATES.json`
- CPU test log `~/experiments/sidra_rsi_m7/logs/E026_validator_CPU_f7aa1fc5.log`

## Decision and hard limits

**PASS original-TRAIN task bank / audit; HOLD Trainer promotion / novel-label admission.** The bank removes the previous immediate false-label blocker for established original TRAIN tasks, but only provides prospective tasks, not new verified model-generated tasks.

Specifically **0 admitted novel Challenger labels, 0 Solver rollouts on these NEW cohorts, 0 optimizer updates, 0 RSI checkpoint generations, 0 optimizer multi-seed experiments, and 0 post-update independent evaluation-A gains** in E-026. The three cohorts/seeds are **selection partitions**, not generational RSI experiments. The single 1/80 strict result in historical M6 episodes is *not* evidence of improved Agent reasoning.

Next decisive work is to run real model actions on this goldblind original-TRAIN bank, audit model/checkpoint provenance and TRAIN-only reward, and perform genuine consecutive online model updates with separate TRAIN holdout/DEV independent screens and at least three trained seeds. A true model-generated Challenger task still requires independently authenticated semantic answer labels; the previously invented Qwen3-8B tasks remain quarantined. Never tune on sealed final-B160; do not merge Draft PR #11 as an RSI success claim.
