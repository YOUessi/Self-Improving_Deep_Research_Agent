# M7-C E-024 — Monotone CPU semantic rejection gates (2026-10-27)

**Real experiment completed.** This follows E-023's 16 actual Qwen3-8B-AWQ semantic reviewer calls on 8 independently quote-replayed, TRAIN-only Challenger proposals. E-023 exposed **two false positives out of three seemingly consistently supported model-review decisions**. Rather than repeat the same flawed LLM, E-024 adds a **negative-only, independently replayed CPU check** that cannot generate new labels or send hidden answers to the model.

## Code, frozen inputs and audit method

- Source snapshot **`197eec0a1496b460850c380bf53f98a2d5656098`** on independent branch `feat/m7-rsi-multi-generation` (draft PR #11). **161 CPU tests passed on Tang**, including new tampered-manifest and negative-only monotonicity checks. No GPU required for E-024.
- `src/sidra/rsi_semantic_negative_gate.py`: restricted checks for (a) matching a short bridge token *inside a larger proper name* (e.g. company **Batdorf & Bronson** versus separate city **Bronson**); (b) company `City, State based` location being absent from the evidence that supplies a requested population; (c) asking who won a championship while neither supplied quote states any championship/title outcome. These are **specific, conservative rejection rules**, not general entity linking or logical entailment.
- `src/sidra/rsi_semantic_triage.py`: applies those negatives only when both source-order Qwen judgments claimed "supported" with the same bridge spelling; positive approvals remain **zero**. It cannot promote any previously negative result.
- `scripts/audit_rsi_semantic_posthoc.py`: **CPU replay on the already-saved 16 model-review results**, without another model call. For each cohort it checks the original SHA-pinned TRAIN corpus/queue, recomputes BM25 source quotes, verifies protected challenge IDs and original evidence packet SHA, matches each saved reviewer output packet SHA, then applies only downgrades. Changed outputs are created into a new directory; original E-023 reports remain immutable.
- `tests/test_rsi_semantic_triage.py` and `tests/test_rsi_semantic_posthoc.py` cover falsified quotes, duplicate documents, attempted optimizer authorization, composite proper names, incorrect city population linkage and evidence-free championship claims.

## Exact reproduced outcomes

| Class | E-023 same-model two-pass reviewer | E-024 after independent CPU rejection gate |
|---|---:|---:|
| Flagged unsupported | 4 | 4 |
| Invalid reviewer proof | 1 | 1 |
| Additional concrete premise errors caught by CPU | 0 | **2** |
| Still pending independent semantic adjudication | **3** | **1** |
| Independently approved novel TRAIN labels | **0** | **0** |
| Actual model optimizer updates or RSI generations | **0** | **0** |

Per-cohort replay: cohort1 **4 records / 0 downgrades**; cohort2 **3 records / 2 downgrades**; cohort3 **1 record / 0 downgrades**. Cohort2's two unsafe positive opinions had exactly the empirically documented errors: **Olympia vs Bronson location/brand collision** and the **Manu Ginóbili championship-win relation never present in the supplied quotes**. Both were explicitly downgraded to `flagged_deterministic_premise_gap`. The poem **Desiderata** remains **unlabelled, pending independent adjudication**, not a model-approved gold fact.

**Observed result:** These narrow negative checks eliminated both *known* false-positive machine approvals in the present 8-candidate audit, without increasing the number of training labels. This is not a statistically validated universal false-positive reduction: the sample is small, the failure modes were used to design the checks, and an unrelated adversarial failure might still pass. No generalized reasoning or multi-hop answer gain is claimed.

## Tang artifacts

All data are kept on Tang, **not committed to GitHub**:

- Original E-023 two-pass model reports: `~/experiments/sidra_rsi_m7/semantic_triage/g{1,2,3}_grounded8_goldblind_qwen3_8b_seed{43,44,45}/`.
- New immutable E-024 replay reports: `~/experiments/sidra_rsi_m7/semantic_triage_posthoc/g{1,2,3}_seed{43,44,45}_negative_cpu/POSTHOC_NEGATIVE_REVIEW{.jsonl,_SUMMARY.json}` (actual filenames are `POSTHOC_NEGATIVE_REVIEW.jsonl` and `POSTHOC_NEGATIVE_REVIEW_SUMMARY.json`).
- Python tests and run log: source snapshot `~/experiments/sidra_rsi_m7/snapshots/197eec0a1496b460850c380bf53f98a2d5656098/` and `~/experiments/sidra_rsi_m7/logs/g{1,2,3}_semantic_posthoc_negative_20261027.log`.

**Next rigorous gate:** an independently labelled **held-out novel two-hop TRAIN challenge benchmark** with true bridge identity/relation annotations and abstention labels; compare E-022 quote-only, E-023 same-model two-pass and E-024 negative gate by supported-task precision/recall with uncertainty, then integrate only independently certified new tasks into on-policy multi-generation Solver–Trainer. Three actual optimizer seeds and independent eval-A must demonstrate durable QA/citation gains before RSI promotion. Keep final-B160 sealed throughout.

**No trained model was promoted and no existing GPU job was interrupted.**
