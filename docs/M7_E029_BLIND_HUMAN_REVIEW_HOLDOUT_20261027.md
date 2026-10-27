> **SUPERSEDED / INDEPENDENCE INVALIDATED by E-030 (2026-10-27):** The frozen E-029 v2 packet remains reproducible, but an all-workstream history audit found **34 of its 64 questions** exposed in earlier actual M7 optimizer/teacher or RSI Solver/reserved original TRAIN artifacts that this E-029 run had not enumerated. A further **10 questions** overlap only the already rejected E-029 v1 packet. **Do not call E-029 v2 an independent holdout or use it to calibrate Verifier.** A fully new E-030 prospectively excluded TRAIN64 packet is documented at [E-030 cross-workstream audit](M7_E030_CROSS_WORKSTREAM_EXPOSURE_AND_REFREEZE_20261027.md). No human labels have been collected for either. This banner corrects the prior *limited-to-four-source* disjointness claim without altering the original experiment record.\n\n# M7 E-029 — Prospective blinded human evidence-adjudication cohort (TRAIN-only, **UNLABELLED**)

**2026-10-27, Tang CPU-only.** Code, tests, frozen 64 cards, two anonymized reviewer packets and a private provenance registry were generated. **NO independent human review has yet occurred; 0/64 reviewed.** This is a *prerequisite* for reliable Verifier evaluation, not semantic-ground-truth quality evidence, a model improvement, or an Agent accuracy gain.

## Key engineering finding — the first proposed holdout was contaminated by known training usage

Before accepting any independent evaluation set, the initial E-029 draft was cross-checked against known project training actions:

- M5 TRAIN action examples: `~/experiments/sidra_m5/data/stop_curriculum_seed43/actions.jsonl`. It includes examples from **736 distinct question IDs** even though the M5 SFT run summary reports only 226 distinct trained questions; we deliberately exclude the *conservative entire source file* to avoid ambiguity.
- M6 genuine TRAIN model rollout A (12 questions), rollout B (8 questions).
- M7 on-policy GRPO real TRAIN rollout (6 questions).
- In the initially proposed 64-question packet, **46 questions overlapped the M5 action source** and **1 question overlapped the M7 online RL questions**. They would not have been a credible prospective unseen project-training holdout. The earlier E-029 packet `e029_independent_blind_train64_seed2029` is **REJECTED / OBSOLETE** for all reliability claims, not overwritten.
- **Fix:** added checksum-verified repeated `--prior-train-rows` exclusions (M5, M6 A/B and M7), TRAIN-split validation, strict row-to-question ID parsing, duplicate-path rejection, and generated a completely separate frozen v2 cohort under a new immutable output path. Tested this fail-closed behavior.

## Frozen v2 cohort — actual result

- Source code snapshot `3ea6e25d006f64da38f82cabc466829c03d0596b`; on Tang `~/experiments/sidra_m7_evidence_run6/source_3ea6e25d006f64da38f82cabc466829c03d0596b`.
- CPU full test suite: **163 passed**. New tests assert original TRAIN provenance, no gold/model label in reviewer packet, no overwritten output, reviewer-ID and frozen-task integrity, and training-use overlap exclusions. GitHub CI must be checked separately.
- Source TRAIN corpus: 1,024 original question pairs; E-026 prior used **12** unique TRAIN questions explicitly excluded; conservatively **739 distinct known project train-used question IDs** from the four M5/M6/M7 artifacts excluded (sets may overlap E-026). The final chosen **64 distinct TRAIN questions are disjoint** from both E-026 and **all four supplied past training/rollout sources**, as enforced by exact question ID checks.
- Stratified: **48 bridge + 16 comparison** original HotpotQA TRAIN question types. For each type, 50% of tasks expose all original supporting sentences and 50% omit *one entire labelled supporting source title* (exact original facts only). Thus **32 full-source / 32 source-omission candidate cards**, ONE card per question in each reviewer's packet. Missing-source cases are *not automatically neutral*, and original answers are *candidates*, NOT proven logical entailments.
- Reviewer A and Reviewer B receive equivalent card sets in different pseudorandom orders, without original question IDs, full/ablated private variant flag, presumed gold semantic label, NLI scores, or editorial expected-label hint. Every card includes the exact original question, a candidate answer and the visible original source sentences (title and sentence index) so a human can reason about sufficiency, contradiction vs ignorance.
- Labels requested from humans: `supported`, `contradicted`, `insufficient_evidence`, `unclear`. The validation interface can check two independently **declared** rater identities and track conflicts, but correctly states that software cannot prove human independence or adjudicate semantic truth automatically. Third-party dispute resolution required before any model calibration.
- **0 human annotations.** No model calls, no SearchAgent actions, no actual Verifier false-positive or false-negative rate measured.

### Immutable local artifact paths / SHA256

Final accepted packet location:
`~/experiments/sidra_m7_evidence_run6/results/e029_hard_holdout_64_after_SFT_RL_exclusions/`.

| Local artifact | SHA256 |
| --- | --- |
| `REVIEWER_A_BLIND.jsonl` | `23ee81bad9ccfc0833f630f229e7825a42715837ac6da6b22d44cc6a4cd0a507` |
| `REVIEWER_B_BLIND.jsonl` | `19571c764c550b36654165eef37b47bb6f50e70d3de450e3a7be69ebed3eea51` |
| `REGISTRY_PRIVATE_DO_NOT_SHARE_WITH_REVIEWERS.jsonl` | `466253172d1640243d36b31667c44a8bc60b9a97b7b2543674abf28e200c2724` |
| `MANIFEST.json` | `1e8343edcc3fb8d2104824da4563f5262f044b11ac519609747305a2b86d0ccc` |

Original source and all four previous training file SHA256 are recorded in the manifest. The **raw reviewer packets and private registry stay on Tang**; they are NOT committed to GitHub, to avoid mixing evaluator source text / private holdout identifiers into agent policy branches.

## Why this matters / next steps

E-026/E-027 showed confident wrong relations and misleading source necessity; E-028 additionally showed order-driven NLI threshold crossing (without any change to evidence). A semantic confidence threshold cannot be calibrated from the same 12 editorial E-026 questions over and over. E-029 freezes an **independent TRAIN-only prospective question cohort with known previous training exposures excluded**, before collecting any predictions or labels.

Next requirements, not yet completed:

1. Obtain two genuinely independent human reviews of ALL 64 tasks, blind to each other and to private full/omitted labels. Do not substitute LLM reviewers or self-agreement for human ground truth. Independently resolve conflicts or mark unresolved (no automatic majority).
2. Keep this frozen 64-question holdout out of future SFT/GRPO/Challenger training; cross-branch integration of its exclusions is **not yet implemented** because this dedicated research branch must not modify other branches.
3. Evaluate existing source-only and model NLI decisions against truly adjudicated evidence support, including false approvals, refusals, tool cost and prompt length. Question-cluster bootstrap by original QIDs, never treat multiple candidate readings as independent.
4. Only after promoting a frozen policy on separate TRAIN development and existing opened Pilot32 should the strictly independent `eval-A64` be examined once. **Sealed final-B160 remains entirely untouched.**

### Reproduction (CPU, no model scoring)

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. ~/projects/CodeRL-Lab/.venv/bin/python \
  scripts/m7_freeze_blind_train_adjudication.py \
  --dataset ~/experiments/sidra_m4/data/train_oracle1024_seed43/converted \
  --original-train ~/experiments/sidra_m4/data/train_oracle1024_seed43/original_train_subset.json \
  --prior-e026-rows ~/experiments/sidra_m7_evidence_run5/results/e026_original_train36_mnli/controlled_rows.jsonl \
  --prior-train-rows ~/experiments/sidra_m5/data/stop_curriculum_seed43/actions.jsonl \
  --prior-train-rows ~/experiments/sidra_m6/results/real_train_rollouts12x4_stop/rollouts_train.jsonl \
  --prior-train-rows ~/experiments/sidra_m6/results/train_rollout_io8x4/rollouts_train.jsonl \
  --prior-train-rows ~/experiments/sidra_m7/results/onpolicy_grpo43_1step_6groups/train_onpolicy_rollouts.jsonl \
  --out <NEW_EMPTY_DIRECTORY> --seed 2029
```

Implementation: `src/sidra/blind_adjudication.py`; tests `tests/test_blind_adjudication.py`; CLI `scripts/m7_freeze_blind_train_adjudication.py`.
