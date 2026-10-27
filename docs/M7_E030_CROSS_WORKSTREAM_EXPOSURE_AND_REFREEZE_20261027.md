# M7 E-030 — cross-workstream TRAIN exposure invalidates E-029 v2; safe replacement frozen

**Executed:** 2026-10-27, Tang CPU-only. **Decision:** **RETRACT E-029 v2 as an independent Verifier holdout**; preserve original files but exclude them from future accuracy/generalization claims. A new disjoint 64-question TRAIN-only blind review packet has been frozen and independently rechecked. **No new human labels, NLI quality gain, model updates, Qwen inference, DEV optimization or answer EM gain.**

## 1. Initial independent verification

Read-only checked M6 report `docs/M6_RESULTS_AND_RL_VALIDITY_20261027.md`, M7 branch HEADs and PR/CI, Tang online status and processes. M6 showed that second-hop retrieval exposure alone did not improve real Qwen answers (real DEV pilot bridge 3/32 EM vs baseline 6/32), and the replay update did not improve strict success. That distinction remains unchanged.

- Work branch: `feat/m7-next-hop-evidence`, PR #12 Draft; dedicated branch only, no merges or writes to GRPO/RSI branches.
- Source SHA used for the final CLI/CPU test validation: `e5e6e0668b18b2457d91e2863c0bd016cccbabb1`; replacement packet was frozen from the identical E-030 algorithm at earlier immutable `3bb8ec0f64ec1f63326222fc1b2b6641cf2360f8` (the subsequent changes added **read-only audit mode and tests**, not different packet selection).
- Tang RTX 4090 Laptop: independent Qwen3-0.6B vLLM still used **9,821 MiB**, free 6,099 MiB. Zero GPU process launches, evictions, forced stops or interference. CPU environment: `~/projects/CodeRL-Lab/.venv`.
- M7-A historical evidence: 11 completed `TRAIN_SUMMARY.json` files, six teacher action files (including their TRAIN-only heldout labels); M5 SFT `actions.jsonl`, M6 two rollout files, M7 early GRPO rollout, RSI three real-model original TRAIN Solver question maps and one separate reserved 32-question original TRAIN set. Exactly **26** distinct external source files: 4 + 6 + 11 + 3 + 1 + 1 obsolete prior-review registry. All are SHA256-pinned in the Tang aggregate report. These are heterogeneous types; **teacher-labelled examples and reserved holdout IDs are not optimizer-trained examples**.

## 2. New discovery: prior E-029 v2 did not exclude concurrent workstreams

The E-029 v2 report had correctly excluded the four historical SFT/rollout sources included *in that particular workflow* (739 distinct TRAIN question IDs); however it omitted the other workstreams' prior M7 teacher/optimizer results and RSI original TRAIN Solver/reserved question IDs.

Cross-checking the actual E-029 v2 private registry (SHA256
`466253172d1640243d36b31667c44a8bc60b9a97b7b2543674abf28e200c2724`) against exact historical inputs:

| Mutually non-exclusive source class | Overlap with E-029 v2's 64 questions |
| --- | ---: |
| Previously declared M5/M6/early M7 `prior_train_jsonl` | 0 |
| Real M7 optimizer `selected_training_ids` across 11 completed summaries | **9** |
| M7 six original TRAIN teacher-labelled action files | **19** |
| RSI real original TRAIN Solver question IDs | **10** |
| RSI reserved original TRAIN question IDs (conservative exclusion) | **5** |
| **Unique E-029 v2 questions appearing in a model/teacher/RSI history** | **34/64** |
| Also present in discarded E-029 v1 review packet | **18/64** |
| **Unique E-029 v2 questions touched by either the histories or v1 packet** | **44/64** |

Counts in different categories overlap: **do not sum the source rows**. Of the 44 E-029 v2 overlapping cases, **34** had genuine earlier workstream exposure (training/teacher/real Solver/reserved); the **remaining 10** had only earlier E-029 v1 card exposure. This is a retrospective data provenance defect, not a newly observed neural training improvement.

**Decision: E-029 v2 must NOT be used as a contamination-free independent evaluator.** It has zero human annotations anyway. The previous assertion “disjoint from four supplied prior artifacts” remains literally true, but was **insufficient for all-workstream independence**. Keeping this negative result matters more than obtaining a flattering Verifier accuracy number.

## 3. Corrected E-030 prospective packet

Created `src/sidra/holdout_history.py`, `scripts/m7_audit_cross_workstream_holdout.py`, `tests/test_holdout_history.py`. The code:

1. Enforces explicit presence of every historical source class and validates exact SHA256 bytes, strict nonempty source row formats, known original TRAIN1024 question IDs, exact optimizer `selected_training_ids`, and unique source paths; a missing/foreign record aborts.
2. Hash-verifies **both blinded reviewer packets and the private original E-029 registry** against their original manifest before using any of its IDs; never accepts an arbitrary unpinned packet.
3. Keeps the 857 unique workstream-exposed IDs distinct from 10 additional unseen-to-workstreams older E-029 v1 card IDs. In addition, excludes **all 64 E-029 v2 question IDs**, even those not used by a model, rather than recycling previously inspected study cards.
4. Deterministically selects replacement original TRAIN examples under a NEW seed **2030**, with **48 bridge + 16 comparison**, **32 full original-source cases + 32 one-source-omission cases**, and one case per question per reviewer. Reviewers see neither private variant, original question ID, intended label nor model predictions. This is a prospective **request for human annotations**, not any semantic ground truth.
5. Checks post-freeze history source hashes once more; if another branch changed an input during refreezing, abort rather than certify stale disjointness. No inherited GitHub code branch is mutated.
6. Provides `--audit-only` for future **read-only, nonzero-on-contamination** preflight before evaluating the newly frozen cohort. Still cannot guarantee protection from new workstream training that starts *after* the check.

### Reproducible real Tang result

- Historical workstream-exposed distinct original TRAIN IDs: **857/1024**. Including the discarded E-029 v1 cards: **867**. Including the original E-029 v2 cards: **887**. Remaining available original TRAIN pool: **137**, of which **99 bridge** and **38 comparison** before new selection; E-026's 12 were separately excluded by the freezer (the selection passed).
- New E-030 replacement cohort: **64 original TRAIN questions**, zero overlap with any of the **887** previously exposed/model/teacher/RSI/old-card IDs (subject to the exact pinned history inputs). Source-title and original supporting-sentence validation passed.
- New frozen selection question-ID digest: `25beb042091bccb185a1c8b0a2638363ceb1cbd06f26d45e18a903f2236124bb`.
- New `MANIFEST.json` SHA256:
  `e4c7aeb3bb997ed700c5c3e4bf83dbecbbbe473ceec41dc1bbeb41891fd74750`.
- New audit aggregate JSON SHA256:
  `60b487f30d0d73f71f692f029e6485f296fa12a6ad7ab5b12871b908755e7060`.
- New private reviewer A/B packet SHA256:
  `0a206073440982c39e98203ce06641153bfa9ac387eb1b36bda352966c995f34` /
  `f5aca3eb9c949232f76fa0e4d9935eec56caf3f36ccd548c286b92890dfe28cc`.
- Private registry SHA256:
  `e7887d94a4498f927e3d7927404bd18c7a6b777f2282404860d570f6301998f7`.

**Final protected Tang-only directory:**
`~/experiments/sidra_m7_evidence_run7/results/E030_PROSPECTIVE_SAFE_TRAIN64_s2030_3bb8ec0f/REPLACEMENT_BLIND_64_TRAIN_ONLY/`.

Full aggregate-only source digest registry:
`~/experiments/sidra_m7_evidence_run7/results/E030_PROSPECTIVE_SAFE_TRAIN64_s2030_3bb8ec0f/AUDIT_AGGREGATES_ONLY.json`.
Reviewers must receive ONLY their own blinded packet, never the private registry or training-exposure list. Original candidate answers are not automatically entailed by original supporting sentences.

### Negative development attempt retained

The first E-030 code revision correctly excluded known training and E-029 v1 cards, but mistakenly **recycled 9 questions from E-029 v2**. This first replacement under
`~/experiments/sidra_m7_evidence_run7/results/E030_REPLACEMENT_TRAIN64_s2030_edb3adef/` is **REJECTED / OBSOLETE**, never a prospective independent cohort. The fix excludes the entire E-029 v2 registry, adds a regression test, and uses a **new immutable output path**. No old artifacts were overwritten or deleted.

## 4. Real tests, fail-closed gates and limitations

- Immutable initial E-030 algorithm `edb3adef...`: Tang full CPU test suite **173 passed**.
- Corrected refreeze code `3bb8ec0f...`: **174 passed**, successfully generated the final replacement.
- Read-only preflight code `e5e6e066...`: **13 targeted tests passed; 176 total CPU tests passed**.
- Real `--audit-only` on old E-029 v2: returned exit **2**, with **34** genuine workstream exposures, 10 other obsolete-card-only overlaps.
- Same `--audit-only` on corrected E-030: returned exit **0**, zero overlaps with old packets and pinned sources. Both gates used only CPU and original TRAIN input.
- Logs: `~/experiments/sidra_m7_evidence_run7/logs/E030_FINAL_3bb8ec0f.log`,
  `E030_AUDIT_ONLY_OLD_e5e6e066.log`, `E030_AUDIT_ONLY_NEW_e5e6e066.log`.
- No real Qwen Search/Read/Finish run, policy change, Qwen model call, token/latency or answer/citation Bootstrap comparison was made here. The contamination audit is a deterministic population membership test, so no sampling CI is appropriate for its exact overlap counts.
- No human annotations, no independently adjudicated entailment truth. Prior E-026 semantic labels remain editorial/provisional; no claims of improved model semantic correctness.
- The new E-030 packet is disjoint only as of the recorded inputs. **GRPO and RSI branches have NOT been modified to enforce its exclusion**. Before any review scoring or model comparison, repeat the fail-closed `--audit-only` with all newly completed history inputs; stop if a new training overlap appears.
- No eval-A64 retesting and no final-B160 access; no DEV answer leakage. Existing E-021 retrieval-only and E-025–029 negative Verifier conclusions remain unchanged.

## 5. Next experiment, not yet run

1. Arrange two genuine independent human reviews of the E-030 packet and third-party arbitration of disagreements; retain `unclear` rather than forcing a positive/negative label.
2. Freeze verifier predictions (model/checkpoint SHA, fixed threshold, hypothesis translation coverage, context truncation) *without human labels as inputs*. Evaluate agreement, false accept/reject and source necessity **only after adjudication**, using question-ID bootstrap, without changing the Verifier using the protected cohort.
3. Separately, compare actual Qwen Search/Read/Finish trajectories on an allowed frozen development pilot under fixed tools, logits, token and latency measurement. Do **not** promote a threshold based on exploratory E-026 or E-029.
4. Keep `eval-A64` and sealed `final-B160` off-limits for iterative optimization.
