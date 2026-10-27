# M7-C E-029 — preregistered observed-only matched second-READ shadow: **ZERO ELIGIBLE SLOTS**

**Run date:** 2026-10-27 (Tang +08:00). **Status: reproduced negative feasibility result; BLOCK same-action-slot improvement claim.**

## Preregistration and source

The test criterion was frozen in `M7C_E029_OBSERVED_ONLY_MATCHED_READ_PREREG_20261027.md` **before** evaluating E-029 real model traces and verifier labels. Changes are in independent `feat/m7-rsi-multi-generation`; no writes to online GRPO.

- E-029 pinned code SHA `bf7990af85cde14a7bd75edfb9bca9d5162b0d4f`.
- E-026 original curriculum pinned SHA `a1c649c5793aeb371f2ce76a756d36aacc9f7c12`.
- Source corpus: **original HotpotQA TRAIN only**, three disjoint E-027 Qwen3-0.6B STOP-LoRA real-model cohorts of 24 each (sampling seeds 43/44/45, same model checkpoint).
- Phase 1 `model-only` reads solely the original sanitized model-only pack, previously observed SEARCH/READ responses and saved model traces; cannot take TRAIN gold or campaign private map paths as inputs. Proposed shadow rows frozen before independent scoring. Phase 2 `verify` replays **all 72** real tool + model action transcripts, canonical TRAIN corpus projections and private cross-cohort/holdout exclusion before touching gold supporting document labels.
- Frozen original E-028 verifier baselines reproduced **72 distinct original TRAIN episodes, 17/72 answer exact match, 0/72 strict correct+full cited original supporting docs**. Historical episodes remain real Qwen3 model calls, but no new real model call or gradient update in E-029.
- New source includes deterministic observed-only selector, immutable two-phase CLI, independent complete-campaign replay and strict anti-gold/forgery unit tests. Tang pinned full CPU test suite **224/224 passed** and `compileall` passed at this SHA.

## Actual data finding (model-only, then independently revalidated)

| Recorded behavior (all 72 authentic prior Solver episodes) | Count |
| --- | ---: |
| Zero READ calls | 1 / 72 |
| Exactly **one** READ call | **71 / 72** |
| Two or more READ calls | **0 / 72** |
| Exactly one SEARCH | 63 / 72 |
| Exactly two SEARCHes | 8 / 72 |
| Zero SEARCH calls | 1 / 72 |
| **Preregistered second-READ same-action-slot comparisons eligible** | **0 / 72** |
| Paired shadow-vs-real support exposure estimate / CI | **undefined (n=0)** |

This is a **substantive blocker**, not an E-029 ranker improvement or a null effect estimate. A second-READ replacement cannot be compared at an actually observed second-READ action position because no real episode ever took that action. Inventing a replacement of a FINISH action would change the action budget and downstream model observations and violate the frozen same-slot protocol.

Observed second-READ count and search-count distributions were computed by an additional **model-only, aggregate-only, read-only inspection** of the three pinned trace files. No per-question TRAIN labels were read in that diagnosis.

**Research implication:** the 0/72 strict citation successes are consistent with a **single-READ bottleneck** on multi-document HotpotQA tasks. This finding alone does not establish that forcing a second READ improves QA: the original supported answer relation may still not be found, and a forced READ can crowd out a good FINISH or waste tool calls.

## Immutable Tang evidence

- Model-only slots + hashes:
  `~/experiments/sidra_rsi_m7/results/E029_MODEL_ONLY_OBSERVED_READ_bf7990af/`
- Independent E-028 complete rerun, E-029 aggregate-only verifier:
  `~/experiments/sidra_rsi_m7/reports/E029_OBSERVED_READ_VERIFIER_bf7990af/`
- 224/224 pytest:
  `~/experiments/sidra_rsi_m7/logs/E029_cpu_bf7990af.log`
- Phase logs: `logs/E029_model_only_bf7990af.log`, `logs/E029_verifier_bf7990af.log`.
- Source snapshot: `~/experiments/sidra_rsi_m7/source_bf7990af85cde14a7bd75edfb9bca9d5162b0d4f/`.
- SHA256 frozen model-only slots: `3e63215216525bae8caa4285cc31634096eb62b33ec0bc6a5d4a1ff3fecc9983`. Independently replayed baseline aggregate SHA256: `af6cf4ef4135ce268a09d80c4f1a3a9cd5869cbe411e33ac0e95d97a7e9ae5f5`.

The independent verifier stored only aggregate output, with no gold-derived per-question examples or private identifiers committed to GitHub.

## Honest gate and next falsifiable study

**Decision:** matched-slot shadow E-029 is **not identifiable** on the frozen campaign (0 eligible slots); do not tune to change eligibility after reading outcomes or claim a positive/negative performance delta.

Next prepare a **prospective**, observed-only `READ-before-FINISH` policy guard: after the first real READ, if the already-visible frontier offers at least two legal unread docs and enough ordinary steps/calls remain, use one extra ordinary READ selected solely from observed titles/snippets/question/READ text. Maintain **the same global maximum 6 decisions / 5 tools** for treatment and baseline, counting the additional READ against that shared budget. Do not rewrite previously recorded continuations. Compare full actual Qwen3 model tool episodes in paired independent original TRAIN questions, with frozen selection/protocol and randomized seeds *when GPU is genuinely uncontended*. Verifier labels remain offline and never enter the guard. A new READ intervention is not proof of multi-generation RSI; it is a retrieval-policy ablation to resolve the evidence bottleneck.

**No new Trainer admission, no actual optimizer steps, no authenticated serving origin, no checkpoint generations, no multi-seed learning improvement, and final-B160 untouched.**
