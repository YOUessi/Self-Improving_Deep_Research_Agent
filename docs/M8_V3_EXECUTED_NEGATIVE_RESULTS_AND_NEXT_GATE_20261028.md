# M8 V3 — matched real Qwen3 negative result and next research gate

**Report recorded 2026-10-28 (Asia/Tokyo). Actual Tang runs completed 2026-10-27 UTC.**
**Code and model provenance:** branch `feat/m8-chain-state-v3`, evaluated snapshot
`6f9318c7a351382ca4aa651dfa64de892623abdd`. Original matched V4 candidate
proposer; Qwen3-0.6B-Base + original TRAIN-only stop-SFT LoRA
`44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`.
Same original public DEV pilot32, question/document hashes
`94db9bdfff053555b9a8e0da9a850b5bee3de73e2406ec85c63461b5ae033e6e`
and `5aabc048b344940ffdce2880a7a57a683d4009777a796fd0602ed683a500cc6c`.
Both arms use actual SearchAgent BM25 Top3, `max_steps=6`,
`max_tool_calls=5`, `max_context=2048`, V4 bounded one-token FINISH catalog,
sampled seeds **151,157**. Data are **previously seen DEV pilot**, NOT a fresh
holdout, final-B160, or external real-web corpus.

## Valid result: V3 fails its preregistered gate

Same source SHA for all four arms, 32 unique original questions repeated under two
seeds, **64 episodes per policy**. Report pairs are clustered by 32 independent
QUESTION IDs, not 64 iid episodes, with 5,000 bootstrap resamples:

| Metric | V4 state OFF | V4 state V3 | V3-minus-OFF | Paired question-cluster 95% CI |
|---|---:|---:|---:|---|
| Exact answer EM | 6/64 (0.09375) | 2/64 (0.03125) | -0.06250 | [-0.140625, 0.0] |
| Strict verified answer+support | 4/64 (0.06250) | 2/64 (0.03125) | -0.03125 | [-0.109375, 0.03125] |
| All original supporting docs SEARCH-exposed | 0.62500 | 0.609375 | -0.015625 | [-0.109375, 0.09375] |
| All original supporting docs actually READ | 27/64 (0.421875) | 27/64 (0.421875) | 0 | [-0.09375, 0.09375] |
| Mean support evidence recall | 0.65625 | 0.632812 | -0.023438 | [-0.101562, 0.0625] |
| Mean tool calls | 4.640625 | 4.390625 | -0.25 | [-0.53125, 0.015625] |
| Mean prompt tokens | 4837.078125 | 4894.6875 | +57.609375 | [-322.671875, 395.140625] |
| Mean latency (ms, Tang local) | 252.0625 | 258.375 | +6.3125 | [-11.546875, 23.078125] |

**Decision = BLOCK.** The preregistered improvement in strict success plus
nonnegative answer EM is NOT met. Treat weaker tool cost and denser state
reward as irrelevant to deployment/GRPO promotion without strict gain. Do
not re-use this already-viewed DEV pilot to tune V3 and announce a positive
result. No V3 GRPO model updates, no multi-generation RSI, no default-policy
change and no access to sealed final-B160.

## Mechanism diagnosis (posthoc on same already-open pilot; NOT independent)

| Observable | OFF (64 episodes) | V3 (64 episodes) |
|---|---:|---:|
| Original support fully READ | 27 | 27 |
| Question-anchored READ appears | 40 | 38 |
| At least one question-anchored linked READ pair | 24 | 23 |
| Correct answer ever appeared in legal FINISH candidates (posthoc gold audit) | 16 | 11 |
| Empty-answer termination / abstain | 1 | 5 |
| Executed Agent errors | 0 | 1 |

The most important result is **answer representability decreased 16→11**
although chain closure was essentially unchanged. V3's chain-conditioned
catalog/FINISH bias cannot fix a missing correct answer option, and may
suppress useful exploration when title aliases or relation direction are
unrecognized. One real V3 rollout failed with `ValueError: duplicate SEARCH
query` during state/action mismatch; it was counted as a failure, not silently
excluded. Do not retrospectively patch and rerun the same pilot as a valid
confirmatory comparison.

The current evidence ledger is a **title co-mention heuristic**, not
sentence-level semantic entailment, contradiction detection, or demonstrated
support for the chosen answer.

## TRAIN-only reward falsification (same original TRAIN160 saved real rollouts)

The preexisting TRAIN160 saved rollouts contain 440 READ actions: **214**
original gold-support and **226** distractor READs. Historical saved-rollout
SHA256 `e85084bd4b4fff67d88647a72d9a5c67c8b3885c4f6597b1606222d63cc9c69d`.
Under V3's new version-aware visible-state transition:

- Valid support: **98/214** READ events showed a positive raw state delta;
  **98/214** gained positive state-delta credit.
- Distractors: **18/226** READs showed *false* positive raw state delta,
  but **0/226** received positive credited state delta because TRAIN gold
  support did not advance. The 18 misleading raw transitions remain a
  scientific concern, not evidence that V3 inference understands relevance.
- This gate uses original TRAIN labels **only after completed real actions**.
  These 160 TRAIN episodes were previously seen and used in V3 rule design.
  It is **not** a prospective evaluation, policy correctness proof, or
  model performance improvement. No reward-learned weight update executed.

## Engineering verification, source of truth, raw artifacts

- Tang with PyTorch installed: **306 tests passed** (one warning).
- Mac minimal Python environment: **278 passed; 28 optional Torch/NumPy skipped**.
- GitHub Actions for `6f9318c7`: completed **success**.
- All ten files in V3 patch published with verified Git blob SHA identical
  to local committed tree; PR remains **DRAFT**.
- Tang immutable diagnostic outputs:
  `~/experiments/sidra_m8/results/M8_V3_DEV32_PAIRED_6f9318c7.json`
  `~/experiments/sidra_m8/results/M8_V3_TRAIN160_REWARD_GATING_DETAIL_6f9318c7.json`
  `~/experiments/sidra_m8/results/M8_V3_DEV32_MECHANISM_DIAGNOSIS_6f9318c7.json`
  and four complete `M8_V3_DEV32_V4_state{off,v3}_s{151,157}_6f9318c7`
  evaluation folders with per-example metrics, real tool traces, source
  manifests, policy adapter and candidate distribution checks.
- Pinned paired comparator output: `grant_train_only_online_grpo_trial=false`.
  No active GPU job from this M8 experiment after completion.

## Next engineering/research tasks — NOT evidence of completed work

1. **Correct-answer representability first:** on an entirely prospective,
   question-disjoint TRAIN-only development cohort, record answer availability
   at each legal FINISH decision under fixed tool budgets and propose a
   non-oracle answer generator which does not consume search/read slots.
   Previously tried V5 alone was unsuccessful in real V4-vs-V5 DEV QA; don't
   simply repeat it. Consider 8B READ-conditioned answering with a
   source-bound claim/evidence verifier only as a distinct, explicitly
   more expensive architecture. The previous hybrid TRAIN64 feasibility
   result must not be repackaged as model-policy or GRPO gains.
2. **Repair state/action contract and alias-gap detection:** distinguish
   complete-title aliases from mere word co-mentions, canonicalize SEARCH
   duplicates using the *same* logic for legal catalog and state bias, test
   polarity and two-source directional claims, and preserve tool-budget
   legality. Add unit tests before any new model work.
3. **Make outcome the optimizer target, not local proxy:** any future
   policy gradient should optimize original TRAIN answer correctness AND
   actual cited supporting evidence, using only genuinely executed tool
   traces; state delta is a low-weight auxiliary **gated by new true support**,
   and must never independently award distractor evidence or abstention.
   Explicitly report cost and abstention as separate measurements.
4. **Avoid contaminated holdouts:** pre-register a fresh question-disjoint
   TRAIN-only dev/cohort and audit exposures across M5/M6/M7/M8, the V5
   frozen TRAIN64 and E-030 64 human-review cards. Keep final-B160 sealed.
   First run a new frozen real-model A/B on that cohort, with paired question
   bootstrap; only authorize TRAIN-only custom GRPO if verified success
   improves and answer EM does not decline.

**Summary:** The v3 implementation is a real traceable engineering artifact.
It did not improve end-to-end multi-hop QA. The only observed improvement is
eliminating false-positive **TRAIN credited** state delta for saved
distractor READ actions; this is insufficient to start GRPO.
