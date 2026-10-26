# M7-A4: versioned evidence-first v3 and TRAIN-only reachable curriculum (2026-10-27)

**Research status:** real online multi-turn model updates and full-task TRAIN successes were recorded for the first time in this line. **No matched public DEV answer/evidence success gain; no checkpoint promoted.** This is an exact-normalized *finite one-token choice* on-policy group-relative policy-gradient experiment, not native TRL GRPOTrainer, arbitrary JSON constrained generation, or multi-generation RSI.

## 1. Falsifiable change and independent code controls

The old `v2` catalog can expose premature nonempty FINISH actions after reading one source, even though HotpotQA generally needs multi-hop evidence. The candidate list's 9-action cap also truncates its tail, suppressing useful FINISH candidates.

New explicit `--answer-proposer v3` preserves legacy and v2 A/B behavior by default. It uses only **actual search/read observations**, NEVER support labels or document text not retrieved:

1. Nonempty FINISH requires two **distinct actually read** document IDs. `finish("")` abstention remains available even with no reads.
2. Every emitted FINISH citation refers only to actually read sources; v3 can retain all observed read docs (within tool budget) instead of blindly discarding older reads.
3. Under at most nine categorical action IDs, reserve up to four finite answer choices and use leftover slots for legal READ and SEARCH. One-digit masked Softmax sampling and differentiable log-probs operate on the identical final catalog.
4. Independent `--curriculum-teacher` verifies the exact original TRAIN corpus/DEV manifest hashes, actual strict-success TRAIN-only teacher manifest, and disjoint heldout TRAIN teacher question IDs. The teacher's private TRAIN gold affects **only the TRAIN task ID selection**. The agent prompt and live Search/Read/Finish rollout never see oracle labels.
5. The final sealed HotpotQA final-B160 was not read, sampled, or used.

Source SHA `494d32c0b90664574e65b8f9712153300953aec7` for the first full online v3 experiment, later helper/test-only SHA `53296b0ef649b79d22ca46b58a3c7e1a01677223` for heldout TRAIN audit. CPU: 158 tests passed at experiment source; 163 tests passed after implementing the heldout audit. GitHub Actions CI passed.

## 2. Honest tradeoff: TRAIN oracle teacher reachability

`scripts/build_and_fit_train_teacher.py make`, same 1,024 original TRAIN questions, same original BM25 SearchAgent and tool budgets, gold accessed **only by offline selector**:

| TRAIN-only oracle teacher diagnostic | Old v2 | New evidence-first v3 |
| --- | ---: | ---: |
| Strict correct-answer + evidence-complete REAL tool trajectories | **327/1024** | **287/1024** |
| Train-fit teacher question groups | 261 | 229 |
| Heldout TRAIN teacher question groups | 66 | 58 |

**Negative design tradeoff:** v3's stricter evidence-first gate actually reduced the number of TRAIN source questions for which this offline greedy teacher finds a fully correct legal path; do not advertise v3 as universally better. This is a `TRAIN` oracle *path existence under this teacher strategy* diagnostic, not an achievable nonoracle model accuracy bound.

Local v3 teacher: `~/experiments/sidra_m7/data/teacher_v3_reachable1024_s43_494d32c0`.

## 3. Actual on-policy v3 experiment: two updates, 160 live TRAIN episodes

Start: original TRAIN-only Qwen3-0.6B stop-SFT LoRA SHA-256 `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`. Selected task pool: **229 disjoint strict-success-reachable TRAIN teacher-fit question IDs**, not the 58 teacher-heldout IDs, not DEV. Train seed 81, 2 updates × 20 groups × 4 freshly sampled full Search/Read/Finish trajectories = **160 real online TRAIN episodes**, original max5 tools / max6 decisions, `lr=3e-6`, 2048 prompt cap, unchanged original TRAIN-only reward.

| Live training measurement | First online update | Second online update (fresh rollouts after parameter update) |
| --- | ---: | ---: |
| Actual episodes | 80 | 80 |
| Full strict correct-answer + evidence outcome | **7/80** | **7/80** |
| Groups with nonconstant TRAIN reward | 11/20 | 7/20 |
| Masked assistant ACTION ID loss labels | 241 | 149 |
| Tool-observation loss labels | **0** | **0** |
| Behavior-action log-prob replay mismatch | **0.0** | **0.0** |
| Gradient norm | 8.4542 | 10.2145 |
| TRAIN rollout reward mean | 0.225625 | 0.191500 |

Both steps generated real action choices using the current trainable LoRA, interacted with real SearchAgent tools, computed post-rollout TRAIN-only reward, and applied an actual optimizer update; the second iteration never reused the first iteration's samples. New saved adapter SHA-256 `91e3bf5ec072fc053417d86952eed71702e44dd2f6bb0969460f62c491b27a7e`; GPU peak allocated **6,452,609,536 bytes**; training runtime **92.338 seconds**. The original v2 online experiment had 0 strict successes in a different task sampler; the new nonzero strict TRAIN rate cannot be credited solely to GRPO or v3 because both curriculum and action constraints changed.

Checkpoint/logs (never GitHub weights): `~/experiments/sidra_m7/results/v3_ONPOLICY_Reachable20x4_u2_s81_494d32c0/`, `~/experiments/sidra_m7/logs/v3_reachable20x4_u2_s81_494d32c0.log`.

## 4. Same-policy frozen public DEV pilot32: **no QA gain**

Same original nonoracle v3 model-choice policy, same retrieved document corpus + question SHA, same sample seed 63, same max-tool/step budget, model architecture, code SHA and generation configuration; adapters differ:

| Public DEV pilot32 | Original TRAIN stop SFT | v3 after two true on-policy updates |
| --- | ---: | ---: |
| Answer exact match | **1/32** | **1/32** |
| Complete evidence / verified success | **1/32** | **1/32** |
| Mean gold supporting-document evidence recall | 0.53125 | 0.578125 |
| Mean actual tool calls | 4.03125 | 4.125 |
| Model abstentions | 4/32 | 2/32 |

**Gate: FAIL**: no statistically supported joint answer+evidence improvement; higher tool cost. No deployment, promotion, or externally validated Agentic GRPO benefit claim. Pair audit: Tang `~/experiments/sidra_m7/results/v3_reachable20x4_u2_s81_PAIR_AUDIT_494d32c0.json`.

## 5. Disjoint TRAIN-only teacher-reachable heldout 58 (complementary validation)

To reduce repeatedly tuning against the public DEV pilot, materialized **58 TRAIN-only teacher-heldout question IDs** disjoint from teacher-fit labels and all 40 online GRPO task IDs, preserving byte-for-byte the **ENTIRE original TRAIN document corpus** (SHA256 `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`) to keep BM25 IDFs unchanged. This is a **biased, teacher-selected TRAIN slice**, NOT frozen DEV or external generalization.

| Heldout TRAIN teacher-reachable (58) | Original stop-SFT | After v3 online updates |
| --- | ---: | ---: |
| Answer EM | **6/58** | **6/58** |
| Complete verified evidence outcome | **5/58** | **5/58** |
| Evidence recall | 0.63793 | **0.69828** |
| Mean tool calls | 4.44828 | **4.70690** |
| Mean read calls | 2.46552 | **2.74138** |

Paired evidence-recall delta **+0.06034**, 5k-sample bootstrap CI95 **[0.01724, 0.10345]**, but verified answers unchanged and mean tool calls increased **+0.25862**, CI95 **[0.06897, 0.48276]**. Relative to this selected slice, the model appears to read more gold-supporting material but **cannot translate it into more correct cited answers**. These bootstrap intervals are exploratory and subject to teacher selection bias.

A real engineering failure occurred while running the first independent audit: the direct-script execution path could not import `scripts.audit_paired_onpolicy`, even though pytest imported the module successfully. Fixed by supporting both direct-script and package invocation; added standalone CLI regression test. First failed log preserved at `~/experiments/sidra_m7/logs/trainholdout58_PAIR_be8ea619.log`; after correction **163 tests passed**, the actual independent heldout pair audit completed under pinned source `53296b0ef649b79d22ca46b58a3c7e1a01677223`. Pair result: `~/experiments/sidra_m7/results/v3_teacher_REACHABLE_HELDOUT_58_PAIR_AUDIT_be8ea619.json`.

## 6. Current research hypothesis

The present bottleneck is **answer-selection / FINISH credit assignment**, not probability mismatch: v3 collects more relevant documents under both public DEV and TRAIN teacher-reachable holdout, but correct cited answers remain flat. A next falsifiable, entirely TRAIN-label-private trial separates group-relative `Search/Read` evidence-recall credit from `Finish` correct-answer/strict-evidence credit. Train with a predeclared versioned `--credit-mode stage_credit_v1` and evaluate on the same disjoint TRAIN teacher-heldout before escalating to frozen public DEV. It is an experimental custom on-policy policy-gradient variant, never claimed to be native TRL GRPO or general RSI.

**No model has been promoted. All data/weights remain on Tang; only source/tests/reports are in GitHub.**
