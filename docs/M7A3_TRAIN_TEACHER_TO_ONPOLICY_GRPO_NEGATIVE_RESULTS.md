# M7-A3 — real TRAIN-only teacher imitation → fresh multi-turn online GRPO (2026-10-27)

**Status: completed negative research cycle. Keep both checkpoint candidates unpromoted.**

All source code and CPU tests were committed to GitHub; all model weights, TRAIN traces and frozen DEV per-example results remain only on Tang, never in git. Source SHA for final stage: `6a0ee669e870a7dc547443bc90e2d96260545448`. GPU one-worker lock `~/experiments/sidra_m7/sidra_gpu_job.lock`; each stage checked actual model processes and free GPU memory. `153 passed` CPU tests in the pinned source snapshot. No sealed final-B160 read or reward use. The experiment remains **custom finite-action, fully probability-matched group-relative beta=0 REINFORCE/PPO-like updates, not native TRL GRPOTrainer**.

## 1. New offline TRAIN teacher: real tools, oracle labels only offline

Program: `scripts/build_and_fit_train_teacher.py make` and `tests/test_train_only_teacher.py`.

Unlike synthetic/Oracle-policy scores, each teacher path is executed through the real scope-restricted BM25 `SearchAgent` using only the exact actions emitted by the unchanged `build_legal_actions(..., answer_proposer=v2)` catalog. The teacher may use original HotpotQA TRAIN gold answers and supporting-document IDs to **label** its choices, but these private fields are never serialized into the student's model prompt; the model sees only original question, real prior Search/Read observations and nonoracle allowed action list. No DEV/TEST labels enter teacher or optimizer. An episode enters the expert-action dataset only if the full genuine rollout finishes with original answer EM, all original supporting sources cited and answer text grounded in citations. A failure is discarded, never retroactively relabelled as a success.

| Executed TRAIN-only teacher result | Value |
|---|---:|
| Source questions | 1,024 |
| Real fully correct/evidence-complete tool trajectories | **327** (31.93%) |
| Verified legal model-choice actions in those paths | 1,632 |
| Disjoint TRAIN training trajectories / actions | 261 / 1,312 |
| Disjoint heldout TRAIN trajectories / actions | 66 / 320 |
| Teacher attempts rejected as incomplete/abstained | 697 |
| Generation wall time | 11.944 s |

Data and full local audit: Tang `~/experiments/sidra_m7/data/teacher1024_v2_seed43_26e00f26c/`, SHA256 TRAIN question `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`, original frozen DEV manifest `3f0c494cca8b44cfef90c78d4d5b1fbb4158dc7387ed0f24316a0a265471d947`. The source files and the frozen manifest were hashed on Tang before running experiments.

**Interpretation:** 327/1024 means a TRAIN-aware *oracle teacher can generate a strictly successful legal path* for that many source questions. It is NOT a nonoracle student's multi-hop accuracy, a model training result or GRPO.

## 2. Identified and fixed a genuine teacher-training sampling bug

In the first `96`-label attempt, task-first round-robin selected **96 first actions, all SEARCH** from different teacher trajectories, with **zero READ or FINISH targets**. This explained why the model was encouraged to keep searching rather than finish correctly; previous results from that checkpoint are invalid as multi-stage imitation evidence. The wrong sampling code was fixed with a deterministic seeded **stage-stratified selector** which interleaves `search/read/finish` and rotates tasks within each stage, plus a dedicated regression test.

First SEARCH-only 96-action SGD run (`2e-5`), retained as a failed ablation:
- Heldout TRAIN expert-action masked CE worsened **2.03378 → 3.66018**.
- Paired public DEV pilot32: answer EM **1→1**, strict answer/evidence **1→1**, mean tool calls **4.28125→5.0**; gate FAIL.
- Trained weights saved on Tang only, not promoted.

Second balanced 96-action run (`32 SEARCH + 32 READ + 32 FINISH`, `2e-6`), TRAIN-heldout masked CE still worsened **2.03378 → 2.39545**, action accuracy marginally changed **0.45625→0.465625**. Not promoted.

## 3. Proper multi-stage offline warm-start: balanced 240 labels with TRAIN-only checkpoint selection

Final trial source `6a0ee669e870a7dc547443bc90e2d96260545448`, teacher labels sampled `80 SEARCH + 80 READ + 80 FINISH`, Qwen3-0.6B-Base + original TRAIN-only stop-SFT LoRA, learning rate `2e-6`, one gradient update per *expert supervised label*, 240 steps, maximum context 2048. Heldout TRAIN expert-action CE checked every 40 steps, and **only heldout TRAIN expert labels** used to pick the saved checkpoint. Best was step 240, not chosen on DEV. Peak GPU allocated 6,881,684,992 bytes; 108.67s.

| Heldout TRAIN expert-choice metric | Before | After |
|---|---:|---:|
| Overall masked action CE (lower better) | 2.03378 | **1.80846** |
| Overall exact teacher-choice accuracy | 45.625% | 43.75% |
| FINISH-choice exact accuracy | 0/66 | **17/66** (25.76%) |
| READ-choice exact accuracy | 65.91% | 46.21% |
| SEARCH-choice exact accuracy | 48.36% | 50.82% |

This is a meaningful supervised **finish-selection** learning signal, with substantial READ regression. It is **not reinforcement learning or evidence of better QA**.

The independent same-source, same-adapter-family, same-policy/seed/DEV pilot32 audit found the following actual regression:

| Paired public DEV pilot32 | Original stop-SFT | After supervised balanced 240 |
|---|---:|---:|
| Answer EM | 1/32 | 1/32 |
| Strict answer+evidence success | **1/32** | **0/32** |
| Mean tool calls | 4.28125 | 3.0625 |
| Gold supporting doc evidence recall | 0.546875 | 0.375 |

**Promotion gate FAIL.** The supervised policy finishes earlier with fewer tools but omits required evidence. The lower TRAIN-heldout action CE did not predict actual public DEV task quality. This is a particularly valuable counterexample to using imitation fit as the main success criterion.

Local stored adapter: `~/experiments/sidra_m7/results/sup_teacher_v2_s43_bal240_lr2em6_besttrain_6a0ee669/`.
Adapter SHA changed `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771` to `77620ac4a906fe44860f45ac04aec4d82cfed5152f78daaef63c652f404f7a7a`. Local matched pair audit: `~/experiments/sidra_m7/results/teacher240_pilot32_s63_PAIR_AUDIT_6a0ee669.json`.

## 4. Resumed TRUE fresh on-policy multi-turn GRPO-style updates **after** warm-start

Source still pinned to `6a0ee669e870a7dc547443bc90e2d96260545448`. Parent LoRA is the balanced teacher checkpoint above (NOT the old initial stop SFT). Dataset strictly TRAIN; each iteration generated **new fresh** real Search/Read/Finish episodes using current LoRA's 1-token masked categorical distribution over state-specific legal IDs. `temperature=1` categorical sampling and differentiable action log-prob use exactly the same mask and normalizer; replay of every sampled log-prob before its update verified maximum error **0.0**. System/tool/observation tokens are conditioning-only, **0 model loss labels on tool text**. Once a gradient update occurs, no earlier trajectories are reused.

Configuration: seed 71, **2 independent online optimizer updates**, 12 groups/update × 4 generations = **96 genuinely sampled full TRAIN episodes**, update lr `3e-6`, original 5-tool/6-step budget, unmodified v2 nonoracle proposer.

| Actual online RL metric | Update 1 | Update 2 |
|---|---:|---:|
| Live tool episodes | 48 | 48 |
| Nonconstant grouped TRAIN rewards | 2/12 | 2/12 |
| Correct and complete answer+evidence paths | **0** | **0** |
| Assistant ACTION ID token loss labels | 39 | 39 |
| Tool-text supervised tokens | **0** | **0** |
| Behavior / optimizer log-prob mismatch | **0.0** | **0.0** |
| Real LoRA gradient norm | 18.18 | 37.07 |
| Mean TRAIN rollout reward | 0.09583 | 0.09375 |

The second iteration's sampling used parameters updated by the first; both updates changed weights. Final adapter SHA `2ac83e4264bcba7854a764304be72548d72b7efa5dfd65cb63c27da13df6f117`. GPU peak allocated 6,701,763,072 bytes; 26.64 seconds.

Frozen public DEV pilot32 was then evaluated with **identical original v2 legal action policy, sampled-mode decoder, same seed 63, data SHA, source commit, model and tool budgets**. Only the adapter checkpoint differs. Strict independent paired audit:

| Paired after-teacher DEV pilot32 | Parent supervised adapter | After 2 fresh online GRPO-style updates |
|---|---:|---:|
| Answer EM | **1/32** | **1/32** |
| Strict answer+evidence | **0/32** | **0/32** |
| Mean tool calls | 3.0625 | **3.40625** |
| Evidence recall | 0.375 | 0.40625 |

**Promotion gate FAIL.** The two actual online updates did not improve correct answers or evidence-complete success; tool use increased again. Paired audit file: Tang `~/experiments/sidra_m7/results/grpo2_afterteacher240_dev32_s63_PAIR_AUDIT_6a0ee669.json`. TRAIN rollouts/model SHA and action-level logs: `~/experiments/sidra_m7/results/finite_v2_seed71_POST_TEACHER240_onlineG12x4_U2_6a0ee669/`, `~/experiments/sidra_m7/logs/finite_v2_seed71_postteacher240_onlineG12x4_u2.log`.

## 5. Research judgment and required next falsifiable changes

1. We verified **true implementation-level** distribution correctness and multiple real on-policy updates; no probability mismatch or dev label leakage exists in this finite action setup. But this is a *restricted one-digit categorical policy* instead of full arbitrary JSON tool generation, and a **custom beta=0 clipped-once group-relative** loss instead of full native TRL/verl GRPO. Do not generalize beyond these constraints.
2. **Actual outcome reward remains sparse**: despite TRAIN-only teacher paths that would solve 327/1024, freshly sampled student rollouts solved 0/96 in the post-teacher GRPO trial. Variable reward is mostly partial support/tool progress, not answer+evidence success. Scaling GRPO blindly is not a demonstrated benefit.
3. **Premature FINISH vs READ tradeoff:** teacher SFT increased heldout TRAIN FINISH choice accuracy from 0 to 25.76% but decreased READ accuracy and public DEV evidence success to zero. Next improvement must jointly train *evidence coverage → correctly timed FINISH*, not just maximize action imitation CE or optimize tool cost.
4. Proposed next ablations (not yet executed): typed hierarchical action choices with exact masked normalization; state-conditioned document/answer candidate ranking; TRAIN-only two-hop curriculum with support-reading rewards and a legality-preserving minimum-evidence policy guard based solely on visible reads, not oracle support IDs. Require same-policy baseline/after tests and heldout generalization; keep the sealed final-B160 entirely untouched.
5. No candidate checkpoint promoted. All negative runs retained, source code tested, and outputs immutable on Tang.
