# M7-A8 — completed pre-registered READ+FINISH hybrid training and NEGATIVE frozen QA validation

**Date:** 2026-10-27 (Tang time +08). **Model:** Qwen3-0.6B-Base + original TRAIN-only stop-SFT LoRA, V4 finite nonoracle legal-action catalog. **This experiment is a hybrid of on-policy group-relative policy gradient and explicitly supervised TRAIN-only READ/FINISH auxiliaries; NOT pure GRPO, NOT RSI.**

## Previously blocked by own idle RSI vLLM, now corrected

The A8 initial GPU attempt had sampled its first 80 TRAIN episodes but OOM'd at first backward because a competing SIDRA RSI E-027 vLLM server started allocating GPU memory. That server was later verified to be **our own orphaned idle service**, not an outside workload. It was safely SIGTERM'd after checking no client connections, with peak GPU occupancy dropping from 9,821MiB to 349MiB. Details under RSI branch `docs/M7C_E030_IDLE_VLLM_GPU_LEAK_CLOSED_20261027.md`.

The new run acquired the **same shared** lease at `/home/you/experiments/sidra_shared/gpu0.lock` across training, inspected GPU and competing model launchers without killing any process, and had 15,568 MiB initially free. Pinned code SHA `b438d390c804958a64761c3c30d5c0bbda0d9635` passed **242 CPU tests**. Independent A9/A10 preflight: exact 40 TRAIN question selection SHA256 `449d09935e9dee7f22e7fb1bafb77a94346c759c81600bed6c23e0358034184c`, 0 overlap with TRAIN64/teacher-heldout/public DEV, full original dataset and parent checkpoint SHA matched, no external GPU jobs. Preflight result: `CPU_INPUT_GATE_PASS_GPU_NOT_RESERVED`, followed by successful shared GPU lock acquisition.

## Pre-registered online training was ACTUALLY COMPLETED

Unchanged protocol from `docs/M7A8_READ_PLUS_FINISH_PREREG.md`:
- Original stop-SFT parent checkpoint SHA256 `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`.
- Original TRAIN-only V4 teacher curriculum, seed153, 20 distinct questions/group × 4 real complete Search/Read/Finish rollouts × 2 optimizer updates, 6 steps and 5 tools max; `lr=3e-6`, group `episode_total` advantage, `finish_aux_weight=0.2`, `read_aux_weight=0.2`; legal one-token categorical choice policy normalization matches behavior sampling exactly.
- **160** genuinely sampled TRAIN episodes; **both** optimizer steps updated LoRA weights. Reward-variable groups **8/20** in update0 and **6/20** in update1. Real TRAIN strict successes **5 in each update**. Behavior replay logprob error **0.0** in both iterations. Posthoc TRAIN support READ preference audited for 141 + 138 legal visited READ states; FINISH preference audited for 63 + 45 legal visited decision states; no gold leaked into live action catalog.
- New adapter checksum SHA256 **`2a6454140d019b048cc1cb1de036e8b6eae8f2e516d50330bb9460e1bfb3f14e`** vs original `44e559...`. Wall-time **138.944s**, peak CUDA allocated **6,709,895,168 bytes**, successful process exit 0.
- Immutable Tang artifacts: `~/experiments/sidra_m7/results/M7A8_READ_FINISH_s153_G20x4_U2_b438d390/` (`TRAIN_SUMMARY.json`, `train_rollouts.jsonl`, adapter weights), log `~/experiments/sidra_m7/logs/M7A8_READ_FINISH_s153_G20x4_U2_b438d390.log`.

## Frozen TRAIN64 primary QA gate — NEGATIVE

Frozen internal 64 original TRAIN questions were frozen BEFORE model training, from 622 historically unused original TRAIN source IDs excluding 402 prior teacher/fits. Exact 64 ID digest `336fa32d645a2b52cf68563a56dc9d6571819971c872a92cfb790e2876fcda81`. No A8 optimizer task intersects these 64; no hidden DEV/final-B labels used. This is **internal TRAIN-only disjoint comparison, not an external independent benchmark**.

| Same-policy sampled finite V4 action metric | Starting stop-SFT | After A8 READ+FINISH hybrid |
| --- | ---: | ---: |
| Answer EM | **0/64** | **0/64** |
| Strict answer + supporting-doc citation | **0/64** | **0/64** |
| Mean cited support recall | 0.562500 | 0.546875 |
| Mean tools | 4.781250 | 4.671875 |
| Mean model calls | 5.781250 | 5.671875 |
| Mean wall latency | 0.261641s | 0.255266s |

Both evaluated by same GitHub source SHA, Qwen3-0.6B parent model, V4 finite legal-choice distribution, seed171, full Search/Read/Finish and tool budget. Independent audit `scripts/audit_frozen_train64.py` replayed original source+17 historically SHA-pinned train/teacher sources, verified no new training leakage, and paired-bootstrap checked 64 original question IDs. Result: **EM delta0 [0,0], strict success delta0 [0,0], cited support recall −0.015625 [−0.046875,0.015625]**, 5,000 paired draws; **promotion_allowed=false**. Saved `~/experiments/sidra_m7/results/M7A8_TRAIN64_PAIRED_AUDIT_s171_b438d390.json`.

## DEV pilot32, two matched stochastic seeds — also NO benefit

Public DEV pilot32 was already viewed; this is reproducibility smoke, **not independent validation**. Same question IDs/model/decoder/actions and source SHA per pair, seed151 and seed157:

| Seed | Starting answer EM | A8 answer EM | Starting strict verified | A8 strict verified | Cited support recall |
| --- | ---: | ---: | ---: | ---: | --- |
| 151 | 4/32 | 3/32 | 3/32 | 2/32 | 0.671875 → 0.640625 |
| 157 | 2/32 | 2/32 | 1/32 | 1/32 | 0.640625 → 0.578125 |

Do not count the same 32 questions twice as 64 independent QA cases. Negative evidence is consistent across both seeds. Saved `~/experiments/sidra_m7/results/M7A8_DEV32_{BASE,A8}_s{151,157}_b438d390/`.

**Decision:** A8 new adapter has genuine online updates and does NOT improve independent answer/citation reliability, so **do not promote it** or use it to tune sealed final-B160. The pure on-policy GRPO claim is further limited by explicit TRAIN-gold auxiliary READ/FINISH imitation (0.2+0.2). Do not present this as demonstrated RSI or generalizable Search Agent benefit.

## Follow-on counterfactual root-cause diagnosis (DEV pilot ONLY)

A later audit, with **oracle exposure of both true supporting documents only for posthoc diagnostic** (NOT online Agent inputs or optimizer), checked whether V4 finite Finish candidate answers could even express the gold answer. In DEV pilot32, **31/32** answers were literally visible in correctly READ source text but only **9/32** appeared among at most four legal nonempty Finish candidates. Thus **22/32** had answer-in-source yet the finite action catalog excluded it. This is a hard representation ceiling that cannot be overcome by changing choice probabilities alone. Source `scripts/audit_finish_candidate_ceiling.py`, tested code SHA `93d0e8bde64aa58f52d709f813e9523898e2824b`, **244 CPU tests passed**; report `~/experiments/sidra_m7/results/M7A11_FINISH_CEILING_PILOT32_v4/`.

An additional **fixed actual-tool-trace** Qwen3-8B-AWQ answer-generator counterfactual was implemented (scripts/m7_fixed_evidence_8b_solver_pilot.py). It must report extra model calls and costs and cannot be claimed as end-to-end Agent/RL improvement. DEV final-B160 remains sealed.

## Next falsifiable research question

Decouple **legal search/read/stop policy learning** (single-token exact-normalized categorical actions) from **source-conditioned freeform answer generation**, with explicit train-only instruction/answer supervision, no DEV gold input, and measured extra inference overhead. Freeze a new unseen question cohort BEFORE any model update. Do not quietly turn this into an oracle-solver result.
