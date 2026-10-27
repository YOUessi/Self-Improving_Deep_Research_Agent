# M7-A5 — real online per-action evidence-delta GRPO-style experiment (2026-10-27)

**Research outcome: NEGATIVE task-quality gate.** Two real fresh online policy updates occurred on TRAIN, but the resulting checkpoint **did not improve public DEV answer/citation success (1/32→1/32)** and **regressed a disjoint, teacher-selected TRAIN holdout (EM 6/58→4/58, strict evidence 5/58→4/58)**. Do not promote or treat this as demonstrated Agentic GRPO quality gain or RSI.

## 0. Environment and exact algorithm claim

- GitHub source for training and paired evaluation: `c05552da3284de89536d053f94a8915553e7669c`; added auditable TRAIN-only failure classifier and tests at `46edc7a8ca54cdc633fb9b0aacd7b2b8d77d789d`.
- **173 passed CPU tests** under the final audit code (Tang `~/experiments/sidra_m7/logs/cpu_finishtaxonomy_46edc7a8ca54cdc633fb9b0aacd7b2b8d77d789d.log`). Training snapshot itself passed 171 tests.
- **Not** native TRL `GRPOTrainer`: custom one-token, legally masked categorical Softmax over at most 9 nonoracle Search/Read/Finish candidates; the *same exact per-state normalized distribution* is used for stochastic sampling and differentiable selected-token log probabilities, no vLLM JSON constrained sampling. This is exploratory group-relative clipped beta=0 on-policy policy gradient, not proof of GRPO framework equivalence or an unrestricted generative Search Agent.
- 4090 Laptop 16 GB GPU on Tang; verified no GPU compute PIDs before training, took `flock` one-worker lock `~/experiments/sidra_m7/sidra_gpu_job.lock`. No other task terminated/overwritten. Weights and original benchmark examples remain solely on Tang.
- Original 1,024 HotpotQA TRAIN questions SHA-256 `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`. Frozen public DEV manifest SHA `3f0c494cca8b44cfef90c78d4d5b1fbb4158dc7387ed0f24316a0a265471d947`. Never read sealed final-B160.

## 1. What was changed and why

Prior `stage_credit_v1` assigns one normalized *end-of-episode evidence recall* advantage to every SEARCH and READ action in the trajectory. Thus an irrelevant late READ can receive the same positive credit as an earlier useful READ.

New opt-in `--credit-mode step_evidence_delta_v2` (default `episode_total` unchanged):

- After **actual SearchAgent** tool execution, for each SEARCH credit *newly surfaced original TRAIN gold supporting document IDs* in returned search hits; for each READ credit *newly actually read original TRAIN gold supporting IDs*. Each ID can earn first-discovery/first-read credit only once per episode.
- FINISH scores `0.4 * original TRAIN answer EM + 0.6 * strict correct-answer-and-evidence outcome`. A wrong but plausible answer **never** receives a correctness reward.
- Normalize each decision credit across independently model-sampled episodes **of the same TRAIN question, same action kind, same within-kind ordinal**. Never compare first READ against third READ across identical episodes. A singleton decision position has zero advantage; a constant group yields no imaginary gradients.
- Preserve original real Search/Read/Finish tool budgets (max 5 tool calls/6 decisions), v3 evidence-first action gates, no DEV/test/Oracle candidate injection, and **exactly one assistant ACTION token loss label per credited action, zero tool-observation label tokens**.
- Strictly reject missing/misaligned logged tool observations, unsupported TRAIN provenance, greedy/off-policy actions, nonfinite credit or mismatched trace vs sampled action; none of this scorer is invoked while the model constructs prompts/allowed actions.

Files: `src/sidra/step_credit.py`, `scripts/train_finite_action_grpo.py`, `tests/test_step_evidence_credit.py`.

### Real bug discovered during CPU tests

The **first** design normalized all actions of one kind together, regardless of position; **two identical trajectories** then wrongly received positive/negative advantages because first SEARCH and second SEARCH had different raw first-discovery returns. A regression test failed and exposed **phantom policy gradients**. Fixed by centering within `(question, kind, ordinal)` and reran full CPU tests. Another test fixture originally shared the same mutable Python action object between trace and sampled catalog, so tampering with either changed both; fixture was corrected to use independent copies. Original failing CI runs were retained; final tests are green.

## 2. Real online two-update TRAIN experiment

Exact matched seed and task selection vs previously saved `stage_credit_v1`: Qwen3-0.6B-Base + original TRAIN-only stop-SFT LoRA, 20 distinct teacher-reachable TRAIN groups × 4 fresh full trajectories per update, 2 updates, seed 81, LR `3e-6`, 2048 max context. The independent TRAIN teacher was used **only to select TRAIN questions**, never as policy input.

| Real measure | Step-delta first update | Step-delta second update |
| --- | ---: | ---: |
| Online model-generated Search/Read/Finish trajectories | 80 | 80 |
| TRAIN gold answer + evidence strict successes | **7/80** | **7/80** |
| Distinct groups with nonconstant overall TRAIN reward | 11/20 | 7/20 |
| Differentiated per-action advantages | **84** | **87** |
| Assistant ACTION-token training targets | 84 | 87 |
| Tool text label targets | **0** | **0** |
| Sampling/replay log-prob mismatch | **0.0** | **0.0** |
| Gradient norm | 7.2694 | 6.3995 |
| Real parameter maximum delta | 3.00072e-6 | 3.00352e-6 |

First iteration per-action credit: 24 SEARCH + 36 READ + 24 FINISH; second: 20 SEARCH + 47 READ + 20 FINISH. **Both were fresh on-policy after parameter update**, not repeated cached replay, while tool results were actually executed. Adapter SHA-256 changed from `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771` to **`cbb46d86bcfee85dcb78a2ff8007e92b96f33420c809eb2d685dd4af8ba6b018`**. Total training **64.422 seconds**.

No longer-running GPU process was interrupted. Complete checkpoint, raw TRAIN action samples, rewards and model/tool traces: Tang `~/experiments/sidra_m7/results/v3_STEPDELTA_s81_G20x4_U2_c05552da/`. Live training log: `~/experiments/sidra_m7/logs/v3_stepdelta_s81_g20x4_u2_c05552da.log`.

## 3. Matched three-way, NEVER gold-trained, independent TRAIN teacher holdout (58 tasks)

All three adapters were evaluated with the **same source SHA, frozen exact original TRAIN index/document SHA, same 58 question IDs, same v3 legal action policy, sample seed 63, tool budget, steps and context cap**. The holdout TRAIN IDs were excluded from both supervised teacher fit and online optimizer selection. This evaluation is **teacher-success-selected TRAIN**, not DEV, not external generalization.

| Outcome on untouched TRAIN teacher-selected holdout | Original stop-SFT | Prior stage-credit | New step-delta |
| --- | ---: | ---: | ---: |
| Answer EM | **6/58** | 6/58 | **4/58** |
| Strict original citation success | **5/58** | 5/58 | **4/58** |
| Original supporting-doc recall | 0.637931 | 0.689655 | 0.672414 |
| Avg actual tool calls | **4.448276** | 4.672414 | 4.620690 |

Paired original→step-delta, 5,000-bootstrap descriptions: EM delta **−2/58**, CI95 **[−0.103448,+0.034483]**; strict citation success delta **−1/58**, CI95 **[−0.086207,+0.034483]**; tool calls delta **+0.172414**, CI95 **[0,+0.362069]**. No positive performance gate.

Three independently regenerated eval directories: Tang `~/experiments/sidra_m7/results/v3_STEPDELTA_HOLDOUT58_{BASE,STAGE,STEP}_s63_c05552da/`. Verified source/label/checkpoint integrity audit: `~/experiments/sidra_m7/results/v3_STEPDELTA_HOLDOUT58_PAIR_AUDIT_c05552da.json`.

## 4. Fixed public DEV pilot32 before-vs-after (negative)

Same current v3 policy and seed63 for all three adapters, identical 32 original public DEV questions, source and budget; zero access to DEV gold while training:

| Outcome | Original stop-SFT | Prior stage-credit | New step-delta |
| --- | ---: | ---: | ---: |
| Answer EM | **1/32** | 1/32 | **1/32** |
| Strict original citation success | **1/32** | 1/32 | **1/32** |
| Supporting-doc recall | 0.53125 | 0.59375 | 0.578125 |
| Avg actual tool calls | **4.03125** | 4.125 | 4.125 |

Paired original→new answer EM and verified success delta **0**, each CI95 **[0,0]**; evidence recall delta +0.046875 with CI95 [0,+0.125], and tool calls delta +0.09375 with CI95 [0,+0.21875]. There is no answer/citation quality improvement; no escalation to sealed final B160 or model promotion.

Paired audit: `~/experiments/sidra_m7/results/v3_STEPDELTA_DEV32_PAIR_AUDIT_c05552da.json`, `pilot_positive_gate=false`. All three evaluation directories: `~/experiments/sidra_m7/results/v3_STEPDELTA_DEV32_{BASE,STAGE,STEP}_s63_c05552da/`.

## 5. Reproducible original TRAIN model-choice bottleneck taxonomy

Additional CPU-only script `scripts/audit_train_finish_bottlenecks.py`, tests `tests/test_audit_train_finish_bottlenecks.py`, generated posthoc aggregate reports from BOTH actually trained 160-episode datasets. It validates original TRAIN dataset SHA and per-step sampled ACTION/real tool trace alignment; it never inserts original gold into model choices. **173 CPU tests passed at pinned audit SHA `46edc7a8ca54cdc633fb9b0aacd7b2b8d77d789d`.**

| Actual TRAIN episode diagnostic | Prior stage-credit | New step-delta |
| --- | ---: | ---: |
| Real episodes | 160 | 160 |
| Ever actually READ **all** original TRAIN gold supporting docs | **52** | **52** |
| Original correct answer present among legal FINISH actions somewhere | 78 | 78 |
| Fully cited correct FINISH available along the sampled path | **50** | **50** |
| Policy actually chose correct-answer FINISH at some point | 19 | 21 |
| Full strict original gold + citation success | **13** | **14** |

**Exclusive cause assignment, new step-delta:** 108/160 lacked a complete set of actually READ original supporting docs; 2/160 had all gold support read but no gold-answer FINISH candidate; 36/160 had a fully cited correct FINISH offered **and ignored it**; 14/160 succeeded. These are on heavily teacher-reachable-selected TRAIN tasks; the taxonomy is along the model's *sampled state path*, not a free-choice upper bound and not heldout QA accuracy.

Original auditable reports on Tang:
- `~/experiments/sidra_m7/results/v3_STAGECREDIT_s81_G20x4_U2_1773b18e_FINISH_BOTTLENECK_AUDIT_46edc7a8.json`
- `~/experiments/sidra_m7/results/v3_STEPDELTA_s81_G20x4_U2_c05552da_FINISH_BOTTLENECK_AUDIT_46edc7a8.json`

## 6. Next falsifiable research goal and nonpromotion decision

**Block both experimental checkpoints from promotion**. Under matched conditions step-delta did not improve correct answers or strict evidence, and it **worsened TRAIN teacher-heldout results**. The remaining high-priority bottlenecks are now directly quantified: actual second-hop evidence READ coverage (**108/160** incomplete) and model FINISH selection (**36/160** missed a correct fully cited available option).

Next development should stay TRAIN-only: improve model-visible **bridge query and relevant READ ranking** without Oracle candidates or DEV label feedback, and independently improve FINISH preference ranking of already legal, fully observed candidate actions using TRAIN-only trusted trajectories. Require new model-sampled full tool episodes after every update, multiple seeds, same-policy disjoint TRAIN holdout followed by preregistered frozen DEV; never use final-B160 as an iterative training/dev gate. Do not represent oracle teacher path existence, a changed LoRA hash, a one-step gradient or a small selected TRAIN diagnostic as Agentic GRPO generalization or multi-generation RSI.
