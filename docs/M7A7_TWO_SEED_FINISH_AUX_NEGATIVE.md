# M7-A7 — controlled FINISH preference vs pure on-policy GRPO (negative DEV gate)

**Executed:** 2026-10-27 on Tang RTX 4090 Laptop 16GiB, local Qwen3-0.6B-Base + frozen TRAIN-only stop-SFT LoRA. All benchmark records and model weights remain on Tang. Final sealed HotpotQA B160 **NOT READ**. This stage did NOT establish GRPO gains or RSI.

## 1. Intervention and experimental classification

Pre-registered design: [M7A7_COUNTERFACTUAL_FINISH_PREREG.md](M7A7_COUNTERFACTUAL_FINISH_PREREG.md). Starting adapter SHA256 `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`.

- **Control:** real on-policy multi-turn Search/Read/Finish trajectories, one-token finite-action categorical sampling and the exact *same masked-logit softmax* in the optimizer; TRAIN-only outcome/trace group reward, no supervisory preference auxiliary.
- **Hybrid:** identical online GRPO plus **lambda=0.2 TRAIN-gold-supervised counterfactual FINISH-choice negative log-likelihood** only at real tool states whose original TRAIN support documents were all previously READ and a correct fully-cited FINISH action was in the *actual observed legal catalog*. This second gradient is **supervision, not pure GRPO**; legal candidate generation remains gold-blind.
- Both: source SHA `d0589ad88facab67537fb81179b17e1f8c83d1c8`, training RNG seed 153, two updates, 20 distinct TRAIN task groups x four genuinely model-sampled multi-turn episodes per update (= 160 episodes **per arm**), same TRAIN question ids, original agent budget of five tool calls/six decisions, Qwen3-0.6B, lr 3e-6. No vLLM grammar JSON mismatch.

| Training audit | Control update 1 / 2 | Hybrid update 1 / 2 |
| --- | --- | --- |
| Real model/tool episodes | 80 / 80 | 80 / 80 |
| Groups with nonconstant TRAIN reward | 8/20 / 6/20 | 8/20 / 7/20 |
| Strict TRAIN trajectory success | 5/80 / 8/80 | 5/80 / 5/80 |
| Assistant ACTION token targets | 174 / 131 | 174 / 139 |
| Tool observation token targets | 0 / 0 | 0 / 0 |
| Behavior vs training log-probability error | 0.0 / 0.0 | 0.0 / 0.0 |
| Actual optimizer parameter update | yes / yes | yes / yes |
| Counterfactual TRAIN-only FINISH states | 0 / 0 | 63 / 43 |

Control final adapter SHA256 `5a7305ed4f7852163f1040fddc15486fb97b6e45216944f48c6ba1c6ac43b083`; hybrid SHA256 `f36c8eb3cbeaf90bfea9b0adb6869ffcdac34304ddc150a031b845e43697ea91`, both different from parent. These SHA changes and gradient magnitudes establish **genuine updates**, not QA gains.

## 2. Actual paired 3-arm evaluation with 2 predeclared RNG seeds

Evaluation source SHA `d0589ad88facab67537fb81179b17e1f8c83d1c8`, frozen public DEV pilot32 (not final or independent generalization) and previously examined oracle-teacher-selected TRAIN heldout58. Identical documents, IDs, v4 legal action policy, categorical sampler, state-keyed RNG, tool/step caps and parent/child adapters within each seed. Model evaluations **sampled 157 and 151**, as prespecified; no training or tuning during these runs.

**Public DEV pilot32** (EM = correct answer, strict = complete answer plus original support citations):

| Stochastic evaluation seed | Starting SFT | Pure GRPO control | Online GRPO + TRAIN-supervised FINISH hybrid |
| --- | --- | --- | --- |
| 151: EM | 4/32 | 4/32 | **3/32** |
| 151: strict support success | 3/32 | 3/32 | **2/32** |
| 157: EM | 2/32 | 2/32 | 2/32 |
| 157: strict support success | 1/32 | 1/32 | 1/32 |
| Average across 2 seed-passes: original support recall | 0.65625 | 0.65625 | **0.59375** |
| Average across 2 seed-passes: tool calls/question | 4.640625 | 4.671875 | 4.578125 |

Each seed reuses **the same 32 questions**. Correct analysis: **32 independent question clusters and 64 stochastic episodes per arm**, NOT 64 independent heldout cases. New `scripts/audit_multi_seed_dev.py` performs fail-closed code/data/adapter/policy checks and resamples question IDs **with both stochastic seed outcomes tied together**. Source/test SHA `5b06a39697a2a41797c20b618c344d3f0ba5f082`.

Question-cluster bootstrap, 5,000 draws, RNG seed 853, hybrid minus original SFT:
- EM: **-0.015625**, CI95 **[-0.046875, 0]**.
- Strict original citation success: **-0.015625**, CI95 **[-0.046875, 0]**.
- Supporting document recall: **-0.0625**, CI95 **[-0.117188, -0.015625]**.
- Tool calls: **-0.0625**, CI95 **[-0.1875, +0.0625]**.
- Pure GRPO control: EM delta **0**; strict delta **0** (both CI95 [0,0]); supporting doc recall delta **0**, CI95 [-0.023438,+0.023438].

Descriptive CIs **do not constitute independent statistical validation** because this public pilot has been inspected repeatedly. These results provide no basis to promote either checkpoint.

**TRAIN teacher-selected heldout58** (biased towards tasks already reachable by an offline TRAIN oracle, disjoint from fit tasks; this is neither unbiased TEST nor public DEV):

| Eval seed | SFT EM / strict | Pure GRPO EM / strict | Hybrid EM / strict |
| --- | --- | --- | --- |
| 151 | 6/58 / 6/58 | 6/58 / 6/58 | 9/58 / 8/58 |
| 157 | 7/58 / 6/58 | 7/58 / 6/58 | 9/58 / 7/58 |

Hybrid-versus-SFT per-seed paired 95% bootstrap includes 0 for both primary outcomes: seed 151 EM delta +3/58 CI [-1/58, +7/58], strict +2/58 CI [-2/58, +6/58]; seed 157 EM delta +2/58 CI [0, +5/58], strict +1/58 CI [0,+3/58]. **This biased TRAIN-holdout signal is not replicated on public DEV** and must not be called robust improvement.

## 3. Observable error attribution — evidence lost after supervised FINISH preference

Summary of actual model traces:

| Set / seed | SFT mean READ calls | Control | Hybrid |
| --- | ---: | ---: | ---: |
| DEV32 / 151 | 2.8125 | 2.8125 | **2.6250** |
| DEV32 / 157 | 2.7188 | 2.7500 | **2.5625** |
| TRAIN-heldout58 / 151 | 2.7414 | 2.7759 | **2.6034** |
| TRAIN-heldout58 / 157 | 2.7414 | 2.7414 | **2.6034** |

The hybrid made fewer real READ tool calls in every comparison and the original-support retrieval score fell on DEV. This is **consistent with** its supervised terminal-choice preference generalizing too broadly to earlier evidence-seeking states, but no causal isolation of that mechanism has yet been proven. Next experiment should explicitly test whether adding TRAIN-only supervision for legal supporting READ choices **before** complete evidence prevents premature FINISH without leaking labels into the runtime policy. That would also remain hybrid supervised RL, never pure GRPO.

## 4. Reproducibility and issues recorded

- On Tang, immutable source at `~/experiments/sidra_m7/source_d0589ad88facab67537fb81179b17e1f8c83d1c8/` trained the two adapters. All 12 (3 arm x 2 datasets x 2 seed) model evaluations: `~/experiments/sidra_m7/results/M7A7_{DEV,HOLD}_{BASE,CTRL,HYB}_s{151,157}_d0589ad8/`; logs same prefix in `~/experiments/sidra_m7/logs`.
- Initial seed-151 HOLD audit invoked a mistyped nonexistent `M7A7_CTRL...` model directory. It correctly failed before producing an audit; reran with real `M7A7_CONTROL...` checkpoint and passed the strict original TRAIN overlap/checkpoint checks. The first error log is retained.
- New multi-seed CI test originally asserted wrong error text for DEV-gold leakage, but production checks correctly rejected that case as `decoder/selection mismatch at trained_on_eval_gold`. Corrected **only the test expectation**; fresh pinned Tang source `5b06a39697a2a41797c20b618c344d3f0ba5f082`: **200/200 CPU tests passed**, GitHub Actions CI succeeded.
- DEV single-seed paired reports: `M7A7_PAIR_DEV_BASE_VS_{CTRL,HYB}_s{151,157}_*.json`; TRAIN-heldout audit `M7A7_PAIR_HOLD_BASE_VS_{CTRL,HYB}_s{151,157}_*.json`. Question-clustered report: `M7A7_TWO_SEED_QUESTION_CLUSTER_AUDIT_5b06a396.json`. The reports, raw question records, checkpoints and model weights stay on Tang; code and aggregate results only go to GitHub.
- GPU safety: second-seed evaluation started with a non-preemptive flock and no existing compute processes; completed successfully without terminating any other job. No sealed final-B160 inspection, no external test.

## 5. Gate decision

**FAIL: no verified joint EM/citation improvement on public DEV.** The supervised FINISH hybrid additionally shows a consistent reduction in READ calls and negative DEV support recall. Keep original stop-SFT checkpoint as baseline, preserve both experimental checkpoints for forensic comparison, leave PR draft. Do not deploy or use DEV scores as TRAIN optimization labels.

Next falsifiable study: TRAIN-only missing-support READ preference on genuine model-visited legal candidate states, combined with or separately from FINISH auxiliary, using a *new preregistered training seed*, explicit supervised-vs-online distinction and fresh TRAIN holdout not used in choosing this intervention; show action/gold isolation and matched normalizer tests before GPU usage. The public DEV32 may remain a sanity check but can no longer be treated as independent evidence.
