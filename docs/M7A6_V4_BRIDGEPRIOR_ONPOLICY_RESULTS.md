# M7-A6: observation-only bridge/read prior + two real on-policy updates (2026-10-27)

**Research status: training executed; independent QA promotion gate FAIL.** The small TRAIN-selected heldout58 improved EM and full evidence success from 3/58 to 6/58 under equal v4 policy but the paired 95% bootstrap interval **touches zero** and tool use increased. The fixed public DEV pilot32 remained **0/32 → 0/32** on both primary QA metrics. **Do not promote this adapter; do not claim GRPO QA/generalization gain or RSI.**

## 1. Change and probability match

At M7-A5, the original TRAIN-only diagnostic observed 108/160 episodes missing at least one actually READ gold support passage and 36/160 cases where a correct fully cited FINISH was available but ignored. A more detailed follow-up TRAIN-only diagnostic, conducted before this change, found among 160 existing trajectories: 80 lacked search access to all original gold support documents, while 35 had a gold document offered as a legal READ candidate but never selected. This motivated separately addressing bridge query generation and read choice.

Opt-in `--answer-proposer v4` is deliberately distinct from unchanged legacy/v2/v3. Implementation:
- `src/sidra/finite_action.py`: first-hop/second-hop query catalog includes top question-relevant bridge entities from text actually seen via READ, using the existing gold-blind `sidra.bridge.bridge_candidates`. Retains v3 two-READ-before-nonempty-FINISH rule, original max 9 legal one-token action IDs, maximum 5 real tools and 6 decisions, original lexical candidate answer proposals and read-before-cite constraint. No extra retrieval budget.
- `src/sidra/action_prior.py`: deterministic bounded **observation-only** per-legal-choice additive logit biases: favour readable question/observed-text-relevant search hits and genuine bridge READs, mildly downweight premature abstention while tools remain. Untrusted observation keys that look like evaluator gold are ignored. No global corpus or hidden support IDs are consulted; no model training labels in action prior.
- The exact sampled and differentiated distribution is
  `pi_theta(action_i|state) = softmax_i( LM_next_token_logit(theta, action_id_i) + observation_prior_bias_i )` over the *same* finite legal candidate action IDs, all other LM vocabulary masked out. **Both** `FiniteActionPolicy.decide()` and `logprob_of_action()` include the same saved prior. `scripts/train_finite_action_grpo.py` validates preupdate behavior-logprob equality and backpropagates only the selected single action-ID target; the question, Search and Read observations are *context only*, never target labels. Prior is part of policy, **not** a post-hoc reward.
- Still a **custom exact-categorical beta=0 online group-relative PPO/GRPO-style** policy-gradient pilot on Qwen3-0.6B, **not** native TRL `GRPOTrainer`, not unrestricted JSON tool generation, and not demonstrated RSI.

## 2. Code tests and audit controls

- Policy/trainer/test source pinned to GitHub **`61703c750f2e291213bda9095c1a1dc2ad98d34d`**; on Tang, **179/179 CPU tests passed** in `~/experiments/sidra_m7/logs/m7a6_cpu_61703c75.log`; GitHub Actions CI succeeded.
- Added v4 tests: no observed/unknown document IDs leak into legal READ, no factual FINISH with only one actual READ, no change in decisions after injecting keys labelled `gold_answer` or `supporting_doc_ids` into an unknown observation field; exact sampling-vs-optimizer log probability under nonzero priors, one action target, zero tool label targets, and rejection of tampered priors before optimizer mutation.
- Initial CPU test run had **178 passed, 1 failed** because test deliberately corrupted a bounded prior by `+3` past the legal `abs(bias)<=4` guard; code correctly rejected it with ValueError *before* reaching the intended distribution-mismatch assertion. Fixed fixture to `+1` inside the valid range and obtained 179 passes. Kept failure evidence.
- First GPU launch guard produced a **false positive**: a plain `ps -eo args|grep` matched the text of the Bash script itself. Corrected guard to inspect actual Python/vLLM process comms and checked `nvidia-smi --query-compute-apps` and `flock`. Confirmed no active model jobs, GPU free ~15.57 GiB before commencing; interrupted no other process.
- During paired audit, an offline tool `paired_compare.py` CLI rejected `--seed/--bootstrap` arguments even though the callable supports both. Used its Python function directly with **seed 817 and 5,000 bootstrap draws**, retaining the first failed audit command. The DEV-only fail-closed audit CLI ran normally.

## 3. Executed online TRAIN pilot on Tang

Original HotpotQA TRAIN-only converted questions SHA256 `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`; preexisting frozen *public DEV ID manifest* checksum `3f0c494cca8b44cfef90c78d4d5b1fbb4158dc7387ed0f24316a0a265471d947`. TRAIN-only offline oracle-success teacher selected **40 TRAIN task IDs** from its 229 TRAIN-fit reachability subset; **58 teacher-heldout TRAIN IDs were disjoint**. Oracle gold was used only by that offline training-task selector and by the post-rollout **TRAIN-only** reward; never by model-visible question, tool history, candidate proposer, prior or sampling. **Final sealed B160 not touched**.

Parent: Qwen3-0.6B-Base + original TRAIN-only stop SFT LoRA; two independent online updates, seed **131**, each **20 distinct TRAIN groups × 4 independently model-sampled real Search/Read/Finish tool trajectories = 80 real episodes**, fresh sampling from current LoRA after each update. LR `3e-6`, `episode_total` group-normalized reward, max 5 tools, max 6 decisions, 2048-token max input, same normalized v4 masked categorical distribution in behavior and optimizer.

| Actual online measurement | Update 0 | Update 1, fresh after first weight update |
| --- | ---: | ---: |
| Real multi-turn tool episodes | 80 | 80 |
| Reward-nonconstant TRAIN groups | **10/20** | **9/20** |
| Strict original TRAIN answer and citation success | **5/80** | **11/80** |
| Gradient-contributing ACTION tokens | **210** | **206** |
| TOOL observation target tokens | **0** | **0** |
| Behavior/sample vs optimizer log-prob mismatch | **0.0** | **0.0** |
| Parameter max absolute delta | 3.00072e-6 | 3.00445e-6 |
| Gradient norm | 6.1093 | 4.8757 |
| TRAIN average reward | 0.180125 | 0.270250 |

**These two 80-episode batches contain different TRAIN questions and are not a before/after matched task-quality comparison.** The increased number of successes is not interpretable alone as policy improvement. Final adapter SHA256 changed from `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771` to `6d110e4cb022d8a6c5e3da01daed92edcd48beaf0fe7bf0d6add0b5a6846c0a1`. Wall time 102.232 sec; allocated CUDA peak 6,876,013,056 bytes.

Training outputs and weights retained **only on Tang**:
`~/experiments/sidra_m7/results/v4_BRIDGEPRIOR_s131_G20x4_U2_61703c75/`
Training log:
`~/experiments/sidra_m7/logs/v4_bridgeprior_s131_g20x4_u2_61703c75.log`.

## 4. True model A/B: identical v4 policy before vs after, seed 137

A separate GPU job ran **four actual model/tool evaluation runs** on frozen identical document indices, question IDs, model architecture, source commit, state-keyed stochastic action decoder, tool/step budgets and policy v4: old SFT adapter vs final online-updated adapter. All immutable manifests checked equal per arm and adapter weights SHA checked on Tang. Evaluation seed **137**; no training or model update on heldout/dev data.

| Metric | TRAIN-only teacher-heldout58 before | TRAIN-only heldout58 after | Public DEV pilot32 before | Public DEV pilot32 after |
| --- | ---: | ---: | ---: | ---: |
| Correct answer EM | 3/58 | **6/58** | **0/32** | **0/32** |
| Strict answer + original citations | 3/58 | **6/58** | **0/32** | **0/32** |
| Original support-doc recall | 0.758621 | 0.775862 | 0.578125 | **0.656250** |
| Mean actual tool calls | 4.603448 | **4.810345** | 4.812500 | 4.781250 |

- Heldout58 paired correct-answer and verified-success delta **+3/58 = +0.05172**, 5,000-resample bootstrap 95% CI **[0, +0.12069]**; interval includes 0. Mean tool calls **+0.20690**, CI95 **[+0.05172, +0.41379]**; more actual tools were needed. The 58 are teacher-success-selected TRAIN questions, **not** random original TRAIN, public DEV or external independent generalization.
- Public DEV pilot32 paired EM and strict citation delta **0.0**, CI95 **[0,0]**; evidence recall **+0.078125**, CI95 **[+0.015625,+0.15625]**, but all answers still failed complete verifier. Tool calls **−0.03125**, CI95 **[−0.21875,+0.18750]**.
- Result **fails the predeclared joint answer + strict evidence success promotion gate**. Stop-SFT remains baseline; V4 trained LoRA is quarantined for further analysis. No sealed final-B160 trial allowed.
- Paired holdout report `~/experiments/sidra_m7/results/V4A6_HOLDOUT58_PAIR_AUDIT_61703c75.json`; strict fail-closed DEV report `~/experiments/sidra_m7/results/V4A6_DEV32_PAIR_AUDIT_61703c75.json`, `pilot_positive_gate=false`. Four raw evaluation dirs `~/experiments/sidra_m7/results/V4A6_{HOLDOUT,DEV}_{BASE,POST}_seed137_61703c75/`.

## 5. Independent TRAIN-only bridge reachability and bottleneck audits

The opt-in CPU teacher v4 comparison was run on the **same original 1,024 TRAIN questions and byte-for-byte identical documents** after adding v4 to the teacher CLI in later GitHub SHA `4215c05e945ebdcd5e2743cbfc0c5b27c67a0975`. A **TRAIN-gold offline oracle** (not the model or GRPO) was able to execute strict-success trajectories for **300/1024 v4 vs 287/1024 v3** under the exact same ordinary BM25 tool/step budget. Those oracle path success fractions are a **candidate-catalog reachability diagnostic only**, not nonoracle QA performance, not a learned policy, and cannot explain an independent answer gain. The v4 teacher files remain under `~/experiments/sidra_m7/data/v4_bridgeprior_teacher1024_s43_4215c05e/`.

Separately, the same original TRAIN-only posthoc failure classifier on the 160 actually sampled v4 online episodes found:
- **97/160** did not READ the complete set of original support documents;
- **42/160** did READ full support and had a fully cited correct FINISH available but never selected;
- **5/160** had all originals read but the offered correct FINISH lacked complete citations;
- **16/160** finished with full original verified answer/citation success.

Because the earlier v3 traces used a **different TRAIN task selection and RNG seed**, comparison against A5's 108/160 evidence misses and 14/160 successes **is not causal**. The current v4 observations nonetheless suggest next bottleneck is accurate FINISH action selection after actual support READ, with better mixed Search/Read/Finish credit but *without* leaked TRAIN gold in the policy.

Posthoc v4 original-TRAIN audit: `~/experiments/sidra_m7/results/v4_BRIDGEPRIOR_s131_G20x4_U2_61703c75_FINISH_TRAIN_DIAG_61703c75.json`.

## 6. Next falsifiable stage and governance

1. **Do not run more updates merely to claim improvement.** Current output demonstrates a real on-policy training mechanism, promising but weak TRAIN-selected QA movement, and zero public DEV answer generalization.
2. Develop TRAIN-only action-stage preference / FINISH reranker tested on a separate, fresh TRAIN holdout that was not used for either TRAIN teacher, online GRPO or repeatedly inspected pilot. Make a multi-seed comparison against baseline at equal v4 selection policy; retain non-improving checkpoints.
3. In any alternative decoder such as vLLM grammar or native TRL tool API, verify *actual behavior distribution* and the policy-gradient log probabilities share the same normalization. Do not silently reintroduce constrained JSON sampling/raw LM mismatch.
4. Never put hidden gold into model prompts, candidate catalogs, logit prior, normal search tool, or proposed unseen-document URLs. Never open the sealed final-B160 for tuning; never commit weight files, source benchmark data, local private paths beyond generic experiment manifests, or secrets. No GPU preemption.

**Summary decision: V4 bridge/query and observation prior have been implemented and trained with honest, probability-matched online rollouts; evaluation shows better evidence reading but no externally credible answer/citation gain. Keep PR draft and model unpromoted.**


### Reproducible CLI audit follow-up (post-experiment)

The paired comparison helper originally exposed `seed` and `bootstrap` only in its Python function, not the terminal CLI; the attempted CLI audit failed before writing a TRAIN-heldout report (DEV audit was unaffected). Patched `scripts/paired_compare.py` so `--seed` and `--bootstrap` are explicit, added a CLI regression test, and ran the entire pinned test suite on Tang at **`eb6bceb961df226027d87c59f8e865643513790f`: 180/180 passed** (`~/experiments/sidra_m7/logs/m7a6_cpu_eb6bceb9.log`). The new code changes **only offline audit CLI and tests**; it does not alter the already evaluated `61703c75` model policy, training checkpoints, or any benchmark results.
