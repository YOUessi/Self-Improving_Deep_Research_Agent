# M7-C E-028 — Independent 72-question REAL TRAIN Solver cross-cohort audit (2026-10-27)

**Decision: PASS E-027 replay + TRAIN-only evidence diagnosis; BLOCK Trainer promotion and RSI success claims.**

The E-026 frozen original HotpotQA TRAIN task bank has three independent *cohort selection* partitions (43/44/45), each with 8 Direct, 8 Bridge and 8 Hard tasks. In E-027 a **real Qwen3-0.6B Base + M5 STOP LoRA** Solver made live vLLM JSON-schema-constrained `SearchAgent` Search/Read/Finish decisions on all 72 questions. These are not three separately trained model generations. The code developed and executed here independently re-opened **all three complete saved real-model runs**, re-created the question-specific document projection from the canonical original TRAIN corpus, replayed BM25 tool actions, action I/O, prompt observation history, sampling seeds and reported token counts, and scored original TRAIN answers/support **only in the separate CPU Verifier**.

## Evidence and preflight

- Work branch: `feat/m7-rsi-multi-generation`, exclusive of online GRPO; PR #11 remains Draft.
- Prior E-027 HEAD during initial read-only GitHub preflight: `8360bc2500328bb245903781f0657589021244e6`, successful push/PR CI.
- Source original immutable E-026 campaign commit `a1c649c5793aeb371f2ce76a756d36aacc9f7c12`. The 72 model-only prompts contain opaque challenge IDs and question text; the corresponding original HotpotQA answer/support only exists in separate trusted TRAIN data.
- Fresh **independent E-027 audit** at `8360bc25`: rechecked all three real cohorts (01, 02, 03); **72/72** complete tool transcript + model action/seed/retrieval replay accepted.
- Actual Tang preflight: RTX 4090 Laptop GPU **9,821 MiB used / 6,099 MiB free**, serving PID 86713 on port 19082. No new GPU experiment was started and no existing vLLM or training process was interrupted. This E-028 run was CPU-only.
- The declared source M5 STOP adapter was separately checked via SHA256 of the actual Tang weight file: `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`. **This file digest matches the run manifest; it is NOT cryptographic proof that the vLLM server used those weights for every historical response.** Independent model serving origin remains unproven.
- All original Qwen3-8B *invented* Challenger labels from E-023/E-025 remain quarantined. This E-028 analysis uses only established original TRAIN labels. There is no DEV / eval-A / sealed final-B gold access.

## Result from real 72 Solver episodes

All runs were real model/tool traces produced in E-027 and not unit-test stubs; E-028 independently validated the original saved artifacts. Decoder, checkpoints, model identifier, tool config and temperature/top-p were consistent across cohorts.

| Original TRAIN cohort | Questions | Sampling seed | Answer EM | Strict correct answer + cited gold support | Finished | Mean original gold-support READ recall | Model calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 24 | 43 | 9/24 | 0/24 | 16 | 0.270833 | 73 |
| 02 | 24 | 44 | 3/24 | 0/24 | 20 | 0.333333 | 75 |
| 03 | 24 | 45 | 5/24 | 0/24 | 19 | 0.291667 | 74 |
| **Total** | **72** | **3 distinct sampling seeds, ONE model checkpoint** | **17/72 = 23.61%** | **0/72 = 0.00%** | **55/72** | **0.298611** | **222** |

All 72 episodes passed independent tool+model transcript integrity replay. Total legal-finish 55, abstain 16, error 1; non-error 71/72. The original TRAIN retrieval difficulty *proxy* aggregates to:

| Difficulty proxy | TRAIN questions | Answers exactly correct | Strict gold evidence success |
|---|---:|---:|---:|
| Direct: all gold support initially retrievable | 24 | 11 | 0 |
| Bridge: partial gold support initially retrievable | 24 | 5 | 0 |
| Hard: no gold support initially in Top-K | 24 | 1 | 0 |

**Key verified failure mode:** **17 correctly answered tasks still failed strict support citation verification.** Full necessary support is frequently not even READ (mean gold-support READ recall 0.299), so improving literal JSON validity or answer EM alone is insufficient. The exact difference `17 correct minus 0 strict` is a measured, reproducible *evidence bottleneck*, not evidence of post-training improvement. The above TRAIN metrics are diagnostics, not generalization.

## New independently tested engineering controls

Added `src/sidra/rsi_solver_campaign_scorecard.py`, `scripts/audit_rsi_solver_campaign.py`, and `tests/test_rsi_solver_campaign_scorecard.py`:

1. **Entire frozen campaign or refuse:** requires each pre-registered cohort, original TRAIN holdout checksum, unique TRAIN source task identities and disjoint cohorts; partial reports are not silently accepted.
2. **Rerun verifier:** revalidates full E-027 saved model I/O, replayed search and read results, read-before-cite, original TRAIN-only reward and question/corpus SHA for each cohort. Requires exact frozen cohort-to-seed association.
3. **Cross-cohort compatibility:** requires identical model ID, declared checkpoint SHA, constrained decoder mode, temperature/top-p and tool config, plus distinct independent sampling seeds and unique saved trace/model-I/O hashes.
4. **Aggregate-only public output:** per-cohort and per-tier counts, not per-question hidden labels or a new `Trainer` example file.
5. **Fail closed against false research claims:** requires zero optimizer updates in these source runs, retains model-origin attestation false and blocks GRPO promotion. The JSON code explicitly reports `real_model_generations=0` and `eligible_for_online_GRPO_training_from_these_files=false`.

Validation: `PYTHONPATH=src python3 -m pytest -q` returned **208 passed** at GitHub source `69098758c2dfecfd4230021ae1b3a61b7ef1fb70`. Regression includes 16 malicious inconsistency mutations (model/decoder/seed/checkpoint/forged I/O/gold in Solver/reused transcript/false optimizer claim), sealed final-B path refusal, and create-only output refusal. The real 72-example scorecard completed successfully on Tang and is retained as:

- `~/experiments/sidra_rsi_m7/reports/E028_real72_independent_campaign_scorecard_69098758.json`
- `~/experiments/sidra_rsi_m7/reports/E027_current836_cohort_{01,02,03}_independent_audit.json`
- `~/experiments/sidra_rsi_m7/logs/E028_CPU_69098758.log`
- `~/experiments/sidra_rsi_m7/logs/E028_real72_campaign_scorecard_69098758.log`
- Immutable code snapshot `~/experiments/sidra_rsi_m7/source_69098758c2dfecfd4230021ae1b3a61b7ef1fb70/`

## Problems observed and solutions

**E028-P1, launcher test imports:** first attempt used `python3 -m pytest -q` without `PYTHONPATH=src` in the uninstalled checkout. It failed to import `sidra` during test collection, with no changes to data or experiment state. Reran with `PYTHONPATH=src`; all **189/189** existing E-027 tests passed. Subsequent E-028 full source used the same environment and passed 208/208.

**E028-P2, shell command line assembly:** initial campaign CLI attempted to break long arguments across shell lines without `\\` continuation. `argparse` correctly rejected the missing required `--pack/--run/--out`; no report was written. Retried as one shell command with the full list of three packs and three runs; fresh immutable output was created, all 72 passed integrity checks. Both failure logs preserved, not disguised as test results.

**E028-P3, evidence starvation:** answers without full cited TRAIN support: **17/72**; strict outcomes: **0/72**, despite nominal answer EM23.61%. Tool validity alone cannot repair the research objective. **OPEN.** The solution requires legitimate second-hop READ/citation planning based on *observed* documents and role-isolated source data; the verifier may use TRAIN labels offline but the Solver must never see them.

## Decision and concrete next gates

**BLOCK the 72 one-shot constrained-decoder runs as GRPO training episodes.** They contain only ONE trajectory per question, have no within-question multi-generation group-relative reward evidence, and their vLLM guided JSON token distributions do not have verified matching model log-probabilities. Model name, checkpoint SHA and saved API traces are not proof of independently attested serving origin. Running an optimizer on these saved examples and calling it on-policy multi-turn GRPO would be invalid.

Next:
1. Build an **observed-only, legal 2nd-READ planner** on original TRAIN tasks; test its impact against a same-tool-budget baseline on new held-out original TRAIN IDs, not on sealed final-B or by leaking labelled support IDs to Solver.
2. Reuse the online GRPO working line's **matched-behavior** policy-gradient implementation only after branch-level review; use **four fresh rollouts per original TRAIN question per checkpoint**, verify nonconstant within-question TRAIN reward, record sampled action logprobs/masks and real LoRA changes.
3. Run genuine consecutive checkpoint generations with deterministic recorded rollback and independent original TRAIN holdout / permitted public DEV pilot screening; compare across **three independent training seeds**, not just three fixed original cohort selection seeds.
4. The separate invented-task Challenger still needs **independent authenticated semantic relation labels**; it remains explicitly blocked, not treated as proof of RSI.

**0 new models trained in E-028, 0 new generated labels approved, 0 completed RSI checkpoint generations, 0 independent RL multi-seed gains demonstrated.** PR remains Draft. Refrain from sealed final-B160 until a pre-registered final evaluation exists.
