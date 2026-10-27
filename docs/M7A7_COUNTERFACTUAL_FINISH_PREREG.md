# M7-A7 pre-registration: FINISH-choice supervision on real visited states

Created before A7 GPU training / model evaluation, 2026-10-27. No sealed HotpotQA final-B160 data or labels may be read, selected, tuned, or opened.

## Mechanistic hypothesis and observed TRAIN-only diagnostic

The completed M7-A6 online V4 dataset (160 freshly sampled TRAIN episodes, not an independent evaluation set) contains **113 real visited decision states in 58 episodes where an actual legal FINISH with the correct original TRAIN answer, all originally supporting citations, and substring support in actually READ text was available**. Among those 58 episodes, 42 missed complete success despite having a suitable terminal choice. The causal hypothesis is an erroneous action ranking after sufficient evidence, not necessarily failure to provide an answer candidate.

A7 introduces **optional post-hoc TRAIN-gold supervised auxiliary** on these states; it is NOT a new GRPO reward or a proof of RL gains. The student legal-action catalog, v4 observation-only logits bias, step/tool budgets, and actual Search/Read/Finish tool rollout remain unchanged. Only after a completed TRAIN episode does the offline verifier label a correct pre-existing FINISH among the already-seen legal choices. Requires all original gold support document IDs actually READ **before** the decision, citations subset of those READ IDs, exact normalized TRAIN answer, and that answer visible in the cited READ text. Never supplies a new action, synthetic/tool-completed state, hidden gold in prompts, DEV labels, or a privileged policy-time feature.

Sampling distribution and the online group-relative loss still use the exact same legal one-token action-ID Softmax + nonoracle V4 logit biases. The optional auxiliary is `-λ log Σ_{a∈C_gold_legal} πθ(a|real state)`, differentiated on the same **legal action IDs** in the *same single optimizer step* as group-relative on-policy policy-gradient. Tool returns are context-only, never labels. **Any gain would belong to the combined on-policy+supervised hybrid, not pure GRPO.**

## Frozen comparison protocol (no outcome-based checkpoint picking)

- Source: tagged code SHA recorded per run, one unmodified original Qwen3-0.6B-Base + TRAIN-only stop-SFT LoRA; no vLLM JSON grammar model.
- HotpotQA TRAIN-only curriculum: same original frozen verified V4 teacher-success TRAIN subset; use original TRAIN gold only for task selection, private terminal reward, and optional optimizer-side auxiliary.
- First pilot: random seed **153**, **20 distinct TRAIN groups ×4 independent live model-sampled tool episodes ×2 online optimizer updates** for each of (A) plain matched-categorical group-relative GRPO, (B) same online GRPO **plus λ=0.2 post-hoc TRAIN-only FINISH supervised auxiliary**; `episode_total` credit and `lr=3e-6`, max 5 Search/Read tools + 6 decisions, V4 answer proposer. Both begin from identical frozen SFT checkpoint. Each optimizer step requires nonconstant actual TRAIN outcome reward, nonzero gradient, original old-policy logprob match tolerance < 1e-3, real adapter SHA change; train trajectories/evidence retained privately on Tang.
- Paired evaluation: original SFT, pure GRPO control, and hybrid, identical V4 action policy, seed **151**, frozen public DEV pilot32 (previously viewed; only a development smoke gate) and teacher-heldout TRAIN58 (already visited; not independent external generalization). Same dataset bytes / selected IDs / SEARCH/READ/FINISH budgets / decoder. Evaluate EM, strict verified citation success, supporting-doc recall, tool count, successful completed steps, selected FINISH distribution, and paired bootstrap CIs (e.g. 5,000 draws, seed 853).
- The exact original full DEV pilot32 and TRAIN58 must not be used for supervision or model selection inside training. A second seed **157** paired evaluation is planned if resources allow; no switching evaluation seed in response to an adverse result.
- Promotion: **NO** checkpoint accepted from TRAIN teacher fit/GRPO reward alone. Requires positive independently replicated QA EM **and** full citation success, no unacceptable tool-cost increase, and a paired confidence assessment excluding zero on truly independent data. Even if pilot32 improves, repeated inspection of the pilot prohibits calling it final generalization. Keep original SFT model as deployed baseline.
- If hybrid improves but control does not, describe it as TRAIN-gold-assisted imitation atop online sampling, **not** pure GRPO or RSI. If it regresses on evidence/answer, document it and keep models quarantined.

## Safety and reproducibility

Check live Tang GPU processes before any train/eval; hold a non-preemptive exclusive project GPU lock, avoid GPU conflicts with any workline, never remove another process. GitHub only gets code, tests, protocol, summarized hashes and negative results, never checkpoints, full benchmark records, private paths/keys or original final-B160. Every run stores independent source SHA, CLI, stdout/stderr log, adapter SHA, frozen manifest hashes and evaluation per-question rows **only on Tang**.

The preliminary A6 TRAIN-only diagnostic was **113 eligible states in 58/160 episodes**, not 113 teacher-positive model completions and not a DEV or model improvement claim.
