# M7-C E-030: prospective observed-only READ-before-FINISH policy guard (preregistration)

**Date:** 2026-10-27 (Tang +08:00). Written **before any E-030 GPU QA evaluation**. E-029 independently established **71/72 historical original TRAIN Solver episodes performed exactly one READ, 1/72 performed none, and 0/72 performed two**, so its preregistered same-second-READ shadow comparison had zero eligible slots. The next design MUST NOT misrepresent an inserted READ as a free, same-slot substitution.

## Intervention frozen before outcome labels

- A deterministic **model-only** policy wrapper is allowed to insert **at most one legal READ immediately after the first observed READ**, only if **at least two distinct unseen document IDs have already been returned by genuine SEARCH**, and the original remaining step/tool budget allows a READ followed by FINISH. Ranking is frozen in E-029's `choose_observed_second_read`: observed title/text/question/snippet and capped observed search scores; no TRAIN answer, gold supporting IDs, global unseen corpus, private task ID map or Verifier decision is an input.
- Extra READ **consumes one of the original five tool calls and one of the original six decision steps**. The model is shown the actual new READ observation on its subsequent decision; no forced answer/citation is generated. Never replay recorded post-intervention model actions as if they came from a new real model trajectory.
- The base Qwen3 model, LoRA, decoding, legal action ID constraints, SearchAgent, search Top3 and maximum 6 steps/5 tools must be identical in matched treatment and baseline. Model-only pack/test data must never carry original TRAIN labels into Solver. Offline Verifier alone may score original TRAIN answer and supporting-document recall.
- This guard is a **rule-based evidence-retrieval intervention**, neither an RL update nor self-improvement. It must not automatically become the default policy; first demonstrate prospective full-trajectory benefit with real paired model calls and frozen fresh TRAIN IDs.

## Predeclared evaluation gate (not run in E-030 engineering)

1. Construct a separate, **frozen and unused original TRAIN evaluation cohort**, conservatively excluding historical M5 SFT action-source question IDs, M6 rollouts, M7 real training, E-026/E-027 curriculum and holdout, E-029 human-adjudication holdout, plus all subsequently used TRAIN question IDs. Reject if fewer than the preregistered target exist; never substitute DEV/eval-A/final-B.
2. Same-model baseline vs wrapper with identical questions, checkpoint SHA, sampling seeds, original BM25 tools and hard budgets. Both sides must execute **live, genuine Search/Read/Finish**, not offline synthetic continuations. Run independent two-seed question-cluster bootstrap for original TRAIN answer EM, strict answer plus complete original support citations, original supporting-document READ recall, model/tool calls, premature abstention and overruns.
3. A positive result must improve strict correctly cited answers without unacceptable tool/cost regression and should replicate on multiple seeds and disjoint tasks. A positive READ recall alone is not enough for deployment; even strict gains are not proof of learning until genuine matched-behavior GRPO optimizer updates and successive checkpoint generations meet independent gates.
4. GPU check on Tang immediately before any real run; **do not preempt** the unrelated active vLLM server or online GRPO. No final-B160 access, no user data/model weights committed to GitHub.

## CPU engineering acceptance

The wrapper must pass deterministic SearchAgent integration tests with a fake policy (tests are **not** model results), anti-gold-tool-state tampering, no valid frontier, near-exhausted budget, one extra READ maximum, no duplicate read, original global tool budget and unchanged baseline behavior without a proposal. Public GitHub CI (Python 3.10/3.12) and Tang pinned SHA CPU suite required.

**This experiment is untested on real model QA as of preregistration; do not promote it, claim RSI success, or use earlier E-027 gold labels to tune its ranking.**
