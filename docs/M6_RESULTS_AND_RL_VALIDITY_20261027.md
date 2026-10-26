# M6 research execution — second-hop retrieval and TRAIN-only group-relative policy pilot

**Date:** 2026-10-27. This document reports actual Tang/GitHub evidence; it does not assert completion of online Agentic GRPO or RSI.

## E-016: Second-hop entity retrieval, DEV pilot32 only

Method: a deterministic gold-blind `BridgeQueryPlanner` extracts named-entity candidates from the question and the first actually READ page (or from a controlled top-1 document in the retrieval-only probe). It never examines the gold answer/supporting titles to choose a query; the query is executed through the normal `SearchAgent.search` budget.

- Retrieval-only pilot32, first BM25 Top3: both supporting passages **14/32**, mean support recall **0.6875**.
- After extra second-hop query Top3: both passages **22/32**, recall **0.8125**. This condition used **two searches vs one**, so this does not establish a more efficient search.
- Real Qwen3-0.6B stop-curriculum LoRA + dynamic tool IDs + grounded guard on same DEV pilot32: **bridge** answer EM **3/32**, strict verified **2/32**, mean citation recall **0.578125**, tools **4.0** vs the **non-bridge** baseline answer EM **6/32**, verified **2/32**, recall **0.515625**, tools **3.1875**.
- **Decision:** do **not** enable the bridge wrapper by default. Extra retrieval availability did not translate to answer success, and the extra search increased tool use. Preserve the negative result; do not evaluate sealed final-B160 to tune this pilot.

Source code: `src/sidra/bridge.py`, `src/sidra/bridge_policy.py`; pilot `scripts/probe_bridge_queries.py`. Tests include no-gold usage and tool budget. Actual bridge model run: GitHub SHA `ad5636cd5b60cb168459424129f044e62820d579`, 86 CPU tests passed, Tang `~/experiments/sidra_m6/results/bridge_stop_pilot32`.

## E-017: REAL TRAIN-only multi-turn policy rollouts and reward variance

Model: checksum-verified Qwen3-0.6B-Base + real M5 stop LoRA adapter `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`; vLLM tool-action schema and dynamic READ/CITE ID whitelist; SearchAgent executes actual BM25 `search/read/finish`. Gold used only by TRAIN reward after the episode, not by the model.

- **Train preflight A:** 12 distinct TRAIN questions × 4 independent sampled episodes, temp0.75, top_p0.9, seed43 → **48 real episodes**. **5/12 groups** showed reward variation, mean reward **0.12**, 1/48 fully correct+verified, mean READ support recall **0.416667**. 20 abstain, 2 errors, 26 finish. This is *not* a policy update.
- **Train preflight B:** 8 questions × 4 episodes, temp0.85 → **32 episodes**. **2/8 groups** with reward variation, mean reward **0.10**, **0/32 strict successes**, mean READ recall **0.421875**. No updates. All actual assistant action completion strings and corresponding tool-context prompt messages retained on Tang for masked-gradient research.
- Source files: `scripts/sample_train_rollouts.py`, `src/sidra/train_reward.py`. Rollout artifacts on Tang under `~/experiments/sidra_m6/results/real_train_rollouts12x4_stop/` and `train_rollout_io8x4/`. No dev, eval-A or final-B gold labels used in rewards.
- **Signal warning:** reward is mostly constant/low; mixed groups are only 42%/25%, and actual task success is rare. More relevant TRAIN-only task sampling and multi-hop read planning are needed before large-scale GRPO.

## E-018: One ACTUAL gradient step, GRPO-style clipped policy replay (NOT full online GRPO)

- Source TRAIN-only preflight B, eight 4-episode groups (two variable); exact recorded model completions and prompts; no teacher/oracle actions substituted into RL optimization.
- Algorithm: group-normalized terminal reward advantage broadcast to sampled assistant ACTION tokens; re-score under initial frozen Qwen3-0.6B + stop LoRA reference, then one LoRA AdamW update with a token-level clipped surrogate (epsilon0.2, lr8e-6). **Tool observation tokens are context-only, no gradient targets**.
- Verified hardware run: GitHub code SHA `35186ff3dfb3807023335f6452ba0ddb5e7ecb36`; **91 CPU tests passed**; 24 action strings and 840 assistant tokens participated in backpropagation; tool-output tokens used as labels **zero**. Gradient norm **1.11392**; maximum trainable-parameter change **8.00006e-6**; one optimizer step; peak allocated CUDA **5,886,943,232 bytes**.
- Input stop LoRA SHA256 `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771` → new saved LoRA SHA256 **`6669d104a420811ce68d11a7c8c94961f63df9b17a6618d463a4f8a766cb98e5`**. Real optimizer step/weight change are verified by code and checkpoint integrity.
- **CRITICAL VALIDITY LIMITATION:** vLLM uses JSON-Schema *constrained decoding*, but the replay surrogate computes *unconstrained* Transformers token likelihoods. The behavioral policy distribution and training probability ratios are not exactly matched. This is therefore a **biased group-relative clipped REPLAY pilot**, NOT standard on-policy full Agentic GRPO. The single weight update itself does not prove reward improvement or the full algorithm.
- New model must be evaluated against the starting stop LoRA with **identical prompt, tools, limits and decoded schema**; pilot comparison launched on Tang. No final-B evaluation permitted while research design is changing.

## M6 stopping gates

1. Frozen QA comparison demonstrates strict verified + correct answer gains, not only retrieval support exposure.
2. Achieve nondegenerate TRAIN policy-rollout reward groups, and implement *on-policy* multi-turn GRPO with valid constrained-action old/new log-probabilities or an official TRL/verl agent environment.
3. Evaluate multi-seed variation, cost, adversarial citations and original HotpotQA sentence-level supporting facts (current verifier is passage-level).
4. Only after a valid online update: Challenger → Solver → Verifier → Trainer multi-generation model revisions, and one sealed final-B160 evaluation per pre-registered protocol. Do not present this replay pilot as RSI.


## E-019: Matched pretrained stop LoRA vs ONE replay-updated LoRA (2026-10-27)

The train-only one-step group-relative replay produced a genuinely different adapter (SHA256 `6669d104a420811ce68d11a7c8c94961f63df9b17a6618d463a4f8a766cb98e5`), but model parameters changing is NOT an agent performance result. We therefore served both starting stop LoRA and replay LoRA via the *same* vLLM 0.31.0 process on Tang (Qwen3-0.6B-Base; non-thinking mode; greedy; JSON-schema constrained actions with dynamic document ID whitelists; identical `GroundedPolicy(min_reads=2)`, BM25 Top3, max steps6, tools5).

- **Evaluation dataset:** 32 frozen HotpotQA **DEV pilot** questions only; code snapshot `35186ff3dfb3807023335f6452ba0ddb5e7ecb36`. `final-B160` remains sealed; no post-replay tuning on eval-A64 was attempted.
- **Before replay (stop LoRA):** answer EM **6/32**, strictly verified **2/32**, cited gold-support recall **0.515625**, model calls **4.1875**/question, tool calls **3.1875**, latency **4.0374s**/question, all 32 normal Finish.
- **After replay (new LoRA):** answer EM **6/32**, strictly verified **2/32**, recall **0.515625**, model calls **4.21875**, tool calls **3.21875**, latency **4.0298s**/question, all 32 Finish.
- **Paired 5,000-bootstrap:** both strict verified and answer EM delta **exactly zero**, intervals [0,0]; tool-call delta +0.03125 (interval [−0.0625,0.15625]); latency −0.0076s (interval [−0.0642,0.0353]). **No observed task-level improvement.**
- **Raw immutable outputs:** Tang `~/experiments/sidra_m6/results/m6_pilot32_stop/`, `m6_pilot32_replay1/`, and `paired_replay1_vs_stop_pilot32.json`.
- **Decision:** reject this *one-step biased replay* as evidence of GRPO benefit; preserve the model, code and negative result but do NOT promote the new LoRA to default or test it against the sealed final set. Prioritize a valid on-policy constrained-action tool rollout trainer and more informative TRAIN-only reward sampling.

**Interpretation:** the concrete next objective is not merely more optimizer steps; it is correct masked on-policy action likelihood under the same behavior distribution, enough variable-reward groups and a nonnegative independent result. No RSI improvement is claimed.
