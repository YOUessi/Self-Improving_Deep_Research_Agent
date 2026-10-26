# M7-A: Train-only on-policy multi-turn group-relative LoRA — first verified run (2026-10-27)

This stage targets **genuine on-policy** multi-turn search/read/finish training, not the biased M6 vLLM-constrained replay. It is a small custom **GRPO-style beta=0** clipped policy-gradient algorithm, not yet a full native TRL/verl Agentic GRPO experiment or statistically validated RSI.

## Probability distribution correction

- **Behavior model:** local Qwen3-0.6B-Base + verified TRAIN-only M5 stop-SFT LoRA.
- **Same-policy sampling:** every tool ACTION generated live by the *current* model using `torch.generate(do_sample=True, temperature=1, top_p=1, top_k=0, repetition_penalty=1)` with **NO grammar/logits masking**. Thus raw LM Softmax probabilities are the actual behavior distribution.
- **Learning:** after full SearchAgent tool rollouts, record each generated assistant action token sequence, compute old token log-probs from the same pre-update model, and backpropagate a group-normalized terminal TRAIN reward with a clipped PPO/GRPO-style objective. **System/question and retrieved tool text are context only**; all gradients target generated assistant action tokens, with old policy re-sampled after every update.
- **Tradeoff:** raw unconstrained 0.6B model emits many invalid JSON actions, so task reward is sparse. This is an *honest on-policy learning problem*, not a claim that guided decoding has been implemented with matching log-probs.
- **Research omissions:** beta=0 (no KL regularizer), one step in initial run, no multi-seed confidence bounds, no reference-policy scheduling, no TRL native environment_factory. These are still required for a rigorous large-scale Agentic GRPO conclusion.

## Executed on Tang — actual result

Source GitHub SHA `0b320e84da3c7106e307758c839e3919d1e8d875`; **98 CPU tests passed** before CUDA work. Actual TRAIN-only data under `~/experiments/sidra_m4/data/train_oracle1024_seed43/converted`, original frozen DEV manifest checked for disjoint question IDs.

Single online update:
- 6 TRAIN questions × 4 model-generated tool episodes = **24** real fresh complete trajectories.
- 6/6 groups had nonconstant reward (mostly tool-validity/partial credit), **0/24 complete answer+evidence successes**, mean total reward **0.058333**.
- Termination **16 error, 7 abstained, 1 finish**.
- **68 assistant action strings** comprising **2,705 generated action tokens** contributed to gradients; **0 tool text label tokens**.
- Group-relative clipped optimizer **1 actual AdamW step**, lr `1e-5`, grad norm `0.66439766`, max adapter parameter delta `1.00005418e-5`. GPU peak allocated **6,620,223,488 bytes**, wall time **65.51s**.
- Trained LoRA SHA256: original `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771` → new `776dbfbad97d21967fad2290c0326bde02a3788473eabe12eb1f5e0a67973c96`.
- Full original prompts/actions, TRAIN rewards, update reports and adapter retained on Tang `~/experiments/sidra_m7/results/onpolicy_grpo43_1step_6groups`. No weights or benchmark raw data committed to GitHub.

**No model quality gain is claimed.** Separate old/new equal-condition evaluation on frozen DEV pilot32 must verify whether action validity, correct answer EM, original sentence evidence and total tool budget improve. External datasets and sealed final-B160 remain unused.

## Next gates

1. **Complete same-runtime paired pilot32 evaluation**; control decoding mode separately: raw HF full-softmax vs a strict vLLM JSON action schema has different action output policies, though either can make a fair before/after comparison if held fixed.
2. Increase eligible TRAIN reward *variance* without using DEV gold: two-hop TRAIN sampling, label-safe partial reading and supporting sentence checks, task difficulty stratification, independent random seeds.
3. Integrate stateful tool environment in locally installed **TRL 1.14.1** (`GRPOTrainer(environment_factory=...)`) or compare rigorous custom on-policy masked loss against TRL implementation with matched log-probability policy, KL controls and step-by-step rewards.
4. After multiple on-policy policy improvements on independent QA and verified sentence-level evidence, advance toward Challenger–Solver–Verifier–Trainer repeated generations. This first one-step adapter is **not RSI**.


## Follow-up: actual same-decoder DEV pilot32 (negative result)

An independent current-task evaluator ran **both** original stop SFT LoRA and the new online-updated LoRA under identical local Transformers greedy `enable_thinking=False` decoding, `max_new_tokens128`, genuine BM25 tools, `max_steps6`, `max_tool_calls5`, and exactly the frozen 32 HotpotQA public DEV pilot questions. Dataset SHA and source commit were identical in both immutable manifests; the actual run used GitHub snapshot `0b320e84da3c7106e307758c839e3919d1e8d875`.

| Pilot32 outcome | Starting stop SFT | After one on-policy group-relative LoRA update |
| --- | ---: | ---: |
| Answer exact match | **0/32** | **0/32** |
| Strict full-citation success | **0/32** | **0/32** |
| Terminated with tool/action ERROR | 32/32 | 26/32 |
| Abstained | 0/32 | 6/32 |
| Average tool calls | 2.03125 | 2.0625 |
| Average model calls | 3.03125 | 3.0625 |
| Mean latency, seconds | 2.394875 | 2.282188 |

**Error taxonomy:** Original model: 31/32 errors were **citing UNREAD document IDs** and 1 malformed JSON. Updated: 25/32 errors were unread citations and 1 malformed JSON; in six cases the model instead abstained. This is a syntax/validity change, **not any successful answer/evidence generalization**. Under constrained JSON/DocID serving the same starting SFT has previously performed better, but the current raw full-softmax optimizer intentionally used unconstrained actions for probability consistency; evaluator decoder affects absolute outcome, so compare only matched policies.

- Raw files: Tang `~/experiments/sidra_m7/results/dev_pilot32_greedy128_{SFT,POST}/`; original TRAIN rollouts and optimizer artifact remain at `~/experiments/sidra_m7/results/onpolicy_grpo43_1step_6groups/`.
- **Conclusion:** first on-policy training step was *mechanically valid*, gave 6/6 variable TRAIN reward groups and real nonzero gradients, but yielded **ZERO** answer-EM or evidence success improvement on fixed pilot32. Do **not** promote the new adapter to default or evaluate sealed final-B160.
- **Next research gate:** constrain action choices (especially legal read/cite) **while matching sampling normalization and model log-probabilities**, or use native TRL tool environments with appropriate function-call fine-tuning. Then run multiple TRAIN updates/seeds and independently verify a nonzero answer/citation benefit. Do not mask invalid tool decisions without reflecting that transformation in the behavior policy.
