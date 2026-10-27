# M8 V2 evidence-chain gate (preregistered before real-model A/B)

**Recorded 2026-10-27 after v1 negative real Agent and TRAIN160 offline counterexample analysis, BEFORE V2 GPU Agent A/B.**

## Why V1 was rejected

Fixed, same-Qwen3-0.6B SFT LoRA, original V4 finite legal answer catalog,
real Search/Read/FInish on already seen DEV pilot32, seeds151/157:

- V4 off: answer EM **6/64 episodes**, strict success **4/64**.
- V4 state-v1: EM **3/64 episodes**, strict success **2/64**.
- 32 unique question clusters (two seeds are repeated measures):
  EM delta −4.6875pp, CI [−14.0625,+1.5625]pp;
  strict delta −3.125pp, CI [−10.9375,+3.125]pp.
- Completed 160 genuine original TRAIN policy rollouts audit: v1 proxy assigned
  positive delta to **214/214 real gold-support READs and 147/226 DISTRACTOR
  READs (65.04% false-positive among distractor reads)**.
- **Decision:** NO GRPO promotion. A real unified state must be discriminative
  before its dense reward can teach the correct multi-hop process.

## Gold-blind V2 mechanism

Replace broad "more distinct READ docs" proxy with an explicit *question
anchor → observed source → actually READ linked second source* evidence chain.

- An anchor article title must appear as a whole token phrase in the original
  question; an entity/title mention is not automatically true.
- An evidence link must mention a second complete, nontrivial article title
  in one of two actually READ source texts.
- No gold support ID or correct answer is used by the V2 inference policy,
  SEARCH gap proposer or conditional action logits.
- Source diversity alone is NO LONGER rewarded. Distractor reads unrelated
  to a question-anchored bridge have no progress delta.
- A shared V2 ledger remains in the model prompt and recorded categorical
  prior logits; **identical additive bias** re-used in differentiable old/new
  action normalization. No extra model/token labels from tool observations.
- TRAIN gold may still be consulted *after* real policy tool actions in
  `evidence_state_delta_v2` credit, mixed with bounded state delta weight
  0.25. This would be custom multi-turn GRPO-style RL, not pure model reward
  if later combined with supervised read/finish auxiliaries.

## Off-model TRAIN-only sanity check

Re-scored the SAME 160 real original TRAIN SearchAgent rollouts from M7-A8,
using labels only in the OFFLINE audit:
- 214 original support reads: state v1 positive 214, V2 positive **98**.
- 226 distractor reads: state v1 positive 147, V2 positive **18**.
- False progress among distractor READs: 65.0% → **8.0%**;
  support-read sensitivity also fell 100% → **45.8%**.
- This is NOT heldout success: those 160 episodes come from already used
  TRAIN questions, and the V2 rule was designed after V1's failure.

## Fixed V2 vs OFF real-model A/B gate

Use exact SAME parent Qwen3-0.6B-Base + stop-SFT LoRA
(`44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`),
same V4 answer catalog, `max_steps=6`, `max_tool_calls=5`,
BM25 Top3, prompt context 2048, same **32 original publicly viewed
HotpotQA DEV pilot** questions, and both seeds **151/157**.

- Baseline: `--answer-proposer v4 --evidence-state off`.
- Intervention: `--answer-proposer v4 --evidence-state v2`.
- All four runs MUST share exact Github source snapshot, dataset SHA, model
  revision, LoRA SHA, tool budget, and random seed keyed per question.
- Report answer EM, strict gold-support citation success, Search exposure
  and actual support READ, linked evidence proxy and tools/models/tokens/latency.
- Bootstrap clustered on **32 original questions**, not 64 iid cases.

**V2 online GRPO go/no-go** predeclared: ONLY run any new TRAIN-only state-v2
policy-gradient updates if matched real Agent strict verified improves AND
answer EM does not worsen in the two-seed average; also report false-positive
state delta risk. A mere reduction in distractor reward is not sufficient.

If no improvement: keep all negative experiments, **do not tune final-B160**,
and improve proof/claim representation or jointly verified answer module
instead of piling on training steps.

## Boundaries

HotpotQA distractor candidate passage retrieval ≠ real web search.
Whole-title mention ≠ NLI entailment. V2 reward's precision/recall depends
on question form, which can omit explicit article-title mentions.
The audit reuses TRAIN trajectories and cannot prove generalization.
Final-B160 remains sealed.
