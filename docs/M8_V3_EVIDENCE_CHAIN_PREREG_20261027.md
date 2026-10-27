# M8 V3 evidence-chain policy and TRAIN-only reward: preregistration

**Date:** 2026-10-27. **Parent repository SHA:** `fc5910a425b66318190c1a5e576fb5393f1fe2d3`.
**Status at writing:** Code and CPU tests only; **NO V3 real-model A/B, no V3 GPU training, no quality improvement claimed.**

## Falsifiable goal and prior negative evidence

Our objective is **correct answer WITH actual cited multi-hop evidence**, not proxy
retrieval, answer-guessing, NLI score, and RL advantage improved independently.

The already-open DEV pilot32 × two seeds (151,157), same Qwen3-0.6B Base +
stop-SFT adapter, gave V4 OFF EM 6/64 and strict 4/64 vs V1 EM 3/64 and
strict 2/64. V1 was rejected. A separate retrospective TRAIN160 audit found
147/226 unrelated READs got positive V1 state delta, vs 18/226 V2; but V2
recognized only 98/214 actual support READs. This is a precision/recall
tradeoff on reused TRAIN rollouts, not held-out policy success.

**Found code-contract defect:** original `action_prompt` injects a ledger only
when `evidence_state_mode=="v1"`; V2 supplies logits/catalog state but omits
the model-facing ledger. Existing V2 comparator also always reconstructs
V1 potential. **V1 and V2 policy implementations remain untouched** to
preserve their preregistered meaning. The comparator now reconstructs the
SAME requested version in both arms, and reports that version explicitly.

## V3 mechanism (frozen before any real-model V3 A/B)

Only `question` and **actually executed** SEARCH hits and READ texts enter
`derive_evidence_state(...,version="v3")`. It constructs:

1. **Anchor READ:** a source whose complete nontrivial title occurs as a
   whole phrase in the question (a conservative lexical starting point).
2. **Unresolved linked hit:** an exposed, unread hit whose complete title is
   mentioned in actually READ anchor text. Prefer READ of this hit over new
   searches; when none is exposed, generate a second-hop SEARCH proposal
   from the anchored READ instead of the most recent distractor READ.
3. **Closed cited chain:** at least two actually READ distinct sources with
   a strong complete-title link and a question anchor. This is **not**
   semantic entailment. Answer candidates get positive policy logits only
   if their text is present inside this chain AND citations contain both
   linked READ IDs. Unrelated answer-bearing READs do not acquire support
   merely because a different source pair is linked.
4. **One policy state:** the same state drives bounded legal catalog ranking,
   state-conditioned model-facing evidence-gap/next-READ prompt, additive
   finite-action choice logits, and the after-episode TRAIN reward. Sampling
   and differentiable re-scoring use exactly the SAME legal IDs and logits.
   Original V4 and V5 policies are unaffected unless V3 is opt-in.

Potential: `0.10*exposed_anchor + 0.25*read_anchor
+ 0.20*visible_unread_linked_hit + 0.45*actually_read_linked_pair`,
bounded to [0,1]. Distractor READs with no linked evidence transition have
zero potential gain. **Still a proxy**: a misleading title coincidence can
create a false positive. Full-text unknown source, evaluator labels, original
support IDs or gold answer are never added to inference state.

## Post-execution TRAIN-only joint credit

`evidence_state_delta_v3` is a custom grouped on-policy RL credit mode:
- SEARCH and READ: `0.75*true_gold_support_increment +
  0.25*max(0,delta_observed_state)*1[true_gold_support_increment>0]`.
  Thus a misleading graph delta with no **new original TRAIN support**
  NEVER produces reward credit. This gold gating happens only after the
  real SearchAgent tool action. Group baselines are per original question,
  action kind, and within-kind ordinal across independent siblings.
- FINISH: `0.20*TRAIN_answer_EM + 0.80*TRAIN_strict_success`.
  Strict already requires original support cited and read, observed answer
  in cited evidence, and exact answer. No fabricated entailment grade.
- Dev/Test labels are forbidden in reward scoring. The model predicts
  exactly **one** legal action-ID token; no source/text token targets.
  This is **custom GRPO-style**, not native TRL GRPO, and it is hybrid
  TRAIN gold shaping, not a fully unsupervised discovery signal.

## Predeclared experiment and stop conditions

1. CPU correctness gate: synthetic chain/distractor tests, hidden-label
   injection rejection, backward V1/V2 invariance, sample/train masked
   probability equality, custom reward gold-gating and no tool-label tokens.
   CPU tests are **necessary but never sufficient** evidence of performance.
2. Off-model original TRAIN160 audit: identical saved real action traces;
   report all READ false-progress events, useful support recall, and V3
   credited distractor delta. Since V3 rules were informed by earlier
   TRAIN data, **do not call this independent/generalized improvement**.
3. Fixed paired real-model A/B on the **already-open DEV pilot32 only**:
   original Qwen3-0.6B Base + immutable stop-SFT LoRA; source commit SHA,
   dataset and adapter SHA, max_steps=6, max_tool_calls=5, BM25 top3,
   context=2048, same original V4 nonoracle answer catalog, sample mode,
   seeds **151** and **157** keyed per question.
   Arms: `--evidence-state off` vs `--evidence-state v3`.
   PRIMARY: verified strict answer+citation and answer EM.
   SECONDARY: original support fully searched and fully read
   (evaluated only after rollout), action availability, search/read
   counts, prompt tokens, model calls, elapsed time, abstention.
   Bootstrap paired differences over **32 original question clusters**,
   not 64 independent episodes. Do not tune on this pilot after results.
4. Only if mean strict success **improves** with EM non-decreasing under
   this preregistered matched comparison, may a separately frozen,
   question-disjoint TRAIN-only on-policy V3 credit experiment start.
   Other gating: exact action-mask probability checks, nonconstant
   within-question reward, actual policy weight delta, no external GPU
   job preemption, no proof-of-correctness from lexical title links.
   Otherwise **NO GRPO, NO V3 default, NO multi-generation RSI**.
5. Final-B160 remains sealed. Existing E-030 independent human-review
   set must not enter training; no model-assisted labels masquerading as
   independent semantic annotations.

Known limitations: question forms without explicit source titles yield
no anchor, mismatched aliases break link discovery, a title citation is not
fact-level entailment, and both search context and answer-candidate coverage
can be bounded below the correct answer. Negative results MUST remain logged.
