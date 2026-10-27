# M7-C E-029 pre-registration — gold-blind matched-slot second-READ shadow

Date: 2026-10-27 (Tang +08:00). **Pre-registered before running any E-029 retrospective outcome comparison.** E-028's real 72 original TRAIN Solver episodes have **17/72 exact answers but 0/72 strict answer+evidence success**, and the source model has low supporting-document READ recall. E-029 is a **CPU-only, retrospective, action-slot diagnostic**, not an actual model continuation or post-training gain.

## Frozen input and trust constraints

- Work ONLY in `feat/m7-rsi-multi-generation`, never edit the online GRPO worktree. Re-audit **all three original E-027 real Solver runs (24 TRAIN questions/seed 43, 44, 45)** with the E-028 independent trace + model-action + corpus/provenance auditor *before* any E-029 result calculation. The three seeds belong to the same frozen model checkpoint; they are **not three optimizer seeds or RSI generations**.
- The CPU planner receives exactly (a) model-visible question text, (b) SearchAgent's previously returned SEARCH hit IDs/titles/snippets/scores, (c) actually READ title/text, (d) observed step/tool budget. It cannot receive original TRAIN ID, gold answer, supporting-title IDs, verifier outcomes, unseen document contents, or DEV/eval-A/final-B metadata.
- A shadow comparison exists **only when the next real model action is READ**, there was already at least one previous READ, **at least two different legal unread SEARCH-returned IDs are available**, and there is at least one step left to finish. Select the first such slot per question, at most once per episode. **Replace one real READ decision with one observed-only proposal; no extra SEARCH, READ, or budget is granted.** Do not replay actual later observations as if the proposed action had occurred.
- Selector: frozen deterministic title/word overlap ranking against question and already-read text plus small capped observed BM25 score; tie-break by first exposure order and opaque document ID. No gold-based candidate restriction, score, thresholds, or learned reward. All observed fields must match strict SearchAgent shapes; unrecognized keys fail closed.
- Independent Verifier only **after proposal is frozen** maps each opaque candidate to original source TRAIN support, checks exact source corpus and calculates whether the real READ or shadow READ would expose one previously unread labelled supporting document at that one slot. Labels are never returned to the selector.

## Predeclared measurements

1. Audited complete episode count and original E-028 replay outcome must match known 72/72 audited, 17/72 answer EM and 0/72 strict TRAIN original citation success before inspecting E-029 score.
2. Eligible same-budget READ slots, slots where missing original support was *available to READ* in the already-visible frontier, choice-change count, actual marginal support READ count versus proposed marginal support READ count.
3. Paired per-question differences with question-level bootstrap (5,000 resamples, seed 829). No claims about complete trajectory reward, answer EM, strict QA, model inference cost or improved generalization: the rest of the real trajectory is **not valid under a substituted action**.
4. Aggregate only by frozen Direct/Bridge/Hard difficulty proxy. Never export individual question IDs, answers, support labels, documents, private mapping, model prompts, weights or final labels.
5. Reject on SHA mismatch, incomplete cohort, gold in selector input, illegal/unobserved read, tampered trace/model response, duplicate question/holdout overlap or changed budget. Write immutable create-only audit files.

## Decision gate

E-029 can at best identify whether deterministic observed-only source selection is worth a later **prospectively registered**, real-model, same-tool-budget head-to-head on an **unopened, disjoint original TRAIN cohort**. This historical 72-question set was already inspected during E-028 and is NOT independent validation. If there is no positive paired same-slot support-exposure signal, record a negative result rather than tune against these labels. GPU is currently occupied by an unrelated vLLM process, so no GPU experiment or interruption is permitted. Even positive same-slot source exposure is NOT GRPO, RSI or proof of semantic entailment.
