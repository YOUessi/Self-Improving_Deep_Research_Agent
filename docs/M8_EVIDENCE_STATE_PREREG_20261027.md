# M8 pre-registered evidence-state closed-loop study (2026-10-27)

## Exact research question

Can ONE observable evidence state used by the Qwen3-0.6B finite Search/Read/Finish
policy increase actual cited multi-hop QA success, rather than improving
retrieval availability, answer coverage and rewards in disconnected subsystems?

This document is recorded BEFORE the full real-model A/B results.

## Mechanism and trust boundary

`src/sidra/evidence_state.py` reconstructs a **gold-blind evidence ledger** at
every decision from the question and actual Search/Read observations alone:
- exposed (but unread) doc IDs/titles/snippets, actually READ IDs/titles/text,
  queried strings and true first-party tool budgets;
- lexical question anchor coverage, linked source titles occurring in other
  actually-read source text, and observed second-hop entity query gap;
- lexical candidate answer support within actually read sentences, which is a
  heuristic and **not fact entailment**.

In opt-in `FiniteActionConfig(answer_proposer="v4",
evidence_state_mode="v1")`, this SAME state enters (1) the candidate
Search/Read ordering and bridge query, (2) compact model-facing visible
evidence state and additive Search/Read/Finish logits, and (3) an AFTER-ROLLOUT
TRAIN-only state-delta credit function `assign_evidence_state_credit`.
The existing action prior plus ledger bias is recorded in each action sample
and is replayed unchanged by `logprob_of_action` during optimizer updates;
one single-token digit action ID among at most nine actions.

No QA gold answer, original support IDs, evaluator verdict, hidden titles or
unread full texts enter any of these inference functions. TRAIN labels enter
ONLY the posthoc `src/sidra/step_credit.py` scoring, never the state or the
model's observations. This is **custom GRPO-style + observed-state credit**, NOT
native TRL GRPO, and the lexical potential may reward irrelevant reads; it is
weighted only 0.25 relative to labelled TRAIN support progress.

## Causal policy A/B — first gate

The SAME original Qwen3-0.6B-Base checkpoint + unchanged TRAIN-only stop-SFT
LoRA SHA `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`,
original finite V4 answer proposer, real per-question HotpotQA distractor
BM25 search/read, SearchAgent budget `max_steps=6`,
`max_tool_calls=5`, `search_top_k=3`, `max_context=2048`.

Dataset is the **32 already-open DEV pilot** questions from the frozen DEV256
partition, NOT original TRAIN64 and NOT sealed final-B160. Each condition
is evaluated with question-keyed random seeds **151 and 157**:
- Control: `--answer-proposer v4 --evidence-state off`.
- Intervention: `--answer-proposer v4 --evidence-state v1`.

Code and model snapshot must be identical across all four arms, using the
branch's immutable GitHub SHA after this preregistration. An unequal dataset,
model, LoRA SHA or tool budget aborts the paired comparison.

Primary: per-QA normalized answer EM **and** strict answer+support citation
verified. Secondary: actual gold-support SEARCH exposure and READ coverage
(scored offline), evidence-state cross-document links (observable only),
search/read/model/Token cost, latency, abstain/error.

Compute paired 5,000-bootstrap CI clustered by **32 unique original question
IDs**, not pretending 2 seeds=64 independent questions. The policy-state change
can affect both tool trajectory and answer choice; report them separately.

**Online GRPO go/no-go:** state-aware policy must show STRICT VERIFIED
improvement averaged across seeds AND no decline in answer EM to proceed to
any new TRAIN-only online GRPO. If no gain, do NOT launch GRPO merely because
the internal evidence-state potential increased: diagnose bridge relevance,
support completeness and candidate answer error instead.

## Next gate only if real A/B passes

Frozen TRAIN-only tasks with no overlap with existing reward/final evaluation
cohorts, same optimizer/seed, `--credit-mode evidence_state_delta_v1` vs
step-based baseline, 2+ actual GRPO updates, student action-token-only
gradients, zero supervised tool-text tokens, matched old/new log-prob checks
and final question-disjoint comparison. No DEV gold used to select tasks or
compute optimizer credit.

## Audit of failure modes

- A title-reference or lexical overlap is NOT proof of entailment.
- A successful 8B answer may require missing support documents; never
  substitute model memory for an actually READ verified citation.
- Potential increase on an irrelevant READ is a reward-hacking risk, so
  compare gold-support read coverage and answer/citation success separately.
- Never overwrite raw experiments. All code/tests on GitHub; Tang GPU only for
  actual model sampling/training with shared advisory lock and free VRAM gate.
- All 160 sealed DEV final-B questions remain untouched. These are not
  externally held-out real-web Search Agent experiments.
