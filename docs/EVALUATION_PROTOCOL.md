# Reproducible evaluation protocol (freeze before reporting a gain)

## Baselines

- `fixture-onehop` vs `fixture-multihop` check only pipeline *integrity*.
- M2 actual comparison: same base LLM, same candidate corpus and candidate-passage scope, (i) one-shot answer prompt; (ii) single retrieval; (iii) multi-round retrieval; (iv) verifier-assisted; (v) SFT; (vi) SFT + RL. All under specified resource and tool budgets.
- Do not compare a stronger model/greater tool budget against a weaker control and attribute differences to RSI.

## Split and contamination

- Train only on train. Freeze dev/test and document hashes before experiments. Never export supervised targets from non-train traces; code raises an error if attempted.
- If using HotpotQA, record source version, license, distractor mode, converted sample count, and candidate passage scope. Use development split for selection, then a separate frozen evaluation split only once for final reporting. Never relabel public dev examples as train to claim independence.
- `group` field prevents entity-level train/dev/test overlap in JSONL, but lexical/template similarity still requires separate distribution-shift evaluation.

## Metrics

| Metric | Definition / limitation |
|---|---|
| EM | normalized predicted vs gold answer; empty outputs never count as correct |
| token-F1 | bag-of-word normalized overlap; inadequate alone |
| evidence P/R | cited doc IDs vs gold support IDs, if provided |
| verified success | correct answer, cited answer string found in evidence, all labeled supporting docs cited |
| tool efficiency | search/read calls under fixed budgets |
| latency | wall-clock time; meaningful only with real model serving |
| robustness | repeated seeds, adversarial irrelevant docs, malformed JSON, prompt injection probes |

All results should report counts (n) and failure/abstention rate, not only averages. Add bootstrap confidence intervals and repeated model runs in M2.

## RSI requires more than reflection

At iteration t: freeze a challenger sampling distribution and task budget; generate *new train tasks*, execute solver, automatically verify (independent checks), train `policy_{t+1}` with accepted trajectories, and evaluate on unchanged external test. Compare no-update/self-reflection/SFT/GRPO with equal compute. Analyze reward hacking and self-generated answer contamination. Until a parameter-updated model beats an untouched control on held-out external questions, refer to results as a **proposed RSI pipeline**, not proven self-improvement.
