# M7 E-028 — Original HotpotQA TRAIN source-order invariance falsification

**Run:** 2026-10-27 (Tang CPU-only, no model update). **Decision:** diagnostic-only; **reject** using citation-order consensus as a validated semantic truth certificate or default Verifier. No new online SearchAgent QA run, DEV gold, eval-A64 or sealed final-B160.

## Question

Does mDeBERTa-v3 MNLI change its predicted entailment of a **fixed claim** if the order of the **same exact original HotpotQA TRAIN supporting sentences** is reversed? A correct semantic relation should not depend on which of the independent quoted sources was listed first. This is a controlled *order invariance* check, distinct from E-027's evidence removal (which changes evidence sufficiency).

We use the original fixed E-026 36 case/12-question frozen TRAIN cohort, not new editorial annotations. Exactly 27 cases have two or three independent quoted supporting facts and are eligible for a nontrivial order reversal; 9 single-source cases are **excluded, not counted as robust successes**. Each original quoted source line is checked against its own official `(title,sentence_index)` key, with unchanged bytes and unchanged hypothesis. No original E-026 gold answer or editorial class is passed into MNLI; those fields are examined **only after NLI scoring**.

## Execution / manifests

- Implementation: `src/sidra/premise_order_audit.py`, `scripts/m7_premise_order_falsification.py`, `tests/test_premise_order_audit.py`.
- Immutable GitHub tested snapshot: `1c572696399190e5712f99022abdfea299668f47`.
- Tang CPU full test suite **158 passed** (formerly 154), `~/experiments/sidra_m7_evidence_run6/logs/e028_cpu_tests_1c572696399190e5712f99022abdfea299668f47.txt`. GitHub CI status is separate.
- CPU NLI model: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`, original E-026 weight SHA256 **`65af59b1ff4450b09ecbf13ca35c840dbf038b26ff8e10e5ea89ca724828ed1e`**; same 384 pair-token cap; explicit no-truncation gates. Weights are verified against the original E-026 summary, and no network downloads/label tuning were performed.
- Original E-026 `summary.json` SHA256: `2f4e5850aacf2e75c90a5370d43f0f2dcf9bc627d3ccee20a5490e2dd05be859`; `controlled_rows.jsonl` SHA256: `097238a75409e653c99f1aafe72a348fc5c7c8f73b3c6cc77175d8efa1354ef4`.
- New E-028 output: `~/experiments/sidra_m7_evidence_run6/results/e028_original_train36_order_reverse/` (`per_case.jsonl`, `summary.json`). Result `per_case.jsonl` SHA256 `2b277405666d23bac2e54bccca515cbfc1411bdd90ef14751a722b440a5066e4`, `summary.json` SHA256 `11d05e39272f832e8a87b84ce0ec24c3a2154bece6d077d56d68ee3b27fc0647`.
- **27 actual new CPU NLI forward passes**, wall time 2.6722 seconds (including Python and bootstrap scoring). Existing 36 E-026 probabilities used as frozen before scores; no replay of the original 36.
- Fixed acceptance rule inherited from prior E-025: `argmax=entailment AND p_entail >= 0.85`. **No threshold fitted to these data**. Nonparametric percentile bootstrap (5,000, seed 2027) resamples the **12 independent original TRAIN question clusters**, not individual sentence/case rows.

## Actual comparison: identical evidence sets, reversed order

| Outcome | 27 eligible TRAIN contrasts |
| --- | ---: |
| Three-way MNLI argmax label changed | **0/27** |
| High-confidence entailment acceptance changed at fixed 0.85 cutoff | **1/27** |
| Question-cluster bootstrap 95% interval for acceptance-flip fraction | **[0, 0.12]** |
| Original accepted (eligible cases) | 13/27 |
| Reversed-order accepted | 12/27 |
| Mean absolute change in p_entail | **0.015503**, question-cluster 95% interval **[0.002400, 0.037971]** |
| Maximum single-case change in p_entail | **0.276387** |

**Reproduced order-sensitive case:** `mall-bridge_entailment`. Original order gives `p_entail=0.923079`, reversed source order `p_entail=0.646693` (delta `-0.276387`). The argmax label stays entailment, but with unchanged supporting sentences the fixed `p>=0.85` Verifier would change from **accept** to **not confidently accepted**. This is a direct model-order sensitivity finding; editorial labels have not been independently adjudicated.

**Counterexamples to “take a second order-consensus vote” as a semantic verifier:** the two editorial `unsupported_bridge` neutral near-misses remain high-confidence *in BOTH source orders*:

- `duggar-unsupported_bridge`: **0.994826 → 0.994252**.
- `facup-unsupported_bridge`: **0.940611 → 0.939510**.

Thus consensus of two source orders would **reject one editorial positive** but fail to block these two already-known unsupported relationship claims. This is not proof about the true labels (not independently adjudicated) but directly falsifies any proposal that order agreement alone resolves the observed NLI semantics problem.

## Research decision / next gate

1. **Do not** install a two-order agreement gate by default; this cohort has no improvement in known high-confidence unsupported-link cases, and the stricter gate introduces a demonstrated additional positive abstention at the same source facts. No new model quality/answer-EM/semantic truth improvement is claimed.
2. Before calibrating or promoting semantic Verifier thresholds, create a **separate disjoint TRAIN** claim/premise cohort with blind two-reviewer independent adjudication and a conflict-resolution stage; never call LLM scoring consensus “independent human labels.” Require relation swaps, wrong bridge entity links, negation scope, missing support, and positive entailment examples.
3. Preserve SearchAgent's `search/read/finish` state machine and original 6-step / 5-tool cap; future **actual online model** comparisons must separately report reachable/READ support, answer EM, original sentence citation coverage, model-semantic prediction, real tokens and latency.
4. Keep Pilot32 unchanged (the same 32 have already been used); freeze the approach before any eval-A64 independent comparison. **final-B160 remains sealed**.

**Reproduce (CPU-only):**

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONPATH=src:. \
  ~/projects/CodeRL-Lab/.venv/bin/python scripts/m7_premise_order_falsification.py \
  --e026-rows ~/experiments/sidra_m7_evidence_run5/results/e026_original_train36_mnli/controlled_rows.jsonl \
  --e026-summary ~/experiments/sidra_m7_evidence_run5/results/e026_original_train36_mnli/summary.json \
  --model ~/.cache/huggingface/hub/models--MoritzLaurer--mDeBERTa-v3-base-mnli-xnli/snapshots/8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c \
  --out <NEW_EMPTY_E028_DIRECTORY> --bootstrap 5000
```
