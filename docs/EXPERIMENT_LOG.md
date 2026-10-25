# Experiment ledger (do not overwrite older entries)

## E-000 — CPU synthetic smoke tests (2026-10-26)

- **Objective:** validate end-to-end search/read/finish workflow, metrics, output JSONL and SFT example export.
- **Environment:** temporary Linux builder, Python 3.13, pure Python runtime; no external API calls and no GPU training.
- **Data:** `scripts/generate_fixture.py --n 40 --seed 13` (24 train / 8 dev / 8 test; 120 synthetic candidate documents). Dataset is an invented Explorer→Region→Mineral chain. It is NOT HotpotQA, not a realistic Search Agent benchmark.
- **Command:** `PYTHONPATH=src python -m sidra.cli eval --dataset examples/fixture --split dev --policy fixture-multihop --out <new-run-dir>`; comparator uses `fixture-onehop`.
- **Dev results (synthetic smoke only):** multihop EM=8/8, verified success=8/8, 2 searches+2 reads per question; onehop EM=0/8, verified success=0/8, 1 search+1 read per question. **Do not use as evidence of learned superiority**: the policies are hard-coded parsers for these artificial fixtures.
- **Train-only export smoke:** 24 verified train traces, 120 action-supervision examples; SFT training NOT executed.
- **Tests:** 15 CPU tests passed in 0.09s after fixing a test quoting syntax error.
- **Issues:** see P-001 through P-005 in `docs/PROBLEM_LOG.md`.
- **Next gate:** run actual HotpotQA distractor with a real LLM endpoint under fixed tool budgets; report confidence intervals and prompt/control equivalence. No real experiment exists yet.

## E-001 — Actual pretrained model evaluation

**NOT RUN.** Reserve ID for first GPU-backed frozen real benchmark (record commit, model hash, tokenizer, host GPU, split, raw traces, budget, retries and failures).

## E-002 — LoRA SFT, Agentic GRPO, RSI self-improvement

**NOT RUN.** Training/export scripts do not constitute a measured training result. Reserve separate records for each actual run.
