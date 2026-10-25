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

## E-001 — Pretrained GPU models on synthetic smoke fixtures (2026-10-26)

- **Objective:** verify real Transformers model can operate `search/read/finish` on GPU and characterize failure modes; no claims of Search Agent generalization.
- **Host:** Tang, NVIDIA RTX 4090 Laptop GPU (16 GB), Python 3.10.12, PyTorch 2.10.0+cu128, Transformers 4.57.6, Pillow 9.0.1.
- **Dataset:** same 40-question artificial Explorer→Region→Mineral fixture; **2 dev questions** (`fixture_000`, `fixture_004`), not a real multi-hop QA benchmark.
- **Models:** cached Qwen/Qwen2.5-1.5B-Instruct, snapshot `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; cached Qwen/Qwen2.5-3B-Instruct, snapshot `aa8e72537993ba99e69dfaafa59ed015b17504d1`. No model updates.
- **Parameters:** greedy generation, `max_new_tokens=280`, tool budget=7, max steps=8, BM25 top-k=3. Adapter based on committed GitHub M1 with Pillow/Accelerate compatibility fixes (`89595e78488963f40496dd2f477b55f7f31bf380`).
- **1.5B result:** answer EM 0/2, verified 0/2; each question issued 1 search then errored due non-JSON action; error classification was preserved.
- **3B result:** answer EM 1/2, verified 0/2; for fixture_000 it read only `region_000` and answered `Ore000` without citing the explorer document. For fixture_004 it incorrectly used the region for question 000.
- **Storage:** `Tang:~/experiments/sidra_m1/results/pretrained_qwen2p5_m1_2q/` and `.../pretrained_qwen2p5_3b_2q/`, with traces, metrics and manifests. Historical manifest fields `git_commit=unavailable` reflect direct connector staging; updated code supports explicit commit metadata.
- **Conclusion:** pretrained model integration works, but tool discipline and multi-hop evidence completion **fail**. No evidence of self-improvement, no GRPO, no SFT, no actual HotpotQA score. Issues P-008 through P-010.

## E-001b — Real multi-hop benchmark (HotpotQA distractor)

**NOT RUN.** Requires benchmark data acquisition, a frozen evaluation protocol and real pretrained model baseline.

## E-002 — LoRA SFT, Agentic GRPO, RSI self-improvement

**NOT RUN.** Training/export scripts do not constitute a measured training result. Reserve separate records for each actual run.


## E-001c — Pretrained 3B synthetic 8-question dev check (2026-10-26)

- **Purpose:** expand the prior 2-question GPU integration smoke check to the complete synthetic fixture dev split; it is *not* an external multi-hop QA benchmark.
- **Provenance:** Tang `~/experiments/sidra_m1/results/pretrained_qwen2p5_3b_dev8/`, manifest `created_at_utc=2026-10-25T21:18:53.368217+00:00`, `git_commit=9da592898a3ec6140799b2c939ae057b4c9d5f76`.
- **Model:** cached `Qwen/Qwen2.5-3B-Instruct` (pretrained; no fine-tuning), Transformers GPU adapter. Greedy actions with `max_new_tokens=280`, `max_steps=8`, `max_tool_calls=7`, `search_top_k=3`, seed=42.
- **Data:** all **8 synthetic dev** questions from `examples/fixture`, not HotpotQA or web search.
- **Observed:** answer Exact Match **1/8 = 0.125**; token F1 **0.125**; mean evidence precision **0.3125**; mean evidence recall **0.1875**; **verified success 0/8**. Average search calls **1.0**, average read calls **1.375**, average tool calls **2.375**; all 8 terminated with `finish`. Mean wall-time `1198.5 ms`/question (only for this local test).
- **Failure analysis:** the one exactly correct answer cited only one of two required supporting documents; other runs had incorrect answers and/or zero evidence recall. The model commonly searched once and finished before collecting both-hop support. Verify detailed cases in Tang `per_example.jsonl` and `traces.jsonl`; raw traces and synthetic dataset are not committed.
- **Interpretation:** a successful inference integration test but **failed task/evidence behavior**. Not evidence of RSI, SFT or GRPO improvements; no model weights were updated. Main next gate: run a genuine fixed HotpotQA distractor baseline after acquiring the dataset and pinning the evaluation protocol.
