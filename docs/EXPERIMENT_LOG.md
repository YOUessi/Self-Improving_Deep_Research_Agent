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


## E-002 — M2 role-specialization CPU validation (2026-10-26)

- **Task:** add Planner / Researcher / Solver / source-only Verifier under the same bounded SearchAgent; audit local model weights before selecting pretrained checkpoints.
- **Model:** none; CPU unit/integration tests only. The synthetic two-hop fixture uses deterministic scripted roles, NOT a learned model.
- **Test gate:** GitHub Actions `run 40236905821` completed successfully with **26 passed** on Python 3.10 and Python 3.12. Compilation check also passed.
- **Controls:** early finish blocked, citations require read sources, no benchmark gold answer in prompts/role history, config-only Qwen3 cache rejected, missing weight shard detected.
- **Limitations:** the implementation is sequential role specialization on one model backend. No independent concurrent role inference, HotpotQA benchmark, valid model-versus-model comparison, SFT, GRPO, or RSI improvement has been run for M2.
- **Next:** synchronize the full GitHub snapshot to Tang through an approved method, then compare single and multi-role on the SAME frozen real benchmark with equal tool budgets and reported inference calls/tokens.


## E-003 — HotpotQA distractor, real passages, BM25 retrieval profile (2026-10-26)

- **Research question:** how often does the question-only BM25 initial search recover both evidence passages, before any model-side search reformulation?
- **Original source:** [HotpotQA distractor validation](https://github.com/hotpotqa/hotpot), via HF-mirror copy of `RAGLAB/data/eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json`. SHA-256 **4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea**, identical to the official HotpotQA dataset manifest. Source has 7,405 dev questions. Data is CC BY-SA 4.0 and is **not committed**.
- **Sample:** 24 dev examples selected by ascending SHA-256 of `42:question_id` from the full original dev split, independent of labels or retrieval success. Selected IDs, seed, count and raw data checksum saved locally in `~/experiments/sidra_m3/data/seed42_dev24/original_subset.manifest.json`. The original `dev` label was retained; **no training data** is exported from these examples.
- **Corpus:** 10 supplied candidate passages per question; 240 documents total; labels are evaluator-only, each question has 2 supporting document titles. This is *distractor candidate-passage* evaluation, **NOT full-Wikipedia or web retrieval**.
- **Commands:** `scripts/select_hotpotqa.py --n 24 --seed 42`; `scripts/convert_hotpotqa.py --split dev`; `PYTHONPATH=src python scripts/profile_hotpot_retrieval.py --limit 24 --seed 42`. Tang used GitHub source snapshot `acfb4b32f59f2d1369486761907b342425cdaac5`; 32 CPU tests passed there.
- **Top-1:** support recall **0.375**, all-support completion **0/24**.
- **Top-3:** support recall **0.6458333**, all-support completion **9/24 = 0.375**.
- **Top-5:** support recall **0.7708333**, all-support completion **14/24 = 0.583333**.
- **Top-10:** support recall **1.0**, all-support completion **24/24 = 1.0**, because the provided distractor candidate set contains gold passages.
- **Raw results:** Tang `~/experiments/sidra_m3/results/retrieval_seed42_24/` (`per_example.jsonl`, `summary.json`). Do not treat these as LLM answer accuracy or training gain. Sample is intentionally small.
- **Open gate:** install/verify AWQ inference backend (vLLM), run actual Qwen3-8B-AWQ inference on frozen 24 dev examples, compare matched single/multi-role tool and total inference budgets, and preserve every failed case. The actual model baseline is **NOT YET RUN**.


## E-004 — Qwen3-8B BF16 local GPU inference, single-action diagnostic (2026-10-26)

- **Goal:** independently verify that a complete Qwen3-8B BF16 checkpoint can perform non-thinking chat-template inference on Tang, before testing a multi-hop benchmark.
- **GitHub code:** `scripts/probe_qwen3_bf16_offload.py`, SHA `cee7363af8237a41b4e9de03ed02316788c88a4f`.
- **Model:** official checksum-verified Qwen3-8B BF16 in `/home/you/models/Qwen3-8B`; device: RTX 4090 Laptop 16GB, PyTorch+Transformers through existing CodeRL-Lab .venv.
- **Runtime:** Accelerate `device_map=auto` with 11GiB GPU cap and CPU offload, no model update. Model load **4.23s**; generate **19 tokens / 13.45s**; peak CUDA allocated **11,770,267,136 bytes**. First 24 model layers ran on GPU and remaining 12 layers plus lm_head on CPU.
- **Output:** correctly structured JSON tool action `{"kind":"search","query":"Tokyo Tower"}` under `enable_thinking=False`.
- **Artifacts:** Tang `~/experiments/sidra_m3/results/qwen3_8b_bf16_probe.json` and associated stdout log. **This is a smoke test, not HotpotQA success or multi-Agent performance**.
- **Next:** the same real model is now being evaluated with SIDRA's `search/read/finish` loop on frozen 24-item HotpotQA distractor subset. Initial run only includes **2 dev questions**, capped to 6 steps and 5 tool calls. Its outcome must be logged separately.


## E-005 — Real HotpotQA dev, Qwen3-8B BF16 single-Agent integration (2026-10-26)

- **Source:** Tang GitHub snapshot `62e3874013d38c90b8677e9d9683b5598787fd52`, original verified HotpotQA distractor dev SHA-256 `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`, frozen ID-hash sample seed 42, converted candidate passages.
- **Model:** Qwen3-8B BF16 official checkpoint, real pretrained/post-trained model inference with Transformers, `enable_thinking=False`. Tang 16GB GPU with `--offload-gpu-gib 12` and CPU layer offload. **No model parameter updates**.
- **Parameters:** `--policy transformers --max-new-tokens 64 --max-steps 6 --max-tool-calls 5 --search-top-k 3 --limit 2 --seed 42`. This is **two** dev questions only, exploratory integration, not a statistically meaningful model benchmark.
- **Exact result:** answer EM **2/2 = 1.0**, token F1 **1.0**, evidence precision **1.0**, supporting-doc recall **0.5**, strict verified success **0/2**. Both episodes terminated with `finish`.
- **Trajectory pattern:** for both questions, model did `search -> read -> finish` after citing **only one** of the two gold supporting passages. Correct answers (`Key West`, `Dungeness crab`) were in the cited evidence but the full two-hop justification was not collected.
- **Costs:** average 3 model calls, 1719 prompt tokens, 94 completion tokens, 2 tool calls (1 search+1 read), 48.08 seconds per question under CPU/GPU offload.
- **Raw artifacts:** `Tang:~/experiments/sidra_m3/results/qwen3bf16_single_dev2/` (`traces.jsonl`, `per_example.jsonl`, `summary.json`, `manifest.json`).
- **Conclusion:** Qwen3 genuine multi-hop QA *output* works, but evidence-complete tool policy still fails. **Do not claim perfect Agent performance from 2/2 answer EM**. A matched multi-role test on the same exact questions, tool budgets, model and code SHA was launched separately; its result is not included in E-005.


## E-006 — Multi-role Qwen3-8B BF16 on same real HotpotQA dev2 (2026-10-26)

- **Code:** same pinned GitHub snapshot `62e3874013d38c90b8677e9d9683b5598787fd52`, dataset, two question IDs, checkpoint, BF16 CPU/GPU offload, temperature=0, `max_new_tokens=64`, `max_steps=6`, `max_tool_calls=5`, `search_top_k=3` as E-005; only policy changed from `transformers` to `multi-transformers --min-citations 2`.
- **Question IDs:** `5a7363ec5542991f29ee2dd7`, `5a75d9b4554299109176e5a7`, frozen dev-only input; no gold labels exposed to either policy.
- **Multi-role:** answer EM **0/2**, F1 **0**, evidence precision/recall **0**, strict verified success **0/2**, all episodes abstained or errored. Mean model calls **3.5**, prompt tokens **2327.5**, completion tokens **125**, tool calls **2.5**, elapsed **63.685s** per question.
- **Matched single (E-005):** answer EM **2/2**, strict verified success **0/2**; model calls **3.0**, prompt tokens **1719**, completion tokens **94**, tool calls **2.0**, elapsed **48.0755s** per question.
- **Multi-role failures:** on first question, Planner search + Researcher read succeeded, but next Researcher call produced unparsable action JSON and terminated as error. On second, Researcher repeated reading the same document, then attempted premature finish, which the role guard converted to abstention.
- **Conclusion:** sequential role specialization **worsened answer quality and cost** on these two real examples; it did **not** improve strict evidence completeness. This is an exploratory n=2 integration result, NOT a statistically powered assessment of multi-agent vs single-agent algorithms.
- **Artifacts:** Tang `~/experiments/sidra_m3/results/qwen3bf16_multi_dev2/` plus single-run outputs and per-example JSONL. Negative results retained; do not rename this into a success experiment.
- **Next gate:** fix structured-action robustness and repeated reads without using gold answers or growing the tool budget; then evaluate on disjoint held-out IDs from the frozen dev24 subset and report both model-calling costs and strict evidence success.


## E-007 — Disjoint HotpotQA dev2 matched baseline vs repaired multi-role (2026-10-26)

- **Aim:** test structural fixes (malformed JSON 1 bounded retry, duplicate-read and premature-finish replanning) on different genuine QA examples, not those used to diagnose E-006 failures.
- **Provenance:** GitHub SHA `69614339669341eb9aa7e5c0881e83179fa61997`, Tang source mirror, **36 CPU tests passed**. Official HotpotQA distractor dev SHA-256 `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`, frozen seed42 24-question subset, sorted question offset **2**, limit **2**, thus disjoint from E-005/E-006 first two diagnostic items.
- **Question IDs:** `5a79f6825542996c55b2dcb4`, `5a7b3ec95542995eb53be8d3`.
- **Matched settings:** same Qwen3-8B BF16, non-thinking mode, Tang 4090 Laptop with GPU12GiB+CPU offload, greedy generation, `max_new_tokens=64`, steps6, max tool calls5, search top-k3. Only `policy` changes `transformers` vs `multi-transformers`; multi policy has role-specific instructions and at most one JSON format repair/replanning, counted as **extra model calls**.
- **Single pretrained:** answer EM **0/2**, strict verified **0/2**, evidence recall **0**, mean model calls **4.0**, prompt tokens **3021**, completion tokens **142**, mean tool calls **3.0**, latency **72.4175 seconds**. Both episodes ended in error: an unparsable model action JSON after 3–5 calls.
- **Repaired multi-role pretrained:** answer EM **0/2**, verified **0/2**, evidence recall **0**, mean model calls **5.0**, prompt tokens **4315**, completion tokens **206**, mean tool calls **3.0**, latency **104.5455 seconds**. Both episodes still ended in malformed JSON errors, despite bounded retry.
- **Trace characterization:** first task requires comparison of Linda Stender and Leonard Lance birth years; both policies performed search, read, further search, read, then errored. Second task searched/read Edward C. Hugger and Alexander Acosta and likewise failed format parsing. Only action objects and verifier decisions logged, no private chain-of-thought.
- **Raw outputs:** Tang `~/experiments/sidra_m3/results/holdout_offset2_transformers` and `~/experiments/sidra_m3/results/holdout_offset2_multi-transformers`, each with `traces.jsonl`, `per_example.jsonl`, `summary.json`, `manifest.json`. Never overwrite these run directories.
- **Conclusion:** on the disjoint 2-question integration check, multi-role did not improve success and consumed more inference tokens and latency. **Neither result is statistically conclusive**, and no SFT/GRPO/RSI training occurred. Fix structure via tokenizer-constrained action schema or additional output tokens; run fresh held-out data with all costs accounted.
