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


## E-008 — Qwen3-8B BF16 generation-length ablation on the same dev2 (2026-10-26)

- **Motivation:** E-007 ended both single-Agent episodes in malformed action JSON; test whether the action generation cap was too short.
- **Code/data:** exactly the same Tang snapshot `69614339669341eb9aa7e5c0881e83179fa61997`, same HotpotQA original SHA-256 `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`, same seed42, same frozen dev offset2/limit2, Qwen3-8B BF16, nonthinking greedy, BM25 Top3, 6 steps, 5 tool calls. **Only max-new-tokens changed from 64 to 160.**
- **At 64 tokens (E-007):** answer EM 0/2, verified 0/2, evidence recall 0.0, both runs terminated with parse errors, mean 4 model calls, 3021 prompt tokens, 142 completion tokens, 72.4175s/question.
- **At 160 tokens (E-008):** answer EM **0/2**, verified **0/2**, evidence precision **1.0**, evidence recall **1.0**, both runs **finish** normally, mean 4.5 model calls, 3575.5 prompt tokens, 230.5 completion tokens, **116.804s/question**. Mean token F1 0.15.
- **Per case:** `5a79f6825542996c55b2dcb4` generated `1951` with both cited supporting documents but wrong final answer; `5a7b3ec95542995eb53be8d3` generated a verbose response identifying people/appointment context, also wrong answer though it cited both gold supporting documents. Both were true `search/read/finish` trajectories, not rule fixture agents.
- **Interpretation:** increasing the output cap fixed parse termination for **these two cases only** and permitted complete citations, but did NOT improve answer correctness and increased latency. The exact causal mechanism for parse errors is not proven (length vs schema robustness); strict JSON Schema constrained decoding should be separately compared in the live vLLM backend.
- **Artifacts:** Tang `~/experiments/sidra_m3/results/holdout_offset2_transformers_gen160/` with full `summary.json`, `traces.jsonl`, `per_example.jsonl`, `manifest.json`. Model not trained. Do not claim RSI.


## E-009 — Real Qwen3-8B-AWQ vLLM inference with constrained tool JSON (2026-10-26 evening)

- **Hardware:** Tang RTX 4090 Laptop, 16GB VRAM. Official checksum-verified Qwen3-8B-AWQ under `/home/you/models/Qwen3-8B-AWQ`.
- **Isolated runtime:** `~/experiments/sidra_m3/.venv_vllm`, vLLM **0.31.0**, PyTorch **2.13.0+cu130**, CUDA available. Prior failures from slow PyPI wheels were resolved by verified local wheel reuse, with remaining dependencies installed in this venv. There are `pip check` warnings related to inherited Mediapipe/colcon in the system-site-packages environment, unrelated to the successful AWQ smoke; package isolation should be strengthened later.
- **Server:** loopback-only `127.0.0.1:18980`, `vllm serve /home/you/models/Qwen3-8B-AWQ --served-model-name sidra-qwen3-8b-awq --gpu-memory-utilization 0.70 --max-model-len 2048 --enforce-eager`; model weights loaded with AutoAWQ-Marlin kernel and FlashAttention 2; `GET /v1/models` returned **HTTP 200**. Measured GPU memory at server idle around **11.5 GB**.
- **Tool-action request:** real HTTP `POST /v1/chat/completions` with Qwen3 `enable_thinking=false` and OpenAI-compatible strict JSON Schema (`kind` enum search/read/finish; query/doc_id/answer/citations fields).
- **Response:** syntactically valid JSON satisfying the declared schema, `kind="search"`, `query="Tokyo Tower location"`, measured response time **1.632s**, prompt **64 tokens**, completion **68 tokens**.
- **Important failure flag:** despite search action, model also filled `answer="Tokyo Tower is located in Tokyo, Japan."`, `citations=["https://en.wikipedia.org/wiki/Tokyo_Tower"]` before executing any search. These were unearned assertions and are NOT valid evidence citations. The runner must enforce action-kind semantics and read-before-cite; syntactic JSON Schema alone cannot prove grounded behavior.
- **Raw output:** `Tang:~/experiments/sidra_m3/results/awq_schema_smoke_20261026.json`. This is an isolated **single-turn inference smoke**; not multi-hop benchmark quality, SFT, RL, or RSI.
- **Follow-up:** server restarted with a 4096-token context to accommodate multi-hop tool history, and a matched single vs multi-role dev QA experiment was launched from SHA `a7db2111898419986103a1af33bbb090a44410ca`, on four previously unused ID-hashed HotpotQA distractor dev questions (offset4, limit4). Do not report outcomes before raw results complete.


## E-010 — AWQ-vLLM JSON action schema failure and discriminated-union fix (2026-10-26 evening)

- **Model/host:** locally served, checksum-verified Qwen3-8B-AWQ; vLLM 0.31.0, PyTorch 2.13.0+cu130 on Tang RTX 4090 Laptop. Loopback `127.0.0.1:18980`, Qwen3 nonthinking mode, max_model_len=4096. Real HotpotQA distractor dev, frozen seed42; diagnostic QA positions **4–7** (4 questions), previously unused by BF16 work.
- **Loose-schema control:** GitHub source snapshot `a7db2111898419986103a1af33bbb090a44410ca`; one flat schema required all five fields for search/read/finish, with unbounded strings. 4/4 episodes failed to parse FIRST action, `completion_tokens=400`/episode, **0 tools used**, verified=0/4. Single average latency 5.265s, multi 10.71525s with two failed inference calls.
- **Root cause reproduction:** directly replayed first question with the same Qwen3 prompt and loose schema. vLLM returned HTTP 200 with `finish_reason=length`, 400 completion tokens, and an unfinished `query` string containing repetitive whitespace; JSON parse error at end of response. Raw diagnostic summary is stored only on Tang `results/awq_schema_action_debug_20261026.json`.
- **Fix:** GitHub commit `b7067faa937cd137f32487dec9e1277ff43a7d97` (source after schema and test update). Replace flat schema with **oneOf three disjoint action objects**: search requires only bounded query (maxLength120); read requires bounded doc_id (maxLength120); finish requires answer (maxLength100) and max5 bounded citations. All variants disallow extra properties; gold labels remain evaluator-only.
- **Direct identical first-question probe:** with union schema + max_tokens160, returned complete `{"kind":"search","query":"City of Boston Christmas tree source Ainslee Glen Nova Scotia"}` in **21 completion tokens / 0.718s**, `finish_reason=stop`; saved at `results/awq_schema_union_probe.json`. This proves syntactic repair only.
- **Same 4 diagnostic questions, repaired schema:** single Agent answer EM **1/4 = 0.25**, verified **0/4**, supporting-doc recall **0.5**, average 2.25 tool calls, 3.25 model calls, 121.25 output tokens, 1.904s/question, three finishes and one error. Multi-role answer EM **0/4**, verified **0/4**, support recall 0, average 3.0 tool calls, 5.5 model calls, 214.5 output tokens, 3.6395s/question, all four abstained.
- **Interpretation:** repair changes FIRST-step behavior from 4/4 schema parse errors and zero tools to an executable Agent workflow. It **does not** establish improved multi-hop reasoning, evidence correctness or multi-role benefit. The 4 QA items were the diagnostic set used to fix the schema, not a clean independent test. A separate offset8–15, eight-QA comparison was queued as the next independent test.
- **Raw artifacts:** Tang `results/qwen3awq_schema_openai_offset4_4`, `results/qwen3awq_schema_multi-openai_offset4_4`, `results/qwen3awq_union_openai_offset4_4`, `results/qwen3awq_union_multi-openai_offset4_4` and their manifests/traces. Do not overwrite.


## E-011 — Independently selected eight real HotpotQA dev questions, Qwen3-8B-AWQ strict actions (2026-10-26)

- **Purpose:** validate the length-bounded discriminated action JSON schema on questions **not used** for the E-010 parser/schema fix. Compare one-model single-Agent and sequential multi-role (Planner/Researcher/Solver/EvidenceAudit) in a matched vLLM runtime.
- **Data and selection:** same official HotpotQA distractor dev SHA-256 `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`, frozen ID-hashed seed42 24-question set, **sorted offset=8, limit=8**, thus disjoint from diagnostic offset4–7. This is *per-question 10-candidate distractor passage retrieval*, NOT full Wikipedia.
- **Code:** exact canonical source SHA `b7067faa937cd137f32487dec9e1277ff43a7d97` mirrored to Tang and tested (38 passed); not trained. Both runs matched: Qwen3-8B-AWQ, vLLM 0.31.0, `enable_thinking=false`, strict bounded `oneOf` action JSON, greedy temperature0, `max_steps=6`, `max_tool_calls=5`, BM25 `search_top_k=3`, same question IDs, single GPU serving endpoint localhost:18980 with max_model_len4096.
- **Single Agent:** answer EM **2/8 = 0.25**, token F1 0.270833, strict verified success **0/8**, evidence recall 0.3125, mean model calls **4.25**, input tokens **3717.75**, output tokens **149.25**, tool calls **3.25**, elapsed **2.6775 s/question**; 5 finished, 3 errored.
- **Sequential multi-role:** answer EM **2/8 = 0.25**, token F1 0.25, strict verified success **1/8 = 0.125**, evidence recall 0.1875, mean model calls **5.25**, input tokens **5225**, output tokens **178.75**, tool calls **3.5**, elapsed **3.450875 s/question**; 2 finished, 6 abstained.
- **The one verified case:** question ID `5a8a5c0d554299515336137b` was answer-exact for both, but single-Agent cited only half of gold supporting documents (evidence recall0.5); multi-role cited both (recall1.0), passing the current benchmark verifier. The verifier uses answer exact-match, gold support completeness, read-before-cite and lexical presence; it is NOT a full natural-language entailment proof.
- **Interpretation:** one positive multi-role verified-evidence example but **no overall answer-EM gain**; multi-role incurred extra calls, 40.5% more average input tokens and ~29% more elapsed wall-time. Eight QA examples are insufficient for a claim of statistical advantage. Do not advertise this as RSI/post-training improvement.
- **Raw outputs:** `Tang:~/experiments/sidra_m3/results/qwen3awq_union_openai_holdout_offset8_8/` and `.../qwen3awq_union_multi-openai_holdout_offset8_8/`; summary/per-example/traces/manifests are unmodified, private artifacts not committed. M3 issue: in the original multi-role manifest `model_id` appeared as null because the wrapper hid `model`; the underlying endpoint and checkpoint were verified from run commands, process logs and code. This provenance issue was fixed for future runs in `MultiAgentPolicy` after E-011; historical manifests are not silently edited.


## E-012 — Qwen3-0.6B Base versus 40-step TRAIN-only LoRA/SFT on frozen pilot32 (2026-10-26)

- **Training completed:** 40 optimizer steps, LoRA rank16 on Qwen3-0.6B-Base, 1,299 action training examples (312 distinct train questions), 192 internally validated action examples (47 distinct train questions), no dev labels in optimizer. Adapter written to Tang `~/experiments/sidra_m4/results/qwen3_06b_action_sft40_seed43/`, SHA-256 `db6687aaa8b9f01ed288676ff7d9745c27e753ed2adc7833a1ec3fe8d689fda5`. Train loss **0.114768**, train wall-time **63.49 s**, 10,092,544 trainable parameters. This is a real weight update, **NOT GRPO or RSI**.
- **Provenance:** train script from GitHub snapshot `224719917f7e4fa5e868ed0bfe480762c9e18495`; both eval policies used snapshot `16054b8850adabf87dabd2a1ad0aa230a063559d`. Tang CodeRL-Lab venv, Transformers 5.18.0 / TRL 1.14.1 / PEFT 0.21.2; Qwen3-0.6B-Base model snapshot `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. Dev pilot32 is a fixed HASH-selected subset, disjoint from TRAIN.
- **Identical evaluator settings:** local `transformers` policy, Qwen3 non-thinking chat template, greedy decoding, `max_new_tokens=160`, `max_steps=6`, `max_tool_calls=5`, BM25 top3, no title boost. The only change is loading the trained LoRA adapter in the treatment.
- **Untuned Base result (n=32):** answer EM **0/32**, strict verified **0/32**, error **28/32**, tool-budget exhausted **4/32**, read calls **0.0**, tool calls **1.5625**, model calls **2.5625**, latency **2.535 s/question**, all had no final answer.
- **LoRA/SFT result (n=32):** answer EM **1/32**, strict verified **1/32**, error **5/32**, tool-budget exhausted **26/32**, finish **1/32**, mean read calls **2.90625**, tool calls **4.875**, model calls **5.875**, latency **5.648 s/question**; abstention rate **31/32 = 96.875%**.
- **Interpretation:** train-only SFT improved ability to produce and execute actions and produced one verified success, but the student generally loops until the budget is exhausted. This is a negative/weak result for reliable long-horizon reasoning and search-stop decision. Results are only on pilot32, not on heldout; do not attribute the difference solely to reasoning capability rather than changed action-format behavior. Compare within same 0.6B base; **do not compare its absolute EM directly to the 8B-AWQ serving system** as an equal-model ablation.
- **Raw immutable local data:** `~/experiments/sidra_m4/results/sft06b_pilot32_baseline/` and `.../sft06b_pilot32_adapted/`, each with full manifest, traces, per-example metrics and summary.
- **Next gate:** improve the student action grammar and finish/stop supervision; inspect loop failure cases on pilot only, then independently validate on eval-A64 with a fixed revision. Do not open sealed final-B160 for tuning.


## E-013 — TRAIN-only SFT stop/abstain curriculum, two real 0.6B LoRA runs (2026-10-27)

- **Motivation:** M4 Qwen3-0.6B SFT pilot32 produced 70 Search + 112 Read but only ONE Finish, 26 tool-budget exhausted. Offline trace-forensics confirmed 35 *actual duplicate read* events and 12 duplicate searches. Its 3,304 teacher TRAIN actions were Search1,096 / Read1,472 / Finish736 (**22.28% Finish**) and SFT prompt hardcoded `remaining_steps=8-step` while benchmark runtime has `max_steps=6`. No dev answers used in failure-repair corpus.
- **Control curriculum:** `scripts/build_stop_curriculum.py --mode format` uses exact same 736 verified TRAIN question IDs and 3,304 actions, normalizes `remaining_steps=6-step`, and uses per-kind minimal JSON tool actions. The TRAIN-only SFT source SHA256 `ff1a06377ecf99af1b7f9fc937910c2ffb94927c9a191ceb8497dc63c0086abe` was held fixed.
- **Stop curriculum:** same 3,304 base actions + **736** last-turn verified Finish examples (only from actually read TRAIN evidence), **276** no-evidence last-step abstentions, **150** one-read insufficient-budget abstentions. 4,466 examples total: Finish1,898 (**42.50%**), Read1,472, Search1,096. No DEV data/labels or fabricated positive retrieval texts enter optimization.
- **Training:** independent Qwen3-0.6B-Base checkpoint `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, seed43, 40 optimizer steps, LoRA rank16 BF16, lr0.00015, batch1 grad_accum8, max length1536, 1,600 max pre-token-filter samples. TRAIN/inner-val partitioned by question ID.
- **Format-only LoRA:** 1,306 effective train and 194 internal-val action samples; train loss **0.1049003**, 64.22s Tang GPU, adapter SHA256 `cf926d0c2b033bed986b409c8ca71ce8fc26f04e85b3d08e5090d8275007fff9`. 
- **Stop LoRA:** 1,249 train and 213 internal-val samples, loss **0.0967193**, 68.32s, adapter SHA256 `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`. Differences in per-question counts reflect curriculum over-sampling/cap; do not attribute loss differences to stop skill.
- **Code:** pinned GH SHA `616acbba3588fa000a99e37c6ae8971a57a81a58`, **69 CPU tests passed** on Tang. Separate adapters and raw logs are under `Tang:~/experiments/sidra_m5/results/qwen3_06b_sft40_*/`; no weights pushed to Github.
- **Interpretation:** real independent parameter updates, but **not GRPO/RSI**; post-training quality must be evaluated with equal real tools and sampling. No claim is made from lower training loss alone.

## E-014 — 0.6B same-vLLM pilot32 static JSON schema vs observation-bound DocIDs (2026-10-27)

- **Inference:** Qwen3-0.6B-Base with 3 individually loaded LoRA adapters (old/format/stop) through ONE vLLM 0.31.0 service (localhost port18981, greedy, Qwen3 non-thinking). Same 32 frozen genuine HotpotQA dev pilot questions, BM25 Top3, 6 max steps, 5 tools, same `OpenAICompatiblePolicy`, same structured JSON action schema. This is PILOT, not final-B heldout.
- **Static action schema (no online document-ID enumeration):**
  - Base: answer0/32, verified0/32, budget exhaustion32/32 (5 search calls per question, zero reads).
  - Old SFT: answer0/32, verified0/32, **29/32 execution error**, 3 budget exhaustion. Among errors, invalid/unexposed doc reads and unread citations were predominant.
  - Format-only SFT: answer0/32, verified0/32, **31/32 budget exhaustion**, 1 error. Extra correct syntax did not help stopping.
  - Stop SFT: answer0/32, verified0/32, **32/32 execution error**, often premature finish citing un-READ documents. Training Finish frequency alone is not sufficient for grounded behavior.
- **Protocol improvement:** added `src/sidra/observed_action_schema.py` and `--constrain-document-ids`: dynamically permits only previously exposed *unread* IDs for read and previously READ IDs for citations. Gold QA labels/support annotations are NEVER consulted. New tests cover no-hit, duplicate read exclusion, and independence of malicious/label-like document metadata (GitHub/Tang CPU tests 75 passed).
- **Dynamic-ID 32-QA repeated comparison, still on PILOT32:**
  - Base: EM0/32 verified0/32; 32 budget exhaustion, 5 searches per task, no reads.
  - Old SFT: EM0/32 verified0/32; 23 abstentions, 6 errors, 3 budget exhaustion; calls4.125, read0.84375, latency4.555s/question.
  - Format SFT: EM0/32 verified0/32; 30 budget exhaustion, 1 error, 1 finish; calls5.844, read2.0, latency3.955s.
  - Stop SFT: **EM7/32 (21.875%)**, verified **0/32**, **32/32 normal Finish**, *no execution error*; mean 1.0 read, 1.0625 search, model calls3.0625, latency2.581s. It commonly finished after only ONE read, so citation support was still incomplete. This is **premature stop** despite improved answer and efficiency.
- **Matched stop-vs-old within dynamic-ID pilot32:** paired answer EM delta +21.875pp, 95% paired bootstrap [9.375pp,37.5pp], whereas strict verified delta=0; stop used fewer calls and input tokens. Small PILOT, only one seed, not generalization proof.
- **Raw, never overwritten:** `Tang:~/experiments/sidra_m5/results/vllm_pilot32_*/`, `docenum_pilot32_*/`, `paired_docenum_stop_vs_old_pilot32.json`, `action_loops_docenum_stop_vs_old_pilot32.json`.
- **Next:** apply identical train-independent evidence guard (at least two real reads) to old/stop models under dynamic ID constraints, score strict passage support as PRIMARY outcome, then choose one policy and evaluate once on independent eval-A64. Sealed final-B160 remains untouched. Do not claim Agentic RL or multi-hop capability gains from answer-only pilot results.
