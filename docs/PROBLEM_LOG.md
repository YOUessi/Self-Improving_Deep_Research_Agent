# Problems, root causes, mitigations and evidence

Use one entry for every meaningful failure or design risk. Do not delete failed attempts. Each entry includes a command/trigger, observations, root cause, correction, evidence and remaining limitations.

## P-001 — GitHub initial tree could not be read (resolved)

- **Observed:** GitHub Contents API returned HTTP 404 `This repository is empty` when checking root `/contents/`.
- **Root cause:** user-created GitHub repository had no initial commit; contents endpoints require a Git tree.
- **Correction:** initialize `README.md` on `main` (commit `ee5b2066605bf206db6ea48c65eef80538a49bcf`), then create isolated branch `feat/m1-traceable-search-agent`.
- **Verification:** repository metadata returned admin/write permissions; branch created successfully.
- **Impact:** none; no existing files were overwritten.

## P-002 — Local pip editable install failed from network-restricted builder (workaround, CI pending)

- **Trigger:** `python -m pip install -e '.[test]'` in the temporary Linux builder.
- **Observed:** failed to download the isolated build dependency `setuptools>=68` due DNS/network restriction.
- **Root cause:** builder cannot reach package index; unrelated to SIDRA source.
- **Correction:** run `PYTHONPATH=src python -m pytest -q` with pytest already installed; GitHub Actions uses an online runner to validate package installation.
- **Verification:** tests completed under PYTHONPATH fallback (details below). Remote CI should be checked separately.
- **Remaining risk:** pinned `setuptools` availability on Tang must be checked; use an existing compatible venv if necessary.

## P-003 — A test file contained malformed Python string quoting (resolved)

- **Trigger:** first CPU pytest collection.
- **Observed:** `SyntaxError: unterminated string literal` in `tests/test_agent.py` around malformed-JSON parser case.
- **Root cause:** missing closing quote in a test literal (not the action JSON parser).
- **Correction:** repair the literal; rerun the full CPU test suite.
- **Verification:** `15 passed in 0.09s` (Python 3.13, local Linux; CPU tests only).

## P-004 — Synthetic smoke metrics can be misleading (mitigated by explicit labeling)

- **Observed:** generated templated dev fixture: hand-coded fixture multihop policy 8/8; fixture one-hop 0/8. These are **not** real LLM tasks.
- **Cause:** policy and dataset share an intentionally artificial grammar; test measures software pipeline and tool constraints only.
- **Correction:** policy names contain `fixture_*_smoke_only`; run manifest records `stage=smoke_fixture` and `no_gpu_training_claimed=true`; real HotpotQA distractor and a model-backed policy are a separate future experiment.
- **Verification:** `docs/EXPERIMENT_LOG.md` includes fixture provenance and values; README includes repeated warning.
- **Remaining risk:** real model results, external benchmark, GRPO and RSI improvements are NOT yet measured.

## P-005 — Potential answer/data leakage and fabricated citations (mitigated by tests)

- **Failure mode:** leaking `gold_answer` to the policy, training on dev/test examples, or citing a document that was never read would invalidate any claimed result.
- **Correction:** SearchAgent passes only question text + retrieved observations; API prohibits citation to unread IDs; export SFT raises on non-train examples and requires independently re-scored success; group split isolation is checked.
- **Verification:** automated tests for fake citations, prohibited read, group leakage and dev-to-SFT rejection pass.
- **Remaining risk:** semantic entailment is NOT proven by substring attribution check; add human-audited evidence evaluation before publication.

## P-007 — Tang could not clone GitHub directly (workaround applied)

- **Observed:** `git ls-remote` and direct HTTPS probes from Mac and Tang stalled/timed out; remote Git access could not be relied upon.
- **Correction:** commit and SHA-verify all files using GitHub connector; selectively relay the **already committed GitHub files** to Tang through authorized remote file operations. Record the exact GitHub snapshot SHA in a local `GITHUB_COMMIT.txt`.
- **Verification:** Tang reproduced 8/8 synthetic fixture smoke success, matching the CPU builder; later ran a pretrained Qwen 1.5B/3B inference baseline.
- **Remaining risk:** Tang local files are an execution mirror, not a separate canonical Git repository. Use `SIDRA_GIT_SHA` or `GITHUB_COMMIT.txt` in run metadata.

## P-008 — Incompatible system Pillow prevented Qwen2 import (resolved with a local shim)

- **Observed:** Transformers 4.57.6 on Tang crashed with `AttributeError: module 'PIL.Image' has no attribute 'Resampling'`.
- **Cause:** Tang has Pillow 9.0.1, but Transformers' image utility assumes newer Pillow API.
- **Correction:** in `policies_transformers.py`, before Transformers import, set process-local `PIL.Image.Resampling = PIL.Image` only if missing. Do not modify the system site-packages.
- **Verification:** direct `Qwen2ForCausalLM` import succeeded; Qwen2.5-1.5B and Qwen2.5-3B both loaded and generated actions on the GPU.

## P-009 — Missing accelerate blocked automatic device map (resolved for small models)

- **Observed:** `ValueError: ... device_map ... requires accelerate` during Qwen loading.
- **Cause:** `accelerate` is not installed in Tang's current Python, and network installation is unavailable.
- **Correction:** for **small cached instruct models that fit in 16 GB VRAM**, avoid `device_map` and call `.to("cuda")` after loading, in the optional pretrained Transformers policy.
- **Verification:** 1.5B and 3B models completed GPU evaluations.
- **Remaining risk:** the approach is not suitable for large/quantized models or distributed workloads. Install and version-pin Accelerate/vLLM separately before larger experiments.

## P-010 — Pretrained LLM fails strict evidence/tool-use on synthetic tasks (OPEN: research problem)

- **Observed:** 1.5B instruct model produced a valid initial search but malformed second-step action JSON on both dev questions (0/2 EM). 3B instructed model produced 1/2 EM, but neither answer passed complete evidence verification (0/2).
- **Cause hypothesis:** action formatting is unstable and the policy does not reliably discover/read both dependency documents; also chooses a premature final answer or repeats reads.
- **Action:** preserve failed traces, refine prompts/structured-output testing and plan/evidence policies, compare against no-retrieval/single-hop and fixed-budget controls, then only train on verified train traces. Do **not** mark as fixed until real held-out verification improves.
- **Verification:** raw synthetic GPU trace files are retained on Tang under `~/experiments/sidra_m1/results/`; a summary is in `docs/EXPERIMENT_LOG.md`.


### P-010 follow-up — Full synthetic dev split confirms incomplete evidence (2026-10-26; still OPEN)

- **Reproduction:** Tang Qwen2.5-3B-Instruct run `results/pretrained_qwen2p5_3b_dev8/` using GitHub snapshot `9da592898a3ec6140799b2c939ae057b4c9d5f76`.
- **Observed:** 8/8 episodes terminated with `finish`, but only 1/8 answered exactly correctly and **0/8 passed independent support validation**; mean supporting-evidence recall 0.1875, mean search actions 1.0.
- **Likely failure classes:** single-query search followed by premature answer; incomplete multi-hop reading; incorrect entity/doc selection; citing text that does not fully support the required chain.
- **Status:** **OPEN**. No prompt/harness fix, model improvement, SFT, GRPO or RSI success is claimed. Preserve the original failing traces; test prompt/schema controls on dev-only data before training, then evaluate on an independently frozen real benchmark.


## P-011 — Qwen3-8B cache is incomplete (OPEN)

- Observed on Tang: both `Qwen3-8B` and `Qwen3-8B-AWQ` snapshot folders contain only `config.json`; no tokenizer or model weights. Folder names had previously been mistaken for runnable checkpoints.
- Correction: implemented `scripts/audit_models.py` and regression tests that detect config-only models and missing shards.
- Verification: direct filesystem inspection. Qwen3-0.6B-Base and 1.7B-Base have complete weights, but they are base models.
- Open: obtain a complete, validated instruction-tuned Qwen3 model before fair comparison.


## P-012 — Tang lacks SIDRA AWQ/training dependencies (OPEN)

- Observed: checked Tang Python environment has PyTorch 2.10.0+cu128 / Transformers 4.57.6 but does not import `awq`, `vllm`, `accelerate`, `peft` or `trl`. HTTPS probe to Hugging Face timed out.
- Root cause: SIDRA quantization and training environments are not yet provisioned; CodeRL-Lab may have a separate environment.
- Mitigation: no silent installs or implicit model downloads. Use verified weights only and share one loaded checkpoint across roles to fit the 16 GB GPU.
- Next: inspect CodeRL's pinned environment; provision and test a separate reproducible SIDRA runtime before GPU SFT/GRPO.


## P-013 — Multi-role agents may leak evaluation labels or inflate budgets (CPU mitigated)

- Risk: planner/researcher/solver/verifier could share hidden gold labels, cite unread documents or receive more tool calls than a single-agent comparison.
- Mitigation: new `MultiAgentPolicy` routes tools through the original bounded SearchAgent. Three role prompts use question and observations only. `EvidenceAudit` requires distinct read citations and lexical support.
- Tests: role sequence, no gold labels in prompt/history, early-finish rejection, unread-citation rejection. CI passes.
- Remaining: lexical presence is NOT semantic entailment; real-model role collaboration may require extra inference calls and has not demonstrated a performance benefit.

## P-014 — M2 Tang mirror is not fully synchronized (OPEN)

- Observed: committing and running CI on GitHub succeeds; remote synchronization of one policy source file was rejected by the connected tool's safety checks. Tang cannot reliably fetch GitHub directly.
- Correction: do not claim M2 GPU inference from a partial mirror. Preserve the existing M1 Qwen2.5 baseline as the last actual model test.
- Next: use a permitted, verifiable synchronization method before running the matched multi-role model comparison.


### P-012 follow-up — CodeRL virtual environment is usable for future SIDRA training

- Further inspection found `/home/you/projects/CodeRL-Lab/.venv/bin/python` with Transformers 5.18.0, TRL 1.14.1, PEFT 0.21.2, Accelerate 1.15.0, Datasets 5.1.0 and PyTorch 2.10.0+cu128.
- Thus P-012's missing-library observation applies ONLY to Tang's system Python. A functioning, separately managed CodeRL training environment exists; `vllm` is still absent there.
- Decision: reuse that **environment and post-training methodology** for a distinct Search Agent experiment after dataset and verifier gates; do not overwrite CodeRL checkpoints or treat coding LoRA weights as Search Agent tuning.


## P-015 — Hugging Face HTTPS timeout while downloading Qwen3-8B (workaround operating; final verification pending)

- **Observed on Tang (2026-10-26):** DNS for `huggingface.co` returned `157.240.16.50`; curl TCP/443 timed out. Mac separately resolved `huggingface.co` to `108.160.165.173` and also timed out. Both are suspicious results, but root cause cannot be proven solely from these observations.
- **Comparison:** HTTPS to GitHub, `hf-mirror.com` and official ModelScope returned 200. Direct GitHub code repository is not a substitute for the Qwen3 model weight repositories.
- **Workaround:** isolated ModelScope 1.40.2 environment, official models `Qwen/Qwen3-8B` and `Qwen/Qwen3-8B-AWQ`, downloaded into separate `/home/you/models/` directories. Avoid system DNS changes and avoid modifying CodeRL-Lab's runtime.
- **Verification:** `scripts/verify_modelscope_models.py` pins official file byte sizes and SHA-256 digests. Both download processes started and transferred real `.safetensors.incomplete` bytes. **Neither checkpoint is marked ready until transfers finish and all SHA-256 checks pass**.
- **Reproduction / logs:** `docs/MODEL_DOWNLOAD_20261026.md`; Tang download logs under `~/experiments/sidra_download/logs/`.


### P-015 closure — ModelScope workaround fully verified (2026-10-26)

- Both Qwen3-8B BF16 (5 weight shards) and Qwen3-8B-AWQ (2 weight shards) completed with download process exit code 0.
- Independent full-file SHA-256 check matched the official ModelScope manifest for **every weight shard, tokenizer.json and index.json**: total 11/11 verified files, exit code 0, no unfinished shards.
- Source of verification: `docs/QWEN3_MODEL_VALIDATION_20261026.json` and Tang `~/experiments/sidra_download/verification_report.json`; CodeRL-Lab's existing environment successfully parsed both configs/tokenizers.
- **Resolution:** model downloads successful through the official ModelScope channel. The Hugging Face DNS/HTTPS outage itself remains unresolved and no system DNS changes were made. AWQ GPU inference is a separate pending validation.


## P-016 — Fetching updated GitHub branch from Tang stalled (resolved via pinned archive)

- **Observed:** original SIDRA GitHub clone completed and was validated, but a later `git fetch --quiet origin feat/m3-real-qwen3-hotpotqa-baseline` remained stalled for several minutes despite a successful `git ls-remote`; the remote `git remote-https` process remained idle.
- **Correction:** terminated only the stalled fetch and downloaded the exact committed source SHA `acfb4b32f59f2d1369486761907b342425cdaac5` as a GitHub `codeload.github.com` tarball, extracted into a new immutable-looking source snapshot directory, wrote `GITHUB_COMMIT.txt` for run provenance. This is a mirror of GitHub, not a new source of truth.
- **Verification:** extraction succeeded, Tang ran `PYTHONPATH=src python3 -m pytest -q` with **32 passed**. HotpotQA conversion and BM25 profiling ran from that snapshot.
- **Remaining:** repeat this snapshot verification for every subsequent GitHub commit used for GPU tests; never mix code files across commits.

## P-017 — vLLM dependency download stuck at z3-solver wheel (partially resolved)

- **Observed:** installing `vllm==0.31.0` in isolated `~/experiments/sidra_m3/.venv_vllm` using default PyPI repeatedly stopped during a 29.3MB `z3-solver==4.15.4.0` download. A prior unrelated infra project experienced the same stall.
- **Diagnosis:** direct PyPI wheel download was severely throttled, whereas Aliyun PyPI mirror returned fast partial content. This is a package distribution speed issue, not evidence that AWQ weights are damaged.
- **Mitigation:** terminated the stalled pip operation (not the other project's install), downloaded the exact z3 wheel from Aliyun, verified **SHA256=7e103a6f203f505b8b8b8e5c931cc407c95b61556512d4921c1ddc0b3f41b08e** against PyPI metadata, installed it to SIDRA's isolated venv with `--no-deps`, and restarted vLLM installation with `--index-url https://mirrors.aliyun.com/pypi/simple/`.
- **Remaining:** vLLM import, CUDA/AWQ compatibility and actual model inference must pass before closing the issue. Do not claim the model has run yet.


### P-017 follow-up — Recover vLLM 0.31 wheel from verified local pip artifact

- PyPI's official `vllm==0.31.0` wheel is **320,555,825 bytes**, official SHA-256 `39132b433967f0962d03d8a5f633b9bee6b32708f566c7d5637b79f8d0ef3433`.
- Tang retained two previous fully downloaded copies under `/tmp/pip-unpack-*/vllm*.whl`; SHA-256 matched. Copied one to SIDRA `~/experiments/sidra_m3/wheels/`, installed with `pip install --no-deps` into **only** `~/experiments/sidra_m3/.venv_vllm`.
- Slow default PyPI `z3-solver` download was mitigated using Aliyun mirror; wheel SHA-256 was checked against PyPI metadata (P-017 original).
- `pip install vllm==0.31.0` was resumed with Aliyun as the explicit index to complete dependencies; package import, CUDA compatibility and Qwen3-AWQ serve are **not yet validated**. The large optional dependency closure can take longer than the base wheel.
- Do not silently consider a `pip --no-deps` wheel install to mean a working inference environment. Check `pip check`, `import vllm` and actual inference separately.


## P-018 — Real multi-role evaluation worsened on first two HotpotQA examples (OPEN)

- **Evidence:** E-005 vs E-006 at GitHub code SHA `62e3874013d38c90b8677e9d9683b5598787fd52`; Qwen3-8B BF16/CPU offload, identical questions, prompt budget, top-k search and tool limits.
- **Observed:** single-agent answers 2/2 exactly correct but verified 0/2; multi-role answers 0/2 and verified 0/2, with more model calls and higher latency.
- **Failure 1:** Researcher third step returned no parseable action JSON; runner correctly terminated with a logged error.
- **Failure 2:** Researcher repeated a `read(doc_id)` that had already been performed, then attempted early finish; the role guard enforced abstention because two distinct sources had not been collected.
- **Hypotheses:** role instructions conflict with structured action schema, no recovery after malformed action, no deduplication of read actions, evidence gating applied too late.
- **Status:** **OPEN**. Preserve failed traces. Implement general structural fixes (not gold-driven answer patches), regression tests, and a new experiment on disjoint frozen dev question IDs. No performance gain may be claimed from this first multi-role version.

## P-019 — First multi-role launch had unbound shell variable (resolved)

- **Observed:** first Tang shell attempt used `set -u` with an unset `sha` variable; it exited immediately before launching a model or writing results.
- **Correction:** explicitly set `sha=62e3874013d38c90b8677e9d9683b5598787fd52` in the command. The corrected run produced E-006.
- **Verification:** first command had no generated checkpoint/evaluation artifacts; second completed with saved JSONL and manifest at the pinned SHA.


## P-020 — More JSON action parse failures on disjoint HotpotQA questions (OPEN)

- **Reproduction:** Tang E-007, GitHub SHA `69614339669341eb9aa7e5c0881e83179fa61997`, non-thinking Qwen3-8B BF16 CPU+GPU, `--offset 2 --limit 2`, `max_new_tokens 64`, exact matched tool budget.
- **Observed:** both single and multi-role policies produced malformed JSON at later turns. Single EM=0/2, strict verified=0/2; multi EM=0/2, strict verified=0/2 even with one bounded format retry; multi incurred +1 average model call and ~32 seconds/question additional latency.
- **Evidence:** both runs' raw step traces and manifests preserved on Tang; no policy reads evaluator gold fields. A short generation cap and lack of hard JSON-schema constrained decoding are potential causes; no root cause proven yet because raw completions are not logged.
- **Next controlled change:** add output-shape enforcement without leaking gold and compare fixed `max_new_tokens` settings, record per-step token counts and parse error frequencies, then test on fresh held-out question IDs. Do NOT declare an algorithmic gain based on retries or a tiny n=2 sample.
