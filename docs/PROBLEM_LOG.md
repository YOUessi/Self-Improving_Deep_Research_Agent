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
