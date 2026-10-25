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

## P-006 — Bulk GitHub upload exceeded one-call orchestration limit (resolved)

- **Trigger:** initial batch attempted to create all 32 Git blobs inside a single code orchestration invocation.
- **Observed:** after the first 15 successful blobs, the orchestration hit a per-call tool invocation cap; no tree/commit was made, so the feature branch was still unchanged.
- **Root cause:** connector orchestration has a bounded number of tool calls per invocation.
- **Correction:** compute canonical Git blob SHA-1 for each local UTF-8 source; upload remaining blobs in smaller batches, then create **one atomic Git tree and commit** and fast-forward the feature branch with expected-SHA lease.
- **Verification:** GitHub returned feature commit `97f45b011ac4d0cba5bed5f694f376a7963620ff`, containing all 32 files. Direct `fetch_file` retrieved `src/sidra/agent.py` and `docs/PROBLEM_LOG.md` from the branch with matching blob hashes.
- **Impact:** intermediate blobs existed unattached; no partial project snapshot was pushed to the branch. The branch was only advanced after the complete 32-file tree was available.
