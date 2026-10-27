# M7-A9 — preregistered TRAIN64 historical replay, actual CPU validation, GPU non-preemption

**Date:** 2026-10-27 (Tang time +08:00). **Status:** CPU engineering gate COMPLETE; **M7-A8 GPU experiment remains BLOCKED by an independent vLLM service.** This stage creates **no** optimizer step, new adapter, model QA improvement or final/test conclusion. Existing M7-A8 preregistration remains unchanged.

## 1. Read-only verification before new work

- Branch: `feat/m7-onpolicy-agent-grpo`; PR #10 remains draft. Start-of-run branch SHA `1b09e6b79db5147a46a659ee48cb003246e30dee`; latest CI there successful in Python 3.10 / 3.12.
- Tang online, RTX 4090 Laptop GPU 16GiB. At first read-only inspection, **9,821 MiB allocated and ~6,099 MiB free**. A separate vLLM server parent PID `86713` and engine PID `86847` were active; independent RSI CPU-only tests were also running. **No GPU model launch/train/eval or foreign process signal attempted.**
- Existing M7-A8 failed experiment log confirms why full A8 is pending: after 80 true TRAIN episodes at earlier 2026-10-27 09:53 +08:00, an unexpectedly concurrent vLLM engine allocated about 9.2GiB; first optimizer backward raised `CUDA out of memory` (requested ~412MiB, ~251MiB free). This left NO A8 trained adapter and NO completed A8 train manifest.
- Frozen original TRAIN1024 questions SHA256 `53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8`, source documents SHA256 `c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111`; public DEV freeze manifest SHA256 `3f0c494cca8b44cfef90c78d4d5b1fbb4158dc7387ed0f24316a0a265471d947`. No DEV labels, eval-A64 or sealed final-B160 were opened for this work.

## 2. New source-of-truth provenance checker

The old `scripts/audit_frozen_train64.py` verified checksums against **values inside the same editable FREEZE_MANIFEST**, plus disjointness from the currently evaluated fit, but did not independently replay the original 64-question selection against the **historical teacher/training inputs**. A future modification to both freeze data and its manifest could bypass that local consistency check.

Added `scripts/audit_a8_frozen_integrity.py` with reproducible fail-closed checks:

1. SHA256 pins the original source TRAIN documents and question file, frozen dataset bytes, and **all six** original TRAIN teacher-action source files and **eleven** real previously completed TRAIN optimizer summary files. No raw QA is committed.
2. Reconstructs the historical set of **402 distinct previously teacher-labeled or optimizer-used TRAIN question IDs** from those 17 actual immutable sources. The original 1,024 TRAIN questions leave **622** eligible unused IDs.
3. Deterministically ranks the eligible questions by SHA256(`M7A8-FROZEN-HOLDOUT-919:<question_id>`), selects exactly 64, recomputes ordered ID digest and requires exact byte-for-byte equality between every frozen question record and the original TRAIN record.
4. Verifies the pre-registered frozen selection digest `336fa32d645a2b52cf68563a56dc9d6571819971c872a92cfb790e2876fcda81`, frozen question bytes SHA `c310a1cc2f23ab87925d24c6392ea51afcfbde29018f40300a588972eed60c92`, and source PIN `9d81deca7dc4efe72170369f89d37cb4fe49d815`. A mismatch or missing input aborts.
5. Re-checks all completed *newer* M7 TRAIN_SUMMARY files and blocks if a later optimizer used even one of the frozen 64 question IDs. This avoids mistakenly describing previously fitted examples as unused.
6. Never reads public DEV answer labels, eval-A64 or sealed final-B160. This remains **within TRAIN** and is not a true external or untouched benchmark.

Changed `scripts/audit_frozen_train64.py` CLI to require `--m7-root` and `--original-train-source`, and **run the independent pinned replay before scoring** a paired TRAIN64 comparison. Added tampering, original-label drift, historical hash, future-fit contamination, and CLI bypass-rejection unit tests.

## 3. Real machine validation and retained artifacts

Pinned testing SHA **`6d21955ab69340cccb1af9706a698cef811581a5`**.

- On Tang: **222 / 222 CPU pytest tests PASSED**, `compileall` PASSED; log `~/experiments/sidra_m7/logs/M7A9_cpu_pytest_6d21955a.log`, immutable source snapshot `~/experiments/sidra_m7/source_6d21955ab69340cccb1af9706a698cef811581a5/`.
- Independently replayed the **real original** 1,024-question TRAIN source and **all 17** SHA-pinned teacher/optimizer histories, reproducing exactly **402 historical exclusions, 622 unused questions, 64 frozen IDs, and both frozen dataset SHA256 checksums**. Result `status=PASS`; report stored on Tang `~/experiments/sidra_m7/results/M7A8_PREREG_FREEZE_INTEGRITY_AUDIT_a239e362.json`. This is a real data provenance check, not a claim of new training.
- The first new-source CPU bootstrap attempt failed cleanly because its shell used an unset variable (`$sha`) with `set -u`. No model process or data rewrite occurred. Retried in a newly named immutable source directory with the pinned literal SHA; the tests and provenance replay then passed.
- Afterward GPU remains at **9,821 MiB allocated / 6,099 MiB free**. A real read-only `assert_no_external_gpu_jobs()` run explicitly rejected parent PID `86713` and engine PID `86847` as `llm_server`, without printing command-line secrets or interacting with either PID.

## 4. Next research gate — unchanged preregistration

**Do not run M7-A8 on the contested GPU.** When the foreign vLLM workload exits naturally and device is actually available, rerun the pre-registered M7-A8 branch from a fresh immutable code SHA and **the original stop-SFT** adapter. Keep TRAIN seed 153, V4 nonoracle catalog and identical masked categorical behavior/loss normalization, 20 distinct TRAIN tasks x 4 genuinely sampled complete Search/Read/Finish episodes x 2 updates, same 5-tool/6-step budget, lr3e-6, episode-total group-relative reward, FINISH and supporting-READ TRAIN-only supervision weights 0.2 each, actual GPU lock, two process-level exclusive checks and no preemption. Save group reward variability, action-token masking, old/new logprob replay error, actual optimizer delta, final adapter SHA and full per-episode TRAIN logs; abort if any check fails.

Only after a successful train manifest / new model SHA may the paired scorer be run with mandatory **pinned 17-file historical replay** on the frozen TRAIN64 (one seed 171) and viewed public DEV32 (seeds 151,157). Use paired question-cluster intervals, strict answer+original support, retrieval completeness, FINISH errors, calls/cost. Treat TRAIN64 as disjoint *internal TRAIN* only and the public DEV32 as repeatedly inspected. A hybrid supervisory effect is **not** evidence of pure GRPO gain. No checkpoint may be promoted without independently verified QA benefit.

**No final-B160 access, no model weights or private QA uploaded, no competing process interruption.**
