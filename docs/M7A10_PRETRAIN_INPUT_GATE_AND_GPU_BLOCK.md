# M7-A10 — CPU-only A8 launch provenance, exact TRAIN membership and contested GPU

**2026-10-27 (+08 Tang local).** This is a real source/experiment guard, **not an online GRPO update or a new QA outcome**. All existing M7-A8 pre-registration parameters remain **unchanged** (see `M7A8_READ_PLUS_FINISH_PREREG.md`). Final-B160 and eval-A64 remain sealed.

## Why a new gate is useful

M7-A9 proved that the internal TRAIN64 was selected correctly based on original TRAIN1024 and seventeen SHA-pinned prior teacher/optimizer histories. That check did not itself prove that the **next attempted GPU launch** would use exactly the original stop-SFT parent, the right TRAIN-only teacher, 40 distinct allowed optimizer questions disjoint from frozen TRAIN64, an unmodified DEV manifest, and the preregistered A8 command. A wrong path/weight file or a new subset could make an otherwise well-executed GPU job uninterpretable.

Added **`scripts/audit_a8_train_launch.py`** (CPU-only) and **`tests/test_audit_a8_train_launch.py`**. They enforce, before any CUDA import/model launch:

1. SHA pin the original 1024 TRAIN question/document files, frozen public DEV manifest and source snapshot (via `GITHUB_COMMIT.txt`). Require the **existing A9 independent historical replay** to return PASS (all 6 original teacher files + 11 prior completed real optimizer summaries, and any later fit pollution).
2. Pin the exact original stop-SFT adapter binary SHA256 `44e559a3d6bc61f46b5a736d0c8ff4f896c281e2139ff12f967ed4b7cc69d771`; reject any parent weight/config mutation.
3. Validate the TRAIN gold teacher's provenance, required `student_model_sees_gold_answer=false`, original TRAIN+DEV hashes, V4 action proposer, 6-step/5-tool limits, every teacher action's `student_model_gold_injected=false`.
4. Recompute the 40 *actual* deterministic optimizer QIDs using `select_train_only()` and A8 seed153, groups20, updates2. Reject duplicate/non-TRAIN IDs, source vs frozen DEV overlap, teacher train vs heldout overlap, and the 64 frozen TRAIN evaluators appearing in teacher train/heldout/selected IDs.
5. Report the exact preregistered experiment values (V4, max_context2048, groups20x4, 2 updates, lr3e-6, FINISH/READ auxiliary weights each 0.2, episode_total). Inspect external GPU launchers with the existing **read-only non-preemptive** `find_external_gpu_jobs()`; never return permission to use GPU merely because CPU input checks pass. Do not dump any raw question, gold label, prompt, vLLM argv or secret.

## Actual GitHub and Tang validation

- Pinned tested/committed source SHA: **`909c591d03bf18c9314fc743410cdd3f27549636`**.
- Tang full CPU suite: **232 passed in 2.02s**; compileall passed. GitHub CI two runs for this commit: **SUCCESS**.
- Immutable snapshot: `~/experiments/sidra_m7/source_909c591d03bf18c9314fc743410cdd3f27549636/`.
- Logs: `~/experiments/sidra_m7/logs/M7A10_CPU_TEST_909c591d03bf18c9314fc743410cdd3f27549636.log`; `~/experiments/sidra_m7/logs/M7A10_A8_PREFLIGHT_909c591d03bf18c9314fc743410cdd3f27549636.log`.
- Output audit: `~/experiments/sidra_m7/results/M7A10_A8_PREFLIGHT_909c591d03bf18c9314fc743410cdd3f27549636.json`.

The executed preflight returned **`CPU_INPUT_GATE_PASS_GPU_BUSY`**:
- Original TRAIN count **1024**. Teacher curriculum training questions **240 unique**, separate teacher heldout **60 unique**; frozen internal TRAIN64 **64**.
- Selected exactly **40** distinct optimizer questions; selected ID SHA256 `449d09935e9dee7f22e7fb1bafb77a94346c759c81600bed6c23e0358034184c`.
- Selected ∩ frozen TRAIN64 = **0**; selected ∩ teacher heldout = **0**; selected ∩ public DEV = **0**.
- A9 historical replay: **PASS**; original frozen ID digest `336fa32d645a2b52cf68563a56dc9d6571819971c872a92cfb790e2876fcda81`.
- Stop-SFT parent SHA pinned and confirmed; experiment controls exactly match pre-registration.

### Failure caught during implementation

The first direct `python scripts/audit_a8_train_launch.py` invocation failed because the new script used a `scripts.*` import while `PYTHONPATH=src` and Python's current module search path did not include the project root for direct script execution. This was a **CPU preflight packaging failure**, **not a GPU model failure**. Fixed CLI/package dual-import routing in SHA `909c591d...`; rebuilt a separate immutable source snapshot, reran the entire CPU suite, and then the **actual A8 input audit passed**. Original failed logfile was retained. This issue cannot silently alter the original training protocol.

## GPU gate: still BLOCKED without preemption

At last inspection Tang GPU reports **9821MiB allocated / 6099MiB free**, with foreign vLLM parent PID **86713** and EngineCore PID **86847** (around 9464MiB GPU allocation by the engine). These are NOT ours to stop. CPU-only inspection reports two external `llm_server` processes. A8 was **not launched**, no optimizer steps ran, no adapter produced, no QA paired evaluation run. The preregistered A8 prior OOM log remains an important failure record.

## Next eligible real task

When the competing vLLM workload terminates naturally, re-check both process ownership and GPU memory immediately before train, acquire the project's exclusive nonblocking `flock`, and rerun this input preflight on the newest pinned tested source. Only if both gates pass, launch **unchanged** A8: original stop-SFT, TRAIN seed153, 20×4 fresh tool rollouts per update, 2 updates, FINISH+READ TRAIN-only 0.2 auxiliary weights, max5 calls/6 steps, strict categorical action-mask probability equality. Abort on competing GPU launch without signaling others; record distribution mismatch, reward variance, TRAIN-only supervision, optimizer delta and checkpoint SHA.

Only after real completion compare same-policy arms on frozen internal TRAIN64 (seed171) and viewed DEV32 (seeds151/157) with clustered paired intervals, QA EM, strict original support, all tool costs. The TRAIN64 is **not externally independent**, the A8 hybrid is **not pure GRPO**, and a zero/negative QA delta cannot be reported as a training gain.

No final-B160, eval-A64, user-private data, model weights or raw QA copied to GitHub.


## Follow-up: immutable 40-question optimizer selection pin

The first preflight reported the selected 40-question sequence SHA but did not reject a different **still-TRAIN-only** optimizer subset if later source code changed the deterministic selection function. Added `EXPECTED_A8_TRAIN_SELECTION_SHA256` and `assert_exact_preregistered_optimizer_selection()`, which now **fail closed** if either the number or SHA of the actual ordered A8 optimization tasks differs. Regression tests explicitly substitute a new (otherwise legal) digest and a wrong count. The previous TRAIN64/DEV/teacher isolation checks still run.

Re-tested after this guard using newest pinned source **`7ddb6d8e3ffab7b4afe4d4cab717c147d3645aab`** on Tang: **242/242 CPU tests passed (2.13s)**; reran the complete **real** original TRAIN1024+teacher+frozen TRAIN64+parent adapter CPU preflight. Result **`CPU_INPUT_GATE_PASS_GPU_BUSY`**, selection SHA `449d09935e9dee7f22e7fb1bafb77a94346c759c81600bed6c23e0358034184c` verified against the hard-coded pin, selected/frozen overlap **0** and two foreign GPU workers detected. No GPU model launch, no optimizer step, no QA promotion, no final-B160.

Proof log `~/experiments/sidra_m7/logs/M7A10_CPU_TEST_7ddb6d8e3ffab7b4afe4d4cab717c147d3645aab.log`; preflight `~/experiments/sidra_m7/results/M7A10_A8_PREFLIGHT_7ddb6d8e3ffab7b4afe4d4cab717c147d3645aab.json`. Resource gate remains strictly non-preemptive; the old A8 training protocol remains intact.
