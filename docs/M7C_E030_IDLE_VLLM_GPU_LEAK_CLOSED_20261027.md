# E-030 — Orphaned SIDRA RSI vLLM server occupying GPU (2026-10-27)

## Confirmed root cause

This was **our own SIDRA RSI E-027 test server**, not a separate user's GPU task or a foreign project. We should not have referred to it as "another task" without checking its launch provenance.

- Launch time: `2026-10-27 09:53:51+08:00`.
- Parent: `vllm serve` PID `86713` (PPID `3605`, Desktop Commander remote); engine PID `86847`.
- Model: locally cached Qwen3-0.6B-Base, served as `sidra-qwen3-06b-base` with STOP-LoRA, loopback `127.0.0.1:19082`, `--gpu-memory-utilization 0.58`.
- Actual stdout/stderr: `~/experiments/sidra_rsi_m7/logs/E027_vllm_qwen3_06b_stop.log`.
- At independent verification 11:13:40 local, ~**9,821 MiB** total GPU occupied (vLLM EngineCore 9,464 MiB), ~**6,099 MiB** VRAM free. The server had been alive for ~80 minutes. It had **no ESTABLISHED client connection**, and there were no executing SIDRA eval, GRPO or Challenger clients. Only the listening loopback socket remained.
- The server is a **lingering completed-experiment service**, not evidence that another authorized workload must retain GPU.

## Corrective action — executed, not merely planned

At `2026-10-27 11:14:20+08:00`:
1. Confirmed PID `86713` was still this exact `vllm serve` command, with served model name and port 19082; confirmed zero established connections and no active SIDRA model client.
2. Sent **SIGTERM only to PID 86713**. Did not signal/stop other programs.
3. Verified parent and its EngineCore child PID `86847` exited. Port 19082 closed. vLLM service log shows graceful MPClient/FastAPI shutdown.
4. Independently rechecked at 11:15:07: **349 MiB GPU used, 15,571 MiB free, 1% utilization, zero GPU compute processes, zero SIDRA vLLM ports (18980, 18981, 19082)**.
5. No TRAIN/DEV datasets, results, checkpoints or Git branches were deleted or overwritten.

## Process control change needed

**Mandatory for future RSI GPU work:** do not launch a persistent `vllm serve` and leave it running after experiment completion. All SIDRA GPU workstreams should:
- Acquire the *same* GPU lock `/home/you/experiments/sidra_shared/gpu0.lock` across the entire server + evaluation lifetime and check available VRAM before launch. The GRPO branch has a tested implementation at `src/sidra/gpu_lease.py` and `scripts/run_with_gpu_lease.py` (242 CPU tests, real contention verified).
- Start only a loopback-scoped, workstream-owned vLLM server and wait for health/ready.
- Track the spawned PID; put orderly server shutdown in a `try/finally` or shell `trap` so it runs on success, failure or interruption. Verify the listening socket and engine memory have disappeared; never kill unverified foreign processes.
- If another workstream already serves a compatible model, obtain agreed ownership or avoid starting a competing server. Do not treat a pre-existing SIDRA service as a permanent dependency unless a live workload actually needs it.
- On busy GPU: proceed with CPU/CI work and retry; never preempt unrelated processes.
- Never open sealed HotpotQA final-B160 for model selection, even when GPU becomes available.

**Status:** immediate orphan service **CLOSED and GPU VRAM RELEASED**. The systematic launcher/automatic shutdown requirement is documented but not yet installed across every existing RSI/GRPO launcher; do not claim all future leaks are impossible.
