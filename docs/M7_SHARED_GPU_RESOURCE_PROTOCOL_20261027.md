# Shared GPU lease for M7 parallel workstreams — 2026-10-27

## Problem it addresses

Previous online-GRPO experiments on Tang were interrupted by an unrelated
vLLM engine appearing after the training rollout phase: CUDA out-of-memory
aborted the first backward pass, leaving no valid updated checkpoint.
M7-A9 documented this and correctly kept the blocked run out of accuracy claims.

**All three M7 workstreams must use ONE shared GPU lease**, even though they
write distinct GitHub branches and experiment folders. CPU tests do not need
this GPU lock.

- Default shared GPU0 lock path:
  `/home/you/experiments/sidra_shared/gpu0.lock`
- Guard implementation: `src/sidra/gpu_lease.py`
- Runnable CLI: `scripts/run_with_gpu_lease.py`
- Unique M7 GPU jobs must use this common lock from launch through exit.
- The lock file must NEVER be deleted while another process could hold it.
- A non-cooperating process does not honor `flock`. The preflight reads
  `nvidia-smi` and refuses to launch if free VRAM is below the requirement,
  but **it cannot prevent an external process from allocating VRAM later**.
  Cooperation by all SIDRA workstreams is necessary for true conflict avoidance.

## Executable examples

From an immutable GitHub source snapshot, with environment correctly set:

```bash
# Original 0.6B LoRA on-policy GRPO (~7–12GB needed).
PYTHONPATH=src:. python3 scripts/run_with_gpu_lease.py \
  --workstream grpo --min-free-mib 12000 -- \
  /home/you/projects/CodeRL-Lab/.venv/bin/python \
  scripts/train_finite_action_grpo.py [EXISTING-TRAIN-ARGS]

# RSI Challenger inference needing GPU (also guarded, *same lock*).
PYTHONPATH=src:. python3 scripts/run_with_gpu_lease.py \
  --workstream rsi --min-free-mib 12000 -- \
  /home/you/projects/CodeRL-Lab/.venv/bin/python \
  scripts/run_rsi_grounded_challenger.py [EXISTING-TRAIN-ONLY-ARGS]
```

CPU-only work can proceed concurrently without GPU lease. The wrapper launches
the exact command without shell string interpolation and holds the flock across
the child process lifetime. It does not kill, reprioritize, or evict anyone.

## Exit-code contract and scheduled retries

| Exit code | Meaning | Action |
| --- | --- | --- |
| `0` | Guarded command succeeded | Check its experiment artifacts/checkpoint |
| `73` | Cooperative lease busy **OR** insufficient free VRAM | Do not start GPU work; perform CPU/CI work; retry next scheduled window |
| `74` | Probe/process infrastructure error | Record as failure; do not assume GPU available |
| Other | Original child process return code | Preserve training logs and investigate rather than relabel success |

Use one shared lock; creating a different lock in each experiment directory
does **not** coordinate workstreams. Record pinned Git SHA, model/adapters and
raw experiment outputs separately; the lock receipt itself deliberately
contains no command arguments, API keys or secret environment data.

## Real Tang verification

- Pinned GitHub source SHA: `7ddb6d8e3ffab7b4afe4d4cab717c147d3645aab`.
- Local full CPU regression: **242 passed** in Tang Python; nine focused
  lease unit tests passed.
- Tang was hosting a separate Qwen3 vLLM process (memory usage **9,821 MiB**,
  free **6,099 MiB**). Guarded `/bin/true` with `--min-free-mib 12000`
  returned exactly **73**, without executing a GPU job or interrupting
  the existing service.
- Real process-concurrency check, only running CPU `sleep 4` and `true`
  under `--min-free-mib 1` to avoid GPU allocation:
  first owner `grpo` acquired lock and exited **0**;
  simultaneous contender `rsi` was rejected with **73**;
  both logs saved in `~/experiments/sidra_m7/logs/lease_test_20261027/`.
- **Safety limit:** this test demonstrates a functional advisory lock and
  fail-fast VRAM preflight; it does NOT imply the existing vLLM server or
  all automated workstreams already use this wrapper.

The proper next training attempt should be run through this guard, with
the original pre-registered TRAIN-only M7-A8 protocol, once sufficient VRAM
is *naturally* available. Do not open sealed final-B160, interrupt any
other job, or claim that orchestration is algorithmic accuracy improvement.
