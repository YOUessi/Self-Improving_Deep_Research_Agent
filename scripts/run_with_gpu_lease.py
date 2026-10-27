"""Run a command only while holding a shared, nonpreemptive SIDRA GPU lease.

Examples (all M7 workstreams MUST use the SAME lock path):
  PYTHONPATH=src python scripts/run_with_gpu_lease.py \\
    --workstream grpo --min-free-mib 12000 -- python scripts/train_...py ...
  PYTHONPATH=src python scripts/run_with_gpu_lease.py \\
    --workstream rsi --min-free-mib 11000 -- python scripts/run_...py ...

Exit 73: lock busy or insufficient GPU memory. Retry at next scheduled run.
Exit 74: probe or execution infrastructure error.
A shared lease does NOT control unrelated processes that fail to cooperate.
Never kill an existing process just to make a GPU lease available.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from sidra.gpu_lease import GpuLeaseBusy, gpu_lease

DEFAULT_LOCK = Path("/home/you/experiments/sidra_shared/gpu0.lock")


def run_guarded(
    command: list[str], *,
    workstream: str,
    lock_file: Path = DEFAULT_LOCK,
    min_free_mib: int = 12000,
    gpu_index: int = 0,
) -> int:
    if not command:
        raise ValueError("A real command is required after --")
    try:
        with gpu_lease(
            lock_file, workstream=workstream, min_free_mib=min_free_mib,
            gpu_index=gpu_index,
        ) as receipt:
            print(
                "SIDRA_GPU_LEASE_ACQUIRED",
                f"workstream={workstream}",
                f"gpu={gpu_index}",
                f"free_mib={receipt['memory_at_acquire']['free_mib']}",
                flush=True,
            )
            # Wait for the child to exit, keeping flock in this parent until
            # completion. No shell=True, no signal to foreign processes.
            return subprocess.run(command, check=False).returncode
    except GpuLeaseBusy as exc:
        print(f"SIDRA_GPU_LEASE_BUSY: {exc}", file=sys.stderr)
        return 73
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"SIDRA_GPU_LEASE_ERROR: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 74


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workstream", required=True)
    ap.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    ap.add_argument("--min-free-mib", type=int, default=12000)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    raise SystemExit(
        run_guarded(
            command, workstream=args.workstream, lock_file=args.lock_file,
            min_free_mib=args.min_free_mib, gpu_index=args.gpu_index,
        )
    )


if __name__ == "__main__":
    main()
