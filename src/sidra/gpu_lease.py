"""Cooperative GPU lease and read-only NVIDIA preflight for SIDRA workstreams.

Shared lock only coordinates SIDRA jobs that use this helper. External jobs
are additionally detected by available GPU memory; this cannot prevent a
*non-cooperating* process from starting later.

Never unlink the lock file: removing it can break flock inode coordination.
This module NEVER kills a foreign process, changes GPU drivers, or forces
allocation/eviction.

Use one SHARED absolute path across GRPO, Search Evidence and RSI workflows,
even when GitHub working branches and experiment directories are independent.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator


class GpuLeaseBusy(RuntimeError):
    """Resource unavailable; do not start or preempt an existing GPU job."""


@dataclass(frozen=True)
class GpuMemory:
    used_mib: int
    free_mib: int
    total_mib: int
    name: str


def inspect_gpu(gpu_index: int = 0) -> GpuMemory:
    """Query rather than allocate GPU memory. Any failed probe blocks launch."""
    if gpu_index < 0:
        raise ValueError("GPU index must be nonnegative")
    proc = subprocess.run(
        ["nvidia-smi", f"--id={gpu_index}",
         "--query-gpu=memory.used,memory.free,memory.total,name",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=8, check=True,
    )
    rows = [row.strip() for row in proc.stdout.splitlines() if row.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one GPU, got {len(rows)}")
    pieces = [x.strip() for x in rows[0].split(",", maxsplit=3)]
    if len(pieces) != 4:
        raise RuntimeError("Invalid nvidia-smi memory response")
    used, free, total = (int(x) for x in pieces[:3])
    if used < 0 or free < 0 or total < 1 or used + free > total + 128:
        raise RuntimeError("Inconsistent GPU memory reading")
    return GpuMemory(used, free, total, pieces[3])


def _clean_label(s: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,80}", s):
        raise ValueError("workstream must be a simple non-secret identifier")
    return s


@contextmanager
def gpu_lease(
    lock_file: Path,
    *,
    workstream: str,
    min_free_mib: int = 12000,
    gpu_index: int = 0,
    probe: Callable[[int], GpuMemory] = inspect_gpu,
) -> Iterator[dict]:
    """Fail-closed: check cooperative lease and available memory under lock.

    :raises GpuLeaseBusy: lease occupied or GPU has insufficient free memory.
    """
    workstream = _clean_label(workstream)
    if min_free_mib < 1:
        raise ValueError("min_free_mib must be positive")
    lock_file = Path(lock_file)
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GpuLeaseBusy(
                f"Shared GPU lease held elsewhere: {lock_file}; do not preempt it"
            ) from exc
        try:
            gpu = probe(gpu_index)
            if gpu.free_mib < min_free_mib:
                raise GpuLeaseBusy(
                    f"GPU{gpu_index} has only {gpu.free_mib} MiB free; "
                    f"requires >= {min_free_mib} MiB. Wait; never stop foreign jobs."
                )
            receipt = {
                "pid": os.getpid(),
                "workstream": workstream,
                "gpu_index": gpu_index,
                "start_unix": round(time.time(), 3),
                "memory_at_acquire": asdict(gpu),
                "required_free_mib": min_free_mib,
                "lease_path": str(lock_file),
                "note": "Cooperative flock + initial GPU memory check, not a reservation",
            }
            # The lock is held over the entire guarded command. The marker
            # intentionally omits command-line args, URLs, keys and environment.
            fd.seek(0)
            fd.truncate(0)
            fd.write(json.dumps(receipt, ensure_ascii=False) + "\n")
            fd.flush()
            os.fsync(fd.fileno())
            yield receipt
        finally:
            fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
