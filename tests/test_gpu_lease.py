"""GPU coordination regression tests never allocate GPU or kill processes."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from sidra.gpu_lease import GpuLeaseBusy, GpuMemory, gpu_lease, inspect_gpu
from scripts.run_with_gpu_lease import run_guarded


def fake_probe(_gpu_index=0):
    return GpuMemory(used_mib=351, free_mib=15568,
                     total_mib=15919, name="Fake RTX")


def test_nonblocking_shared_lock_refuses_second_workstream(tmp_path):
    lock = tmp_path / "all_sidra_branches_gpu0.lock"
    with gpu_lease(lock, workstream="grpo", min_free_mib=10000,
                   probe=fake_probe) as receipt:
        assert receipt["workstream"] == "grpo"
        marker = json.loads(lock.read_text())
        assert marker["memory_at_acquire"]["free_mib"] == 15568
        assert "secret" not in lock.read_text()
        with pytest.raises(GpuLeaseBusy, match="held elsewhere"):
            with gpu_lease(lock, workstream="rsi", min_free_mib=10000,
                           probe=fake_probe):
                pytest.fail("Contending job incorrectly obtained the lease")
    # Releasing the same inode allows the next independent workstream.
    with gpu_lease(lock, workstream="search_evidence",
                   min_free_mib=10000, probe=fake_probe):
        pass


def test_external_gpu_allocation_refuses_to_start(tmp_path):
    lock = tmp_path / "gpu.lock"
    with pytest.raises(GpuLeaseBusy, match="only 6000 MiB"):
        with gpu_lease(
            lock, workstream="grpo", min_free_mib=12000,
            probe=lambda gpu: GpuMemory(9800, 6000, 15900, "Busy GPU"),
        ):
            pytest.fail("A contested GPU must never be used")
    # Probe failure must release the lock for an independent later retry.
    with gpu_lease(lock, workstream="rsi", min_free_mib=100,
                   probe=fake_probe):
        pass


def test_failed_command_keeps_lock_until_exit(monkeypatch, tmp_path):
    from scripts import run_with_gpu_lease as module

    def substitute_gpu_lease(lock_file, **kwargs):
        return gpu_lease(
            lock_file, workstream=kwargs["workstream"],
            min_free_mib=kwargs["min_free_mib"],
            probe=fake_probe,
        )

    monkeypatch.setattr(module, "gpu_lease", substitute_gpu_lease)
    command = ["python3", "-c", "raise SystemExit(7)"]
    assert run_guarded(command, workstream="grpo",
                       lock_file=tmp_path / "gpu.lock",
                       min_free_mib=100) == 7
    # A subsequent command can proceed without deleting the lockfile.
    assert run_guarded(["python3", "-c", "print('lease-ok')"],
                       workstream="rsi",
                       lock_file=tmp_path / "gpu.lock",
                       min_free_mib=100) == 0


def test_inspect_gpu_is_read_only_and_fail_closed(monkeypatch):
    commands = []

    def fake_run(argv, **kwargs):
        commands.append((argv, kwargs))
        return SimpleNamespace(stdout="351, 15568, 15919, RTX GPU\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = inspect_gpu(0)
    assert result.free_mib == 15568
    assert result.used_mib == 351
    assert commands[0][0][:2] == ["nvidia-smi", "--id=0"]
    assert commands[0][1]["timeout"] == 8


@pytest.mark.parametrize("bad", [
    "A,B", " leading", "", "name with spaces", "../wrong",
])
def test_invalid_workstream_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        with gpu_lease(tmp_path / "gpu.lock", workstream=bad,
                       min_free_mib=1, probe=fake_probe):
            pass
