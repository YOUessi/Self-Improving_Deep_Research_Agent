"""Fail-closed GPU ownership checks for *other* processes on Tang.

Shared flock is necessary but insufficient: a second vLLM server can start
loading while nvidia-smi reports only the 350 MiB display baseline. This
detects its CPU-side parent *before* CUDA allocation, and is called before
our own GPU training / before each weight update.

Do NOT signal, preempt, or otherwise modify a foreign PID. This is only a
cooperative gate; programs ignoring the project's flock can still race.
"""
from __future__ import annotations

import os
from pathlib import Path


def classify_gpu_process(comm:str,cmdline:str)->str|None:
    """Conservative known GPU-consuming launchers, ignoring shell text."""
    name=comm.casefold().strip()
    argv=cmdline.casefold().strip()
    # Parent process often starts before CUDA appears in nvidia-smi.
    if name.startswith(("vllm","ray::")) or "vllm::engine" in name:
        return "llm_server"
    # Bare shell commands can *quote* snippets mentioning CUDA/vLLM: do not
    # classify unrelated bash/zsh/grep/ssh processes as model jobs.
    if not (
        name.startswith("python")
        or name.startswith(("torchrun","accelerate","deepspeed"))
    ):
        return None
    if any(part in argv for part in (
        "/vllm serve", " -m vllm", " vllm serve", "vllm.entrypoints",
        "vllm::enginecore",
    )):
        return "llm_server"
    if any(part in argv for part in (
        "train_finite_action_grpo.py train", "train_onpolicy_tool_grpo.py",
        "train_action_sft.py", "replay_masked_grpo_pilot.py",
        " -m sidra.cli eval", "torchrun ", "deepspeed ",
        "accelerate launch", " -m torch.distributed.run",
    )):
        return "other_gpu_experiment"
    return None


def find_external_gpu_jobs(*,proc_root:Path=Path("/proc"),
                           own_pid:int|None=None)->list[dict]:
    """Return only safe PID/type metadata, no potentially secret argv."""
    self_pid=os.getpid() if own_pid is None else own_pid
    foreign=[]
    if not proc_root.exists():
        raise RuntimeError("Linux /proc unavailable; GPU ownership cannot be verified")
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal():
            continue
        pid=int(proc.name)
        if pid==self_pid:
            continue
        try:
            comm=(proc/"comm").read_text(encoding="utf-8").strip()
            args=(proc/"cmdline").read_bytes().replace(b"\0",b" ").decode(
                "utf-8",errors="replace")
        except (FileNotFoundError,ProcessLookupError,PermissionError,OSError):
            # Process may have exited during the check. A live /proc PID with
            # inaccessible commandline must not be silently treated as safe.
            if proc.exists():
                foreign.append({"pid":pid,"kind":"uninspectable_process"})
            continue
        kind=classify_gpu_process(comm,args)
        if kind:
            foreign.append({"pid":pid,"kind":kind})
    return sorted(foreign,key=lambda row:row["pid"])


def assert_no_external_gpu_jobs(*,proc_root:Path=Path("/proc"),
                                own_pid:int|None=None)->None:
    jobs=find_external_gpu_jobs(proc_root=proc_root,own_pid=own_pid)
    if jobs:
        # Do not print command lines, tokens, credentials, or user paths.
        short=", ".join(f"{row['pid']}:{row['kind']}" for row in jobs[:12])
        raise RuntimeError(
            "Foreign GPU job already starting/running (no preemption): "+short
        )
