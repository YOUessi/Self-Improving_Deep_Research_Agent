"""Cooperative GPU guard catches vLLM during CPU-only warmup."""
from pathlib import Path

import pytest

from sidra.gpu_coordination import (
    classify_gpu_process,find_external_gpu_jobs,assert_no_external_gpu_jobs,
)


def make_proc(root:Path,pid:int,name:str,argv:list[str]|None):
    p=root/str(pid)
    p.mkdir(parents=True)
    (p/"comm").write_text(name+"\n")
    if argv is not None:
        (p/"cmdline").write_bytes(
            b"\0".join(x.encode() for x in argv)+b"\0")


def test_gpu_launchers_detected_without_cuda_allocation(tmp_path):
    make_proc(tmp_path,100,"vllm",[
        "/usr/bin/python","/venv/bin/vllm","serve","model"])
    make_proc(tmp_path,101,"VLLM::EngineCor",["VLLM::EngineCore"])
    make_proc(tmp_path,102,"python",[
        "python","scripts/train_finite_action_grpo.py","train","--seed","21"])
    make_proc(tmp_path,103,"python",[
        "python","-I","-m","gpu_agent","web"])
    make_proc(tmp_path,104,"bash",[
        "bash","-c","echo vllm serve; grep train_finite_action_grpo"])
    make_proc(tmp_path,105,"python",[
        "python","-m","pytest","-q"])
    make_proc(tmp_path,106,"python",[
        "python","-m","sidra.cli","eval","--policy","transformers"])
    make_proc(tmp_path,107,"vllm",[
        "/venv/bin/vllm","serve","--secret","do-not-log"])
    result=find_external_gpu_jobs(proc_root=tmp_path,own_pid=107)
    assert [row["pid"] for row in result]==[100,101,102,106]
    assert all("do-not-log" not in str(row) for row in result)
    with pytest.raises(RuntimeError,match="no preemption"):
        assert_no_external_gpu_jobs(proc_root=tmp_path,own_pid=107)


def test_gpu_guard_allows_benign_workers_and_self_only(tmp_path):
    make_proc(tmp_path,201,"python",[
        "/usr/bin/python","-m","pytest","-q"])
    make_proc(tmp_path,202,"bash",[
        "bash","-c","vllm serve && torchrun"])
    make_proc(tmp_path,203,"python",[
        "python","scripts/train_finite_action_grpo.py","train"])
    assert find_external_gpu_jobs(proc_root=tmp_path,own_pid=203)==[]
    assert_no_external_gpu_jobs(proc_root=tmp_path,own_pid=203)


def test_gpu_guard_fails_closed_for_uninspectable_python_process(tmp_path):
    make_proc(tmp_path,301,"python",None)
    (tmp_path/"302").mkdir()
    (tmp_path/"302"/"comm").write_text("kworker\n")
    result=find_external_gpu_jobs(proc_root=tmp_path,own_pid=0)
    assert result==[{"pid":301,"kind":"uninspectable_gpu_launcher"}]
    with pytest.raises(RuntimeError,match="301"):
        assert_no_external_gpu_jobs(proc_root=tmp_path,own_pid=0)


def test_classifier_does_not_treat_text_mentions_as_jobs():
    assert classify_gpu_process("bash","echo 'vllm serve'") is None
    assert classify_gpu_process("python","python -m gpu_agent web") is None
    assert classify_gpu_process("python","python -m vllm serve /model")=="llm_server"
    assert classify_gpu_process("python","python -m torch.distributed.run")=="other_gpu_experiment"
