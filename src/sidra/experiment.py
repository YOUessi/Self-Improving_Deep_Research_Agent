"""Evaluation with immutable-looking manifest and raw per-example artifacts."""

from __future__ import annotations

import hashlib
import os
import re
import json
import subprocess
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from sidra.agent import AgentConfig, Policy, SearchAgent
from sidra.io import load_dataset, save_jsonl
from sidra.metrics import score_trace, summarize


def sha256_file(path: Path) -> str:
    with path.open("rb") as fp:
        digest = hashlib.sha256()
        for block in iter(lambda: fp.read(1 << 20), b""):
            digest.update(block)
        return digest.hexdigest()


def git_sha() -> str:
    # Tang may receive a verified GitHub snapshot via the connector while direct git
    # network access is unavailable. An explicit SHA takes precedence over local git.
    override = os.environ.get("SIDRA_GIT_SHA", "")
    if re.fullmatch(r"[0-9a-fA-F]{40}", override):
        return override.lower()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    mirror_sha = Path("GITHUB_COMMIT.txt")
    if mirror_sha.exists():
        sha = mirror_sha.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            return sha.lower()
    return "unavailable"


def evaluate(dataset_dir: Path, split: str, policy: Policy, output_dir: Path,
             config: AgentConfig | None = None, limit: int = 0, seed: int = 42,
             offset: int = 0, *, runtime: str = "legacy",
             graph_role_mode: str = "multi", graph_evidence_state_mode: str = "off",
             graph_min_citations: int = 2, graph_checkpoint_db: Path | None = None,
             graph_role_policies: dict | None = None) -> dict:
    if runtime not in {"legacy", "langgraph"}:
        raise ValueError("runtime must be legacy or langgraph")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if graph_checkpoint_db is not None:
        if runtime != "langgraph":
            raise ValueError("SQLite checkpoints require --runtime langgraph")
        if graph_checkpoint_db.exists():
            raise FileExistsError(graph_checkpoint_db)
        if graph_checkpoint_db.resolve().is_relative_to(output_dir.resolve()):
            raise ValueError("checkpoint database must live outside output directory")
    documents, questions = load_dataset(dataset_dir)
    selected = sorted((q for q in questions if q.split == split), key=lambda q: q.question_id)
    if offset < 0:
        raise ValueError("offset must be non-negative")
    selected = selected[offset:]
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No questions in requested split {split}")
    lookup = {d.doc_id: d for d in documents}
    traces, rows, graph_threads = [], [], []
    with ExitStack() as stack:
        if runtime == "langgraph":
            from sidra.langgraph_agent import GraphResearchAgent
            saver = None
            if graph_checkpoint_db is not None:
                from langgraph.checkpoint.sqlite import SqliteSaver
                graph_checkpoint_db.parent.mkdir(parents=True, exist_ok=True)
                saver = stack.enter_context(SqliteSaver.from_conn_string(str(graph_checkpoint_db)))
            agent = GraphResearchAgent(
                documents, policy, config, role_mode=graph_role_mode,
                role_policies=graph_role_policies,
                min_citations=graph_min_citations,
                evidence_state_mode=graph_evidence_state_mode,
                checkpointer=saver, thread_namespace=f"eval-{seed}",
            )
        else:
            agent = SearchAgent(documents, policy, config)
        for question in selected:
            trace = agent.run(question)
            traces.append(trace.as_dict())
            rows.append(score_trace(question, trace, lookup))
            if runtime == "langgraph":
                graph_threads.append({"question_id": question.question_id,
                                      "thread_id": agent.last_thread_id})
    output_dir.mkdir(parents=True, exist_ok=False)  # never silently overwrite a prior run
    save_jsonl(output_dir / "traces.jsonl", traces)
    if runtime == "langgraph":
        save_jsonl(output_dir / "graph_threads.jsonl", graph_threads)
    save_jsonl(output_dir / "per_example.jsonl", rows)
    summary = summarize(rows)
    fixture_description = (dataset_dir / "README.md").read_text(encoding="utf-8") \
        if (dataset_dir / "README.md").exists() else ""
    synthetic = "synthetic smoke fixture" in fixture_description.lower()
    if "fixture_" in policy.name:
        label = "smoke_fixture"
    elif synthetic:
        label = "pretrained_model_on_synthetic_smoke"
    elif getattr(policy, "adapter_path", None):
        label = "pretrained_plus_train_only_sft_adapter_evaluation"
    else:
        label = "untrained_baseline"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent_runtime": runtime,
        "graph_role_mode": graph_role_mode if runtime == "langgraph" else None,
        "graph_evidence_state_mode": graph_evidence_state_mode if runtime == "langgraph" else None,
        "graph_min_citations": graph_min_citations if runtime == "langgraph" else None,
        "graph_role_models": ({role: getattr(p, "model_name", getattr(p, "model", p.name))
                              for role, p in (graph_role_policies or {}).items()}
                              if runtime == "langgraph" else None),
        "graph_checkpoint_backend": ("sqlite" if graph_checkpoint_db else "none"),
        "graph_checkpoint_sha256": (sha256_file(graph_checkpoint_db)
                                   if graph_checkpoint_db else None),
        "stage": label,
        "dataset_dir": str(dataset_dir),
        "data_sha256": {name: sha256_file(dataset_dir / name)
                        for name in ["documents.jsonl", "questions.jsonl"]},
        "split": split, "model_or_policy": policy.name, "seed": seed,
        "model_id": getattr(policy, "model_name", getattr(policy, "model", None)),
        "generation_max_new_tokens": getattr(policy, "max_new_tokens", None),
        "synthetic_fixture": bool(synthetic),
        "git_commit": git_sha(), "agent_config": asdict(agent.config),
        "sample_count": len(selected), "evaluation_offset": offset,
        "no_gpu_training_claimed": True,
        "no_weight_updates_in_this_evaluation": True,
        "adapter_path": getattr(policy, "adapter_path", None),
        "agent_roles": (["planner", "researcher", "solver", "source_verifier"]
                        if runtime == "langgraph" and graph_role_mode == "multi"
                        else ["single_policy"] if runtime == "langgraph"
                        else ["planner", "researcher", "solver", "evidence_audit"]
                        if policy.name.startswith("multi_role_") else ["single_policy"]),
        "minimum_distinct_citations": getattr(policy, "min_citations", None),
        "caution": "Synthetic fixture results cannot show generalization or RSI gains." if synthetic
                    else "This is a pretrained baseline, not post-training or RSI.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
