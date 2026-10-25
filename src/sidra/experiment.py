"""Evaluation with immutable-looking manifest and raw per-example artifacts."""

from __future__ import annotations

import hashlib
import json
import subprocess
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
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def evaluate(dataset_dir: Path, split: str, policy: Policy, output_dir: Path,
             config: AgentConfig | None = None, limit: int = 0, seed: int = 42) -> dict:
    documents, questions = load_dataset(dataset_dir)
    selected = sorted((q for q in questions if q.split == split), key=lambda q: q.question_id)
    if limit:
        selected = selected[:limit]
    if not selected:
        raise ValueError(f"No questions in requested split {split}")
    agent = SearchAgent(documents, policy, config)
    lookup = {d.doc_id: d for d in documents}
    traces, rows = [], []
    for question in selected:
        trace = agent.run(question)
        traces.append(trace.as_dict())
        rows.append(score_trace(question, trace, lookup))
    output_dir.mkdir(parents=True, exist_ok=False)  # never silently overwrite a prior run
    save_jsonl(output_dir / "traces.jsonl", traces)
    save_jsonl(output_dir / "per_example.jsonl", rows)
    summary = summarize(rows)
    label = "smoke_fixture" if policy.name.startswith("fixture_") else "untrained_baseline"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": label,
        "dataset_dir": str(dataset_dir),
        "data_sha256": {name: sha256_file(dataset_dir / name)
                        for name in ["documents.jsonl", "questions.jsonl"]},
        "split": split, "model_or_policy": policy.name, "seed": seed,
        "git_commit": git_sha(), "agent_config": asdict(agent.config),
        "sample_count": len(selected), "no_gpu_training_claimed": True,
        "caution": "Synthetic smoke fixtures are not evidence of external agent improvement.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
