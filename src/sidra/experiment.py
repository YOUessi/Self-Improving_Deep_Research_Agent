"""Evaluation with immutable-looking manifest and raw per-example artifacts."""

from __future__ import annotations

import hashlib
import os
import re
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
             offset: int = 0) -> dict:
    documents, questions = load_dataset(dataset_dir)
    selected = sorted((q for q in questions if q.split == split), key=lambda q: q.question_id)
    if offset < 0:
        raise ValueError("offset must be non-negative")
    selected = selected[offset:]
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
    fixture_description = (dataset_dir / "README.md").read_text(encoding="utf-8") \
        if (dataset_dir / "README.md").exists() else ""
    synthetic = "synthetic smoke fixture" in fixture_description.lower()
    if "fixture_" in policy.name:
        label = "smoke_fixture"
    elif synthetic:
        label = "pretrained_model_on_synthetic_smoke"
    else:
        label = "untrained_baseline"
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
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
        "agent_roles": ["planner", "researcher", "solver", "evidence_audit"]
                       if policy.name.startswith("multi_role_") else ["single_policy"],
        "minimum_distinct_citations": getattr(policy, "min_citations", None),
        "caution": "Synthetic fixture results cannot show generalization or RSI gains." if synthetic
                    else "This is a pretrained baseline, not post-training or RSI.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
