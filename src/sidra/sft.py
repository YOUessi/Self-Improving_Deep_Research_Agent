"""Leakage-gated export of verified train-only step-level action supervision."""

from __future__ import annotations

import json
from pathlib import Path

from sidra.io import iter_jsonl, load_dataset, save_jsonl
from sidra.metrics import score_trace
from sidra.schema import AgentTrace
from sidra.policies import SYSTEM


def _trace(row: dict) -> AgentTrace:
    return AgentTrace(**row)


def build_sft_examples(dataset_dir: Path, trace_path: Path, out: Path) -> dict:
    docs, questions = load_dataset(dataset_dir)
    qmap = {q.question_id: q for q in questions}
    dmap = {d.doc_id: d for d in docs}
    examples: list[dict] = []
    seen = set()
    count_success = 0
    rejected = 0
    for row in iter_jsonl(trace_path):
        trace = _trace(row)
        if trace.question_id in seen:
            raise ValueError(f"Duplicate trace ID {trace.question_id}")
        seen.add(trace.question_id)
        q = qmap.get(trace.question_id)
        if q is None:
            raise ValueError(f"Trace {trace.question_id} is missing from the dataset")
        if q.split != "train":
            raise ValueError(f"Cannot generate supervised training actions from {q.split} split")
        if not score_trace(q, trace, dmap)["verified_success"]:
            rejected += 1
            continue
        count_success += 1
        observations: list[dict] = []
        for step in trace.steps:
            if "action" not in step:
                continue
            action = step["action"]
            prompt = [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": json.dumps({
                          "question": q.question,
                          "remaining_steps": max(1, 8 - int(step["step"])),
                          "tool_history": observations,
                      }, ensure_ascii=False)}]
            examples.append({
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": json.dumps(action, ensure_ascii=False)}],
                "question_id": q.question_id, "step": int(step["step"]),
            })
            if "observation" in step:
                observations.append(step["observation"])
    save_jsonl(out, examples)
    return {"successful_train_questions": count_success, "rejected_train_questions": rejected,
            "action_training_examples": len(examples), "output": str(out)}
