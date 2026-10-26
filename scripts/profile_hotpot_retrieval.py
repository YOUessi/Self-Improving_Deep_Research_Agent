"""Retrieval-only support recall on a frozen HotpotQA candidate-passage split.

This is a CPU diagnostic, not a generated-answer benchmark and not full-Wikipedia search.
The evaluator uses support labels *after* retrieval only; no gold information is
used to produce queries or to rank documents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

from sidra.io import load_dataset, save_jsonl
from sidra.retrieval import BM25Index


def frozen_selection(items, *, limit: int, seed: int):
    """Choose a stable subset independently of labels and retrieval performance."""
    chosen = sorted(items, key=lambda q: (hashlib.sha256(
        f"{seed}:{q.question_id}".encode("utf-8")).hexdigest(), q.question_id))
    return chosen[:limit] if limit else chosen


def evaluate_retrieval(data_dir: Path, output_dir: Path, *, split: str = "dev",
                       limit: int = 24, seed: int = 42,
                       ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    docs, questions = load_dataset(data_dir)
    selected = frozen_selection([q for q in questions if q.split == split],
                                limit=limit, seed=seed)
    if not selected:
        raise ValueError("no matching questions")
    ks = tuple(sorted(set(ks)))
    if not ks or any(k < 1 for k in ks):
        raise ValueError("top-k values must be positive")
    index = BM25Index(docs)
    rows = []
    for question in selected:
        hits = index.search(question.question, scope=question.scope, k=max(ks))
        gold = set(question.supporting_doc_ids)
        row = {"question_id": question.question_id, "split": question.split,
               "support_count": len(gold), "hit_ids": [h.doc_id for h in hits],
               "scope": question.scope}
        for k in ks:
            ids = {hit.doc_id for hit in hits[:k]}
            row[f"support_recall_at_{k}"] = len(gold & ids) / len(gold) if gold else None
            row[f"complete_support_at_{k}"] = int(gold.issubset(ids))
        rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_jsonl(output_dir / "per_example.jsonl", rows)
    summary = {"count": len(rows), "split": split, "dataset": str(data_dir),
               "seed": seed, "selection": "SHA256(seed:question_id) ascending",
               "scope": "per-question distractor passages, NOT full Wikipedia",
               "top_k_values": list(ks)}
    for k in ks:
        summary[f"complete_support_at_{k}"] = mean(row[f"complete_support_at_{k}"] for row in rows)
        nums = [row[f"support_recall_at_{k}"] for row in rows
                if row[f"support_recall_at_{k}"] is not None]
        summary[f"support_recall_at_{k}"] = mean(nums) if nums else None
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--split", choices=("train", "dev", "test"), default="dev")
    p.add_argument("--limit", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    print(json.dumps(evaluate_retrieval(args.dataset, args.out, split=args.split,
                                        limit=args.limit, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
