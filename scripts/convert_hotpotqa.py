"""Convert original HotpotQA *distractor* JSON to scoped document/question JSONL.

Input format: list of {"_id", "question", "answer", "context": [[title,[sentences]],...],
                       "supporting_facts": [[title, sentence_index],...]}. This is NOT a
blind full-wikipedia retrieval task: each question only searches its candidate passages.
Do not mix original train/dev splits or pretend selected passages are web-scale retrieval.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                            for r in rows), encoding="utf-8")


def convert(source: Path, output: Path, split: str, limit: int = 0) -> tuple[int, int]:
    if split not in {"train", "dev", "test"}:
        raise ValueError("split must be train/dev/test")
    records = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("Expected a list of HotpotQA distractor examples")
    if limit > 0:
        records = records[:limit]
    docs, questions = [], []
    for row in records:
        qid = str(row["_id"])
        titles: dict[str, str] = {}
        for ordinal, section in enumerate(row["context"]):
            title, sentences = section
            if title in titles:
                raise ValueError(f"Duplicate context title for {qid}: {title}")
            doc_id = f"{qid}::{ordinal}"
            titles[title] = doc_id
            docs.append({"doc_id": doc_id, "title": title, "text": " ".join(sentences), "scope": qid})
        support_titles = list(dict.fromkeys(title for title, _ in row["supporting_facts"]))
        missing = set(support_titles) - set(titles)
        if missing:
            raise ValueError(f"Missing supporting facts for {qid}: {missing}")
        questions.append({"question_id": qid, "question": row["question"],
                          "gold_answer": row["answer"],
                          "supporting_doc_ids": [titles[t] for t in support_titles],
                          "split": split, "group": qid, "scope": qid})
    save_jsonl(output / "documents.jsonl", docs)
    save_jsonl(output / "questions.jsonl", questions)
    (output / "README.md").write_text(
        "# HotpotQA distractor conversion\n\nCandidate passages are scoped per question. "
        "This is **not** full-wikipedia, web search, or out-of-distribution evaluation. "
        "Ground-truth answers are never shown to policies by the runner. "
        "Never treat the dev set as a train set for SFT/GRPO.\n",
        encoding="utf-8")
    return len(questions), len(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    n, d = convert(args.input, args.out, args.split, args.limit)
    print(f"Converted {n} examples, {d} candidate docs to {args.out}")


if __name__ == "__main__":
    main()
