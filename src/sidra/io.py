"""Validated JSONL reading/writing and split isolation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from sidra.schema import Document, Question


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fp:
        for index, line in enumerate(fp, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected an object")
                    yield row
                except (ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid JSONL {path}:{index}: {exc}") from exc


def save_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_dataset(directory: str | Path) -> tuple[list[Document], list[Question]]:
    root = Path(directory)
    documents = [Document.from_dict(row) for row in iter_jsonl(root / "documents.jsonl")]
    questions = [Question.from_dict(row) for row in iter_jsonl(root / "questions.jsonl")]
    if not documents or not questions:
        raise ValueError("Both documents.jsonl and questions.jsonl must be non-empty")
    ids = [d.doc_id for d in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate document IDs")
    question_ids = [q.question_id for q in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question IDs")
    doc_map = {doc.doc_id: doc for doc in documents}
    group_split: dict[str, str] = {}
    for q in questions:
        if q.split not in {"train", "dev", "test"}:
            raise ValueError(f"Invalid split for {q.question_id}: {q.split}")
        for doc_id in q.supporting_doc_ids:
            if doc_id not in doc_map:
                raise ValueError(f"Question {q.question_id} references missing support {doc_id}")
            if doc_map[doc_id].scope not in {q.scope, "global"}:
                raise ValueError(f"Question {q.question_id}: out-of-scope support {doc_id}")
        if q.group:
            old = group_split.setdefault(q.group, q.split)
            if old != q.split:
                raise ValueError(f"Group leakage: group={q.group} spans {old} and {q.split}")
    return documents, questions
