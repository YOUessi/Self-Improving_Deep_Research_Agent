"""Stable JSONL schema; ground-truth answers are evaluator-only fields."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    scope: str = "global"  # Per-question retrieval when scope == question ID.

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(row["doc_id"]),
            title=str(row["title"]),
            text=str(row["text"]),
            scope=str(row.get("scope", "global")),
        )


@dataclass(frozen=True)
class Question:
    question_id: str
    question: str
    gold_answer: str
    supporting_doc_ids: tuple[str, ...] = ()
    split: Literal["train", "dev", "test"] = "test"
    group: str = ""
    scope: str = "global"

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Question":
        return cls(
            question_id=str(row["question_id"]),
            question=str(row["question"]),
            gold_answer=str(row["gold_answer"]),
            supporting_doc_ids=tuple(map(str, row.get("supporting_doc_ids", []))),
            split=row.get("split", "test"),
            group=str(row.get("group", "")),
            scope=str(row.get("scope", "global")),
        )


@dataclass(frozen=True)
class Action:
    kind: Literal["search", "read", "finish"]
    query: str = ""
    doc_id: str = ""
    answer: str = ""
    citations: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Action":
        kind = raw.get("kind")
        if kind not in {"search", "read", "finish"}:
            raise ValueError(f"Invalid action kind: {kind!r}")
        citation_raw = raw.get("citations", [])
        if not isinstance(citation_raw, list) or not all(isinstance(c, str) for c in citation_raw):
            raise ValueError("citations must be an array of document-ID strings")
        return cls(
            kind=kind,
            query=str(raw.get("query", "")),
            doc_id=str(raw.get("doc_id", "")),
            answer=str(raw.get("answer", "")),
            citations=tuple(citation_raw),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentTrace:
    question_id: str
    question: str
    policy: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    citations: list[str] = field(default_factory=list)
    termination: str = ""
    tool_calls: int = 0
    search_calls: int = 0
    read_calls: int = 0
    elapsed_ms: int = 0
    model_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
