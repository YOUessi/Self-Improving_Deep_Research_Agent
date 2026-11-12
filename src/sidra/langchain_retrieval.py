"""LangChain document splitting, embeddings, vector RAG, and ranked hybrid RRF.

All results are returned as the original SIDRA Hit (parent doc_id, not chunk ID).
Gold answers, supporting_doc_ids, and evaluator labels are never indexed.
Default BM25 remains unchanged; embeddings execute only when explicitly selected.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from typing import Protocol

from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sidra.retrieval import BM25Index, Hit
from sidra.schema import Document


class ScopedRetriever(Protocol):
    def search(self, query: str, *, k: int = 3, scope: str = "global") -> list[Hit]: ...


class LangChainDenseIndex:
    """In-memory vector store with strict scope checks and source-ID collapse.

    This is a research index: no hidden answer keys, no persistence, and no
    cross-question search access. Build on an approved corpus only: external
    embedding providers receive the source passage text at index time.
    """

    def __init__(
        self,
        documents: Iterable[Document],
        embeddings: Embeddings,
        *,
        chunk_size: int = 1200,
        chunk_overlap: int = 120,
    ) -> None:
        docs = tuple(documents)
        if not docs or len({d.doc_id for d in docs}) != len(docs):
            raise ValueError("nonempty corpus with unique doc IDs required")
        if chunk_size < 64 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("invalid LangChain chunk size/overlap")
        self.sources = {d.doc_id: d for d in docs}
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.chunks: list[LCDocument] = []
        for doc in docs:
            # Titles help semantic retrieval, but no evaluator gold is embedded.
            text = doc.title + "\n" + doc.text
            pieces = self.splitter.split_text(text)
            for number, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                self.chunks.append(LCDocument(
                    id=f"{doc.doc_id}::chunk::{number}",
                    page_content=piece,
                    metadata={"doc_id": doc.doc_id, "scope": doc.scope,
                              "title": doc.title, "chunk_index": number},
                ))
        if not self.chunks:
            raise ValueError("corpus contains no indexable text")
        self.store = InMemoryVectorStore(embeddings)
        self.store.add_documents(self.chunks)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def search(self, query: str, *, k: int = 3, scope: str = "global") -> list[Hit]:
        if k <= 0 or not query.strip():
            return []
        if not scope:
            raise ValueError("scope must be nonempty")
        permitted = {"global", scope}
        # Overfetch all chunks so deduplicating multiple matches from one
        # source cannot hide a distinct valid source in the top-k results.
        # LangChain filters the scored pool before returning ranked hits.
        raw = self.store.similarity_search_with_score(
            query, k=len(self.chunks),
            filter=lambda row: row.metadata.get("scope") in permitted,
        )
        best: dict[str, float] = {}
        for chunk, score in raw:
            doc_id = str(chunk.metadata.get("doc_id", ""))
            source = self.sources.get(doc_id)
            if source is None or source.scope not in permitted:
                # Fail closed even if a vectorstore's filter contract changes.
                raise ValueError("vector result violates source provenance/scope")
            if not math.isfinite(float(score)):
                raise ValueError("nonfinite semantic retrieval score")
            best[doc_id] = max(best.get(doc_id, float("-inf")), float(score))
        ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        return [
            Hit(doc_id=doc_id, title=self.sources[doc_id].title,
                snippet=self.sources[doc_id].text[:500], score=score)
            for doc_id, score in ordered[:k]
        ]


class LangChainHybridIndex:
    """BM25 + LangChain dense with *rank* fusion, not mixed raw scores."""

    def __init__(
        self,
        documents: Iterable[Document],
        embeddings: Embeddings,
        *,
        title_boost: float = 0.0,
        chunk_size: int = 1200,
        chunk_overlap: int = 120,
        dense_weight: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        docs = tuple(documents)
        if not 0 <= dense_weight <= 1 or not math.isfinite(dense_weight):
            raise ValueError("dense_weight must be in [0,1]")
        if rrf_k < 1:
            raise ValueError("rrf_k must be >=1")
        self.lexical = BM25Index(docs, title_boost=title_boost)
        self.dense = LangChainDenseIndex(
            docs, embeddings, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.sources = {d.doc_id: d for d in docs}
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k

    def search(self, query: str, *, k: int = 3, scope: str = "global") -> list[Hit]:
        if k <= 0 or not query.strip():
            return []
        pool = min(len(self.sources), max(8, k * 4))
        lexical = self.lexical.search(query, k=pool, scope=scope)
        dense = self.dense.search(query, k=pool, scope=scope)
        ranks: dict[str, float] = defaultdict(float)
        for weight, hits in ((1 - self.dense_weight, lexical),
                             (self.dense_weight, dense)):
            for rank, hit in enumerate(hits, 1):
                ranks[hit.doc_id] += weight / (self.rrf_k + rank)
        result = sorted(ranks, key=lambda doc_id: (-ranks[doc_id], doc_id))
        return [
            Hit(doc_id=doc_id, title=self.sources[doc_id].title,
                snippet=self.sources[doc_id].text[:500], score=ranks[doc_id])
            for doc_id in result[:k]
        ]


def create_retriever(
    documents: list[Document],
    *,
    mode: str = "bm25",
    embeddings: Embeddings | None = None,
    title_boost: float = 0.0,
    chunk_size: int = 1200,
    chunk_overlap: int = 120,
    dense_weight: float = 0.5,
) -> ScopedRetriever:
    """Exact existing BM25 for baseline; explicit dense/hybrid for ablation."""
    if mode == "bm25":
        return BM25Index(documents, title_boost=title_boost)
    if mode not in {"dense", "hybrid"}:
        raise ValueError(f"unsupported retrieval strategy: {mode}")
    if embeddings is None:
        raise ValueError("dense and hybrid require an explicit Embeddings provider")
    if mode == "dense":
        return LangChainDenseIndex(
            documents, embeddings, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    return LangChainHybridIndex(
        documents, embeddings, title_boost=title_boost,
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        dense_weight=dense_weight,
    )
