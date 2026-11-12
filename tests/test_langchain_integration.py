"""Offline LangChain integration: real packages, fake model/embeddings, no API/GPU.

Tests intentionally do not claim dense RAG or multi-agent quality gains.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_openai")
pytest.importorskip("langchain_text_splitters")
pytest.importorskip("langgraph")

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from sidra.agent import AgentConfig, SearchAgent
from sidra.experiment import evaluate
from sidra.io import save_jsonl
from sidra.langchain_policy import (
    LangChainChatPolicy, make_openai_embeddings, provider_url,
)
from sidra.langchain_retrieval import (
    LangChainDenseIndex, LangChainHybridIndex, create_retriever,
)
from sidra.langgraph_agent import GraphResearchAgent
from sidra.policies import FixtureMultiHopPolicy
from sidra.schema import Action, Document, Question
from sidra.train_reward import score_train_rollout


class SmallDeterministicEmbeddings(Embeddings):
    """Synthetic vectors for interface/permission checks, not semantic quality."""

    words = ("atlas7", "region8", "quartz9", "secret", "alpha", "beta")

    def _encode(self, value: str) -> list[float]:
        text = value.casefold()
        return [float(text.count(word)) for word in self.words] + [0.001]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._encode(text)


def corpus():
    docs = [
        Document("p", "Explorer Atlas7", "Explorer Atlas7 was born in Region8"),
        Document("r", "Region Region8", "Region Region8 primary mineral is Quartz9"),
        Document("hidden", "Secret Region8", "Secret Quartz9", scope="different-question"),
    ]
    q = Question("qa-one",
                 "What primary mineral is in the region where explorer Atlas7 was born?",
                 "Quartz9", ("p", "r"), split="train", scope="qa-one")
    return docs, q


def test_dense_vector_pipeline_has_real_langchain_chunks_and_source_mapping():
    docs, q = corpus()
    index = LangChainDenseIndex(
        docs, SmallDeterministicEmbeddings(), chunk_size=64, chunk_overlap=10
    )
    assert index.chunks
    assert all(chunk.metadata["doc_id"] in {"p", "r", "hidden"}
               for chunk in index.chunks)
    assert any("::chunk::" in str(chunk.id) for chunk in index.chunks)
    for query in ("Atlas7", "Region8", "Quartz9"):
        hits = index.search(query, k=3, scope=q.scope)
        assert hits
        assert all(hit.doc_id != "hidden" for hit in hits)
        assert all("::chunk::" not in hit.doc_id for hit in hits)
        assert all(hit.doc_id in {"p", "r"} for hit in hits)
    assert index.search("Secret", k=3, scope=q.scope) == [] or all(
        hit.doc_id != "hidden" for hit in index.search("Secret", k=3, scope=q.scope)
    )


def test_hybrid_rrf_preserves_scope_and_bm25_baseline():
    docs, q = corpus()
    from sidra.retrieval import BM25Index
    baseline = BM25Index(docs).search("Explorer Atlas7", k=2, scope=q.scope)
    via_factory = create_retriever(docs, mode="bm25")
    assert via_factory.search("Explorer Atlas7", k=2, scope=q.scope) == baseline

    hybrid = LangChainHybridIndex(
        docs, SmallDeterministicEmbeddings(), chunk_size=64, chunk_overlap=8
    )
    hits = hybrid.search("Explorer Atlas7", k=3, scope=q.scope)
    assert hits[0].doc_id == "p"
    assert all(hit.doc_id in {"p", "r"} for hit in hits)
    assert all(0 < hit.score < 1 for hit in hits)  # RRF, not raw cosine score
    assert hybrid.search("", k=4, scope=q.scope) == []
    with pytest.raises(ValueError, match="dense_weight"):
        LangChainHybridIndex(docs, SmallDeterministicEmbeddings(), dense_weight=1.2)
    with pytest.raises(ValueError, match="explicit Embeddings"):
        create_retriever(docs, mode="hybrid", embeddings=None)


@pytest.mark.parametrize("mode", ["dense", "hybrid"])
def test_langchain_retrieval_executes_in_real_graph_and_train_scorer(mode):
    docs, q = corpus()
    retriever = create_retriever(
        docs, mode=mode, embeddings=SmallDeterministicEmbeddings(),
        chunk_size=64, chunk_overlap=8,
    )
    agent = GraphResearchAgent(
        docs, FixtureMultiHopPolicy(),
        AgentConfig(max_steps=8, max_tool_calls=7),
        role_mode="multi", retriever=retriever,
    )
    trace = agent.run(q)
    assert trace.termination == "finish", trace.error
    assert trace.answer == "Quartz9"
    assert set(trace.citations) == {"p", "r"}
    # Dense/Hybrid may expose both valid sources in the first SEARCH.
    # Unlike fixture/BM25, a second SEARCH is not always necessary.
    assert trace.read_calls == 2
    assert trace.search_calls in {1, 2}
    assert trace.tool_calls == trace.read_calls + trace.search_calls
    reward = score_train_rollout(q, trace, {d.doc_id: d for d in docs},
                                 max_tool_calls=7)
    assert reward.strict_outcome == 1.0
    assert reward.answer_correct == 1.0
    # The legacy runtime can also use the same retriever: framework is orthogonal.
    old = SearchAgent(
        docs, FixtureMultiHopPolicy(),
        AgentConfig(max_steps=8, max_tool_calls=7),
        retriever=retriever,
    ).run(q)
    assert old.answer == trace.answer
    assert old.tool_calls == trace.tool_calls


def test_eval_manifest_records_retriever_without_labels_in_model_state(tmp_path):
    docs, q = corpus()
    root = tmp_path / "dataset"
    root.mkdir()
    save_jsonl(root / "documents.jsonl", [vars(x) for x in docs])
    save_jsonl(root / "questions.jsonl", [
        {**vars(q), "supporting_doc_ids": list(q.supporting_doc_ids)}
    ])
    out = tmp_path / "eval_dense"
    result = evaluate(
        root, "train", FixtureMultiHopPolicy(), out,
        runtime="langgraph", retriever_mode="dense",
        embeddings=SmallDeterministicEmbeddings(),
        embedding_model_id="test-deterministic-only", retrieval_chunk_size=64,
        retrieval_chunk_overlap=8,
    )
    assert result["verified_success_mean"] == 1.0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["retriever_mode"] == "dense"
    assert manifest["retrieval_embedding_model_id"] == "test-deterministic-only"
    assert manifest["retrieval_chunk_size"] == 64
    raw_trace = (out / "traces.jsonl").read_text()
    assert "gold_answer" not in raw_trace
    assert "supporting_doc_ids" not in raw_trace
    with pytest.raises(ValueError, match="BM25 baseline"):
        evaluate(root, "train", FixtureMultiHopPolicy(), tmp_path / "bad",
                 embeddings=SmallDeterministicEmbeddings())


def test_langchain_chat_model_is_real_action_policy_without_leaking_gold():
    fake = FakeListChatModel(responses=[
        '{"kind":"search","query":"Explorer Atlas7"}',
        '{"kind":"read","doc_id":"p"}',
        '{"kind":"finish","answer":"","citations":[]}',
    ])
    policy = LangChainChatPolicy(
        "http://127.0.0.1:8000/v1", "fake",
        chat_model=fake, max_tokens=40,
    )
    docs, q = corpus()
    trace = SearchAgent(docs, policy).run(q)
    assert trace.termination == "abstained"
    assert [step["action"]["kind"] for step in trace.steps] == [
        "search", "read", "finish"
    ]
    assert trace.model_calls == 3
    assert not trace.citations


def test_chat_action_usage_accounting_and_bound_schema():
    class FakeReply:
        content = '{"kind":"finish","answer":"","citations":[]}'
        usage_metadata = {"input_tokens": 17, "output_tokens": 4}

    class CaptureChat:
        def __init__(self):
            self.payload = []
            self.kwargs = {}

        def bind(self, **kwargs):
            self.kwargs.update(kwargs)
            return self

        def invoke(self, messages):
            self.payload = messages
            return FakeReply()

    chat = CaptureChat()
    policy = LangChainChatPolicy(
        "http://127.0.0.1:8000/v1", "test",
        constrain_actions=True, chat_model=chat,
    )
    action = policy.decide("Where is a university?", [], 3)
    assert action == Action(kind="finish", answer="", citations=())
    assert policy.model_calls == 1
    assert policy.prompt_tokens == 17
    assert policy.completion_tokens == 4
    assert chat.kwargs["response_format"]["type"] == "json_schema"
    assert "remaining_steps" in chat.payload[-1].content
    assert "gold_answer" not in chat.payload[-1].content


def test_provider_security_rejects_credentials_unencrypted_http(monkeypatch):
    assert provider_url("http://localhost:8000/v1/chat/completions", chat=True) == (
        "http://localhost:8000/v1"
    )
    with pytest.raises(ValueError, match="HTTPS"):
        provider_url("http://example.com/v1")
    with pytest.raises(ValueError, match="credentials"):
        provider_url("https://u:p@example.com/v1")
    monkeypatch.delenv("SIDRA_LANGCHAIN_TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="required"):
        LangChainChatPolicy(
            "https://example.com/v1", "model",
            key_env="SIDRA_LANGCHAIN_TEST_KEY",
        )
    # Explicit local endpoint should not send or depend on a remote API key.
    policy = LangChainChatPolicy(
        "http://localhost:8000/v1", "local_model",
        key_env="SIDRA_LANGCHAIN_TEST_KEY",
    )
    assert policy.name.startswith("langchain_")
    embedding = make_openai_embeddings(
        "http://localhost:8000/v1", "local_embedding",
        key_env="SIDRA_LANGCHAIN_TEST_KEY",
    )
    assert embedding.model == "local_embedding"
