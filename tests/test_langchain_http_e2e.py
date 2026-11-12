"""Local HTTP wire test for LangChain ChatOpenAI + OpenAIEmbeddings + LangGraph.

This is a deterministic OpenAI-compatible test server, NOT a live LLM or a
semantic quality benchmark. It exercises the real SDK HTTP protocol and CLI.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_openai")
pytest.importorskip("langgraph")

from sidra.io import save_jsonl
from sidra.schema import Document, Question


class OfflineAPIServer(ThreadingHTTPServer):
    def __init__(self, address):
        super().__init__(address, MockOpenAIHandler)
        self.chat_requests = []
        self.embedding_requests = []


class MockOpenAIHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(n))
            if self.path == "/v1/embeddings":
                self.server.embedding_requests.append(payload)
                texts = payload.get("input", [])
                if isinstance(texts, str):
                    texts = [texts]
                reply = {
                    "object": "list", "model": payload.get("model", "mock"),
                    "data": [
                        {"object": "embedding", "index": i, "embedding": _embed(text)}
                        for i, text in enumerate(texts)
                    ],
                    "usage": {"prompt_tokens": 12, "total_tokens": 12},
                }
            elif self.path == "/v1/chat/completions":
                self.server.chat_requests.append(payload)
                tool_history = json.loads(payload["messages"][-1]["content"])["tool_history"]
                hits = {x["doc_id"] for observation in tool_history
                        if observation.get("kind") == "search"
                        for x in observation.get("hits", [])}
                read = {o["doc_id"] for o in tool_history
                        if o.get("kind") == "read"}
                if not hits:
                    action = {"kind": "search", "query": "Explorer Atlas7"}
                elif "p" not in read:
                    action = {"kind": "read", "doc_id": "p"}
                elif "r" not in read:
                    action = ({"kind": "read", "doc_id": "r"}
                              if "r" in hits else
                              {"kind": "search", "query": "Region8"})
                else:
                    action = {
                        "kind": "finish", "answer": "Quartz9",
                        "citations": ["p", "r"],
                    }
                reply = {
                    "id": "local-test", "object": "chat.completion",
                    "created": 1, "model": payload.get("model", "mock"),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {
                                     "role": "assistant",
                                     "content": json.dumps(action),
                                 }}],
                    "usage": {"prompt_tokens": 27,
                              "completion_tokens": 8, "total_tokens": 35},
                }
            else:
                self.send_error(404)
                return
            output = json.dumps(reply).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)
        except Exception:
            # The client will receive 500 without disclosing test server stack.
            self.send_error(500)


def _embed(text):
    value = text.casefold()
    return [float(value.count("atlas7")), float(value.count("region8")),
            float(value.count("quartz9")), 0.001]


def test_langchain_cli_graph_http_transport_and_scoped_hybrid(tmp_path, monkeypatch):
    docs = [
        Document("p", "Explorer Atlas7", "Explorer Atlas7 was born in Region8"),
        Document("r", "Region Region8", "Region Region8 primary mineral is Quartz9"),
        Document("leak", "Other Question", "Atlas7 Quartz9 secret",
                 scope="other-question"),
    ]
    q = Question(
        "qa-one", "What primary mineral is in the region where explorer Atlas7 was born?",
        "Quartz9", ("p", "r"), split="dev", scope="qa-one"
    )
    data = tmp_path / "dataset"
    data.mkdir()
    save_jsonl(data / "documents.jsonl", [vars(doc) for doc in docs])
    save_jsonl(data / "questions.jsonl", [
        {**vars(q), "supporting_doc_ids": list(q.supporting_doc_ids)}
    ])
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    server = OfflineAPIServer(("127.0.0.1", 0))
    task = threading.Thread(target=server.serve_forever, daemon=True)
    task.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    output = tmp_path / "graph_langchain_http"
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "sidra.cli", "eval",
             "--dataset", str(data), "--split", "dev",
             "--policy", "multi-openai", "--runtime", "langgraph",
             "--graph-role-mode", "multi", "--model-backend", "langchain",
             "--endpoint", endpoint + "/chat/completions",
             "--model", "local-test-chat", "--constrain-actions",
             "--constrain-document-ids", "--retriever", "hybrid",
             "--embedding-endpoint", endpoint,
             "--embedding-model", "local-test-embedding",
             "--retrieval-chunk-size", "64", "--retrieval-chunk-overlap", "8",
             "--max-steps", "8", "--max-tool-calls", "7",
             "--out", str(output)],
            capture_output=True, text=True, timeout=35,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        summary = json.loads((output / "summary.json").read_text())
        assert summary["n"] == 1
        assert summary["exact_match_mean"] == 1
        assert summary["verified_success_mean"] == 1
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["retriever_mode"] == "hybrid"
        assert manifest["retrieval_embedding_model_id"] == "local-test-embedding"
        assert manifest["agent_runtime"] == "langgraph"
        assert manifest["model_or_policy"] == "langchain_chat_openai_sidra_actions"
        trace = json.loads((output / "traces.jsonl").read_text().strip())
        assert trace["answer"] == "Quartz9"
        assert set(trace["citations"]) == {"p", "r"}
        assert trace["model_calls"] >= 3
        assert not any(
            hit["doc_id"] == "leak"
            for step in trace["steps"]
            for hit in step.get("observation", {}).get("hits", [])
        )
        assert len(server.chat_requests) >= 3
        assert len(server.embedding_requests) >= 1
        assert all(
            "gold_answer" not in str(request["messages"])
            and "supporting_doc_ids" not in str(request["messages"])
            for request in server.chat_requests
        )
    finally:
        server.shutdown()
        task.join(timeout=2)
        server.server_close()
