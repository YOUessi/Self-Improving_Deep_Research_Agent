"""LangChain ChatOpenAI + OpenAIEmbeddings adapters for SIDRA agents.

LangGraph owns the workflow; LangChain owns the model/embedding interfaces.
The Tool/Action/Verifier contracts remain SIDRA's and are gold-blind.
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlsplit, urlunsplit
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from sidra.policies import SYSTEM, _json_object
from sidra.schema import Action


def provider_url(endpoint: str, *, chat: bool = False) -> str:
    """Require explicit endpoint; strip chat-completions suffix for ChatOpenAI."""
    parts = urlsplit(endpoint)
    local = parts.hostname in {"localhost", "127.0.0.1", "::1"}
    if parts.scheme not in {"https", "http"} or not parts.netloc:
        raise ValueError("explicit http(s) provider endpoint required")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("credentials/query/fragment in provider endpoint forbidden")
    if parts.scheme == "http" and not local:
        raise ValueError("non-local model providers require HTTPS")
    path = parts.path.rstrip("/")
    if chat and path.endswith("/chat/completions"):
        path = path[:-len("/chat/completions")]
    if not path.endswith("/v1"):
        raise ValueError("provider base URL must terminate at /v1")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _key_for(endpoint: str, key_env: str) -> str:
    key = os.getenv(key_env, "")
    if key:
        return key
    if urlsplit(endpoint).hostname in {"localhost", "127.0.0.1", "::1"}:
        return "SIDRA_LOCAL_NO_AUTH"
    raise ValueError(f"{key_env} is required for remote LangChain provider")


class LangChainChatPolicy:
    """One SIDRA Action per LangChain model call; no implicit agent loop."""

    name = "langchain_chat_openai_sidra_actions"

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        key_env: str = "OPENAI_API_KEY",
        max_tokens: int = 400,
        timeout: float = 90.0,
        constrain_actions: bool = False,
        constrain_document_ids: bool = False,
        chat_model: Any | None = None,
    ) -> None:
        if not model.strip() or max_tokens < 1 or timeout <= 0:
            raise ValueError("model and positive inference limits required")
        if constrain_document_ids and not constrain_actions:
            raise ValueError("document ID constraints require action constraints")
        self.model_name = model
        self.model = model
        self.max_tokens = max_tokens
        self.constrain_actions = constrain_actions
        self.constrain_document_ids = constrain_document_ids
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        if chat_model is None:
            base = provider_url(endpoint, chat=True)
            self.chat = ChatOpenAI(
                model=model, base_url=base,
                api_key=_key_for(base, key_env),
                temperature=0, max_tokens=max_tokens, timeout=timeout,
                max_retries=0, use_responses_api=False,
            )
        else:
            # Purely injected fake/mocked chat for offline infrastructure tests.
            self.chat = chat_model

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action:
        return self.decide_with_system(question, observations, remaining, SYSTEM)

    def decide_with_system(
        self,
        question: str,
        observations: list[dict],
        remaining: int,
        system_prompt: str,
    ) -> Action:
        if remaining < 1:
            raise ValueError("no remaining agent steps")
        body = json.dumps(
            {"question": question, "remaining_steps": remaining,
             "tool_history": observations}, ensure_ascii=False
        )
        payload = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=body),
        ]
        llm = self.chat
        if self.constrain_actions:
            if self.constrain_document_ids:
                from sidra.observed_action_schema import observed_document_schema
                schema = observed_document_schema(observations)
            else:
                schema = _basic_action_schema()
            llm = llm.bind(response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sidra_tool_action", "strict": True, "schema": schema
                },
            })
        self.model_calls += 1
        try:
            message = llm.invoke(payload)
        except Exception as exc:
            # Never include provider response bodies, headers or API key in logs.
            raise RuntimeError(
                f"LangChain provider call failed: {type(exc).__name__}"
            ) from None
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:
            metadata = getattr(message, "response_metadata", None) or {}
            usage = metadata.get("token_usage", {})
        self.prompt_tokens += int(
            usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        )
        self.completion_tokens += int(
            usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
        )
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            raise ValueError("LangChain response must contain text JSON action")
        return Action.from_dict(_json_object(content))


def _basic_action_schema() -> dict:
    return {
        "oneOf": [
            {"type": "object", "properties": {
                "kind": {"const": "search"},
                "query": {"type": "string", "minLength": 1, "maxLength": 120}},
             "required": ["kind", "query"], "additionalProperties": False},
            {"type": "object", "properties": {
                "kind": {"const": "read"},
                "doc_id": {"type": "string", "minLength": 1, "maxLength": 120}},
             "required": ["kind", "doc_id"], "additionalProperties": False},
            {"type": "object", "properties": {
                "kind": {"const": "finish"},
                "answer": {"type": "string", "maxLength": 100},
                "citations": {"type": "array", "items": {
                    "type": "string", "maxLength": 120}, "maxItems": 5}},
             "required": ["kind", "answer", "citations"],
             "additionalProperties": False},
        ],
    }


def make_openai_embeddings(
    endpoint: str,
    model: str,
    *,
    key_env: str = "OPENAI_API_KEY",
) -> OpenAIEmbeddings:
    """Explicit opt-in to an external embedding provider.

    Caller is responsible for permission to send candidate passage text.
    No train/dev/test answer labels or source-question metadata are sent.
    """
    if not model.strip():
        raise ValueError("embedding model must be specified")
    base = provider_url(endpoint)
    return OpenAIEmbeddings(
        model=model, base_url=base,
        api_key=_key_for(base, key_env),
        max_retries=0, request_timeout=90,
        check_embedding_ctx_length=False,
    )
