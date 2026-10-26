"""Verify optional constrained action output request, without a live backend."""
import json

from sidra.policies import OpenAICompatiblePolicy


class Response:
    def __init__(self, value):
        self.payload = json.dumps(value).encode("utf-8")
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, size=-1): return self.payload


def test_qwen3_action_schema_sent_only_when_requested(monkeypatch):
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return Response({"choices": [{"message": {"content":
            '{"kind":"read","query":"","doc_id":"sample","answer":"","citations":[]}'
        }}]})
    monkeypatch.setattr("sidra.policies.urllib.request.urlopen", fake_urlopen)
    model = OpenAICompatiblePolicy(
        "http://127.0.0.1:8900/v1/chat/completions",
        "sidra-qwen3-8b-awq", constrain_actions=True)
    action = model.decide("question", [], 5)
    assert action.kind == "read"
    request = calls[-1]
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    fmt = request["response_format"]
    assert fmt["type"] == "json_schema"
    schema = fmt["json_schema"]["schema"]
    assert fmt["json_schema"]["strict"] is True
    assert "oneOf" in schema
    assert len(schema["oneOf"]) == 3
    search, read, finish = schema["oneOf"]
    assert search["properties"]["kind"]["const"] == "search"
    assert search["required"] == ["kind", "query"]
    assert search["additionalProperties"] is False
    assert search["properties"]["query"]["maxLength"] == 120
    assert read["properties"]["kind"]["const"] == "read"
    assert read["required"] == ["kind", "doc_id"]
    assert read["properties"]["doc_id"]["maxLength"] == 120
    assert finish["properties"]["kind"]["const"] == "finish"
    assert finish["required"] == ["kind", "answer", "citations"]
    assert finish["properties"]["citations"]["maxItems"] == 5
    assert all(x["additionalProperties"] is False for x in schema["oneOf"])


def test_legacy_endpoint_has_no_schema_by_default(monkeypatch):
    calls = []
    def fake_urlopen(request, timeout):
        calls.append(json.loads(request.data))
        return Response({"choices":[{"message":{"content":
            '{"kind":"search","query":"two-hop","doc_id":"","answer":"","citations":[]}'
        }}]})
    monkeypatch.setattr("sidra.policies.urllib.request.urlopen", fake_urlopen)
    policy = OpenAICompatiblePolicy("http://127.0.0.1:8900/v1/chat/completions",
                                    "legacy-instruct")
    policy.decide("question", [], 5)
    assert "response_format" not in calls[-1]
    assert "chat_template_kwargs" not in calls[-1]
