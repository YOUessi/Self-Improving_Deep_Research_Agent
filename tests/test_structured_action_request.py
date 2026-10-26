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
    assert schema["additionalProperties"] is False
    assert schema["properties"]["kind"]["enum"] == ["search", "read", "finish"]
    assert schema["required"] == ["kind", "query", "doc_id", "answer", "citations"]


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
