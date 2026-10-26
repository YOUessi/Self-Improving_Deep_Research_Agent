"""Qwen3 endpoint request compatibility and cost/provenance accounting."""
import json

from sidra.agent import SearchAgent
from sidra.multi_agent import MultiAgentPolicy
from sidra.policies import OpenAICompatiblePolicy
from sidra.schema import Action, Document, Question


class JsonResponse:
    def __init__(self, value):
        self.content = json.dumps(value).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self, size=-1):
        return self.content


def test_qwen3_vllm_request_disables_thinking_and_counts_usage(monkeypatch):
    sent = []

    def fake_urlopen(req, timeout):
        body = json.loads(req.data)
        sent.append(body)
        return JsonResponse({
            "choices": [{"message": {"content": '{"kind":"search","query":"Alpha"}'}}],
            "usage": {"prompt_tokens": 84, "completion_tokens": 18},
        })

    monkeypatch.setattr("sidra.policies.urllib.request.urlopen", fake_urlopen)
    policy = OpenAICompatiblePolicy("http://127.0.0.1:8000/v1/chat/completions",
                                    "sidra-qwen3-8b-awq")
    action = policy.decide("q", [], 6)
    assert action.kind == "search"
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert policy.model_calls == 1
    assert policy.prompt_tokens == 84
    assert policy.completion_tokens == 18
    assert len(policy.action_io) == 1
    record = policy.action_io[0]
    assert record["raw_completion"] == '{"kind":"search","query":"Alpha"}'
    assert record["messages"][1]["role"] == "user"
    assert record["prompt_tokens"] == 84
    assert record["constrained_schema"] is False


def test_non_qwen_model_has_no_qwen_specific_override(monkeypatch):
    requests = []

    def fake_urlopen(req, timeout):
        requests.append(json.loads(req.data))
        return JsonResponse({"choices": [{"message": {"content": '{"kind":"finish","answer":"","citations":[]}'}}]})

    monkeypatch.setattr("sidra.policies.urllib.request.urlopen", fake_urlopen)
    policy = OpenAICompatiblePolicy("http://127.0.0.1:8000/v1/chat/completions", "other-model")
    policy.decide("q", [], 1)
    assert "chat_template_kwargs" not in requests[0]


def test_trace_model_calls_counted_without_answer_gold():
    class CounterPolicy:
        name = "counter"
        def __init__(self):
            self.model_calls = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0

        def decide(self, question, observations, remaining):
            self.model_calls += 1
            self.prompt_tokens += 7
            self.completion_tokens += 2
            if not observations:
                return Action("search", query="Alpha")
            return Action("finish", answer="")

    policy = CounterPolicy()
    agent = SearchAgent([Document("a", "Alpha", "fact")], policy)
    trace = agent.run(Question("q", "Where?", "fact"))
    assert trace.model_calls == 2
    assert trace.prompt_tokens == 14
    assert trace.completion_tokens == 4
    assert trace.termination == "abstained"


def test_multi_role_keeps_underlying_model_identity():
    inner = OpenAICompatiblePolicy(
        "http://127.0.0.1:18980/v1/chat/completions", "sidra-qwen3-8b-awq")
    combined = MultiAgentPolicy(inner)
    assert combined.model_name == "sidra-qwen3-8b-awq"
    assert combined.model_policy.model == combined.model_name
