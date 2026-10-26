"""Gold-only offline labels MUST NOT affect allowed model actions."""
import json
import pytest
from sidra.observed_action_schema import observed_document_schema
from sidra.policies import OpenAICompatiblePolicy


def kinds(schema):
    return [branch["properties"]["kind"]["const"] for branch in schema["oneOf"]]


def get(schema,kind):
    return next(b for b in schema["oneOf"] if b["properties"]["kind"]["const"]==kind)


def test_no_search_hits_means_no_read_variant():
    s=observed_document_schema([])
    assert kinds(s)==["search","finish"]
    assert get(s,"finish")["properties"]["citations"]["maxItems"]==0


def test_only_exposed_unread_documents_may_be_read():
    observations=[
        {"kind":"search","query":"q","hits":[
            {"doc_id":"doc-b","title":"B"},
            {"doc_id":"doc-a","title":"A"}]},
        {"kind":"read","doc_id":"doc-a","text":"fact"},
    ]
    s=observed_document_schema(observations)
    assert get(s,"read")["properties"]["doc_id"]["enum"]==["doc-b"]
    assert get(s,"finish")["properties"]["citations"]["items"]["enum"]==["doc-a"]
    assert get(s,"finish")["properties"]["citations"]["maxItems"]==1
    assert "gold_answer" not in json.dumps(s)


def test_after_all_hits_read_no_duplicate_read_choice():
    observations=[{"kind":"search","query":"q","hits":[{"doc_id":"d"}]},
                  {"kind":"read","doc_id":"d","text":"answer"}]
    s=observed_document_schema(observations)
    assert kinds(s)==["search","finish"]


def test_schema_is_independent_of_untrusted_titles_and_support_annotations():
    a=[{"kind":"search","hits":[{"doc_id":"d","title":"A",
                                   "supporting_doc_ids":["SECRET"]}]}]
    b=[{"kind":"search","hits":[{"doc_id":"d","title":"CHANGED",
                                   "supporting_doc_ids":["OTHER"]}]}]
    assert observed_document_schema(a)==observed_document_schema(b)


class Resp:
    def __init__(self,obj):self.bytes=json.dumps(obj).encode()
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def read(self,size=-1):return self.bytes


def test_online_policy_sends_dynamic_read_enumeration(monkeypatch):
    submitted=[]
    def fake(req,timeout):
        body=json.loads(req.data)
        submitted.append(body)
        return Resp({"choices":[{"message":{"content":
              '{"kind":"read","doc_id":"x"}'}}]})
    monkeypatch.setattr("sidra.policies.urllib.request.urlopen",fake)
    policy=OpenAICompatiblePolicy(
        "http://127.0.0.1:18981/v1/chat/completions",
        "sidra-qwen3-06b-stop",constrain_actions=True,
        constrain_document_ids=True)
    result=policy.decide("a question",[{"kind":"search","query":"q",
                           "hits":[{"doc_id":"x"}]}],5)
    assert result.kind=="read" and result.doc_id=="x"
    s=submitted[0]["response_format"]["json_schema"]["schema"]
    assert get(s,"read")["properties"]["doc_id"]["enum"]==["x"]
    assert get(s,"finish")["properties"]["citations"]["maxItems"]==0


def test_requires_static_schema_when_dynamic_requested():
    with pytest.raises(ValueError,match="requires constrain_actions"):
        OpenAICompatiblePolicy("http://127.0.0.1:18981/v1/chat/completions",
                               "sidra-qwen3-06b-stop",constrain_document_ids=True)
