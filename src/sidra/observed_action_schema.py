"""Gold-blind dynamic JSON action constraints from actually observed tool data.

This is a *tool-safety* decoder, not a semantic verifier:
- read IDs can ONLY be selected from hit IDs returned by previous searches
  and not yet read;
- finish citations can ONLY reference document IDs previously READ;
- no question.gold_answer/supporting_doc_ids are accepted or inspected.
This changes *validity* and token budget, so compare as a separate
intervention under identical models and search budgets.
"""
from __future__ import annotations


def observed_document_schema(observations: list[dict]) -> dict:
    exposed: set[str] = set()
    read: set[str] = set()
    for item in observations:
        if item.get("kind") == "search":
            for hit in item.get("hits", []):
                doc_id = hit.get("doc_id")
                if isinstance(doc_id, str) and doc_id:
                    exposed.add(doc_id)
        elif item.get("kind") == "read":
            doc_id = item.get("doc_id")
            if isinstance(doc_id, str) and doc_id:
                read.add(doc_id)

    # The Schema is derived from the *online observations only*.
    # Document titles, gold labels and support annotations are not needed.
    unread = sorted(exposed - read)
    choices = [
        {
            "type":"object",
            "properties":{
                "kind":{"const":"search"},
                "query":{"type":"string","minLength":1,"maxLength":120},
            },
            "required":["kind","query"],"additionalProperties":False,
        }
    ]
    if unread:
        choices.append({
            "type":"object",
            "properties":{
                "kind":{"const":"read"},
                "doc_id":{"type":"string","enum":unread},
            },
            "required":["kind","doc_id"],"additionalProperties":False,
        })

    citations={"type":"array","maxItems":min(5,len(read)),
               "items":{"type":"string"}}
    if read:
        citations["items"]["enum"]=sorted(read)
    else:
        citations["maxItems"]=0
    choices.append({
        "type":"object",
        "properties":{
            "kind":{"const":"finish"},
            "answer":{"type":"string","maxLength":100},
            "citations":citations,
        },
        "required":["kind","answer","citations"],
        "additionalProperties":False,
    })
    return {"oneOf":choices}
