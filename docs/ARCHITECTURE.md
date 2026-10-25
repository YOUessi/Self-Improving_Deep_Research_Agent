# M1 Architecture and data contracts

```text
questions.jsonl (gold held by Evaluator)
       |
       v
Agent loop (policy sees only question + tool observations)
       | search(query) --> BM25 scoped index --> doc IDs, titles, snippets
       | read(doc_id) --> only IDs previously searched --> document text
       | finish(answer,citations) --> reject citations to unread docs
       v
AgentTrace JSONL ------------> score_trace(question, trace, corpus)
       |                               |
       |                               v
       |                         per_example.jsonl + summary.json
       v
Train-only, verified-success filter --> prompt/completion action supervision
                                              |
                                         Tang LoRA SFT (optional, GPU)
```

## Dataset formats

- `documents.jsonl`: `{"doc_id": "...", "title": "...", "text": "...", "scope": "global"}`.
- `questions.jsonl`: `{"question_id":"...", "question":"...", "gold_answer":"...", "supporting_doc_ids":["..."], "split":"train|dev|test", "group":"...", "scope":"global"}`.

`scope` is either `global` or an explicit question ID. Retrieval sees a scoped candidate set; a question cannot read documents belonging to another question's scope. `group` identifies an entity/template grouping to prohibit split leakage.

## Agent API

`sidra.schema.Action` supports three kinds: `search` (query), `read` (doc ID), `finish` (answer with cited read IDs). Invalid actions, over-budget calls, arbitrary file access, and unread citations terminate safely. Model policy uses standard chat-completions format, with `--endpoint`/`--model` required. No model credentials are baked into files.

## Known limitations

1. M1 uses basic BM25 for local text. No semantic retriever, web connector, reranking or live search.
2. The `fixture_*` policies hard-code *synthetic* template behavior; their scores are NOT a real baseline or evidence of learned reasoning.
3. For OpenAI-compatible backends, malformed JSON and unavailable endpoints terminate trace with an error. We do not silently turn invalid calls into correct answers.
4. Citation checks enforce *read-before-cite*, not full natural-language entailment. `answer_in_cited_evidence` is an approximate substring check and will miss some semantically equivalent answers. Add robust attribution judgements with a human-audited subset in M2.
5. HotpotQA conversion is distractor (candidate passages scoped to each question), not full Wikipedia / live web. Its results must not be compared as if they used identical retrieval settings to open-domain methods.
6. Latency from local synthetic tests is too small to be meaningful. TTFT and token counts require instrumentation of the model server in M2.
