"""Paired, gold-blind next-hop *retrieval-only* probe on TRAIN or open pilot32.

Runs genuine SearchAgent.search/read/finish actions with ONE shared BM25 index
and identical 6-step/5-tool budget. Only evaluator uses support labels AFTER
each trace ends. This does NOT measure model answers, EM or semantic entailment.
Never point this diagnostic at final-B160 or eval-A64 during policy selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from sidra.agent import AgentConfig, SearchAgent
from sidra.bridge import next_bridge_query
from sidra.bridge_policy import choose_unread_hit
from sidra.io import load_dataset, save_jsonl
from sidra.next_hop_policy import plan_next_hop
from sidra.schema import Action


# Frozen *opened* DEV pilot32; NEVER substitute eval-A64 or final-B160.
OPEN_PILOT_DOCS_SHA256 = "5aabc048b344940ffdce2880a7a57a683d4009777a796fd0602ed683a500cc6c"
OPEN_PILOT_QUESTIONS_SHA256 = "94db9bdfff053555b9a8e0da9a850b5bee3de73e2406ec85c63461b5ae033e6e"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class ProbePolicy:
    name = "retrieval_probe"

    def __init__(self, mode: str):
        if mode not in {"baseline", "forced_bridge", "frontier_first"}:
            raise ValueError("unknown mode")
        self.mode = mode
        self.name = "m7_" + mode

    def decide(self, question: str, observations: list[dict], remaining: int) -> Action:
        reads = [o for o in observations if o.get("kind") == "read"]
        searches = [o for o in observations if o.get("kind") == "search"]
        if not observations:
            return Action("search", query=question)
        if not searches:
            return Action("finish", answer="")
        if not reads:
            hits = searches[-1].get("hits") or []
            return (Action("read", doc_id=str(hits[0]["doc_id"])) if hits
                    else Action("finish", answer=""))
        if len(reads) >= 2:
            return Action("finish", answer="")
        if observations[-1].get("kind") == "search":
            candidate = choose_unread_hit(question, observations)
            return (Action("read", doc_id=candidate) if candidate
                    else Action("finish", answer=""))
        if self.mode == "frontier_first":
            plan = plan_next_hop(question, observations,
                remaining_steps=remaining, remaining_tool_calls=5 - len(searches) - len(reads))
            if plan:
                return plan.action
        elif self.mode == "forced_bridge" and remaining >= 3:
            proposal = next_bridge_query(question, observations)
            if proposal and len(searches) + len(reads) <= 3:
                return Action("search", query=proposal.query)
        candidate = choose_unread_hit(question, observations)
        return (Action("read", doc_id=candidate) if candidate
                else Action("finish", answer=""))


def trace_metrics(q, trace) -> dict:
    gold = set(q.supporting_doc_ids)
    exposed = {
        str(hit["doc_id"])
        for step in trace.steps
        for hit in (step.get("observation") or {}).get("hits", [])
    }
    read = {
        str((step.get("observation") or {}).get("doc_id"))
        for step in trace.steps
        if (step.get("observation") or {}).get("kind") == "read"
    }
    read.discard("None")
    if not gold:
        raise ValueError("support labels required only for offline evaluation")
    return {
        "retrieval_complete": int(gold <= exposed),
        "read_support_complete": int(gold <= read),
        "retrieval_support_recall": len(gold & exposed) / len(gold),
        "read_support_recall": len(gold & read) / len(gold),
        "search_calls": trace.search_calls,
        "read_calls": trace.read_calls,
        "tool_calls": trace.tool_calls,
        "cpu_wall_ms": trace.elapsed_ms,
        "termination": trace.termination,
    }


def paired_ci(rows: list[dict], a: str, b: str, field: str, *, seed: int = 2027,
              repeats: int = 5000) -> dict:
    d = [r[a][field] - r[b][field] for r in rows]
    rnd = random.Random(seed)
    n = len(d)
    means = sorted(sum(d[rnd.randrange(n)] for _ in range(n)) / n
                   for _ in range(repeats))
    return {
        "delta": round(sum(d)/n, 6),
        "ci95": [round(means[int(repeats * 0.025)], 6),
                 round(means[min(repeats - 1, int(repeats * 0.975))], 6)],
        "paired_bootstrap_samples": repeats,
    }


def run(dataset: Path, out: Path, *, mode: str, limit: int = 64,
        seed: int = 43) -> dict:
    if out.exists():
        raise FileExistsError(out)
    if mode not in {"train", "pilot"}:
        raise ValueError("mode must be TRAIN-only or opened PILOT32")
    if mode == "pilot":
        if (_sha256(dataset / "documents.jsonl") != OPEN_PILOT_DOCS_SHA256
                or _sha256(dataset / "questions.jsonl") != OPEN_PILOT_QUESTIONS_SHA256):
            raise ValueError("not the frozen open pilot32: SHA256 mismatch; refuse")
    docs, questions = load_dataset(dataset)
    expected_split = "train" if mode == "train" else "dev"
    if not questions or any(q.split != expected_split for q in questions):
        raise ValueError("data split mismatch; refuse unknown or heldout data")
    rng = random.Random(seed)
    questions = sorted(questions, key=lambda q: q.question_id)
    if mode == "train":
        questions = sorted(rng.sample(questions, min(limit, len(questions))),
                           key=lambda q: q.question_id)
    else:
        if len(questions) != 32:
            raise ValueError("pilot MUST be the opened 32-question set")
    agent = SearchAgent(docs, ProbePolicy("baseline"),
                        AgentConfig(max_steps=6, max_tool_calls=5,
                                    search_top_k=3, max_read_chars=3500))
    rows = [{"question_id":q.question_id} for q in questions]
    for variant in ("baseline", "forced_bridge", "frontier_first"):
        agent.policy = ProbePolicy(variant)
        for row, q in zip(rows, questions):
            trace = agent.run(q)
            if trace.termination == "error":
                raise RuntimeError(f"illegal policy action in {variant}: {trace.error}")
            row[variant] = trace_metrics(q, trace)
    metrics = ("retrieval_complete", "read_support_complete",
               "retrieval_support_recall", "read_support_recall",
               "search_calls", "read_calls", "tool_calls", "cpu_wall_ms")
    aggregate = {
        variant: {m: round(sum(row[variant][m] for row in rows) / len(rows), 6)
                  for m in metrics}
        for variant in ("baseline", "forced_bridge", "frontier_first")
    }
    report = {
        "mode": mode,
        "partition": "TRAIN-only random sample" if mode == "train"
                     else "opened DEV pilot32 aggregate-only",
        "n": len(rows), "seed": seed,
        "retrieval": "Actual SearchAgent BM25 search/read/finish, no LLM",
        "steps": 6, "max_tool_calls": 5, "top_k": 3, "max_read_chars": 3500,
        "gold_used_by_policy": False,
        "aggregates": aggregate,
        "paired_frontier_minus_forced": {
            m: paired_ci(rows, "frontier_first", "forced_bridge", m)
            for m in metrics if m != "cpu_wall_ms"
        },
        "paired_frontier_minus_baseline": {
            m: paired_ci(rows, "frontier_first", "baseline", m)
            for m in ("read_support_complete", "read_support_recall",
                      "search_calls", "tool_calls")
        },
        "answer_EM": None,
        "sentence_citation_success": None,
        "semantic_entailment_success": None,
        "warning": ("Retrieval-only, not model inference or a learned improvement. "
                    "Read selection is scripted; do not infer answer EM. "
                    "CPU wall time is local-process only. "
                    "No final-B160 or eval-A64 was accessed."),
    }
    out.mkdir(parents=True, exist_ok=False)
    (out / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if mode == "train":
        save_jsonl(out / "train_only_per_question.jsonl", rows)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("train", "pilot"), required=True)
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    print(json.dumps(run(args.dataset, args.out, mode=args.mode,
                         limit=args.limit, seed=args.seed), indent=2))
