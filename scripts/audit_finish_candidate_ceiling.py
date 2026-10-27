"""DEV PILOT-ONLY: upper-bound audit of finite FINISH candidate availability.

This is a *counterfactual oracle-exposure diagnosis*, not an Agent run.
It deliberately simulates that BOTH true supporting documents have been READ,
using DEV pilot labels offline. It then passes only normal QUESTION + READ
observations to the production nonoracle answer proposer. Gold answers are
consulted ONLY after candidates are enumerated to score availability.

If the right answer is absent even under oracle READ exposure, action-ID GRPO
cannot choose it regardless of its learned choice logits. This diagnoses the
representation ceiling; it does not select new training data or infer model
behavior from frozen TRAIN64, eval-A64, or sealed final-B160.

Usage:
 PYTHONPATH=src python scripts/audit_finish_candidate_ceiling.py \
   --pilot /.../frozen_seed42_32p_224h/pilot/converted \
   --out /.../M7A11_DEV_PILOT32_ANSWER_CEILING.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sidra.finite_action import FiniteActionConfig, build_legal_actions
from sidra.io import load_dataset
from sidra.metrics import normalize_answer


def examine(question, document_lookup, *, proposer: str = "v4") -> dict:
    """After *oracle* READ exposure, test nonoracle action catalog coverage."""
    if question.split != "dev":
        raise ValueError("This experiment only admits DEV pilot (never TRAIN/heldout)")
    gold_ids = list(dict.fromkeys(question.supporting_doc_ids))
    if len(gold_ids) < 1 or not set(gold_ids) <= set(document_lookup):
        raise ValueError("missing labelled DEV pilot support source")
    gold_docs = [document_lookup[qid] for qid in gold_ids]
    visible = []
    # The *offline auditor* uses labels to simulate complete source exposure.
    # The actual proposer gets only the resulting ordinary tool observations.
    visible.append({
        "kind": "search", "query": question.question,
        "hits": [
            {"doc_id": d.doc_id, "title": d.title, "snippet": d.text[:120]}
            for d in gold_docs
        ],
    })
    for d in gold_docs:
        visible.append({
            "kind": "read", "doc_id": d.doc_id, "title": d.title,
            "text": d.text[:3500],
        })
    cfg = FiniteActionConfig(answer_proposer=proposer)
    actions = build_legal_actions(
        question.question, visible, cfg=cfg, remaining=1,
        max_tool_calls=5,
    )
    answers = [a.answer for a in actions if a.kind == "finish" and a.answer]
    gold_norm = normalize_answer(question.gold_answer)
    candidates = [normalize_answer(x) for x in answers]
    sources = " ".join(o.get("text", "") for o in visible if o.get("kind") == "read")
    support_text_visible = bool(gold_norm) and gold_norm in normalize_answer(sources)
    exact_candidates = [i for i, text in enumerate(candidates)
                        if gold_norm and text == gold_norm]
    gold_is_binary = gold_norm in {"yes", "no"}
    return {
        "question_id": question.question_id,
        "original_support_documents_read": len(gold_docs),
        "answer_occurs_in_read_text": bool(support_text_visible),
        "gold_available_as_action": bool(exact_candidates),
        "gold_action_indices": exact_candidates,
        "n_nonempty_finish_actions": len(answers),
        "gold_is_binary": gold_is_binary,
        "gold_is_numeric": bool(re.fullmatch(r"\d+(?:\.\d+)?", gold_norm)),
        "proposer_version": proposer,
        # Never save gold answers, raw evidence, or text in this aggregate
        # research diagnostic. It is DEV evaluator-only.
    }


def diagnose(pilot_data: Path, out: Path, *, proposer: str = "v4") -> dict:
    if out.exists():
        raise FileExistsError("refuse to overwrite a diagnostic result")
    # Refuse frozen heldout data even if passed as a path containing 'dev'.
    if pilot_data.name != "converted" or pilot_data.parent.name != "pilot":
        raise ValueError("Counterfactual oracle exposure is restricted to DEV pilot/converted")
    docs, questions = load_dataset(pilot_data)
    if not questions or any(q.split != "dev" for q in questions):
        raise ValueError("not pure DEV pilot questions")
    look = {d.doc_id: d for d in docs}
    rows = [examine(q, look, proposer=proposer) for q in questions]
    n = len(rows)
    summary = {
        "stage": "M7A11_ORACLE_SOURCE_EXPOSURE_FINISH_CANDIDATE_CEILING",
        "partition": "DEV pilot ONLY; gold read by posthoc evaluator, not model",
        "n": n,
        "proposer_version": proposer,
        "max_catalog_answers": 8 if proposer=="v5" else 4,
        "answer_visible_in_oracle_read_count": sum(r["answer_occurs_in_read_text"]
                                                    for r in rows),
        "correct_answer_available_as_finite_finish_count": sum(
            r["gold_available_as_action"] for r in rows),
        "correct_finish_coverage": round(
            sum(r["gold_available_as_action"] for r in rows)/n, 6),
        "answer_visible_but_no_finish_candidate": sum(
            r["answer_occurs_in_read_text"] and not r["gold_available_as_action"]
            for r in rows),
        "binary_questions": sum(r["gold_is_binary"] for r in rows),
        "binary_with_available_finish": sum(
            r["gold_is_binary"] and r["gold_available_as_action"] for r in rows),
        "nonempty_finite_finish_mean": round(
            sum(r["n_nonempty_finish_actions"] for r in rows)/n, 6),
        "no_hyperparam_tuning_on_train64_or_final_B160": True,
        "interpretation": (
            "If the correct answer is absent even after oracle support READ, "
            "learning to choose the correct action ID cannot solve the case. "
            "This is NOT a model QA accuracy score or a semantic entailment test."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--proposer", choices=["v2","v3","v4","v5"], default="v4")
    a = parser.parse_args()
    print(json.dumps(diagnose(a.pilot, a.out, proposer=a.proposer), indent=2))


if __name__ == "__main__":
    main()
