"""Diagnose tool-loop failure without touching gold-labelled dev answers.

Uses raw trajectories (actions, tool outcomes, termination) and TRAIN-only
action demonstrations. Reports counts, repeated reads, transitions, and finish
frequency. Does not change model weights or evaluate correctness.

python scripts/analyze_stop_loops.py --train-actions <train.jsonl> \
  --before-traces <base_traces.jsonl> --after-traces <sft_traces.jsonl> \
  --out <NEW_report.json>
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path
from sidra.io import iter_jsonl


def action_distribution(rows: list[dict]) -> dict:
    actions=Counter()
    stages=Counter()
    for row in rows:
        a=json.loads(row["completion"][0]["content"])
        actions[a["kind"]]+=1
        stages[f"{row['step']}:{a['kind']}"]+=1
    n=sum(actions.values())
    return {"n":n,"action_counts":dict(sorted(actions.items())),
            "action_fractions":{k:round(v/n,5) for k,v in sorted(actions.items())} if n else {},
            "step_action_counts":dict(sorted(stages.items()))}


def trajectory_distribution(traces: list[dict]) -> dict:
    actions=Counter()
    transitions=Counter()
    endings=Counter()
    terminations=Counter()
    repeated_reads=0
    repeated_searches=0
    calls=0
    for trace in traces:
        ordered=[(step.get("action") or {}).get("kind","error") for step in trace["steps"]]
        actions.update(ordered)
        transitions.update(f"{a}->{b}" for a,b in zip(ordered,ordered[1:]))
        endings.update(ordered[-1:] if ordered else ["empty"])
        terminations[trace["termination"]]+=1
        seen_reads=set()
        seen_searches=set()
        for step in trace["steps"]:
            a=step.get("action") or {}
            if a.get("kind")=="read":
                if a["doc_id"] in seen_reads:repeated_reads+=1
                seen_reads.add(a["doc_id"])
            if a.get("kind")=="search":
                q=a.get("query","").strip().casefold()
                if q in seen_searches:repeated_searches+=1
                seen_searches.add(q)
        calls+=trace.get("model_calls",0)
    n=len(traces)
    return {
        "n":n,"action_counts":dict(sorted(actions.items())),
        "termination_counts":dict(sorted(terminations.items())),
        "last_action_counts":dict(sorted(endings.items())),
        "top_transitions":transitions.most_common(15),
        "repeated_read_events":repeated_reads,
        "repeated_search_events":repeated_searches,
        "finish_episode_rate":round(terminations["finish"]/n,5) if n else None,
        "average_model_calls":round(calls/n,5) if n else None,
    }


def analyze(train_actions:Path,before:Path,after:Path,out:Path)->dict:
    if out.exists(): raise FileExistsError(out)
    a=list(iter_jsonl(train_actions))
    b=list(iter_jsonl(before))
    c=list(iter_jsonl(after))
    if sorted(x["question_id"] for x in b)!=sorted(x["question_id"] for x in c):
        raise ValueError("before/after must have identical QA IDs")
    report={
        "train_supervision":action_distribution(a),
        "base_model":trajectory_distribution(b),
        "trained_adapter":trajectory_distribution(c),
        "limit":"tool behavior only; no dev labels or evaluator answers loaded"}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-actions",type=Path,required=True)
    p.add_argument("--before-traces",type=Path,required=True)
    p.add_argument("--after-traces",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(analyze(a.train_actions,a.before_traces,a.after_traces,a.out),
                     indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
