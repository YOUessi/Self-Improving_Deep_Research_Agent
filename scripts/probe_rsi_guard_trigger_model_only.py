"""E-030 read-only, aggregate-only trigger feasibility on prior real Solver traces.

Only sanitized model-only packs and previous tool traces. No original TRAIN
answers/support, private campaign map, model retraining or final-B160. A
counterfactual rule-trigger count is NOT a counterfactual solved trajectory.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from sidra.experiment import git_sha, sha256_file
from sidra.io import iter_jsonl
from sidra.rsi_observed_read import choose_observed_second_read
from sidra.rsi_solver_cohort import _check_path, _json, _validate_public_pack


def probe(*, packs: list[Path], runs: list[Path], source_sha: str,
          out: Path) -> dict:
    if len(packs) != 3 or len(runs) != 3:
        raise ValueError("all three frozen original-model cohorts required")
    if not re.fullmatch(r"[a-f0-9]{40}", source_sha):
        raise ValueError("invalid pinned original campaign SHA")
    for path in [*packs, *runs, out]:
        _check_path(path)
    if out.exists():
        raise FileExistsError(out)
    cohorts = []
    n = 0
    candidate_n = 0
    first_read_n = 0
    seen = set()
    for cohort, (pack, run) in enumerate(zip(packs, runs, strict=True), 1):
        pm, prompts, _ = _validate_public_pack(pack)
        rm = _json(run / "SOLVER_RUN_MANIFEST.json")
        tf = run / "SOLVER_TRACES.jsonl"
        if (pm.get("cohort") != cohort or pm.get("source_git_sha") != source_sha
            or rm.get("cohort") != cohort or rm.get("source_git_sha") != source_sha
            or rm.get("public_solver_pack_sha256")
                != sha256_file(pack / "SOLVER_PACK_MANIFEST.json")
            or rm.get("traces_sha256") != sha256_file(tf)
            or rm.get("observed_read_before_finish_guard", False) is not False
            or rm.get("optimizer_updates") != 0
            or rm.get("TRAIN_gold_loaded_or_graded_by_solver") is not False
            or rm.get("config", {}).get("max_steps") != 6
            or rm.get("config", {}).get("max_tool_calls") != 5):
            raise ValueError("recorded original Solver baseline source or policy mismatched")
        traces = list(iter_jsonl(tf))
        if (len(prompts) != len(traces) or len(prompts) != 24
            or any(p["challenge_id"] in seen for p in prompts)):
            raise ValueError("incomplete or duplicate original TRAIN cohort")
        seen.update(p["challenge_id"] for p in prompts)
        by_id = {t["question_id"]: t for t in traces}
        if len(by_id) != len(prompts):
            raise ValueError("duplicate Solver run questions")
        first = eligible = 0
        for p in prompts:
            tr = by_id.get(p["challenge_id"])
            if tr is None or tr["question"] != p["question"]:
                raise ValueError("question/trace mismatch")
            observations = []
            for step in tr["steps"]:
                if "observation" not in step:
                    continue
                observations.append(step["observation"])
                if step["observation"].get("kind") == "read":
                    first += 1
                    offered = choose_observed_second_read(
                        tr["question"], observations,
                        max_steps=6, max_tool_calls=5)
                    eligible += int(offered is not None)
                    break
        n += len(prompts)
        first_read_n += first
        candidate_n += eligible
        cohorts.append({
            "cohort": cohort, "recorded_model_episodes": len(prompts),
            "episodes_with_first_read": first,
            "guard_would_offer_second_read": eligible,
            "Solver_trace_sha256": sha256_file(tf),
        })
    result = {
        "stage": "E030_MODEL_ONLY_OBSERVED_READ_TRIGGER_PROBE_NOT_MODEL_QA",
        "code_git_sha": git_sha(),
        "original_campaign_source_sha": source_sha,
        "real_existing_solver_episodes": n,
        "episodes_with_first_READ": first_read_n,
        "episodes_where_guard_would_offer_second_READ": candidate_n,
        "cohorts": cohorts,
        "TRAIN_gold_or_private_map_opened": False,
        "actual_model_continuations_after_guard": 0,
        "new_model_calls": 0,
        "new_optimizer_updates": 0,
        "checkpoint_generations": 0,
        "external_eval_or_finalB_opened": False,
        "conclusion": "Trigger feasibility only; new READ and downstream model QA "
                      "must be measured in prospective live runs before any claim.",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as f:
        json.dump(result, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", action="append", type=Path, required=True)
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(probe(
        packs=args.pack, runs=args.run, source_sha=args.source_sha,
        out=args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
