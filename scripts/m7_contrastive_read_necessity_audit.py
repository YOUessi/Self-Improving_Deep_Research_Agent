"""E-027 CPU-only post-hoc fixed E-026 TRAIN semantic contrast analysis.

This does not re-run NLI or SearchAgent and never reads DEV/eval-A/final-B.
Original model probabilities must come from pinned M7 E-026 TRAIN run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.contrastive_reliability import audit_file


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--e026-rows",required=True,type=Path)
    p.add_argument("--e026-summary",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--bootstrap",type=int,default=5000)
    a=p.parse_args()
    result=audit_file(
        a.e026_rows,a.e026_summary,a.out,bootstrap=a.bootstrap)
    # Keep logging aggregated only; source labels never sent to a policy.
    print(json.dumps({
        "n_cases":result["n_cases"],
        "n_clusters":result["n_question_clusters"],
        "n_paired_support_removals":result["n_support_removal_pairs"],
        "metrics":result["metrics"],
        "false_accept_by_type":result["neutral_case_false_accept_by_type"],
        "decision":result["decision"],
    },indent=2))


if __name__=="__main__":
    main()
