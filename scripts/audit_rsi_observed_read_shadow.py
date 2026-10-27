"""E-029 two-process CPU audit: model-only shadow, then trusted TRAIN Verifier.

Never pass --train-data to model-only stage; the trusted verification step
replays historical SearchAgent BM25, model actions and frozen cohort hashes
before reading TRAIN-only labels. No DEV/eval-A/final-B input accepted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.rsi_observed_read_audit import (
    materialize_model_only_shadow, verify_matched_read_shadow,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="role", required=True)
    for name in ("model-only", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--pack", action="append", type=Path, required=True)
        cmd.add_argument("--run", action="append", type=Path, required=True)
        cmd.add_argument("--source-sha", required=True,
                         help="original frozen E026 campaign source commit")
        cmd.add_argument("--shadow-code-sha", required=True,
                         help="exact current E029 source commit")
        cmd.add_argument("--out", type=Path, required=True)
        if name == "verify":
            cmd.add_argument("--train-data", type=Path, required=True)
            cmd.add_argument("--campaign", type=Path, required=True)
            cmd.add_argument("--model-only", type=Path, required=True)
    args = parser.parse_args()
    common = dict(packs=args.pack, runs=args.run, source_sha=args.source_sha,
                  shadow_code_sha=args.shadow_code_sha, out=args.out)
    if args.role == "model-only":
        result = materialize_model_only_shadow(**common)
    else:
        result = verify_matched_read_shadow(
            train_data=args.train_data, campaign=args.campaign,
            model_only=args.model_only, **common
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
