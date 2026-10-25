"""CLI evaluation and SFT export; no remote API called without --endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidra.agent import AgentConfig
from sidra.experiment import evaluate
from sidra.policies import FixtureMultiHopPolicy, FixtureOneHopPolicy, OpenAICompatiblePolicy
from sidra.sft import build_sft_examples


def main() -> None:
    parser = argparse.ArgumentParser(prog="sidra")
    sub = parser.add_subparsers(dest="command", required=True)
    ev = sub.add_parser("eval", help="Run baseline and store immutable trace/metrics/manifest directory")
    ev.add_argument("--dataset", required=True, type=Path)
    ev.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    ev.add_argument("--policy", choices=["fixture-multihop", "fixture-onehop", "openai", "transformers", "multi-openai", "multi-transformers"], required=True)
    ev.add_argument("--endpoint", help="OpenAI-compatible /v1/chat/completions URL")
    ev.add_argument("--model", help="Model served at --endpoint")
    ev.add_argument("--key-env", default="OPENAI_API_KEY")
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--max-steps", type=int, default=8)
    ev.add_argument("--max-tool-calls", type=int, default=7)
    ev.add_argument("--search-top-k", type=int, default=3)
    ev.add_argument("--limit", type=int, default=0)
    ev.add_argument("--seed", type=int, default=42)
    ev.add_argument("--max-new-tokens", type=int, default=280)
    export = sub.add_parser("export-sft", help="Export only fully verified TRAIN traces")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--traces", type=Path, required=True)
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "export-sft":
        print(json.dumps(build_sft_examples(args.dataset, args.traces, args.out), indent=2))
        return
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    if args.policy.startswith("fixture-") and "fixture" not in (args.dataset / "README.md").read_text(
            encoding="utf-8"):
        parser.error("Fixture policies are strictly for generated synthetic fixtures")
    if args.policy == "fixture-multihop":
        policy = FixtureMultiHopPolicy()
    elif args.policy == "fixture-onehop":
        policy = FixtureOneHopPolicy()
    elif args.policy == "transformers":
        if not args.model:
            parser.error("--model is required for local Transformers inference")
        from sidra.policies_transformers import TransformersPolicy
        policy = TransformersPolicy(args.model, max_new_tokens=args.max_new_tokens)
    else:
        if not args.endpoint or not args.model:
            parser.error("--endpoint and --model are required for real LLM evaluation")
        policy = OpenAICompatiblePolicy(args.endpoint, args.model, key_env=args.key_env)
    config = AgentConfig(max_steps=args.max_steps,
                         max_tool_calls=args.max_tool_calls, search_top_k=args.search_top_k)
    result = evaluate(args.dataset, args.split, policy, args.out, config, args.limit, args.seed)
    print(json.dumps(result, indent=2))
    print(f"Results saved in {args.out}")


if __name__ == "__main__":
    main()
