"""Synthetic multi-hop retrieval fixtures for CPU smoke tests; NOT a research benchmark.

Usage: python scripts/generate_fixture.py --out examples/fixture --n 40 --seed 13
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def save(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                            for item in records), encoding="utf-8")


def make(n: int, seed: int) -> tuple[list[dict], list[dict]]:
    if n < 10:
        raise ValueError("Fixture must have >=10 questions to populate all splits")
    rng = random.Random(seed)
    permutation = list(range(n))
    rng.shuffle(permutation)
    train_n = n * 3 // 5
    dev_n = n // 5
    groups = {value: "train" if idx < train_n else "dev" if idx < train_n + dev_n else "test"
              for idx, value in enumerate(permutation)}
    docs = []
    questions = []
    for i in range(n):
        person, region, mineral = f"Mara{i:03d}", f"Vale{i:03d}", f"Ore{i:03d}"
        person_id, region_id = f"explorer_{i:03d}", f"region_{i:03d}"
        docs.extend([
            {"doc_id": person_id, "title": f"Explorer {person}",
             "text": f"Explorer {person} was born in {region}. Their expedition began decades later.",
             "scope": "global"},
            {"doc_id": region_id, "title": f"Region {region}",
             "text": f"In {region}, the primary mineral is {mineral}. Visitors come for the mountains.",
             "scope": "global"},
            {"doc_id": f"distraction_{i:03d}", "title": f"Expedition report {i:03d}",
             "text": f"Explorer {person} later explored Vale{(i+1)%n:03d}. "
                     f"This note does not describe a birthplace or minerals.", "scope": "global"},
        ])
        questions.append({"question_id": f"fixture_{i:03d}",
                          "question": f"What is the primary mineral mined in the birthplace of explorer {person}?",
                          "gold_answer": mineral,
                          "supporting_doc_ids": [person_id, region_id],
                          "split": groups[i], "group": person, "scope": "global"})
    return docs, questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("examples/fixture"))
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    docs, questions = make(args.n, args.seed)
    save(args.out / "documents.jsonl", docs)
    save(args.out / "questions.jsonl", questions)
    (args.out / "README.md").write_text(
        f"# Synthetic smoke fixture\n\nGenerated with n={args.n}, seed={args.seed}. "
        "Artificial entity chains only; **not evidence of real multi-hop QA capability**. "
        "Never quote fixture accuracy as benchmark improvement.\n", encoding="utf-8")
    print(f"Wrote {len(questions)} questions, {len(docs)} docs to {args.out}")


if __name__ == "__main__":
    main()
