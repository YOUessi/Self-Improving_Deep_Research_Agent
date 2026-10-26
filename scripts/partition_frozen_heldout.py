"""Re-partition already frozen heldout IDs into immutable eval-A and final-B.

Input manifest is created by scripts/freeze_hotpot_dev.py. Split order is the
ORIGINAL hash-ranked order, not evaluation's lexicographic question sorting.
The source file is *original dev*; eval-A/final-B stay split='dev', never train.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from scripts.convert_hotpotqa import convert
else:
    from convert_hotpotqa import convert


def partition(root: Path, dest: Path, eval_a: int=64) -> dict:
    if dest.exists():
        raise FileExistsError(dest)
    m=json.loads((root/"FROZEN_MANIFEST.json").read_text())
    raw_path=root/"heldout/original_dev_subset.json"
    original_bytes=raw_path.read_bytes()
    if hashlib.sha256(original_bytes).hexdigest()!=m["heldout"]["original_subset_sha256"]:
        raise ValueError("heldout data was modified after freezing")
    rows=json.loads(original_bytes)
    ids=[row["_id"] for row in rows]
    if ids!=m["heldout"]["selected_ids"]:
        raise ValueError("heldout ID ordering does not match official freeze manifest")
    if not (0<eval_a<len(rows)):
        raise ValueError("eval-A must leave a nonempty sealed final-B")
    pilot=set(m["pilot"]["selected_ids"])
    if pilot.intersection(ids):
        raise ValueError("pilot and heldout overlap")
    subsets=[("eval_a", rows[:eval_a]),("final_b",rows[eval_a:])]
    result={"source_dev_sha256":m["source_sha256"],"seed":m["seed"],
            "eval_a_size":eval_a,"sealed_final_b_size":len(rows)-eval_a,
            "selection":"Original SHA256(ID) ranking; no label-conditional sampling",
            "dev_only_no_train":True}
    dest.mkdir(parents=True,exist_ok=False)
    for name,sub in subsets:
        d=dest/name;d.mkdir()
        file=d/"original_dev_subset.json"
        file.write_text(json.dumps(sub,ensure_ascii=False),encoding="utf-8")
        convert(file,d/"converted",split="dev")
        (d/"converted/README.md").write_text(
            f"Original HotpotQA distractor dev: {name}. "
            "Must not be used for SFT, GRPO, reward modeling or selecting prompts. "
            "Sealed final-B is reserved until the training protocol is fixed.\n")
        result[name]={
            "ids":[r["_id"] for r in sub],
            "converted_question_sha256":hashlib.sha256(
                (d/"converted/questions.jsonl").read_bytes()).hexdigest(),
            "converted_doc_sha256":hashlib.sha256(
                (d/"converted/documents.jsonl").read_bytes()).hexdigest(),
        }
    (dest/"FROZEN_HELDOUT_SPLIT.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True)+"\n")
    return result


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--frozen-root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--eval-a",type=int,default=64)
    a=p.parse_args()
    result=partition(a.frozen_root,a.out,a.eval_a)
    print("eval-A",len(result["eval_a"]["ids"]),
          result["eval_a"]["converted_question_sha256"])
    print("final-B",len(result["final_b"]["ids"]),
          result["final_b"]["converted_question_sha256"])


if __name__=="__main__":
    main()
