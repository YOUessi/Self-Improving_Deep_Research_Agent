"""Run a real local, corpus-grounded Qwen3 Challenger on frozen TRAIN only.

The output is an UNLABELLED audit queue. A successful quote gate does not
prove a generated question's answer or permit any training reward.

python scripts/run_rsi_grounded_challenger.py \
  --train-data /path/TRAIN/converted \
  --queue /path/g1/CHALLENGER_INPUT.jsonl \
  --manifest /path/g1/QUEUE_MANIFEST.json \
  --verifier-map /path/g1/VERIFIER_ONLY_MAP.jsonl \
  --endpoint http://127.0.0.1:18985/v1/chat/completions \
  --model sidra-qwen3-8b-awq --limit 8 --out /new/path
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from urllib.error import URLError

from sidra.io import load_dataset,save_jsonl
from sidra.retrieval import BM25Index
from sidra.rsi_challenge_scheduler import _sha
from sidra.rsi_grounded_challenger import (
    GroundedCandidate,LocalGroundedChallenger,
    make_grounding_packet,verify_grounded_response,
)


def hash_file(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(
    *,train_data:Path,queue_file:Path,manifest_file:Path,verifier_map:Path,
    backend:LocalGroundedChallenger,seed:int,out:Path,limit:int=8,
)->dict:
    if out.exists():
        raise FileExistsError("refuse overwrite previous model generation")
    manifest=json.loads(manifest_file.read_text())
    if manifest.get("split")!="train" or manifest.get("gold_fields_in_challenger_input") is not False:
        raise ValueError("only trusted TRAIN-only Challenger queues are allowed")
    if (hash_file(queue_file)!=manifest.get("challenger_input_sha256") or
        hash_file(verifier_map)!=manifest.get("verifier_only_map_sha256")):
        raise ValueError("frozen queue/map SHA mismatch")
    docs,questions=load_dataset(train_data)
    if any(q.split!="train" for q in questions):
        raise ValueError("mixed or dev dataset forbidden")
    qmap={q.question_id:q for q in questions}
    queues=[json.loads(x) for x in queue_file.read_text().splitlines() if x.strip()]
    mappings=[json.loads(x) for x in verifier_map.read_text().splitlines() if x.strip()]
    if len(queues)!=manifest["n"] or len(mappings)!=manifest["n"]:
        raise ValueError("queue/map count mismatch")
    map_by_id={}
    for row in mappings:
        if (set(row)!={"challenge_id","source_train_question_id","text_sha256","tier"}
            or row["challenge_id"] in map_by_id):
            raise ValueError("invalid trusted map")
        map_by_id[row["challenge_id"]]=row
    seen=set()
    for entry in queues:
        if set(entry)!={"challenge_id","question"}:
            raise ValueError("Challenger input contains privileged fields")
        identity=entry["challenge_id"]
        if identity in seen or identity not in map_by_id:
            raise ValueError("unknown or repeated candidate ID")
        seen.add(identity)
        mapping=map_by_id[identity]
        q=qmap.get(mapping["source_train_question_id"])
        if (not q or q.question!=entry["question"]
            or _sha(q.question)!=mapping["text_sha256"]):
            raise ValueError("TRAIN source identity/hash does not match trusted frozen queue")
    if seen!=set(map_by_id):
        raise ValueError("queue missing verifier map IDs")
    if limit<1 or limit>len(queues):
        raise ValueError("invalid candidate limit")
    index=BM25Index(docs,title_boost=0.0)
    proposals=[]
    for ix,row in enumerate(queues[:limit]):
        identity=row["challenge_id"]
        q=qmap[map_by_id[identity]["source_train_question_id"]]
        packet=None
        try:
            # Unlike the verifier, this packet constructor reads NO
            # q.gold_answer or q.supporting_doc_ids.
            packet=make_grounding_packet(
                identity,row["question"],scope=q.scope,index=index,top_k=4)
            raw=backend.generate_packet(packet,seed=seed+ix)
            outcome=verify_grounded_response(raw,packet)
        except (URLError,TimeoutError,ValueError,KeyError,IndexError,OSError):
            outcome=GroundedCandidate(
                identity,"","","",(),
                "inference_or_retrieval_error",
                packet.packet_sha256 if packet is not None else "","")
        proposals.append(outcome.as_dict())
        print("GROUNDED_CHALLENGE",ix+1,"/",limit,outcome.verdict,flush=True)
    out.mkdir(parents=True,exist_ok=False)
    dest=out/"UNLABELLED_GROUNDED_CANDIDATES.jsonl"
    save_jsonl(dest,proposals)
    histogram=dict(Counter(x["verdict"] for x in proposals))
    count=histogram.get(
        "quote_extracted_unlabelled_pending_semantic_verification",0)
    result={
        "status":"model_generated_train_only_quote_gate_not_training",
        "generation":manifest["generation"],"seed":seed,
        "n_attempted":limit,"model":backend.model,
        "local_only":True,"question_input_sha256":hash_file(queue_file),
        "train_dataset_questions_sha256":hash_file(train_data/"questions.jsonl"),
        "train_dataset_documents_sha256":hash_file(train_data/"documents.jsonl"),
        "grounded_candidates_output_sha256":hash_file(dest),
        "observed_quote_structural_passes":count,
        "verdict_histogram":histogram,
        "gold_answer_or_supporting_ids_in_model_prompts":False,
        "model_proposed_answer_is_gold_truth":False,
        "semantic_entailment_verified":False,
        "new_trainable_challenges_approved":0,
        "optimizer_updates":0,
        "multi_generation_RSI_completed":False,
        "caution":"Even perfect quotes and an extractive proposed answer "
                  "cannot prove question premise or semantic entailment. "
                  "Never train on these novel proposals without separate verification.",
    }
    (out/"GROUNDED_CANDIDATE_SUMMARY.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--train-data",type=Path,required=True)
    p.add_argument("--queue",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--verifier-map",type=Path,required=True)
    p.add_argument("--endpoint",required=True)
    p.add_argument("--model",required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seed",type=int,default=43)
    p.add_argument("--limit",type=int,default=8)
    p.add_argument("--max-tokens",type=int,default=480)
    args=p.parse_args()
    engine=LocalGroundedChallenger(
        args.endpoint,args.model,max_tokens=args.max_tokens,
        temperature=0.5,top_p=0.9)
    summary=execute(
        train_data=args.train_data,queue_file=args.queue,manifest_file=args.manifest,
        verifier_map=args.verifier_map,backend=engine,seed=args.seed,
        limit=args.limit,out=args.out)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
