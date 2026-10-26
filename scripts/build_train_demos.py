"""Build TRAIN-ONLY executable oracle demonstrations for future SFT.

This script deliberately uses TRAIN gold supporting passages to make teacher
*targets*, NOT to evaluate the model or to generate online inference inputs.
It is supervised learning, not self-improvement, self-play or GRPO.

Select by SHA256(43:question_id) from the official train Parquet shards. The
source checksums are pinned; dev pilot/heldout IDs must be supplied for strict
train/dev separation. In addition to score_trace, require the labeled answer
to be lexically present in actually READ (truncated) tool observations.

Usage:
 python scripts/build_train_demos.py --train-dir <2-parquet-folder> \
  --dev-freeze <frozen-dev/FROZEN_MANIFEST.json> --out <new-dir> --n 1024
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path

from sidra.agent import AgentConfig, SearchAgent
from sidra.io import load_dataset, save_jsonl
from sidra.metrics import normalize_answer, score_trace
from sidra.retrieval import BM25Index
from sidra.schema import Action

if __package__:
    from scripts.convert_hotpotqa import convert
else:
    from convert_hotpotqa import convert


EXPECTED_SHARDS={
    "train-00000-of-00002.parquet":
      "76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68",
    "train-00001-of-00002.parquet":
      "713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f",
}


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""):
            h.update(block)
    return h.hexdigest()


def frozen_train_rows(paths:list[Path], *, n:int=1024, seed:int=43,
                      expected:dict[str,str]|None=None):
    """Streaming minimum-hash selection independent of answers/support labels."""
    try:
        import pyarrow.parquet as pq
    except ImportError as ex:
        raise RuntimeError("requires pyarrow in the CodeRL training environment") from ex
    if n < 1:raise ValueError("n must be positive")
    hashes={}
    heap=[]  # (-hash, question_id, row)
    total=0
    seen=set()
    for path in paths:
        if expected is not None:
            expected_hash=expected.get(path.name)
            if not expected_hash or sha256(path)!=expected_hash:
                raise ValueError("unverified train Parquet source SHA256: "+path.name)
        hashes[path.name]=sha256(path)
        parquet=pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                qid=str(row["id"])
                if qid in seen:raise ValueError("duplicate original train QA id")
                seen.add(qid)
                total+=1
                rank=int.from_bytes(hashlib.sha256(f"{seed}:{qid}".encode()).digest(),"big")
                if len(heap)<n:
                    heapq.heappush(heap,(-rank,qid,row))
                elif rank<-heap[0][0]:
                    heapq.heapreplace(heap,(-rank,qid,row))
    if total<n:raise ValueError("insufficient source train rows")
    selected=[( -v[0],v[1],v[2]) for v in heap]
    selected.sort(key=lambda x:(x[0],x[1]))
    return [v[2] for v in selected],total,hashes


def to_hotpot_original(row:dict)->dict:
    context=row["context"]
    facts=row["supporting_facts"]
    titles=context.get("title",[])
    sentences=context.get("sentences",[])
    if len(titles)!=len(sentences):
        raise ValueError("bad train candidate structure")
    support=facts.get("title",[])
    indices=facts.get("sent_id",[])
    if len(support)!=len(indices):
        raise ValueError("bad supporting-fact structure")
    return {"_id":str(row["id"]),"question":row["question"],"answer":str(row["answer"]),
            "type":row.get("type"),"level":row.get("level"),
            "context":[[title,texts] for title,texts in zip(titles,sentences)],
            "supporting_facts":[[t,int(i)] for t,i in zip(support,indices)]}


class TrainOnlyOraclePolicy:
    """Labels are used ONLY to construct TRAIN action targets."""
    name="train_gold_oracle_teacher_NOT_a_model"
    def __init__(self,actions):
        self.actions=list(actions)
        self.cursor=0
    def decide(self,question,observations,remaining):
        if self.cursor>=len(self.actions):
            return Action("finish",answer="")
        action=self.actions[self.cursor]
        self.cursor+=1
        return action


def build_train(train_dir:Path, dev_manifest:Path, output:Path,
                *, n:int=1024, seed:int=43, search_top_k:int=3)->dict:
    if output.exists():raise FileExistsError(output)
    manifest=json.loads(dev_manifest.read_text())
    dev_ids=set(manifest["pilot"]["selected_ids"])|set(manifest["heldout"]["selected_ids"])
    files=[train_dir/key for key in EXPECTED_SHARDS]
    selected,total,hashes=frozen_train_rows(files,n=n,seed=seed,expected=EXPECTED_SHARDS)
    if {row["id"] for row in selected}&dev_ids:
        raise ValueError("train/dev question-id leakage")
    original=[to_hotpot_original(row) for row in selected]
    output.mkdir(parents=True,exist_ok=False)
    raw=output/"original_train_subset.json"
    raw.write_text(json.dumps(original,ensure_ascii=False),encoding="utf-8")
    data_dir=output/"converted"
    convert(raw,data_dir,"train")
    docs,questions=load_dataset(data_dir)
    docmap={d.doc_id:d for d in docs}
    index=BM25Index(docs)
    traces=[]
    qualified=0
    executed=0
    reasons={}
    def skip(why):
        reasons[why]=reasons.get(why,0)+1
    cfg=AgentConfig(max_steps=6,max_tool_calls=5,search_top_k=search_top_k)
    for q in questions:
        gold=set(q.supporting_doc_ids)
        if len(gold)!=2:
            skip("not_two_distinct_support_passages");continue
        hits=index.search(q.question,scope=q.scope,k=search_top_k)
        visible=[h.doc_id for h in hits]
        first=next((v for v in visible if v in gold),None)
        if first is None:
            skip("no_gold_support_in_initial_search");continue
        second=next(iter(gold-{first}))
        if second in visible:
            actions=[
                Action("search",query=q.question),
                Action("read",doc_id=first),
                Action("read",doc_id=second),
                Action("finish",answer=q.gold_answer,citations=(first,second)),
            ]
        else:
            second_title=docmap[second].title
            # Teacher must not invent a second-hop entity not present in
            # question/actually-read first-hop passage.
            first_text=docmap[first].text[:cfg.max_read_chars]
            if second_title.casefold() not in (
                    q.question+" "+first_text).casefold():
                skip("second_hop_not_lexically_exposed");continue
            next_hits=index.search(second_title,scope=q.scope,k=search_top_k)
            if second not in {h.doc_id for h in next_hits}:
                skip("second_hop_query_misses_support");continue
            actions=[
                Action("search",query=q.question),
                Action("read",doc_id=first),
                Action("search",query=second_title),
                Action("read",doc_id=second),
                Action("finish",answer=q.gold_answer,citations=(first,second)),
            ]
        qualified+=1
        trace=SearchAgent(docs,TrainOnlyOraclePolicy(actions),cfg).run(q)
        executed+=1
        seen=" ".join(s["observation"]["text"] for s in trace.steps if
                      (s.get("observation") or {}).get("kind")=="read")
        if (trace.termination!="finish" or
                not score_trace(q,trace,docmap)["verified_success"] or
                normalize_answer(q.gold_answer) not in normalize_answer(seen)):
            skip("oracle_not_fully_grounded_in_observed_text");continue
        traces.append(trace.as_dict())
    save_jsonl(output/"train_oracle_traces.jsonl",traces)
    from sidra.sft import build_sft_examples
    sft=build_sft_examples(data_dir,output/"train_oracle_traces.jsonl",
                           output/"train_sft_actions.jsonl") if traces else None
    result={"source":"official HotpotQA distractor TRAIN parquet, no DEV content",
            "source_sha256":hashes,"total_original_train_rows":total,
            "selected_train_questions":len(selected),
            "selected_train_id_sha256":hashlib.sha256(
                "\n".join(row["_id"] for row in original).encode()).hexdigest(),
            "seed":seed,"search_top_k":search_top_k,
            "trained_model":False,"generated_by":"oracle TRAIN labels, NOT LLM",
            "qualified_candidate_questions":qualified,
            "executed_candidate_questions":executed,
            "verified_train_oracle_traces":len(traces),
            "skipped_reasons":reasons,
            "action_supervision_examples":sft["action_training_examples"] if sft else 0,
            "dev_pilot_and_heldout_used_for_targets":False,
            "caveat":"Teacher creates easy/label-conditioned successful train trajectories;"
                     " no evidence of Search Agent generalization until independent eval."}
    (output/"TRAIN_DATA_MANIFEST.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train-dir",required=True,type=Path)
    p.add_argument("--dev-freeze",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--n",type=int,default=1024)
    p.add_argument("--seed",type=int,default=43)
    a=p.parse_args()
    print(json.dumps(build_train(a.train_dir,a.dev_freeze,a.out,
                                 n=a.n,seed=a.seed),indent=2))


if __name__=="__main__":
    main()
