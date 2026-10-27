"""M7-A8 preregistered TRAIN64 source-replay audit (CPU only; no models).

Independent source check. Unlike a paired scoring script that merely trusts a
self-reported freeze manifest, this replays the *entire original selection*
from immutable, SHA-pinned TRAIN-only teacher and GRPO input histories.

Never reads DEV labels, eval-A64, or sealed final-B160. No raw QA is printed
or committed. Fails closed when a history source is missing/modified or a
further completed optimizer run accidentally trained on a frozen question.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


# These checksums were obtained from the pre-existing files listed in the
# 2026-10-27 A8 preregistration (before A8 training), NOT from model outputs.
TEACHER_HISTORY={
    "data/teacher1024_v2_seed43_26e00f26c/TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl":
        "f3373d516265ca6745a63b64f40cf41ce3bb3df878dd10fdbc50617ec460325c",
    "data/teacher1024_v2_seed43_26e00f26c/TEACHER_ACTIONS_TRAIN.jsonl":
        "a100ab91763800bdcb1c8f616af29f2968f5ef1f27b290131c32271162757e2c",
    "data/teacher_v3_reachable1024_s43_494d32c0/TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl":
        "1128ad029ddde981306e2ae60014f47ce3f6c142e77842458d07a1f7d88c3c58",
    "data/teacher_v3_reachable1024_s43_494d32c0/TEACHER_ACTIONS_TRAIN.jsonl":
        "76a080ee285c83dad3495d408fa37b0325050629a1aa7218cf6225d74734a111",
    "data/v4_bridgeprior_teacher1024_s43_4215c05e/TEACHER_ACTIONS_HELDOUT_TRAIN.jsonl":
        "ee912eefcdcf044c9138c0bd15233f5dfb2ab30b7d01d80b4e52d2168ac4ca5c",
    "data/v4_bridgeprior_teacher1024_s43_4215c05e/TEACHER_ACTIONS_TRAIN.jsonl":
        "e6af2db0f9ba259f9cdc8a240379b68c15c1a7aed275f68474a42287ebd855bf",
}
FIT_HISTORY={
    "results/M7A7_CONTROL_s153_G20x4_U2_ac6cc67a/TRAIN_SUMMARY.json":
        "e1ad7228ff38c10b9c9397c6a00efe7b409123adc4bb6c540c25b7c481570783",
    "results/M7A7_CONTROL_s153_G20x4_U2_d0589ad8/TRAIN_SUMMARY.json":
        "365c68ee4e634266025ce890306cef636c442975154319c8356dfd008de41db4",
    "results/M7A7_HYBRID_s153_G20x4_U2_d0589ad8/TRAIN_SUMMARY.json":
        "87dbd13dc3f3369d1be9fc5eebe5d75978827813cfdd5b7fcb122fdcfdf85d11",
    "results/finite_budget_grpo_seed54_g6x4/TRAIN_SUMMARY.json":
        "c5b729fda4a8201b80b305a36249029630a4b9d6408858ea85acf131e1be5128",
    "results/finite_legal_grpo_seed53_g6x4_v2/TRAIN_SUMMARY.json":
        "40ef8a3a5885460fa6d3308f1461fe2cb4df2aa00d2a8a92060109e47cf3a521",
    "results/finite_v2_seed63_g12x4_u1/TRAIN_SUMMARY.json":
        "d65c0ef8928aff1798d2f4e0d22354b6344a201eb18a49936394932c44746a00",
    "results/finite_v2_seed71_POST_TEACHER240_onlineG12x4_U2_6a0ee669/TRAIN_SUMMARY.json":
        "f345644638e3648d777b34c6b0242a2a89a588b4588f8c9a64b7bd79fb5555c8",
    "results/v3_ONPOLICY_Reachable20x4_u2_s81_494d32c0/TRAIN_SUMMARY.json":
        "eac4ad2622c3017563545b30da846ef4b7e3a821b023016d95ef65edc19066a4",
    "results/v3_STAGECREDIT_s81_G20x4_U2_1773b18e/TRAIN_SUMMARY.json":
        "4ef5dc9547327c36a1351431db1fdf42fa36ee79ebbf7cd009bb4b2cbc02e706",
    "results/v3_STEPDELTA_s81_G20x4_U2_c05552da/TRAIN_SUMMARY.json":
        "0650f31e0c535730a20a25eb25d3fa928073734939916fa04beb4d19b912f2da",
    "results/v4_BRIDGEPRIOR_s131_G20x4_U2_61703c75/TRAIN_SUMMARY.json":
        "a5f6f83fdb70b039ea3173247db12d20eb54bbd2d4ff97fb1754d8956cde3bfb",
}


@dataclass(frozen=True)
class FreezeProtocol:
    source_questions_sha256: str
    source_documents_sha256: str
    frozen_questions_sha256: str
    frozen_documents_sha256: str
    ordered_qid_sha256: str
    count: int
    excluded_count: int
    selection_seed: int
    source_sha: str


A8=FreezeProtocol(
    source_questions_sha256="53f8b07853f451dc8f7854cef04288de9f7637f6c1b8365fcbee2d0340ee20a8",
    source_documents_sha256="c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111",
    frozen_questions_sha256="c310a1cc2f23ab87925d24c6392ea51afcfbde29018f40300a588972eed60c92",
    frozen_documents_sha256="c2f1060e5921232c72c3636df7df4ec6f74df7736c1dbd264e92aff3a63bf111",
    ordered_qid_sha256="336fa32d645a2b52cf68563a56dc9d6571819971c872a92cfb790e2876fcda81",
    count=64,excluded_count=402,selection_seed=919,
    source_sha="9d81deca7dc4efe72170369f89d37cb4fe49d815",
)


def _sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda:fh.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def _jsonl(path:Path)->list[dict]:
    rows=[]
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj=json.loads(line)
            if not isinstance(obj,dict):
                raise ValueError("history row must be an object")
            rows.append(obj)
    return rows


def _ids_from_records(rows:list[dict],key:str)->set[str]:
    ids=set()
    for x in rows:
        qid=x.get(key)
        if not isinstance(qid,str) or not qid:
            raise ValueError("history contains empty question id")
        ids.add(qid)
    return ids


def _historical_ids(root:Path, manifests:dict[str,str], *,
                    teacher:bool)->set[str]:
    result=set()
    for rel,expect in sorted(manifests.items()):
        path=root/rel
        if not path.is_file() or _sha256(path)!=expect:
            raise ValueError(f"history input missing or modified: {rel}")
        if teacher:
            result.update(_ids_from_records(_jsonl(path),"question_id"))
        else:
            row=json.loads(path.read_text(encoding="utf-8"))
            if (row.get("split")!="train"
                or row.get("stage")!="real_online_finite_constrained_categorical_grpo_style"
                or row.get("dev_gold_used_for_reward") is not False):
                raise ValueError("historical fit is not verified TRAIN-only")
            ids=row.get("selected_training_ids")
            if not isinstance(ids,list) or not ids or len(ids)!=len(set(ids)):
                raise ValueError("historical fit IDs missing or malformed")
            result.update(_ids_from_records([{"question_id":x} for x in ids],"question_id"))
    return result


def select_disjoint_train(ids:set[str],excluded:set[str],seed:int,count:int)->list[str]:
    if not excluded<=ids:
        raise ValueError("historical teacher/fit contains unknown TRAIN id")
    available=ids-excluded
    if len(available)<count:
        raise ValueError("insufficient unused TRAIN questions")
    return sorted(available,key=lambda q:(
        hashlib.sha256(f"M7A8-FROZEN-HOLDOUT-{seed}:{q}".encode()).hexdigest(),q
    ))[:count]


def validate_a8_freeze(*,m7root:Path,source:Path,frozen:Path,
                       protocol:FreezeProtocol=A8,
                       teacher_history:dict[str,str]|None=None,
                       fit_history:dict[str,str]|None=None,
                       inspect_new_fits:bool=True)->dict:
    teacher_history=TEACHER_HISTORY if teacher_history is None else teacher_history
    fit_history=FIT_HISTORY if fit_history is None else fit_history
    if (len(teacher_history)<1 or len(fit_history)<1):
        raise ValueError("must check original teacher and real optimizer history")
    # Enforce SHA independently of the user-editable FREEZE_MANIFEST.json.
    expected={
        source/"questions.jsonl":protocol.source_questions_sha256,
        source/"documents.jsonl":protocol.source_documents_sha256,
        frozen/"questions.jsonl":protocol.frozen_questions_sha256,
        frozen/"documents.jsonl":protocol.frozen_documents_sha256,
    }
    for file,sha in expected.items():
        if not file.is_file() or _sha256(file)!=sha:
            raise ValueError(f"pre-registered dataset checksum mismatch: {file.name}")
    source_rows=_jsonl(source/"questions.jsonl")
    ids=_ids_from_records(source_rows,"question_id")
    if (len(ids)!=len(source_rows)
        or any(x.get("split")!="train" for x in source_rows)):
        raise ValueError("original source must contain unique TRAIN-only questions")
    historical=_historical_ids(m7root,teacher_history,teacher=True)
    historical|=_historical_ids(m7root,fit_history,teacher=False)
    if len(historical)!=protocol.excluded_count:
        raise ValueError("original TRAIN excluded-history count differs")
    ordered=select_disjoint_train(ids,historical,protocol.selection_seed,protocol.count)
    digest=hashlib.sha256("\n".join(ordered).encode()).hexdigest()
    if digest!=protocol.ordered_qid_sha256:
        raise ValueError("pre-registered ordered question ID digest differs")
    actual=_jsonl(frozen/"questions.jsonl")
    frozen_ids=_ids_from_records(actual,"question_id")
    if (len(actual)!=protocol.count or frozen_ids!=set(ordered)
        or any(x.get("split")!="train" for x in actual)):
        raise ValueError("frozen TRAIN64 IDs or split differ from source replay")
    by_id={x["question_id"]:x for x in source_rows}
    if any(x!=by_id[x["question_id"]] for x in actual):
        raise ValueError("frozen TRAIN labels altered relative to original source")
    man=json.loads((frozen/"FREEZE_MANIFEST.json").read_text())
    for name in (
        "source_questions_sha256","source_documents_sha256",
        "frozen_questions_sha256","frozen_documents_sha256",
        "ordered_qid_sha256","selection_seed","count","excluded_count","source_sha"
    ):
        if man.get(name)!=getattr(protocol,name):
            raise ValueError(f"frozen manifest field modified: {name}")
    if (man.get("split")!="train"
        or man.get("model_training_labels_allowed") is not False
        or man.get("teacher_files_checked")!=len(teacher_history)
        or man.get("prior_fit_files_checked")!=len(fit_history)
        or man.get("candidate_count")!=len(ids)-len(historical)):
        raise ValueError("frozen manifest split/count/provenance mismatch")
    if inspect_new_fits:
        for path in sorted((m7root/"results").glob("*/TRAIN_SUMMARY.json")):
            rel=path.relative_to(m7root).as_posix()
            if rel in fit_history:
                continue
            row=json.loads(path.read_text(encoding="utf-8"))
            if row.get("split")!="train":
                raise ValueError("unrecognized later fit was not TRAIN-only")
            fit_ids=row.get("selected_training_ids")
            if not isinstance(fit_ids,list) or any(not isinstance(x,str) for x in fit_ids):
                raise ValueError("unrecognized later fit lacks auditable TRAIN IDs")
            if set(fit_ids)&frozen_ids:
                raise ValueError(f"FROZEN TRAIN64 contaminated by later model training: {rel}")
    return {
        "status":"PASS","stage":"M7A8_TRAIN64_INDEPENDENT_SOURCE_REPLAY_AUDIT",
        "source_question_count":len(ids),"historically_excluded_count":len(historical),
        "eligible_count":len(ids)-len(historical),"frozen_count":len(frozen_ids),
        "teacher_files_pinned":len(teacher_history),
        "training_summaries_pinned":len(fit_history),
        "ordered_qid_sha256":digest,
        "frozen_questions_sha256":protocol.frozen_questions_sha256,
        "frozen_documents_sha256":protocol.frozen_documents_sha256,
        "new_model_training_leakage_checked":bool(inspect_new_fits),
        "sealed_final_B160_opened":False,
        "public_DEV_gold_opened":False,
        "QA_content_uploaded":False,
        "warning":"Internal TRAIN-only disjoint development set, not independent external data.",
    }


def main()->None:
    p=argparse.ArgumentParser()
    p.add_argument("--m7-root",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--frozen",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError(args.out)
    report=validate_a8_freeze(m7root=args.m7_root,source=args.source,
                              frozen=args.frozen)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
