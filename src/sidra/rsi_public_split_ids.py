"""Public DEV question ID-only exclusion — independent of evaluator/semantic CLI.

No DEV labels, answers, supporting facts or sealed final-B files are loaded.
Keep this dependency leaf importable by installed sidra packages: importing
`sidra.rsi_trainer_admission` pulls in a scripts namespace and is unsuitable
when running a script with only PYTHONPATH=src.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def frozen_public_dev_ids(manifest_path:Path)->tuple[set[str],str]:
    resolved=str(manifest_path.resolve()).casefold()
    if any(token in resolved for token in ("final-b","final_b","b160")):
        raise ValueError("never open a sealed final-B file")
    # The M4 manifest is already a public-dev split selection ID manifest.
    # Read only its ID/count fields, never the actual DEV dataset.
    raw=manifest_path.read_bytes()
    obj=json.loads(raw)
    if not isinstance(obj,dict):
        raise ValueError("malformed public DEV manifest")
    seen=set()
    for part in ("pilot","heldout"):
        row=obj.get(part)
        if not isinstance(row,dict):
            raise ValueError("missing public DEV ID-only section")
        ids=row.get("selected_ids")
        if (not isinstance(ids,list) or not ids
            or type(row.get("questions")) is not int
            or row["questions"]!=len(ids)
            or any(not isinstance(x,str) or not x for x in ids)
            or len(ids)!=len(set(ids))
            or seen.intersection(ids)):
            raise ValueError("public DEV ID lists are malformed, duplicate or overlapping")
        seen.update(ids)
    return seen,hashlib.sha256(raw).hexdigest()
