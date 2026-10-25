"""Audit Hugging Face local snapshots without network access or model loading.

Usage: python scripts/audit_models.py [--root ~/.cache/huggingface/hub]
A directory containing config.json alone is not a usable model checkpoint.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def inspect_model(path: Path) -> dict:
    snapshots = sorted((path / "snapshots").glob("*")) if (path / "snapshots").exists() else []
    if not snapshots:
        return {"name": path.name.replace("models--", "").replace("--", "/"),
                "ready": False, "reason": "no_snapshot"}
    snap = snapshots[-1]
    config_path = snap / "config.json"
    try:
        config = json.loads(config_path.read_text()) if config_path.exists() else {}
    except (ValueError, OSError):
        return {"name": path.name, "ready": False, "reason": "bad_config"}
    indexes = list(snap.glob("*.safetensors.index.json")) + list(snap.glob("*.bin.index.json"))
    if indexes:
        try:
            expected = sorted(set(json.loads(indexes[0].read_text())["weight_map"].values()))
        except (KeyError, ValueError, OSError):
            expected = []
    else:
        expected = sorted(f.name for f in snap.glob("*.safetensors")) + sorted(
            f.name for f in snap.glob("pytorch_model*.bin"))
    missing = [f for f in expected if not (snap / f).is_file()]
    quant = config.get("quantization_config") or {}
    backend = str(quant.get("backend", quant.get("quant_method", "none")))
    blockers = []
    if not expected:
        blockers.append("weights_missing")
    if missing:
        blockers.append("weight_shards_missing")
    if quant.get("quant_method") == "awq" and importlib.util.find_spec("awq") is None:
        blockers.append("awq_runtime_not_installed")
    name = path.name.replace("models--", "").replace("--", "/")
    return {
        "name": name,
        "revision": snap.name,
        "model_type": config.get("model_type", "unknown"),
        "quantization": backend,
        "weight_files_expected": len(expected),
        "weight_files_missing": missing,
        "weights_complete": bool(expected) and not missing,
        "runtime_blockers": blockers,
        "ready": not blockers,
        "note": "ready indicates files and known AWQ dependency only; actual inference remains untested",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / ".cache/huggingface/hub")
    args = parser.parse_args()
    models = sorted(args.root.glob("models--Qwen--*"))
    print(json.dumps([inspect_model(p) for p in models], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
