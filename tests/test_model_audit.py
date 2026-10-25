"""Offline model cache audit protects against config-only snapshots."""
import json
from pathlib import Path
import runpy

from sidra.evidence_audit import EvidenceAudit
from sidra.schema import Action


def audit(path: Path):
    inspect = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_models.py"))["inspect_model"]
    return inspect(path)


def fake_cache(tmp_path, name="models--Qwen--Qwen3-8B"):
    model = tmp_path / name
    snap = model / "snapshots" / "abcdef12345"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
    return model, snap


def test_config_only_qwen3_is_not_a_usable_model(tmp_path):
    model, snap = fake_cache(tmp_path)
    result = audit(model)
    assert result["weights_complete"] is False
    assert result["ready"] is False
    assert "weights_missing" in result["runtime_blockers"]


def test_missing_weight_shard_is_detected(tmp_path):
    model, snap = fake_cache(tmp_path)
    index = {"weight_map": {
        "layer.0": "model-00001-of-00002.safetensors",
        "layer.1": "model-00002-of-00002.safetensors",
    }}
    (snap / "model.safetensors.index.json").write_text(json.dumps(index))
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"fake")
    result = audit(model)
    assert result["ready"] is False
    assert "model-00002-of-00002.safetensors" in result["weight_files_missing"]


def test_unsharded_weight_file_presence_only(tmp_path):
    model, snap = fake_cache(tmp_path, "models--Qwen--Qwen3-1.7B-Base")
    (snap / "model.safetensors").write_bytes(b"placeholder")
    result = audit(model)
    assert result["weights_complete"] is True
    assert result["ready"] is True
    assert "actual inference remains untested" in result["note"]


def test_audit_rejects_punctuation_only_answer():
    obs = [
        {"kind": "read", "doc_id": "a", "text": "a record"},
        {"kind": "read", "doc_id": "b", "text": "another record"},
    ]
    result = EvidenceAudit(2).verify(
        Action("finish", answer="...", citations=("a", "b")), obs)
    assert result.approved is False
