"""M7-A8 paired CLI must not score models on an unverified frozen TRAIN64."""
from __future__ import annotations

import sys

import pytest

from scripts.audit_frozen_train64 import main


def _arguments(tmp_path):
    return [
        "audit_frozen_train64",
        "--frozen",str(tmp_path/"frozen"),
        "--train-summary",str(tmp_path/"TRAIN_SUMMARY.json"),
        "--baseline",str(tmp_path/"baseline"),
        "--candidate",str(tmp_path/"candidate"),
        "--out",str(tmp_path/"paired.json"),
    ]


def test_paired_cli_rejects_missing_historical_source_replay(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(sys,"argv",_arguments(tmp_path))
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code==2
    err=capsys.readouterr().err
    assert "--m7-root" in err
    assert "--original-train-source" in err
    assert not (tmp_path/"paired.json").exists()


def test_paired_cli_validates_history_before_evaluating(monkeypatch,tmp_path):
    args=_arguments(tmp_path)+[
        "--m7-root",str(tmp_path/"not-present"),
        "--original-train-source",str(tmp_path/"train"),
    ]
    monkeypatch.setattr(sys,"argv",args)
    with pytest.raises((ValueError,FileNotFoundError)):
        main()
    assert not (tmp_path/"paired.json").exists()
