"""CLI tests for `prodromos plan`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prodromos.plan import cli as plan_cli

_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "tm-spec" / "examples" / "preflight_example.tm.yaml"
)

pytest.importorskip("tm_spec.validator")


def test_cli_route_envelope_exit0(capsys):
    rc = plan_cli.main([str(_EXAMPLE), "--mode", "route", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "plan"
    assert payload["verdict"] == "GO"


def test_cli_emit_preflight(capsys):
    rc = plan_cli.main([str(_EXAMPLE), "--emit", "preflight", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"]["name"] == "prodromos"


def test_cli_invalid_case(tmp_path, capsys):
    bad = tmp_path / "bad.tm.yaml"
    bad.write_text("spec: tm-spec/0.3\nkind: NEBCalculation\n", encoding="utf-8")
    rc = plan_cli.main([str(bad), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "INVALID_CASE"


def test_cli_missing_file(tmp_path, capsys):
    rc = plan_cli.main([str(tmp_path / "nope.tm.yaml"), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "INVALID_CASE"
