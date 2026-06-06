"""Tests for the magnetic-provenance gate (§C/§C-bis), pure offline comparison."""
from __future__ import annotations

from prodromos.magnetic_provenance import (
    compare_ordering,
    normalize_ordering,
    run_magnetic_provenance,
)


def test_normalize_aliases():
    assert normalize_ordering("antiferromagnetic") == "AFM"
    assert normalize_ordering("AFM-G") == "AFM"
    assert normalize_ordering("Ferri") == "FiM"
    assert normalize_ordering(None) is None


def test_troilite_mp_fm_vs_experiment_afm_conflict():
    # MP says FM, MAGNDATA (neutron) says AFM -> type conflict, seed from experiment.
    cmp = compare_ordering("FM", "AFM")
    assert cmp["verdict"] == "CONFLICT_TYPE"
    assert cmp["agree"] is False
    assert cmp["seed_source"] == "magndata"
    assert "DISAGREES" in cmp["warning"]


def test_binary_conflict_nm_vs_afm():
    cmp = compare_ordering("NM", "AFM")
    assert cmp["verdict"] == "CONFLICT_BINARY"
    assert cmp["seed_source"] == "magndata"


def test_agree_no_warning():
    cmp = compare_ordering("AFM", "AFM-G")
    assert cmp["verdict"] == "AGREE"
    assert cmp["agree"] is True
    assert cmp["warning"] is None


def test_mp_only_warns_unverified():
    cmp = compare_ordering("FM", None)
    assert cmp["verdict"] == "MP_ONLY"
    assert cmp["seed_source"] == "mp"
    assert "unverified" in cmp["warning"]


def test_no_data():
    assert compare_ordering(None, None)["verdict"] == "NO_DATA"


def test_run_envelope_conflict_routes_to_magndata():
    env = run_magnetic_provenance(mp_ordering="FM", magndata_ordering="AFM")
    assert env["tool"] == "magnetic_provenance"
    assert env["verdict"] == "CONFLICT_TYPE"
    assert env["result"]["seed_source"] == "magndata"
    assert env["warnings"]
    assert any("experimental MAGNDATA" in a for a in env["next_actions"])


def test_cli_main_smoke(capsys):
    from prodromos.magnetic_provenance import main

    rc = main(["--mp-ordering", "FM", "--magndata-ordering", "AFM"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "CONFLICT_TYPE" in out and "magndata" in out
