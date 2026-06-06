"""Tests for the structure-level magnetic-sublattice crossing predictor (§C-bis)."""
from __future__ import annotations

from prodromos.sublattice_preflight import (
    MagSite,
    assign_signs_by_coordinate,
    nearest_site,
    run_sublattice_preflight,
    sites_from_magmoms,
)

_CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def _sites():
    return [
        MagSite("Fe", (0.10, 0.25, 0.50), sign=+1, moment_uB=4.0),  # 0
        MagSite("Fe", (0.90, 0.75, 0.50), sign=-1, moment_uB=-4.0),  # 1
        MagSite("Fe", (0.50, 0.25, 0.10), sign=+1, moment_uB=4.0),  # 2
    ]


def test_migrant_same_sublattice_is_go():
    # migrant A near site0(+1), B near site2(+1) -> same sublattice -> GO
    env = run_sublattice_preflight(
        _sites(), _CELL, migrant_a=(0.12, 0.25, 0.50), migrant_b=(0.50, 0.25, 0.12),
        mode="migrant",
    )
    assert env["verdict"] == "GO_SINGLE_SHEET"
    assert env["result"]["sublattice_sign_flips"] is False


def test_migrant_sublattice_flip_is_nogo_with_recipe():
    # migrant A near site0(+1), B near site1(-1) -> sign flip -> NO-GO + recipe
    env = run_sublattice_preflight(
        _sites(), _CELL, migrant_a=(0.12, 0.25, 0.50), migrant_b=(0.88, 0.75, 0.50),
        mode="migrant",
    )
    assert env["verdict"] == "NO-GO_SINGLE_SHEET"
    assert env["result"]["endpoint_a"]["sign"] == +1
    assert env["result"]["endpoint_b"]["sign"] == -1
    # the recipe (next_actions) must be emitted, not just the label
    assert any("starting_magnetization" in a for a in env["next_actions"])


def test_polaron_mode_explicit_indices_nogo():
    # nonmagnetic Li migrant; polaron forced on site0(+1) at A, site1(-1) at B -> NO-GO
    env = run_sublattice_preflight(
        _sites(), _CELL, migrant_a=(0.30, 0.30, 0.30), migrant_b=(0.31, 0.31, 0.31),
        mode="polaron", polaron_index_a=0, polaron_index_b=1, migrant_species="Li",
    )
    assert env["verdict"] == "NO-GO_SINGLE_SHEET"
    assert env["result"]["mode"] == "polaron"
    assert any("polaron" in r for r in env["reasons"])


def test_polaron_mode_same_site_is_go():
    env = run_sublattice_preflight(
        _sites(), _CELL, migrant_a=(0.30, 0.30, 0.30), migrant_b=(0.31, 0.31, 0.31),
        mode="polaron", polaron_index_a=0, polaron_index_b=2, migrant_species="Li",
    )
    assert env["verdict"] == "GO_SINGLE_SHEET"


def test_review_when_no_signed_sites():
    flat = [MagSite("S", (0.0, 0.0, 0.0), sign=0)]
    env = run_sublattice_preflight(flat, _CELL, (0.1, 0.1, 0.1), (0.2, 0.2, 0.2))
    assert env["verdict"] == "REVIEW"


def test_assign_signs_by_coordinate_lifepo4_rule():
    # LiFePO4 olivine: Fe 4c at y ~ 0.25 (+) / 0.75 (-)
    sites = [
        MagSite("Fe", (0.28, 0.28, 0.97), sign=0),
        MagSite("Fe", (0.72, 0.72, 0.47), sign=0),
    ]
    signed = assign_signs_by_coordinate(sites, axis=1)
    assert signed[0].sign == +1  # y=0.28 -> band [0,0.5)
    assert signed[1].sign == -1  # y=0.72 -> band [0.5,1.0)


def test_sites_from_magmoms_signs_and_filter():
    sites = sites_from_magmoms(
        elements=["Fe", "Fe", "O"],
        fracs=[(0, 0, 0), (0.5, 0.5, 0.5), (0.25, 0.25, 0.25)],
        magmoms_uB=[4.1, -4.1, 0.01],
        mag_elements={"Fe"},
    )
    assert len(sites) == 2
    assert sites[0].sign == +1 and sites[1].sign == -1


def test_nearest_site_uses_minimum_image():
    # migrant at 0.95 is closer to a site at 0.02 across the boundary than to 0.5
    sites = [MagSite("Fe", (0.02, 0.5, 0.5), +1), MagSite("Fe", (0.50, 0.5, 0.5), -1)]
    i, d = nearest_site((0.95, 0.5, 0.5), sites, _CELL)
    assert i == 0  # MIC wraps 0.95->0.02 (dist 0.7 A) vs 0.5 (4.5 A)


def test_cli_main_smoke(tmp_path, capsys):
    import json

    from prodromos.sublattice_preflight import main

    spec = {
        "cell": _CELL,
        "sites": [
            {"element": "Fe", "frac": [0.10, 0.25, 0.50], "sign": 1},
            {"element": "Fe", "frac": [0.90, 0.75, 0.50], "sign": -1},
        ],
        "migrant_a": [0.12, 0.25, 0.50],
        "migrant_b": [0.88, 0.75, 0.50],
        "mode": "migrant",
    }
    path = tmp_path / "case.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    rc = main(["--input", str(path)])
    assert rc == 0
    assert "NO-GO_SINGLE_SHEET" in capsys.readouterr().out
