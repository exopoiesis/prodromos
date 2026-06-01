import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.saddle_proximity_gate import run_saddle_proximity_gate, _clean_read

CELL = 20.0  # large cubic cell so MIC is trivial for these small clusters


def _write_xyz(path: Path, atoms: list[tuple[str, float, float, float]], comment: str = "") -> Path:
    lines = [str(len(atoms))]
    header = f'Lattice="{CELL} 0 0 0 {CELL} 0 0 0 {CELL}" Properties=species:S:1:pos:R:3 pbc="T T T"'
    if comment:
        header = header + " " + comment
    lines.append(header)
    for sym, x, y, z in atoms:
        lines.append(f"{sym} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def test_symmetric_direct_transfer_ok(tmp_path):
    # S0 at x=0, S1 at x=3.0, H midway at x=1.5 -> symmetric, nearest is an anchor,
    # no 3rd S nearby (S2 far away).
    p = _write_xyz(tmp_path / "saddle_ok.xyz", [
        ("S", 0.0, 5.0, 5.0),    # 0
        ("S", 3.0, 5.0, 5.0),    # 1
        ("H", 1.5, 5.0, 5.0),    # 2
        ("S", 10.0, 10.0, 10.0),  # 3 far off-path S
    ])
    atoms = _clean_read(p)
    env = run_saddle_proximity_gate(atoms, s_i=0, s_k=1)  # H auto-detected (idx 2)
    assert env["verdict"] == "DIRECT_TRANSFER_OK"
    assert env["result"]["h_idx"] == 2
    assert env["result"]["symmetric_ok"] is True
    assert env["result"]["anchor_ok"] is True
    assert env["result"]["mu_bridge_warn"] is False
    assert env["result"]["asym"] < 1e-6


def test_h_migrated_to_third_s_is_off_path(tmp_path):
    # H sits right on top of a 3rd S (idx 2), far from anchors 0 and 1.
    # nearest non-H is S2 (not an anchor) -> anchor_ok False -> OFF_PATH.
    p = _write_xyz(tmp_path / "saddle_offpath.xyz", [
        ("S", 0.0, 5.0, 5.0),    # 0 anchor i
        ("S", 8.0, 5.0, 5.0),    # 1 anchor k
        ("S", 4.0, 9.0, 5.0),    # 2 third S (where H went)
        ("H", 4.0, 10.3, 5.0),   # 3 H ~1.3 A from S2
    ])
    atoms = _clean_read(p)
    env = run_saddle_proximity_gate(atoms, s_i=0, s_k=1, h_idx=3)
    assert env["verdict"] == "OFF_PATH_OR_INTERMEDIATE_INVESTIGATE"
    assert env["result"]["anchor_ok"] is False
    assert env["result"]["nearest_nonH"] == "S2"


def test_mu_bridge_warn_triggers_off_path(tmp_path):
    # Symmetric H between anchors S0/S1, nearest is an anchor (anchor_ok True),
    # BUT a 3rd S (idx 2) sits 1.6 A from H (< 2.0 cutoff) -> mu-bridge warn -> OFF_PATH.
    p = _write_xyz(tmp_path / "saddle_mu.xyz", [
        ("S", 0.0, 5.0, 5.0),    # 0
        ("S", 2.8, 5.0, 5.0),    # 1
        ("H", 1.4, 5.0, 5.0),    # 2  (1.4 A from each anchor)
        ("S", 1.4, 6.6, 5.0),    # 3  third S, 1.6 A from H
    ])
    atoms = _clean_read(p)
    env = run_saddle_proximity_gate(atoms, s_i=0, s_k=1, h_idx=2)
    assert env["result"]["symmetric_ok"] is True
    assert env["result"]["anchor_ok"] is True
    assert env["result"]["mu_bridge_warn"] is True
    assert env["result"]["third_S_d"] < 2.0
    assert env["verdict"] == "OFF_PATH_OR_INTERMEDIATE_INVESTIGATE"
    assert any("mu-" in r or "bridge" in r for r in env["reasons"])


def test_asymmetric_is_off_path(tmp_path):
    # H much closer to S0 than S1 -> asym large -> not symmetric -> OFF_PATH.
    p = _write_xyz(tmp_path / "saddle_asym.xyz", [
        ("S", 0.0, 5.0, 5.0),    # 0
        ("S", 4.0, 5.0, 5.0),    # 1
        ("H", 0.9, 5.0, 5.0),    # 2  d_i=0.9, d_k=3.1, asym=2.2
        ("S", 12.0, 12.0, 12.0),  # 3 far
    ])
    atoms = _clean_read(p)
    env = run_saddle_proximity_gate(atoms, s_i=0, s_k=1, h_idx=2)
    assert env["result"]["symmetric_ok"] is False
    assert env["verdict"] == "OFF_PATH_OR_INTERMEDIATE_INVESTIGATE"


def test_auto_detect_multiple_h_raises(tmp_path):
    p = _write_xyz(tmp_path / "two_h.xyz", [
        ("S", 0.0, 5.0, 5.0),
        ("S", 3.0, 5.0, 5.0),
        ("H", 1.5, 5.0, 5.0),
        ("H", 8.0, 8.0, 8.0),
    ])
    atoms = _clean_read(p)
    with pytest.raises(ValueError):
        run_saddle_proximity_gate(atoms, s_i=0, s_k=1)  # ambiguous H


def test_clean_read_strips_nspins_key(tmp_path):
    # extxyz with a non-standard 'nspins=1' key should not crash _clean_read.
    p = _write_xyz(tmp_path / "with_nspins.xyz", [
        ("S", 0.0, 5.0, 5.0),
        ("S", 3.0, 5.0, 5.0),
        ("H", 1.5, 5.0, 5.0),
    ], comment="nspins=1 nkpts=8 energy=-123.456")
    atoms = _clean_read(p)  # must not raise
    assert len(atoms) == 3


def test_envelope_has_stable_keys(tmp_path):
    p = _write_xyz(tmp_path / "saddle_keys.xyz", [
        ("S", 0.0, 5.0, 5.0),
        ("S", 3.0, 5.0, 5.0),
        ("H", 1.5, 5.0, 5.0),
        ("S", 12.0, 12.0, 12.0),
    ])
    atoms = _clean_read(p)
    env = run_saddle_proximity_gate(atoms, s_i=0, s_k=1)
    assert set(env) == {
        "tool", "version", "status", "verdict", "confidence",
        "reasons", "next_actions", "artifacts", "warnings", "result",
    }
    assert env["tool"] == "saddle_proximity_gate"
    assert env["status"] == "ok"
