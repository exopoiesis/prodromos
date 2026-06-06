"""Tests for the MIC endpoint-alignment gate (roadmap section E)."""
from __future__ import annotations

from ase import Atoms

from prodromos.mic_alignment_gate import run_mic_alignment

_CELL = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]


def _atoms(scaled):
    return Atoms("Fe2", scaled_positions=scaled, cell=_CELL, pbc=True)


def test_aligned_when_no_atom_crosses_boundary():
    a = _atoms([(0.10, 0.5, 0.5), (0.50, 0.5, 0.5)])
    b = _atoms([(0.15, 0.5, 0.5), (0.50, 0.5, 0.5)])
    env = run_mic_alignment(a, b)
    assert env["verdict"] == "ALIGNED"
    assert env["result"]["n_crossing"] == 0


def test_needs_alignment_when_atom_wraps_across_cell():
    # atom 0 moves 0.95 -> 0.05 along x: a true 0.1-cell hop, but raw delta -0.9
    # crosses the boundary -> a naive interpolation routes it 9 A the wrong way.
    a = _atoms([(0.95, 0.5, 0.5), (0.50, 0.5, 0.5)])
    b = _atoms([(0.05, 0.5, 0.5), (0.50, 0.5, 0.5)])
    env = run_mic_alignment(a, b)
    assert env["verdict"] == "NEEDS_MIC_ALIGNMENT"
    assert env["result"]["n_crossing"] == 1
    cross = env["result"]["crossing_atoms"][0]
    assert cross["index"] == 0
    assert cross["cell_shift"] == [-1, 0, 0]
    assert cross["raw_displacement_A"] > 8.0  # the long way round
    assert cross["mic_displacement_A"] < 1.5  # the true minimum-image hop


def test_aligned_positions_round_trip_to_aligned_verdict():
    a = _atoms([(0.95, 0.5, 0.5), (0.50, 0.5, 0.5)])
    b = _atoms([(0.05, 0.5, 0.5), (0.50, 0.5, 0.5)])
    env = run_mic_alignment(a, b)
    aligned = Atoms("Fe2", scaled_positions=env["result"]["aligned_scaled_positions"],
                    cell=_CELL, pbc=True)
    # re-running A vs the aligned B must now be ALIGNED
    assert run_mic_alignment(a, aligned)["verdict"] == "ALIGNED"


def test_review_on_atom_count_mismatch():
    a = _atoms([(0.1, 0.5, 0.5), (0.5, 0.5, 0.5)])
    b = Atoms("Fe", scaled_positions=[(0.1, 0.5, 0.5)], cell=_CELL, pbc=True)
    assert run_mic_alignment(a, b)["verdict"] == "REVIEW"


def test_review_on_no_cell():
    a = Atoms("Fe2", positions=[(0, 0, 0), (2, 0, 0)])  # no cell
    b = Atoms("Fe2", positions=[(0.1, 0, 0), (2, 0, 0)])
    assert run_mic_alignment(a, b)["verdict"] == "REVIEW"


def test_write_aligned_to_disk(tmp_path):
    from ase.io import read

    a = _atoms([(0.95, 0.5, 0.5), (0.50, 0.5, 0.5)])
    b = _atoms([(0.05, 0.5, 0.5), (0.50, 0.5, 0.5)])
    out = tmp_path / "endB_aligned.xyz"
    env = run_mic_alignment(a, b, write_aligned=str(out))
    assert env["verdict"] == "NEEDS_MIC_ALIGNMENT"
    assert out.exists()
    # the written endpoint, read back, is minimum-image aligned to A
    assert run_mic_alignment(a, read(str(out)))["verdict"] == "ALIGNED"


def test_cli_main_smoke(tmp_path, capsys):
    from ase.io import write as ase_write

    from prodromos.mic_alignment_gate import main

    a = _atoms([(0.95, 0.5, 0.5), (0.50, 0.5, 0.5)])
    b = _atoms([(0.05, 0.5, 0.5), (0.50, 0.5, 0.5)])
    pa, pb = tmp_path / "a.xyz", tmp_path / "b.xyz"
    ase_write(str(pa), a)
    ase_write(str(pb), b)
    rc = main([str(pa), str(pb)])
    assert rc == 0
    assert "NEEDS_MIC_ALIGNMENT" in capsys.readouterr().out
