"""mic_alignment_gate -- first-class pre-flight path-sanity gate for NEB endpoints
(roadmap section E).

Before an NEB interpolates between endpoints A and B, the two structures must be in
the SAME periodic image: if any atom's fractional displacement A->B exceeds half a
cell along some axis, a naive linear interpolation routes that atom the long way
*across the cell* (a PBC crossing) instead of the short minimum-image hop. The path
is then fictitiously long and the barrier meaningless. This bit a beta-Li3PS4 case
as a driver-local fix; here it is promoted to a standalone gate that BOTH diagnoses
the crossing AND returns a minimum-image-aligned endpoint B.

Verdicts: ALIGNED / NEEDS_MIC_ALIGNMENT / REVIEW (atom mismatch / no cell).

The aligned B' places every atom in the image nearest to its A counterpart:
    s_B' = s_A + wrap_to_[-0.5,0.5)(s_B - s_A)
so a straight interpolation A->B' is the true minimum-image path.
"""
from __future__ import annotations

import argparse

from prodromos.cli_contract import dump_json, response_envelope


def _mic_report(scaled_a, scaled_b, cell, symbols, cross_threshold: float = 0.5):
    """Per-atom raw vs minimum-image displacement. Pure (numpy)."""
    import numpy as np

    sa = np.asarray(scaled_a, dtype=float)
    sb = np.asarray(scaled_b, dtype=float)
    cell = np.asarray(cell, dtype=float)
    raw = sb - sa
    shift = np.round(raw)  # integer cell jumps to undo
    mic = raw - shift
    crossing_mask = np.any(np.abs(shift) >= 1, axis=1)
    raw_cart = raw @ cell
    mic_cart = mic @ cell
    raw_disp = np.linalg.norm(raw_cart, axis=1)
    mic_disp = np.linalg.norm(mic_cart, axis=1)

    crossing = []
    for i in np.nonzero(crossing_mask)[0]:
        crossing.append({
            "index": int(i),
            "element": symbols[i] if symbols is not None and i < len(symbols) else None,
            "cell_shift": [int(x) for x in shift[i]],
            "raw_displacement_A": round(float(raw_disp[i]), 3),
            "mic_displacement_A": round(float(mic_disp[i]), 3),
        })
    aligned_scaled = sa + mic  # B in the image nearest to A
    return {
        "n_atoms": int(sa.shape[0]),
        "n_crossing": int(crossing_mask.sum()),
        "crossing_atoms": crossing,
        "max_raw_displacement_A": round(float(raw_disp.max()), 3) if raw_disp.size else 0.0,
        "max_mic_displacement_A": round(float(mic_disp.max()), 3) if mic_disp.size else 0.0,
        "aligned_scaled_positions": aligned_scaled.tolist(),
    }


def run_mic_alignment(atoms_a, atoms_b, cross_threshold: float = 0.5, write_aligned: str | None = None) -> dict:
    """Core gate over two ASE ``Atoms`` endpoints (same atom ordering).

    Returns a response envelope. On ``NEEDS_MIC_ALIGNMENT`` the result carries the
    minimum-image-aligned endpoint-B scaled positions; pass ``write_aligned`` (a path)
    to also write the aligned B to disk (format inferred by ASE from the extension).
    """
    sym_a = list(atoms_a.get_chemical_symbols())
    sym_b = list(atoms_b.get_chemical_symbols())
    if len(sym_a) != len(sym_b):
        return response_envelope(
            tool="mic_alignment_gate", verdict="REVIEW", confidence="low",
            reasons=[f"endpoint atom counts differ ({len(sym_a)} vs {len(sym_b)}) -- "
                     "endpoints must share atom ordering for an NEB"],
            result={"n_atoms_a": len(sym_a), "n_atoms_b": len(sym_b)},
        )
    if sym_a != sym_b:
        n_mismatch = sum(1 for a, b in zip(sym_a, sym_b) if a != b)
        return response_envelope(
            tool="mic_alignment_gate", verdict="REVIEW", confidence="low",
            reasons=[f"endpoint atom ORDERING differs ({n_mismatch} positions) -- "
                     "reorder B to match A before an NEB (this gate assumes shared ordering)"],
            result={"n_symbol_mismatches": n_mismatch},
        )
    import numpy as np

    cell = np.asarray(atoms_a.get_cell(), dtype=float)
    if not bool(np.any(cell)) or abs(np.linalg.det(cell)) < 1e-8:
        return response_envelope(
            tool="mic_alignment_gate", verdict="REVIEW", confidence="low",
            reasons=["endpoint A has no (or degenerate) cell -- the minimum-image "
                     "convention is undefined; supply a periodic cell"],
            result={"cell": cell.tolist()},
        )

    rep = _mic_report(
        atoms_a.get_scaled_positions(wrap=False),
        atoms_b.get_scaled_positions(wrap=False),
        cell,
        sym_a,
        cross_threshold=cross_threshold,
    )

    if rep["n_crossing"] == 0:
        return response_envelope(
            tool="mic_alignment_gate", verdict="ALIGNED", confidence="high",
            reasons=[f"all {rep['n_atoms']} atoms move by less than half a cell along every "
                     f"axis (max displacement {rep['max_mic_displacement_A']} A) -> endpoints "
                     "are already minimum-image aligned; a straight NEB interpolation is sane"],
            result=rep,
        )

    if write_aligned:
        from ase import Atoms
        from ase.io import write as ase_write

        aligned = Atoms(symbols=sym_a, scaled_positions=rep["aligned_scaled_positions"],
                        cell=cell, pbc=atoms_a.get_pbc())
        ase_write(write_aligned, aligned)
        rep["aligned_written_to"] = write_aligned

    worst = max(rep["crossing_atoms"], key=lambda c: c["raw_displacement_A"])
    return response_envelope(
        tool="mic_alignment_gate", verdict="NEEDS_MIC_ALIGNMENT", confidence="high",
        reasons=[
            f"{rep['n_crossing']}/{rep['n_atoms']} atom(s) cross a periodic boundary between "
            f"endpoints (worst: atom {worst['index']} {worst['element']} raw "
            f"{worst['raw_displacement_A']} A vs minimum-image {worst['mic_displacement_A']} A, "
            f"cell shift {worst['cell_shift']}) -> a naive NEB interpolation routes them the long "
            f"way across the cell and the barrier is meaningless",
        ],
        next_actions=[
            "align endpoint B to A's image before the NEB: s_B' = s_A + wrap(s_B - s_A) "
            "(result.aligned_scaled_positions); re-run the gate -> ALIGNED",
            "or pass write_aligned=<path> to have this gate write the aligned endpoint B",
        ],
        result=rep,
    )


def _read(path: str):
    from ase.io import read
    return read(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="MIC endpoint-alignment gate: are NEB endpoints in the same periodic image?"
    )
    p.add_argument("endpoint_a", help="endpoint A structure (any ASE-readable format)")
    p.add_argument("endpoint_b", help="endpoint B structure")
    p.add_argument("--write-aligned", default=None,
                   help="write the minimum-image-aligned endpoint B to this path")
    p.add_argument("--cross-threshold", type=float, default=0.5)
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    env = run_mic_alignment(
        _read(args.endpoint_a), _read(args.endpoint_b),
        cross_threshold=args.cross_threshold, write_aligned=args.write_aligned,
    )
    if args.output:
        dump_json(env, args.output)
    if args.json:
        dump_json(env)
    elif not args.output:
        print(f"verdict\t{env['verdict']}\tconfidence\t{env['confidence']}")
        for r in env["reasons"]:
            print(f"reason\t{r}")
        for a in env["next_actions"]:
            print(f"next\t{a}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
