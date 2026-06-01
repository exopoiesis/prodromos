"""N-11 -- spin-collapse verdict: end the nspin=1 <-> nspin=2 flip-flop for Fe-S V_Fe+H.

M0 (electron_parity_gate) answers the *a-priori* parity question: an odd-electron
cell (e.g. pyrite Fe31S64H1) cannot run nspin=1 with FIXED occupations, so it
emits NSPIN2_MANDATORY with a metallic-smearing caveat. But parity ALONE does not
decide the operational question that actually drives the deploy choice:

    "Does the local TM moment COLLAPSE (so nspin=1 == nspin=2, smooth PES) or
     PERSIST (so the nspin=1 PES is pathological and nspin=2 is mandatory)?"

This is exactly the flip-flop seen across campaigns: pyrite/mack collapse
(nspin=1 turned out fine), pentlandite persists (nspin=1 was wrong all along).
The cheapest *decisive* experiment is ONE nspin=2 single-point: read the resulting
absolute magnetization per TM atom and route on it.

Criterion (REVISED-1xx, Fe-S V_Fe+H family):
    mabs_per_tm = M_abs / n_tm  [uB / transition-metal atom]
      * mabs_per_tm <  threshold (default 0.30 uB/TM)  -> COLLAPSED
            -> NSPIN1_OK: the seeded moment died, nspin=1 == nspin=2, smooth PES.
               (mack, pyrite V_Fe+H behaviour.)
      * mabs_per_tm >= threshold                       -> PERSISTS
            -> NSPIN2_REQUIRED: a localized moment survives, the nspin=1 PES is
               pathological, must run spin-polarised production.
               (pentlandite behaviour, ~1.8 uB/TM.)

The 0.30 uB/TM threshold sits well below any genuine Fe-S local moment (~1.5-3.5
uB/Fe) and well above smearing/numerical residue (<~0.1 uB/TM), so the verdict is
robust to SCF noise.

Verdicts: NSPIN1_OK / NSPIN2_REQUIRED.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from prodromos.cli_contract import response_envelope, dump_json

DEFAULT_THRESHOLD = 0.30  # uB per transition-metal atom
CRITERION_REF = (
    "Fe-S V_Fe+H spin-collapse criterion (N-11): mabs_per_tm < 0.30 uB/TM -> "
    "moment collapses (nspin=1 == nspin=2); >= 0.30 -> moment persists (nspin=2 "
    "mandatory). Calibrated on mack/pyrite (collapsed) vs pentlandite (~1.8 uB/TM)."
)


def run_spin_collapse_verdict(
    mabs: float | None = None,
    n_tm: int | None = None,
    mabs_per_tm: float | None = None,
    mtot: float | None = None,
    parity: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Core routing gate. Ingests ONE cheap nspin=2 single-point result.

    Provide EITHER (`mabs` absolute magnetization + `n_tm` TM-atom count) OR a
    pre-computed `mabs_per_tm`. `mtot`/`parity` are optional context (echoed +
    used only for sanity warnings). Returns a response_envelope dict.
    """
    reasons: list[str] = []
    warnings: list[str] = []

    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    # Resolve mabs_per_tm from whichever inputs were given.
    if mabs_per_tm is None:
        if mabs is None or n_tm is None:
            raise ValueError(
                "provide either --mabs-per-tm, or both --mabs and --n-tm"
            )
        if n_tm <= 0:
            raise ValueError(f"--n-tm must be a positive integer, got {n_tm}")
        if mabs < 0:
            # absolute magnetization is non-negative by definition; tolerate sign noise
            warnings.append(f"--mabs is negative ({mabs}); using |mabs|")
            mabs = abs(mabs)
        mabs_per_tm = mabs / n_tm
    else:
        if mabs_per_tm < 0:
            warnings.append(f"--mabs-per-tm is negative ({mabs_per_tm}); using |value|")
            mabs_per_tm = abs(mabs_per_tm)
        if mabs is not None and n_tm and n_tm > 0:
            derived = mabs / n_tm
            if abs(derived - mabs_per_tm) > 1e-6:
                warnings.append(
                    f"--mabs-per-tm ({mabs_per_tm}) disagrees with --mabs/--n-tm "
                    f"({derived:.6f}); using the explicit --mabs-per-tm"
                )

    if parity is not None:
        parity = parity.lower()
        if parity not in ("odd", "even"):
            warnings.append(f"unrecognised --parity '{parity}' (expected odd/even); ignored")
            parity = None

    collapsed = mabs_per_tm < threshold

    if collapsed:
        verdict = "NSPIN1_OK"
        confidence = "high"
        reasons.append(
            f"mabs_per_tm={mabs_per_tm:.4f} uB/TM < threshold={threshold:.2f} -> "
            f"local moment COLLAPSED. nspin=1 is equivalent to nspin=2 (smooth PES); "
            f"matches mack/pyrite V_Fe+H behaviour."
        )
        reasons.append(CRITERION_REF)
        next_actions = ["nspin=1 production OK; nspin=2 control optional"]
        if parity == "odd":
            warnings.append(
                "parity is ODD: nspin=1 is only defensible with metallic smearing "
                "(occupations='smearing'); do NOT use fixed occupations."
            )
    else:
        verdict = "NSPIN2_REQUIRED"
        confidence = "high"
        reasons.append(
            f"mabs_per_tm={mabs_per_tm:.4f} uB/TM >= threshold={threshold:.2f} -> "
            f"local moment PERSISTS. The nspin=1 PES is pathological; spin-polarised "
            f"production is mandatory. Matches pentlandite behaviour (~1.8 uB/TM)."
        )
        reasons.append(CRITERION_REF)
        next_actions = [
            "nspin=2 production + per-atom starting_magnetization from relaxed AFM",
            "run magnetic_endpoint_gate on nspin=2-relaxed endA/endB (single-sheet check)",
            "band/dimer must stay on one sheet",
        ]

    result = {
        "mabs": mabs,
        "n_tm": n_tm,
        "mabs_per_tm": mabs_per_tm,
        "threshold": threshold,
        "collapsed": bool(collapsed),
        "mtot": mtot,
        "parity": parity,
    }
    return response_envelope(
        tool="spin_collapse_verdict",
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        next_actions=next_actions,
        warnings=warnings,
        result=result,
    )


def print_gate(env: dict) -> None:
    r = env["result"] or {}
    print(f"verdict\t{env['verdict']}\tconfidence\t{env['confidence']}")
    print(f"mabs_per_tm\t{r.get('mabs_per_tm'):.4f}\tthreshold\t{r.get('threshold')}\tcollapsed\t{r.get('collapsed')}")
    if r.get("mabs") is not None:
        print(f"mabs\t{r.get('mabs')}\tn_tm\t{r.get('n_tm')}")
    for x in env["reasons"]:
        print(f"reason\t{x}")
    for x in env["next_actions"]:
        print(f"next\t{x}")
    for x in env["warnings"]:
        print(f"warning\t{x}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="N-11 spin-collapse verdict: route nspin=1 vs nspin=2 from ONE cheap "
                    "nspin=2 single-point (Fe-S V_Fe+H)")
    p.add_argument("--mabs", type=float, default=None,
                   help="absolute magnetization (Bohr mag/cell) from the nspin=2 single-point")
    p.add_argument("--n-tm", type=int, default=None,
                   help="number of transition-metal atoms (Fe/Ni/Co/Mn) in the cell")
    p.add_argument("--mabs-per-tm", type=float, default=None,
                   help="pre-computed absolute magnetization per TM atom (uB/TM); "
                        "alternative to --mabs/--n-tm")
    p.add_argument("--mtot", type=float, default=None, help="optional total magnetization (context)")
    p.add_argument("--parity", choices=["odd", "even"], default=None,
                   help="optional electron parity (from electron_parity_gate) for a smearing caveat")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                   help=f"collapse threshold in uB/TM (default {DEFAULT_THRESHOLD})")
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    env = run_spin_collapse_verdict(
        mabs=args.mabs, n_tm=args.n_tm, mabs_per_tm=args.mabs_per_tm,
        mtot=args.mtot, parity=args.parity, threshold=args.threshold,
    )
    if args.output:
        dump_json(env, args.output)
    if args.json:
        dump_json(env)
    elif not args.output:
        print_gate(env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
