"""N-15 -- H-transfer barrier paper-readiness gate.

Answers the question: "Is this H-transfer barrier publication-grade, or merely an
electronic-energy estimate that lacks the frequency/ZPE corrections required by
any quantitative H-transfer discussion?"

Verdict logic
-------------
PAPER_GRADE  <=>  ALL of the following:
    1. has_dft_freq is True
    2. n_imag_modes == 1            (exactly one imaginary mode at the saddle)
    3. imag_mode_H_fraction >= h_fraction_threshold
                                    (the imaginary mode is H-dominated, confirming
                                     the mode belongs to the transfer coordinate)
    4. dZPE_eV is not None          (zero-point energy correction has been computed)

ELECTRONIC_ONLY  if any condition fails; reasons list what is missing.

ZPE note
--------
For H-transfer reactions, the ZPE correction to the classical barrier is typically
negative: ΔZPE‡ ≈ −100 to −150 meV (the tight H-stretching modes at the reactant
are lost at the loose transition state, lowering the effective barrier by ~20-50 %).
When dZPE_eV is provided, barrier_with_zpe = barrier_eV + dZPE_eV is reported,
and the sign is annotated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from prodromos.cli_contract import response_envelope, dump_json

# Typical ZPE correction range for H-transfer in Fe-S systems [eV]
_ZPE_TYPICAL_LOW = -0.15   # most negative (largest lowering)
_ZPE_TYPICAL_HIGH = -0.10  # least negative

DEFAULT_H_FRACTION_THRESHOLD = 0.5


def run_h_barrier_paper_readiness(
    barrier_eV: float,
    has_dft_freq: bool,
    n_imag_modes: int | None = None,
    imag_mode_H_fraction: float | None = None,
    dZPE_eV: float | None = None,
    h_fraction_threshold: float = DEFAULT_H_FRACTION_THRESHOLD,
) -> dict:
    """N-15 paper-readiness gate for H-transfer barriers (MCP-callable entry point).

    Parameters
    ----------
    barrier_eV:
        Classical electronic energy barrier in eV (positive = endothermic saddle).
    has_dft_freq:
        Whether a DFT frequency calculation at the saddle has been performed.
    n_imag_modes:
        Number of imaginary vibrational modes found at the saddle.  Should be 1
        for a genuine first-order saddle point.
    imag_mode_H_fraction:
        Fraction of the imaginary mode's kinetic energy carried by H atoms
        (0-1).  Values >= h_fraction_threshold confirm the mode lies along the
        H-transfer coordinate.
    dZPE_eV:
        ZPE correction to the barrier: ΔZPE‡ = ZPE(TS) - ZPE(reactant) in eV.
        Typically −0.10 to −0.15 eV for H-transfer.  When provided,
        barrier_with_zpe = barrier_eV + dZPE_eV is computed.
    h_fraction_threshold:
        Minimum H kinetic-energy fraction for the imaginary mode to be considered
        H-dominated (default 0.5).

    Returns
    -------
    response_envelope dict.
    """
    if barrier_eV < 0:
        # Negative barrier is physically unusual but not impossible (downhill reaction
        # with a shallow saddle); allow it but warn.
        pass

    reasons: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    # ----- evaluate each criterion -----

    crit_freq = has_dft_freq
    crit_single_imag = (n_imag_modes is not None) and (n_imag_modes == 1)
    crit_h_dom = (
        (imag_mode_H_fraction is not None)
        and (imag_mode_H_fraction >= h_fraction_threshold)
    )
    crit_zpe = dZPE_eV is not None

    paper_grade = crit_freq and crit_single_imag and crit_h_dom and crit_zpe

    # ----- detailed reasons -----

    if crit_freq:
        reasons.append("DFT frequency calculation at the saddle is present.")
    else:
        missing.append("no DFT frequency calculation (has_dft_freq=False)")
        reasons.append(
            "Missing: DFT frequency calculation at the saddle.  "
            "Without frequencies the saddle cannot be confirmed as a first-order "
            "transition state, and ZPE corrections are unavailable."
        )

    if n_imag_modes is None:
        missing.append("n_imag_modes not provided")
        reasons.append(
            "Missing: number of imaginary modes not provided.  "
            "A genuine H-transfer TS must have exactly 1 imaginary mode."
        )
    elif n_imag_modes == 1:
        reasons.append(
            f"n_imag_modes={n_imag_modes}: exactly one imaginary mode confirming "
            "first-order saddle character."
        )
    else:
        missing.append(f"n_imag_modes={n_imag_modes} (expected 1)")
        reasons.append(
            f"n_imag_modes={n_imag_modes}: expected exactly 1 for a clean H-transfer TS.  "
            "Multiple imaginary modes indicate a higher-order saddle (step back along "
            "the extra soft modes) or an incomplete geometry optimisation."
        )

    if imag_mode_H_fraction is None:
        missing.append("imag_mode_H_fraction not provided")
        reasons.append(
            "Missing: H kinetic-energy fraction of the imaginary mode not provided.  "
            "This is needed to confirm the mode lies along the H-transfer coordinate "
            "rather than a lattice distortion or S-S breathing mode."
        )
    elif imag_mode_H_fraction >= h_fraction_threshold:
        reasons.append(
            f"imag_mode_H_fraction={imag_mode_H_fraction:.2f} >= threshold "
            f"{h_fraction_threshold:.2f}: imaginary mode is H-dominated, "
            "consistent with the H-transfer coordinate."
        )
    else:
        missing.append(
            f"imag_mode_H_fraction={imag_mode_H_fraction:.2f} < threshold "
            f"{h_fraction_threshold:.2f}"
        )
        reasons.append(
            f"imag_mode_H_fraction={imag_mode_H_fraction:.2f} is below threshold "
            f"{h_fraction_threshold:.2f}.  "
            "The imaginary mode is not H-dominated; it may describe a lattice "
            "distortion rather than the H-transfer coordinate.  "
            "Re-examine the mode eigenvector and choose the correct saddle."
        )

    if dZPE_eV is None:
        missing.append("dZPE_eV not provided")
        reasons.append(
            "Missing: ZPE correction (dZPE_eV).  "
            "For H-transfer barriers, ΔZPE‡ is typically −100 to −150 meV, "
            "lowering the effective barrier by 20-50 %.  "
            "Without it the reported barrier is purely classical."
        )
    else:
        lowering_meV = dZPE_eV * 1000
        if _ZPE_TYPICAL_LOW <= dZPE_eV <= _ZPE_TYPICAL_HIGH:
            zpe_note = (
                f"dZPE={lowering_meV:+.1f} meV — within the typical H-transfer range "
                f"({_ZPE_TYPICAL_LOW*1000:.0f} to {_ZPE_TYPICAL_HIGH*1000:.0f} meV). "
                f"ZPE lowers the classical barrier by {abs(dZPE_eV)*1000:.1f} meV "
                f"({abs(dZPE_eV)/barrier_eV*100:.1f} %)."
                if barrier_eV != 0 else
                f"dZPE={lowering_meV:+.1f} meV — within typical H-transfer range."
            )
        elif dZPE_eV < _ZPE_TYPICAL_LOW:
            zpe_note = (
                f"dZPE={lowering_meV:+.1f} meV — larger correction than the typical "
                f"H-transfer range ({_ZPE_TYPICAL_LOW*1000:.0f} to "
                f"{_ZPE_TYPICAL_HIGH*1000:.0f} meV); verify the mode assignment."
            )
        else:
            zpe_note = (
                f"dZPE={lowering_meV:+.1f} meV — smaller (or positive) ZPE correction "
                f"than typical H-transfer range; verify the mode assignment."
            )
            if dZPE_eV > 0:
                warnings.append(
                    f"dZPE_eV={dZPE_eV:.4f} is positive; H-transfer usually lowers the "
                    "barrier via ZPE.  Double-check the TS vs reactant ZPE assignment."
                )
        reasons.append(f"ZPE correction provided: {zpe_note}")

    # ----- build result dict -----

    barrier_with_zpe: float | None = None
    zpe_effect_note: str | None = None
    if dZPE_eV is not None:
        barrier_with_zpe = barrier_eV + dZPE_eV
        # Use the detailed zpe_note already built above (includes typical-range context).
        zpe_effect_note = zpe_note  # type: ignore[possibly-undefined]

    if barrier_eV < 0:
        warnings.append(
            f"barrier_eV={barrier_eV:.4f} is negative (downhill reaction).  "
            "Confirm the reaction direction and endpoint assignment."
        )

    # ----- confidence -----

    n_criteria_met = sum([crit_freq, crit_single_imag, crit_h_dom, crit_zpe])
    if paper_grade:
        confidence = "high"
    elif n_criteria_met >= 2:
        confidence = "medium"
    else:
        confidence = "low"

    verdict = "PAPER_GRADE" if paper_grade else "ELECTRONIC_ONLY"

    if paper_grade:
        reasons.append(
            "All four paper-grade criteria satisfied: "
            "DFT frequency, single imaginary mode, H-dominated mode, ZPE correction present."
        )
        next_actions = [
            "report barrier_with_zpe as the primary result in the manuscript",
            "include the classical barrier as a supplementary comparison",
            "cite the frequency calculation details (code, cutoff, k-mesh) in the SI",
        ]
    else:
        reasons.append(
            f"Paper-grade criteria NOT fully satisfied ({n_criteria_met}/4). "
            "This barrier is an electronic-energy-only estimate."
        )
        next_actions = [
            f"address missing criteria: {'; '.join(missing)}",
            "run a DFT frequency calculation at the located saddle geometry",
            "verify the imaginary mode is H-dominated (project eigenvector onto H atoms)",
            "compute ΔZPE‡ = ZPE(TS) - ZPE(reactant) for the full ZPE-corrected barrier",
        ]

    result = {
        "barrier_eV": barrier_eV,
        "has_dft_freq": has_dft_freq,
        "n_imag_modes": n_imag_modes,
        "imag_mode_H_fraction": imag_mode_H_fraction,
        "h_fraction_threshold": h_fraction_threshold,
        "dZPE_eV": dZPE_eV,
        "barrier_with_zpe": barrier_with_zpe,
        "zpe_effect_note": zpe_effect_note,
        "criteria_met": n_criteria_met,
        "criteria_total": 4,
        "missing": missing,
    }

    return response_envelope(
        tool="h_barrier_paper_readiness",
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        next_actions=next_actions,
        warnings=warnings,
        result=result,
    )


def print_gate(env: dict) -> None:
    r = env.get("result") or {}
    print(f"verdict\t{env['verdict']}\tconfidence\t{env['confidence']}")
    print(
        f"barrier_eV\t{r.get('barrier_eV')}\t"
        f"barrier_with_zpe\t{r.get('barrier_with_zpe')}\t"
        f"criteria\t{r.get('criteria_met')}/{r.get('criteria_total')}"
    )
    if r.get("zpe_effect_note"):
        print(f"zpe_note\t{r.get('zpe_effect_note')}")
    if r.get("missing"):
        print(f"missing\t{'; '.join(r.get('missing', []))}")
    for x in env["reasons"]:
        print(f"reason\t{x}")
    for x in env["next_actions"]:
        print(f"next\t{x}")
    for x in env["warnings"]:
        print(f"warning\t{x}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "N-15 H-transfer barrier paper-readiness gate: checks whether a "
            "reported H-transfer barrier meets the criteria for publication "
            "(DFT freq, single imaginary mode, H-dominated mode, ZPE correction)."
        )
    )
    p.add_argument(
        "--barrier-ev", type=float, required=True,
        help="classical electronic energy barrier in eV"
    )
    p.add_argument(
        "--has-dft-freq", action="store_true", default=False,
        help="flag: a DFT frequency calculation at the saddle has been performed"
    )
    p.add_argument(
        "--n-imag-modes", type=int, default=None,
        help="number of imaginary vibrational modes at the saddle (should be 1)"
    )
    p.add_argument(
        "--imag-mode-h-fraction", type=float, default=None,
        help="fraction of imaginary mode kinetic energy on H atoms (0-1)"
    )
    p.add_argument(
        "--dzpe-ev", type=float, default=None,
        help="ZPE correction DELTA_ZPE = ZPE(TS) - ZPE(reactant) in eV; typically -0.10 to -0.15"
    )
    p.add_argument(
        "--h-fraction-threshold", type=float, default=DEFAULT_H_FRACTION_THRESHOLD,
        help=f"minimum H fraction for the imaginary mode (default {DEFAULT_H_FRACTION_THRESHOLD})"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    env = run_h_barrier_paper_readiness(
        barrier_eV=args.barrier_ev,
        has_dft_freq=args.has_dft_freq,
        n_imag_modes=args.n_imag_modes,
        imag_mode_H_fraction=args.imag_mode_h_fraction,
        dZPE_eV=args.dzpe_ev,
        h_fraction_threshold=args.h_fraction_threshold,
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
