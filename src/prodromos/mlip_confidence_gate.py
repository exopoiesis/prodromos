"""mlip_confidence_gate -- predict WHEN a foundation-MLIP migration barrier is
trustworthy, and route the untrustworthy hosts to DFT (roadmap §B).

The battery-reproduction study (BATTERY_REPRO_RESULTS.md) found a foundation MLIP
(spin-blind, e.g. MACE/CHGNet) gives MAE 0.154 eV on intercalation barriers overall,
BUT is off by 0.48 eV on MgV2S4 -- with no clean "which host is safe" rule from
energetics alone. The discriminating signal is ELECTRONIC: near-degenerate /
itinerant 3d manifolds (V3+/V4+) are where the spin-blind MLIP fails, because the
true PES depends on a magnetic / orbital state the foundation model does not resolve.

This gate turns that negative finding into a positive, actionable verdict: given the
host's TM chemistry (+ optional band gap / oxidation context), it flags hosts whose
foundation-MLIP barrier should NOT be trusted and must go to DFT.

Verdicts:
    TRUST_MLIP    -- closed-shell or robust large-gap localized-moment host; the
                     spin-blind MLIP barrier is defensible.
    DFT_REQUIRED  -- near-degenerate itinerant 3d / multivalent redox host; the MLIP
                     barrier is unreliable -> route to DFT.
    REVIEW        -- insufficient electronic descriptors to decide.
"""
from __future__ import annotations

import argparse

from prodromos.cli_contract import dump_json, response_envelope
from prodromos.electron_parity_gate import (
    OPEN_SHELL_TM,
    counts_from_formula_tokens,
    counts_from_structure,
    infer_closed_shell,
)

# Early 3d/4d with near-degenerate, easily-itinerant t2g manifolds: the documented
# foundation-MLIP failure class (MgV2S4 = V). Spin/orbital state is delicate.
NEAR_DEGENERATE_TM = {"Ti", "V", "Cr", "Nb", "Mo"}

# Redox-active TM that readily take >=2 oxidation states in a cathode -> mixed
# valence / redox polaron -> the MLIP must resolve a charge/spin state it does not.
MULTIVALENT_REDOX_TM = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"}

# Small-gap threshold: below this the host is effectively itinerant.
ITINERANT_GAP_EV = 0.5
# Clear-insulator threshold: above this a localized-moment picture is robust.
INSULATOR_GAP_EV = 1.0


def run_mlip_confidence_gate(
    symbol_counts: dict[str, int],
    charge: float = 0.0,
    band_gap_eV: float | None = None,
    migrant: str | None = None,
    multivalent: bool | None = None,
) -> dict:
    """Core gate.

    Parameters
    ----------
    symbol_counts : cell composition, e.g. ``{"Mg": 1, "V": 2, "S": 4}``.
    charge : net cell charge (e) for the formal-oxidation inference.
    band_gap_eV : optional MLIP/DFT band gap; < 0.5 eV => itinerant (risk),
        >= 1.0 eV => clear insulator (localized moments robust).
    migrant : optional migrating-ion species (e.g. ``"Li"``/``"Mg"``); its presence
        signals a redox-cathode context (the host TM must change oxidation).
    multivalent : optional explicit override of the multivalent-host flag.
    """
    tms = sorted(s for s in symbol_counts if s in OPEN_SHELL_TM)
    reasons: list[str] = []
    if not tms:
        return response_envelope(
            tool="mlip_confidence_gate",
            verdict="TRUST_MLIP",
            confidence="medium",
            reasons=["no open-shell TM in the host -> no spin/orbital state for the "
                     "foundation MLIP to miss; the barrier is defensible."],
            result={"tm_species": [], "band_gap_eV": band_gap_eV},
        )

    ox = infer_closed_shell(symbol_counts, charge=charge)
    closed_shell = ox["status"] == "ok" and ox["closed_shell"]

    near_deg = sorted(set(tms) & NEAR_DEGENERATE_TM)
    redox = sorted(set(tms) & MULTIVALENT_REDOX_TM)
    is_multivalent = multivalent if multivalent is not None else bool(redox)
    itinerant = band_gap_eV is not None and band_gap_eV < ITINERANT_GAP_EV
    clear_insulator = band_gap_eV is not None and band_gap_eV >= INSULATOR_GAP_EV
    redox_context = migrant is not None

    detail = {
        "tm_species": tms,
        "near_degenerate_tm": near_deg,
        "multivalent_redox_tm": redox,
        "is_multivalent": is_multivalent,
        "band_gap_eV": band_gap_eV,
        "itinerant": itinerant,
        "clear_insulator": clear_insulator,
        "redox_context": redox_context,
        "oxidation_inference": ox,
        "closed_shell": closed_shell,
    }

    # 1. closed-shell d0/d10 -> nonmagnetic -> MLIP fine.
    if closed_shell:
        reasons.append(f"closed-shell host ({ox['reason']}) -> nonmagnetic; the spin-blind "
                       "foundation MLIP has no magnetic state to miss -> barrier trustworthy.")
        return response_envelope(
            tool="mlip_confidence_gate", verdict="TRUST_MLIP", confidence="medium",
            reasons=reasons, result=detail,
        )

    # 2. near-degenerate itinerant 3d (the documented failure class) -> DFT.
    if near_deg and (itinerant or band_gap_eV is None):
        reasons.append(
            f"near-degenerate / itinerant 3d host ({', '.join(near_deg)}) -- the documented "
            f"foundation-MLIP failure class (MgV2S4 off 0.48 eV); the barrier depends on a "
            f"spin/orbital state the spin-blind MLIP does not resolve."
        )
        return response_envelope(
            tool="mlip_confidence_gate", verdict="DFT_REQUIRED", confidence="high",
            reasons=reasons,
            next_actions=["compute the barrier with spin-polarized DFT (NEB) -- do not quote "
                          "the foundation-MLIP value for this host"],
            result=detail,
        )

    # 3. multivalent redox host in a cathode context -> redox polaron / mixed valence.
    if is_multivalent and redox_context:
        reasons.append(
            f"multivalent redox host ({', '.join(redox)}) in a migration (cathode) context: the "
            f"hop is charge-compensated by a redox polaron (Fe2+/Fe3+, V3+/V4+) -- a charge/spin "
            f"state the foundation MLIP does not resolve -> MLIP barrier unreliable."
        )
        nxt = ["compute with spin-polarized DFT; also screen the polaron sublattice with "
               "sublattice_preflight (mode='polaron')"]
        return response_envelope(
            tool="mlip_confidence_gate", verdict="DFT_REQUIRED", confidence="medium",
            reasons=reasons, next_actions=nxt, result=detail,
        )

    # 4. clear-insulator, robust localized HS moment (e.g. Fe3+ d5, Mn2+ d5) -> MLIP ok-ish.
    if clear_insulator:
        reasons.append(
            f"clear insulator (gap {band_gap_eV:g} eV) with a localized-moment TM "
            f"({', '.join(tms)}) -> the magnetic state is robust and well-separated; the "
            f"foundation-MLIP barrier is defensible (verify a single DFT endpoint if quoting)."
        )
        return response_envelope(
            tool="mlip_confidence_gate", verdict="TRUST_MLIP", confidence="low",
            reasons=reasons,
            next_actions=["optionally confirm one endpoint with DFT before quoting the barrier"],
            result=detail,
        )

    # 5. undecided.
    reasons.append(
        f"open-shell TM ({', '.join(tms)}) but no band gap / oxidation context to classify "
        f"itinerant-vs-localized -> cannot certify the foundation-MLIP barrier."
    )
    return response_envelope(
        tool="mlip_confidence_gate", verdict="REVIEW", confidence="low",
        reasons=reasons,
        next_actions=["supply band_gap_eV (and migrant if a cathode hop) to classify, "
                      "or default to DFT for the barrier"],
        result=detail,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="MLIP-confidence gate: is a foundation-MLIP migration barrier trustworthy?"
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--structure", help="xyz/cif/traj of the host cell")
    src.add_argument("--symbols", nargs="+", help="explicit counts, e.g. --symbols Mg1 V2 S4")
    p.add_argument("--charge", type=float, default=0.0, help="net cell charge in e")
    p.add_argument("--band-gap", type=float, default=None, help="band gap in eV (<0.5 itinerant)")
    p.add_argument("--migrant", default=None, help="migrating ion (signals a cathode redox context)")
    p.add_argument("--multivalent", action="store_true", default=None,
                   help="force the multivalent-host flag")
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args(argv)

    counts = (
        counts_from_structure(args.structure)
        if args.structure
        else counts_from_formula_tokens(args.symbols)
    )
    env = run_mlip_confidence_gate(
        counts,
        charge=args.charge,
        band_gap_eV=args.band_gap,
        migrant=args.migrant,
        multivalent=args.multivalent,
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
