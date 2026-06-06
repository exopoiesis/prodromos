"""M0 -- a-priori electron-parity / spin-requirement gate (structure-only, $0).

The cheapest gate in the pipeline. Runs BEFORE any DFT, on the bare cell, to
predict whether the calculation REQUIRES nspin=2. This closes the gap that all
the other magnetic gates (magnetic_output_parser / endpoint_gate / band_gate)
share: they only work on ALREADY-COMPUTED DFT outputs. M0 needs only the atoms.

Physics (rigorous):
    Total valence electrons  N_e = sum(Z_val) - charge.
    Total spin S obeys       2S == N_e (mod 2).
      * N_e ODD  -> S is half-integer -> total magnetization is an ODD integer
                    (>=1 uB). nspin=1 (which forces S=0) is MATHEMATICALLY
                    IMPOSSIBLE -> QE smears a half-electron at E_F -> flat/noisy
                    PES (a documented pyrite "frozen energy" failure mode).
      * N_e EVEN -> S integer. nspin=1 is *allowed by parity*, but an
                    open-shell TM (Fe/Co/Ni/Mn/...) can still order AFM/FM, so
                    nspin=2 is RECOMMENDED and must be verified by the magnetic
                    output gates.

CRITICAL usage note: run this on the ACTUAL DFT cell, i.e. the H-containing
ENDPOINT (Fe31S64H1), NOT the pristine host (Fe32S64). A metal vacancy keeps
parity (even Z_val); an adsorbed H FLIPS it. Pyrite: pristine even -> nspin=1
ok; endpoint odd -> nspin=2 mandatory.

Verdicts: NSPIN2_MANDATORY / NSPIN2_RECOMMENDED / NSPIN1_OK / REVIEW (unknown species).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from prodromos.cli_contract import response_envelope, dump_json

# PseudoDojo ONCV PBE standard valences (z_valence). Matches /opt/pp/oncv_pbe.
# Parity-critical: most are even, but semicore Mn=15, Co=17, Cu=19 are ODD.
DEFAULT_VALENCE = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "Ne": 8,
    "Na": 9, "Mg": 10, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7, "Ar": 8,
    "K": 9, "Ca": 10, "Sc": 11, "Ti": 12, "V": 13, "Cr": 14, "Mn": 15,
    "Fe": 16, "Co": 17, "Ni": 18, "Cu": 19, "Zn": 20,
    "Ga": 13, "Ge": 14, "As": 15, "Se": 16, "Br": 17,
    "Mo": 14, "Ru": 16, "Rh": 17, "Pd": 18, "Ag": 19, "Cd": 20,
    "Sn": 14, "Sb": 15, "Te": 16, "I": 17,
}

# Open-shell 3d/4d transition metals: even-parity cells can still be AFM/FM.
OPEN_SHELL_TM = {
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
}

# --- A-priori closed-shell electronic descriptor (formal-oxidation d-count) ----
# Group number (= neutral-atom (n-1)d + ns valence count) for d-block elements.
# d_count(ion) = group - oxidation_state; d0 and d10 are diamagnetic closed shells
# *independent of crystal-field / spin-state* (unlike LS d6), so they can be
# rejected as non-magnetic without DFT. This is the cheap descriptor §A asks for:
# it lets M0 reject closed-shell d0 (Ti4+, V5+) / d10 (Zn2+, Cu+, Ag+) TM cells
# that composition-parity alone over-conservatively flags NSPIN2_RECOMMENDED.
D_BLOCK_GROUP = {
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Cu": 11, "Zn": 12,
    "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10,
    "Ag": 11, "Cd": 12,
}

# Formal anion oxidation states (simple ionic assumption -- polychalcogenide /
# persulfide S-S bonds (e.g. pyrite S2^2-) break this, which yields a non-integer
# or wrong TM oxidation and therefore NO override: safe by construction).
ANION_OX = {
    "O": -2, "S": -2, "Se": -2, "Te": -2,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
    "N": -3, "P": -3, "As": -3,
    "H": 1,  # protonation context (S-H): H is +1. Hydrides are rare here.
}

# Main-group cations with a single common oxidation state (spectators). Multivalent
# p-block (Sn, Pb, Sb, Bi, Tl) are deliberately omitted -> inference abstains.
FIXED_CATION_OX = {
    "Li": 1, "Na": 1, "K": 1, "Rb": 1, "Cs": 1,
    "Be": 2, "Mg": 2, "Ca": 2, "Sr": 2, "Ba": 2,
    "Al": 3, "Ga": 3, "In": 3, "Sc": 3, "Y": 3,
}


def infer_closed_shell(
    symbol_counts: dict[str, int], charge: float = 0.0
) -> dict:
    """Best-effort formal-oxidation d-count for the d-block species.

    Solves the single remaining d-block species' oxidation state from charge
    balance (anions at ``ANION_OX``, fixed main-group cations at
    ``FIXED_CATION_OX``, cell charge ``charge``). Returns::

        {"status": "ok"|"ambiguous", "species": <sym>|None, "oxidation": int|None,
         "d_count": int|None, "closed_shell": bool, "reason": str}

    ``closed_shell`` is True only for an unambiguous integer d0 or d10 -- the
    diamagnetic, spin-state-independent shells. Abstains (ambiguous, closed=False)
    when there is not exactly one d-block species to solve, when a non-d-block
    species is unaccounted (unknown oxidation), or when the balance is non-integer
    / out of [0, group] range.
    """
    d_species = [s for s in symbol_counts if s in D_BLOCK_GROUP]
    if len(d_species) != 1:
        return {"status": "ambiguous", "species": None, "oxidation": None,
                "d_count": None, "closed_shell": False,
                "reason": f"need exactly 1 d-block species to solve, found {len(d_species)}"}
    tm = d_species[0]
    fixed_sum = 0.0
    for s, n in symbol_counts.items():
        if s == tm:
            continue
        if s in ANION_OX:
            fixed_sum += ANION_OX[s] * n
        elif s in FIXED_CATION_OX:
            fixed_sum += FIXED_CATION_OX[s] * n
        else:
            return {"status": "ambiguous", "species": tm, "oxidation": None,
                    "d_count": None, "closed_shell": False,
                    "reason": f"unaccounted species {s!r} (no fixed oxidation state)"}
    n_tm = symbol_counts[tm]
    ox = (charge - fixed_sum) / n_tm
    if abs(ox - round(ox)) > 1e-9:
        return {"status": "ambiguous", "species": tm, "oxidation": None,
                "d_count": None, "closed_shell": False,
                "reason": f"non-integer formal oxidation {ox:.3f} for {tm} (S-S / mixed valence?)"}
    ox_i = int(round(ox))
    d_count = D_BLOCK_GROUP[tm] - ox_i
    if d_count < 0 or d_count > 10:
        return {"status": "ambiguous", "species": tm, "oxidation": ox_i,
                "d_count": d_count, "closed_shell": False,
                "reason": f"{tm} d-count {d_count} out of [0,10] -- implausible balance"}
    closed = d_count in (0, 10)
    return {"status": "ok", "species": tm, "oxidation": ox_i, "d_count": d_count,
            "closed_shell": closed,
            "reason": f"{tm}^{ox_i:+d} -> d{d_count}" + (" (closed shell)" if closed else "")}


def run_electron_parity_gate(
    symbol_counts: dict[str, int],
    charge: float = 0.0,
    valence_overrides: dict[str, int] | None = None,
    metallic: bool = False,
    smearing: str | None = None,
) -> dict:
    """Core gate. `symbol_counts` e.g. {'Fe': 31, 'S': 64, 'H': 1}.

    `charge` = net cell charge in units of e (positive = electrons removed).
    `metallic` = True if the system is treated with metallic smearing
    (occupations='smearing' in QE). When True and verdict is NSPIN2_MANDATORY,
    an additional collapse-test spec is emitted in next_actions.
    `smearing` = smearing type string (e.g. 'gaussian', 'cold', 'mv'); if
    provided it is treated as equivalent to metallic=True.
    Returns a response_envelope dict.
    """
    valence = dict(DEFAULT_VALENCE)
    if valence_overrides:
        valence.update(valence_overrides)

    unknown = sorted(s for s in symbol_counts if s not in valence)
    reasons: list[str] = []
    warnings: list[str] = []
    next_actions: list[str] = []

    if unknown:
        warnings.append(f"unknown species (no valence): {', '.join(unknown)} -- pass --valence {unknown[0]}=Z")
        return response_envelope(
            tool="electron_parity_gate",
            verdict="REVIEW",
            confidence="low",
            status="ok",
            reasons=[f"cannot determine electron count: missing valence for {unknown}"],
            next_actions=[f"provide valences via --valence (e.g. --valence {unknown[0]}=Z) or --pseudo-dir"],
            warnings=warnings,
            result={"symbol_counts": symbol_counts, "unknown_species": unknown},
        )

    n_e = sum(valence[s] * n for s, n in symbol_counts.items()) - charge
    # charge may be float; parity only defined for integer electron count
    if abs(n_e - round(n_e)) > 1e-9:
        warnings.append(f"non-integer electron count N_e={n_e} (fractional --charge?)")
    n_e_int = int(round(n_e))
    parity = "odd" if n_e_int % 2 else "even"
    tms = sorted(s for s in symbol_counts if s in OPEN_SHELL_TM)

    # detect whether a metallic/smearing context was signalled
    _is_metallic = metallic or (smearing is not None and smearing.strip() != "")

    # A-priori closed-shell descriptor: reject d0/d10 TM cells the parity gate
    # would otherwise over-conservatively flag. effective_tms = the open-shell TMs
    # that are NOT inferred to be a diamagnetic closed shell.
    ox_inf = infer_closed_shell(symbol_counts, charge=charge)
    closed_tm: list[str] = []
    if ox_inf["status"] == "ok" and ox_inf["closed_shell"] and ox_inf["species"]:
        closed_tm = [ox_inf["species"]]
    effective_tms = [t for t in tms if t not in closed_tm]
    has_eff_tm = bool(effective_tms)

    # vacancy-odd vs open-shell-TM-odd discriminator: an ODD electron count from a
    # vacancy / off-stoichiometry / lone pair on a diamagnetic-cation system (no
    # effective open-shell TM) under metallic SMEARING smears at E_F -> non-magnetic
    # -> nspin=1 is acceptable. (Manuscript caveat 5: PbS / In-S / GaS, and d0/d10
    # TM hosts.) Without smearing the parity constraint is hard -> nspin=2.
    _vacancy_odd_ok = (parity == "odd") and (not has_eff_tm) and _is_metallic

    if parity == "odd" and not _vacancy_odd_ok:
        verdict = "NSPIN2_MANDATORY"
        confidence = "high"
        nspin_required = 2
        total_mag_parity = "odd"
        min_abs_total_mag = 1
        reasons.append(
            f"N_e={n_e_int} is ODD -> S is half-integer. With FIXED occupations (insulator) "
            f"nspin=1 forces S=0 and is impossible. With metallic SMEARING the half-electron "
            f"is smeared at E_F, so nspin=1 is acceptable IF the system is genuinely non-magnetic "
            f"(moment collapses to 0) -- verify cheaply. (pyrite V_Fe+H: odd but smeared -> "
            f"non-magnetic -> nspin=1 was correct.)"
        )
        next_actions.append("run a cheap nspin=2 single-point with seeded starting_magnetization")
        next_actions.append("if the seeded moment COLLAPSES to ~0 (metallic, smeared) -> nspin=1 OK; "
                            "if a LOCALIZED moment persists -> nspin=2 (re-relax endpoints) mandatory")
        if _is_metallic:
            # N-08: ready-to-run collapse-test spec for metallic/smearing context
            next_actions.append(
                "COLLAPSE TEST (metallic/smearing context): run 1 nspin=2 U=0 single-point with "
                "seeded starting_magnetization on this endpoint; if |Mtot|,|Mabs| -> 0 then "
                "nspin=1 certified (use spin_collapse_verdict.py: check magnetization_settled "
                "field -- NSPIN1_OK when mabs_per_tm < 0.30 uB/TM), else re-do the barrier "
                "at nspin=2. See spin_collapse_verdict.magnetization_settled for interpretation."
            )
        if effective_tms:
            next_actions.append(f"seed/inspect local moments on {', '.join(effective_tms)} (AFM vs FM)")
        next_actions.append("re-relax endpoints at nspin=2 (nspin=1-relaxed geometry is invalid)")
        next_actions.append("verify post-DFT with magnetic_endpoint_gate.py / magnetic_band_gate.py")
    elif _vacancy_odd_ok:
        verdict = "NSPIN1_OK"
        confidence = "medium"
        nspin_required = 1
        total_mag_parity = "odd"
        min_abs_total_mag = 1
        _cs = f" ({ox_inf['reason']})" if closed_tm else ""
        reasons.append(
            f"N_e={n_e_int} is ODD but there is NO effective open-shell TM{_cs} and metallic "
            f"smearing is in use -> the odd electron is a vacancy / off-stoichiometry / lone-pair "
            f"artefact smeared at E_F, not an open-shell moment -> nspin=1 is acceptable. "
            f"(vacancy-odd, NOT TM-odd; manuscript caveat 5.)"
        )
        next_actions.append("confirm cheaply: nspin=2 single-point should give |Mabs| -> 0")
    elif has_eff_tm:
        verdict = "NSPIN2_RECOMMENDED"
        confidence = "medium"
        nspin_required = 2
        total_mag_parity = "even"
        min_abs_total_mag = 0
        reasons.append(
            f"N_e={n_e_int} is EVEN (parity allows nspin=1), BUT open-shell TM present "
            f"({', '.join(effective_tms)}) -> AFM/FM ordering possible. nspin=1 may miss local moments."
        )
        next_actions.append("run nspin=2 single-point with starting_magnetization; compare E vs nspin=1")
        next_actions.append("verify with magnetic gates; nspin=1 only if moments collapse to 0")
    else:
        verdict = "NSPIN1_OK"
        confidence = "high"
        nspin_required = 1
        total_mag_parity = "even"
        min_abs_total_mag = 0
        if closed_tm:
            reasons.append(
                f"N_e={n_e_int} is EVEN and the only d-block species is a closed shell "
                f"({ox_inf['reason']}) -> diamagnetic, spin-state-independent -> nspin=1 is "
                f"defensible despite the TM (a-priori d-count override of NSPIN2_RECOMMENDED)."
            )
        else:
            reasons.append(f"N_e={n_e_int} is EVEN and no open-shell TM -> closed-shell nspin=1 is defensible.")

    result = {
        "symbol_counts": symbol_counts,
        "charge_e": charge,
        "n_electrons": n_e_int,
        "parity": parity,
        "nspin_required": nspin_required,
        "total_magnetization_parity_constraint": total_mag_parity,
        "min_abs_total_magnetization_uB": min_abs_total_mag,
        "open_shell_tm": tms,
        "effective_open_shell_tm": effective_tms,
        "closed_shell_tm": closed_tm,
        "oxidation_inference": ox_inf,
        "valence_used": {s: valence[s] for s in symbol_counts},
        "metallic_smearing_context": _is_metallic,
    }
    return response_envelope(
        tool="electron_parity_gate",
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,
        next_actions=next_actions,
        warnings=warnings,
        result=result,
    )


# ---- input helpers ----
def counts_from_structure(path: str | Path) -> dict[str, int]:
    """Read an xyz/cif/traj via ASE and return {symbol: count}."""
    from ase.io import read
    atoms = read(str(path))
    return dict(Counter(atoms.get_chemical_symbols()))


def counts_from_formula_tokens(tokens: list[str]) -> dict[str, int]:
    """Parse ['Fe31', 'S64', 'H1'] or ['Fe', '31', ...] style into counts."""
    import re
    counts: dict[str, int] = {}
    for tok in tokens:
        m = re.fullmatch(r"([A-Z][a-z]?)(\d+)", tok)
        if not m:
            raise SystemExit(f"bad --symbols token '{tok}' (expected like Fe31)")
        counts[m.group(1)] = counts.get(m.group(1), 0) + int(m.group(2))
    return counts


def parse_valence_overrides(items: list[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"bad --valence '{it}' (expected SYM=Z)")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v)
    return out


def print_gate(env: dict) -> None:
    r = env["result"] or {}
    print(f"verdict\t{env['verdict']}\tconfidence\t{env['confidence']}")
    print(f"N_e\t{r.get('n_electrons')}\tparity\t{r.get('parity')}\tnspin_required\t{r.get('nspin_required')}")
    print(f"total_mag_parity\t{r.get('total_magnetization_parity_constraint')}\tmin_|M|_uB\t{r.get('min_abs_total_magnetization_uB')}")
    if r.get("open_shell_tm"):
        print(f"open_shell_tm\t{','.join(r['open_shell_tm'])}")
    for x in env["reasons"]:
        print(f"reason\t{x}")
    for x in env["next_actions"]:
        print(f"next\t{x}")
    for x in env["warnings"]:
        print(f"warning\t{x}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="M0 electron-parity / spin-requirement gate (structure-only)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--structure", type=Path, help="xyz/cif/traj of the ACTUAL DFT cell (use the H-containing endpoint, not pristine)")
    src.add_argument("--symbols", nargs="+", help="explicit counts, e.g. --symbols Fe31 S64 H1")
    p.add_argument("--charge", type=float, default=0.0, help="net cell charge in e (positive = electrons removed)")
    p.add_argument("--valence", nargs="+", default=None, help="valence overrides SYM=Z (e.g. --valence Fe=16 S=6)")
    p.add_argument("--metallic", action="store_true", default=False,
                   help="flag: system uses metallic smearing (enables collapse-test spec in next_actions)")
    p.add_argument("--smearing", type=str, default=None,
                   help="smearing type string (e.g. gaussian, cold, mv); implies --metallic")
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    counts = counts_from_structure(args.structure) if args.structure else counts_from_formula_tokens(args.symbols)
    env = run_electron_parity_gate(
        counts, charge=args.charge,
        valence_overrides=parse_valence_overrides(args.valence),
        metallic=args.metallic,
        smearing=args.smearing,
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
