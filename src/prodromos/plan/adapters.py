"""tm-spec document -> gate run_fn inputs.

This is the inverse of ``tm_spec.extract.compose`` (which goes script ->
.tm.yaml): here we go .tm.yaml (case) -> the flat keyword arguments a gate's
``run_*`` consumes. When a gate's required inputs are absent from the case, the
adapter returns a :class:`NeedsData` marker; the route interpreter then emits a
NEEDS_DATA terminal recommending which datum to obtain (rather than crashing
mid-walk -- consilium S4/D8).

First increment: only the gates a *pre-flight* case can actually feed are wired
to real input extraction (electron-parity from the formula, endpoint-provenance
from workflow.endpoints[*].geometry_origin, external-reference from elements).
The magnetic / saddle / band / lint / h-barrier gates consume parsed DFT
outputs or local files that a pre-flight document does not yet carry, so their
adapters return NeedsData with the concrete missing artefact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Map tm-spec calculation.level.spin -> nspin integer (advisory echo only).
_SPIN_TO_NSPIN = {"none": 1, "collinear": 2, "non-collinear": 4}
_OPEN_SHELL_TM = {"Fe", "Co", "Ni", "Mn", "Cr", "V", "Ti", "Cu"}


@dataclass(frozen=True)
class NeedsData:
    """Marker: this gate cannot be evaluated from the current case.

    ``missing`` lists the case fields / artefacts that are absent; ``recommend``
    is a single human-actionable next step that the route terminal surfaces.
    """

    gate: str
    missing: list[str] = field(default_factory=list)
    recommend: str = ""


# --------------------------------------------------------------------------
# formula parsing
# --------------------------------------------------------------------------
_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> dict[str, int]:
    """Parse a flat chemical formula like ``Fe31S64H1`` -> ``{'Fe':31,...}``.

    A bare symbol counts as 1. Composite/symbolic strings (containing ``(``,
    ``+`` or ``x``, as produced by extract._compute_formula for non-pyrite/mack
    prototypes) yield an empty dict -> the caller treats that as NeedsData.
    """
    if not formula or any(c in formula for c in "(+ x"):
        return {}
    counts: dict[str, int] = {}
    pos = 0
    for m in _FORMULA_TOKEN.finditer(formula):
        if m.start() != pos:  # gap -> unparseable garbage between tokens
            return {}
        pos = m.end()
        sym, num = m.group(1), m.group(2)
        counts[sym] = counts.get(sym, 0) + (int(num) if num else 1)
    if pos != len(formula):
        return {}
    return counts


def _structure_formula(doc: dict) -> str | None:
    structure = doc.get("structure") or {}
    f = structure.get("formula")
    return f if isinstance(f, str) and f.strip() else None


def _endpoints(doc: dict) -> dict:
    wf = doc.get("workflow") or {}
    eps = wf.get("endpoints") or {}
    return eps if isinstance(eps, dict) else {}


def _spin_string(doc: dict) -> str | None:
    lvl = (doc.get("calculation") or {}).get("level") or {}
    spin = lvl.get("spin")
    return spin if isinstance(spin, str) else None


def _is_metallic(doc: dict) -> bool:
    lvl = (doc.get("calculation") or {}).get("level") or {}
    smearing = lvl.get("smearing") or {}
    kind = smearing.get("kind") if isinstance(smearing, dict) else None
    return bool(kind)


# --------------------------------------------------------------------------
# per-gate adapters
# --------------------------------------------------------------------------
def to_electron_parity_inputs(doc: dict) -> dict | NeedsData:
    """G11: needs symbol_counts (from structure.formula) + smearing context."""
    formula = _structure_formula(doc)
    counts = parse_formula(formula) if formula else {}
    if not counts:
        return NeedsData(
            gate="electron-parity",
            missing=["structure.formula (flat, e.g. Fe31S64H1)"],
            recommend=(
                "provide a flat structure.formula so the electron count / parity "
                "can be derived (current formula is missing or symbolic)"
            ),
        )
    smearing_kind = None
    lvl = (doc.get("calculation") or {}).get("level") or {}
    sm = lvl.get("smearing") or {}
    if isinstance(sm, dict):
        smearing_kind = sm.get("kind")
    return {
        "symbol_counts": counts,
        "metallic": _is_metallic(doc),
        "smearing": smearing_kind,
    }


def to_endpoint_provenance_inputs(doc: dict) -> dict | NeedsData:
    """G09: needs at least one endpoint's geometry_origin.

    The gate evaluates one endpoint at a time; we feed the WORST (least trusted)
    endpoint so a single non-dft_relaxed endpoint flips the verdict.
    """
    eps = _endpoints(doc)
    if not eps:
        return NeedsData(
            gate="endpoint-provenance",
            missing=["workflow.endpoints"],
            recommend="add workflow.endpoints with geometry_origin on each endpoint",
        )
    origins = {name: (ep or {}).get("geometry_origin") for name, ep in eps.items()}
    if any(v is None for v in origins.values()):
        missing = [f"workflow.endpoints.{n}.geometry_origin" for n, v in origins.items() if v is None]
        return NeedsData(
            gate="endpoint-provenance",
            missing=missing,
            recommend="set geometry_origin (dft_relaxed | mlip_relaxed | ...) on every endpoint",
        )
    # Pick the worst: any non-dft_relaxed dominates.
    worst_name, worst_prov = next(
        ((n, p) for n, p in origins.items() if str(p).lower() != "dft_relaxed"),
        next(iter(origins.items())),
    )
    energy = (eps.get(worst_name) or {}).get("E_eV")
    return {
        "geometry_origin": worst_prov,
        "energy_eV": energy,
        "label": worst_name,
    }


def to_external_reference_inputs(doc: dict, *, live: bool = False) -> dict | NeedsData:
    """G19: needs elements (derivable from formula or composition)."""
    formula = _structure_formula(doc)
    counts = parse_formula(formula) if formula else {}
    elements = sorted(counts)
    if not elements:
        structure = doc.get("structure") or {}
        comp = structure.get("composition") or {}
        # composition is a {phase: "formula"} map; harvest element symbols.
        for v in comp.values():
            if isinstance(v, str):
                elements.extend(parse_formula(v))
        elements = sorted(set(elements))
    if not elements:
        return NeedsData(
            gate="external-reference",
            missing=["structure.formula or structure.composition"],
            recommend="add a formula/composition so the element set can be queried against NOMAD/OPTIMADE",
        )
    sg = (doc.get("structure") or {}).get("space_group") or {}
    sg_label = sg.get("symbol") if isinstance(sg, dict) else None
    return {
        "elements": elements,
        "reduced_formula": formula,
        "space_group": sg_label,
        "live": live,
    }


def _needs_dft_output(gate: str, artefact: str) -> NeedsData:
    return NeedsData(
        gate=gate,
        missing=[artefact],
        recommend=(
            f"run the upstream step to produce {artefact}; this gate consumes "
            "parsed DFT outputs, which a pre-flight case does not yet carry"
        ),
    )


def to_spin_collapse_inputs(doc: dict) -> dict | NeedsData:
    """G02: needs a cheap nspin=2 single-point's magnetization (post-pilot)."""
    return _needs_dft_output(
        "spin-collapse", "an nspin=2 single-point magnetization (mabs + n_tm)"
    )


def to_symmetry_preflight_inputs(doc: dict) -> dict | NeedsData:
    """G03: needs pristine/endA/triple structure files on disk."""
    return _needs_dft_output(
        "symmetry-preflight",
        "pristine + endA + canonical-triple .extxyz structure files for the L1 gate",
    )


def to_magnetic_endpoint_inputs(doc: dict) -> dict | NeedsData:
    return _needs_dft_output(
        "magnetic-endpoint", "two parsed endpoint magnetic summaries (nspin=2 SCF outputs)"
    )


def to_magnetic_band_inputs(doc: dict) -> dict | NeedsData:
    return _needs_dft_output(
        "magnetic-band", "parsed per-image magnetic outputs from a NEB band"
    )


def to_magnetic_parser_inputs(doc: dict) -> dict | NeedsData:
    return _needs_dft_output("magnetic-parser", "a DFT SCF output file to parse")


def to_magnetic_recommend_inputs(doc: dict) -> dict | NeedsData:
    return _needs_dft_output(
        "magnetic-recommend", "parsed endpoint/band magnetic summaries"
    )


def to_neb_advisor_inputs(doc: dict) -> dict | NeedsData:
    """G16: needs a running-NEB signature (band + optimizer history)."""
    return _needs_dft_output(
        "neb-advisor", "an NEB signature (band energies + optimizer/fmax history)"
    )


def to_saddle_proximity_inputs(doc: dict) -> dict | NeedsData:
    """G17: needs a saddle/TS geometry (ASE Atoms) + S anchor indices."""
    return _needs_dft_output(
        "saddle-proximity", "a saddle-candidate geometry (Atoms) + S-anchor indices"
    )


def to_lint_dft_script_inputs(doc: dict) -> dict | NeedsData:
    """G18: needs the paired deploy script on disk."""
    wf = doc.get("workflow") or {}
    script = wf.get("paired_script")
    if not script:
        return NeedsData(
            gate="lint-dft-script",
            missing=["workflow.paired_script"],
            recommend="point workflow.paired_script at the deploy script to lint it",
        )
    # Even when named, the file is generally not resolvable from the case dir in
    # a pre-flight context; defer to the next increment (script-resolution).
    return _needs_dft_output(
        "lint-dft-script", f"the deploy script {script!r} resolvable on disk"
    )


def to_h_barrier_readiness_inputs(doc: dict) -> dict | NeedsData:
    """G20: needs a computed barrier + DFT saddle frequency (post-NEB)."""
    return _needs_dft_output(
        "h-barrier-readiness", "a computed barrier + DFT saddle frequency (n_imag, H-fraction, dZPE)"
    )


# Dispatch table: gate subcommand -> adapter function.
ADAPTERS = {
    "electron-parity": to_electron_parity_inputs,
    "endpoint-provenance": to_endpoint_provenance_inputs,
    "external-reference": to_external_reference_inputs,
    "spin-collapse": to_spin_collapse_inputs,
    "symmetry-preflight": to_symmetry_preflight_inputs,
    "magnetic-endpoint": to_magnetic_endpoint_inputs,
    "magnetic-band": to_magnetic_band_inputs,
    "magnetic-parser": to_magnetic_parser_inputs,
    "magnetic-recommend": to_magnetic_recommend_inputs,
    "neb-advisor": to_neb_advisor_inputs,
    "saddle-proximity": to_saddle_proximity_inputs,
    "lint-dft-script": to_lint_dft_script_inputs,
    "h-barrier-readiness": to_h_barrier_readiness_inputs,
}


def tm_doc_to_gate_inputs(doc: dict, gate: str) -> dict | NeedsData:
    """Extract inputs for ``gate`` from a tm-spec ``doc``.

    Returns a kwargs dict for the gate's run_fn, or a :class:`NeedsData` marker
    when the case lacks the required data.
    """
    adapter = ADAPTERS.get(gate)
    if adapter is None:
        return NeedsData(
            gate=gate,
            missing=[f"no adapter registered for gate {gate!r}"],
            recommend=f"register an adapter for {gate!r} in prodromos.plan.adapters",
        )
    return adapter(doc)
