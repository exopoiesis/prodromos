"""``prodromos from-inputs`` -- convert QE/ABACUS *input files* into a tm-spec/0.3 stub.

The onboarding entry point. A scientist who just downloaded prodromos starts with
*only their own input files* (a Quantum ESPRESSO ``pw.x`` ``.in`` or an ABACUS
``INPUT`` + ``STRU`` directory). This module turns those into a tm-spec/0.3
document that validates against the bundled schema and can be fed straight into
``prodromos plan``.

This is the input-file counterpart of ``tm_spec.extract`` (which parses *deploy
scripts*). We deliberately reuse the spirit of that module -- a small set of
helpers, valid-enum defaults so the result passes schema, ``[TODO_HUMAN]``-style
placeholders for everything that genuinely needs a human -- but here the source
of truth is the engine input file, not a Python driver.

Mapping highlights (see module-level docs in convert_to_tmspec):
    QE  &control.calculation  scf -> SinglePointCalculation
                              relax/vc-relax -> RelaxCalculation
                              neb -> NEBCalculation
    ABACUS INPUT.calculation  scf -> SinglePointCalculation
                              relax -> RelaxCalculation
                              cell-relax -> RelaxCalculation

Usage::

    prodromos from-inputs path/to/pw.in            # auto-detect QE
    prodromos from-inputs path/to/abacus_run/      # auto-detect ABACUS (dir)
    prodromos from-inputs INPUT --code abacus --json
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import yaml

SPEC_VERSION = "0.3"

# QE calculation -> tm-spec kind
_QE_CALC_TO_KIND = {
    "scf": "SinglePointCalculation",
    "nscf": "SinglePointCalculation",
    "bands": "SinglePointCalculation",
    "relax": "RelaxCalculation",
    "vc-relax": "RelaxCalculation",
    "md": "MDCalculation",
    "vc-md": "MDCalculation",
    "neb": "NEBCalculation",
}

# ABACUS calculation -> tm-spec kind
_ABACUS_CALC_TO_KIND = {
    "scf": "SinglePointCalculation",
    "nscf": "SinglePointCalculation",
    "relax": "RelaxCalculation",
    "cell-relax": "RelaxCalculation",
    "md": "MDCalculation",
}

# QE occupations='smearing' + smearing keyword -> tm-spec smearing_kind enum.
_QE_SMEARING_MAP = {
    "gaussian": "gaussian",
    "gauss": "gaussian",
    "methfessel-paxton": "methfessel-paxton",
    "m-p": "methfessel-paxton",
    "mp": "methfessel-paxton",
    "marzari-vanderbilt": "marzari-vanderbilt",
    "cold": "marzari-vanderbilt",
    "m-v": "marzari-vanderbilt",
    "mv": "marzari-vanderbilt",
    "fermi-dirac": "fermi",
    "fd": "fermi",
    "tetrahedra": "tetrahedra",
}

# ABACUS smearing_method -> tm-spec smearing_kind enum.
_ABACUS_SMEARING_MAP = {
    "gauss": "gaussian",
    "gaussian": "gaussian",
    "mp": "methfessel-paxton",
    "mp2": "methfessel-paxton",
    "mv": "marzari-vanderbilt",
    "cold": "marzari-vanderbilt",
    "fd": "fermi",
    "fermi": "fermi",
    "fixed": None,
    "tetrahedra": "tetrahedra",
}

_PROTOTYPE_HINTS: dict[int, str] = {
    205: "AB2_cP12_205_a_c",
    225: "AB_cF8_225_a_b",
    129: "AB_tP4_129_a_c",
}

_MINERAL_TAGS = (
    "pyr", "mack", "pent", "trog", "marc", "violar", "cubanit",
    "chalcopyrite", "bornite", "millerite", "grei", "smyth", "pyrrhot",
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _strip_fortran_str(v: Any) -> Any:
    """Strip Fortran string quotes ('PBE' -> PBE). Leave non-strings alone."""
    if isinstance(v, str):
        s = v.strip()
        if len(s) >= 2 and s[0] in "'\"" and s[-1] == s[0]:
            return s[1:-1]
        return s
    return v


def _as_float(v: Any) -> float | None:
    try:
        return float(str(v).replace("d", "e").replace("D", "e"))
    except (TypeError, ValueError):
        return None


def _as_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _mineral_tag(name: str) -> str:
    low = name.lower()
    for tag in _MINERAL_TAGS:
        if tag in low:
            return tag
    return "import"


def _hill_formula(symbols: list[str]) -> str:
    """Hill-system flat formula from a list of per-atom symbols (Fe31S64H1)."""
    from collections import Counter

    counts = Counter(symbols)
    parts: list[str] = []
    # Hill: C first, H second, then alphabetical; for inorganic-only, alphabetical.
    ordered: list[str] = []
    if "C" in counts:
        ordered.append("C")
        if "H" in counts:
            ordered.append("H")
        ordered += sorted(s for s in counts if s not in ("C", "H"))
    else:
        ordered = sorted(counts)
    for s in ordered:
        n = counts[s]
        parts.append(s if n == 1 else f"{s}{n}")
    return "".join(parts)


def _structure_block_from_atoms(atoms: Any, *, supercell: list[int] | None = None) -> dict:
    """Build a tm-spec `structure` block from an ASE Atoms object."""
    symbols = list(atoms.get_chemical_symbols())
    structure: dict[str, Any] = {"formula": _hill_formula(symbols)}

    cell = atoms.cell
    try:
        lengths = [round(float(x), 6) for x in cell.lengths()]
        angles = [round(float(x), 4) for x in cell.angles()]
        if any(lengths):
            structure["cell"] = {
                "a": f"{lengths[0]} A",
                "b": f"{lengths[1]} A",
                "c": f"{lengths[2]} A",
                "angles": f"{angles} deg",
            }
            structure["lattice_vectors_A"] = [
                [round(float(c), 6) for c in row] for row in cell.array
            ]
    except Exception:
        pass

    structure["pbc"] = [bool(p) for p in atoms.pbc]
    if supercell:
        structure["supercell"] = supercell
    return structure


# --------------------------------------------------------------------------
# Quantum ESPRESSO
# --------------------------------------------------------------------------
def _read_qe_namelist(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return (namelist_sections, card_lines) using ASE's robust reader."""
    from ase.io.espresso import read_fortran_namelist

    with open(path, encoding="utf-8") as fh:
        sections, card = read_fortran_namelist(fh)
    return {k: dict(v) for k, v in sections.items()}, list(card)


def _qe_kpoints(card_lines: list[str]) -> dict | None:
    """Parse the K_POINTS card into a tm-spec k_points object."""
    for i, ln in enumerate(card_lines):
        if ln.strip().upper().startswith("K_POINTS"):
            mode = ln.split()[1].lower() if len(ln.split()) > 1 else "automatic"
            if "gamma" in mode:
                return {"mesh": [1, 1, 1], "shift": [0, 0, 0], "mode": "gamma"}
            if i + 1 < len(card_lines):
                nums = card_lines[i + 1].split()
                vals = [_as_int(x) for x in nums]
                if "automatic" in mode and len(vals) >= 6 and None not in vals[:6]:
                    return {"mesh": vals[:3], "shift": vals[3:6]}
                if len(vals) >= 3 and None not in vals[:3]:
                    return {"mesh": vals[:3], "shift": [0, 0, 0]}
            return {"mode": mode}
    return None


def _convert_qe(path: Path, date: str, kind_override: str | None) -> dict:
    from ase.io import read

    sections, card = _read_qe_namelist(path)
    control = sections.get("control", {})
    system = sections.get("system", {})

    calc = str(_strip_fortran_str(control.get("calculation", "scf"))).lower()
    kind = kind_override or _QE_CALC_TO_KIND.get(calc, "SinglePointCalculation")

    # --- structure (via ASE) ---
    try:
        atoms = read(str(path), format="espresso-in")
        structure = _structure_block_from_atoms(atoms)
    except Exception as exc:  # noqa: BLE001 -- fall back to a placeholder formula
        structure = {"formula": "[TODO_HUMAN]", "pbc": [True, True, True]}
        structure["_parse_note"] = f"ASE could not read geometry: {exc}"

    # --- level of theory ---
    level: dict[str, Any] = {}
    input_dft = _strip_fortran_str(system.get("input_dft"))
    level["xc"] = input_dft if input_dft else "PBE"  # QE default is PBE
    if not input_dft:
        level["_xc_note"] = "input_dft absent; QE default PBE assumed"

    ecutwfc = _as_float(system.get("ecutwfc"))
    ecutrho = _as_float(system.get("ecutrho"))
    if ecutwfc is not None:
        basis: dict[str, Any] = {"kind": "plane_waves", "cutoff_Ry": ecutwfc}
        if ecutrho is not None:
            basis["rho_cutoff_Ry"] = ecutrho
        level["basis"] = basis

    occ = str(_strip_fortran_str(system.get("occupations", ""))).lower()
    if occ == "smearing":
        sm_key = str(_strip_fortran_str(system.get("smearing", "gaussian"))).lower()
        sm_kind = _QE_SMEARING_MAP.get(sm_key, "gaussian")
        degauss = _as_float(system.get("degauss"))
        smear: dict[str, Any] = {"kind": sm_kind}
        if degauss is not None:
            smear["width_Ry"] = degauss
        level["smearing"] = smear

    nspin = _as_int(system.get("nspin"))
    if nspin == 4 or _strip_fortran_str(system.get("noncolin")) in (True, ".true.", "true"):
        level["spin"] = "non-collinear"
    elif nspin == 2:
        level["spin"] = "collinear"
    else:
        level["spin"] = "none"

    # Hubbard (old lda_plus_u or new dftU not parsed deeply; flag presence)
    if _strip_fortran_str(system.get("lda_plus_u")) in (True, ".true.", "true"):
        hub: dict[str, Any] = {}
        for k, v in system.items():
            if k.lower().startswith("hubbard_u"):
                hub[k] = _as_float(v)
        level["hubbard"] = hub or {"_note": "lda_plus_u enabled (values not parsed)"}

    calculation: dict[str, Any] = {"method": "DFT+U" if "hubbard" in level else "DFT"}
    calculation["level"] = level

    kpts = _qe_kpoints(card)
    if kpts:
        calculation["k_points"] = kpts

    electrons = sections.get("electrons", {})
    conv: dict[str, Any] = {}
    conv_thr = _as_float(electrons.get("conv_thr"))
    if conv_thr is not None:
        conv["scf_Ry"] = conv_thr
    fcheck = _as_float(control.get("forc_conv_thr"))
    if fcheck is not None:
        conv["forc_conv_thr_Ry_per_au"] = fcheck
    if conv:
        calculation["convergence"] = conv

    calculation["code"] = {"name": "QuantumESPRESSO"}

    doc = _assemble_doc(
        kind=kind,
        structure=structure,
        calculation=calculation,
        date=date,
        source_path=path,
        source_code="QuantumESPRESSO",
        source_calc=calc,
        nspin=nspin,
        cell_relax_kind=("all" if calc == "vc-relax" else "ions_only"),
    )
    return doc


# --------------------------------------------------------------------------
# ABACUS
# --------------------------------------------------------------------------
def _parse_abacus_input(path: Path) -> dict[str, str]:
    """Parse an ABACUS INPUT file (whitespace-separated key value, # comments)."""
    out: dict[str, str] = {}
    in_params = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line.upper().startswith("INPUT_PARAMETERS"):
            in_params = True
            continue
        # Tolerate files without the header banner.
        parts = line.split(None, 1)
        if len(parts) == 2:
            out[parts[0].strip().lower()] = parts[1].strip()
        elif len(parts) == 1 and not in_params:
            continue
    return out


def _parse_abacus_stru(path: Path) -> tuple[list[str], list[list[float]] | None]:
    """Parse an ABACUS STRU file -> (per-atom symbols, lattice_vectors_A or None).

    Returns the element list expanded per atom (so counts give the formula) and
    the lattice vectors scaled by LATTICE_CONSTANT (Bohr -> Angstrom).
    """
    bohr_to_A = 0.52917721067
    text = path.read_text(encoding="utf-8")
    lines = [ln.split("#")[0].rstrip() for ln in text.splitlines()]

    def _section(name: str) -> int | None:
        for i, ln in enumerate(lines):
            if ln.strip().upper() == name:
                return i
        return None

    # LATTICE_CONSTANT (in Bohr by default)
    latt_const = 1.0
    idx = _section("LATTICE_CONSTANT")
    if idx is not None:
        for ln in lines[idx + 1:]:
            s = ln.strip()
            if not s:
                continue
            latt_const = _as_float(s.split()[0]) or 1.0
            break

    # LATTICE_VECTORS
    vectors: list[list[float]] | None = None
    idx = _section("LATTICE_VECTORS")
    if idx is not None:
        vecs: list[list[float]] = []
        for ln in lines[idx + 1:]:
            s = ln.strip()
            if not s:
                if vecs:
                    break
                continue
            nums = [_as_float(x) for x in s.split()[:3]]
            if len(nums) == 3 and None not in nums:
                vecs.append([v * latt_const * bohr_to_A for v in nums])  # type: ignore[operator]
            if len(vecs) == 3:
                break
        if len(vecs) == 3:
            vectors = [[round(c, 6) for c in row] for row in vecs]

    # ATOMIC_POSITIONS: element blocks
    #   <Element>
    #   <magnetization>
    #   <n_atoms>
    #   <x y z [flags]>  x n_atoms
    symbols: list[str] = []
    idx = _section("ATOMIC_POSITIONS")
    if idx is not None:
        cursor = idx + 1
        # skip the coordinate-mode line (Direct/Cartesian/...)
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor < len(lines):
            cursor += 1  # consume mode line
        while cursor < len(lines):
            s = lines[cursor].strip()
            cursor += 1
            if not s:
                continue
            # element label line (single non-numeric token)
            tok = s.split()
            if len(tok) == 1 and not _is_number(tok[0]):
                elem = tok[0]
                # next non-empty: magnetization
                cursor = _skip_blank(lines, cursor)
                cursor += 1  # magnetization
                cursor = _skip_blank(lines, cursor)
                if cursor < len(lines):
                    n_at = _as_int(lines[cursor].strip().split()[0]) or 0
                    cursor += 1
                    symbols.extend([elem] * n_at)
                    # skip the n_at position rows
                    consumed = 0
                    while cursor < len(lines) and consumed < n_at:
                        if lines[cursor].strip():
                            consumed += 1
                        cursor += 1
    return symbols, vectors


def _is_number(tok: str) -> bool:
    return _as_float(tok) is not None


def _skip_blank(lines: list[str], cursor: int) -> int:
    while cursor < len(lines) and not lines[cursor].strip():
        cursor += 1
    return cursor


def _abacus_kpt(kpt_path: Path) -> dict | None:
    """Parse an ABACUS KPT file (Gamma/MP mesh)."""
    if not kpt_path.exists():
        return None
    lines = [ln.split("#")[0].strip() for ln in kpt_path.read_text(encoding="utf-8").splitlines()]
    lines = [ln for ln in lines if ln]
    # Format: K_POINTS / 0 / Gamma|MP / nx ny nz sx sy sz
    for i, ln in enumerate(lines):
        toks = ln.split()
        if len(toks) >= 6 and all(_is_number(t) for t in toks[:6]):
            vals = [_as_int(t) for t in toks[:6]]
            if None not in vals:
                return {"mesh": vals[:3], "shift": vals[3:6]}
    return None


def _resolve_abacus_inputs(path: Path) -> tuple[Path, Path | None, Path | None]:
    """From a path (dir or INPUT file) locate INPUT, STRU, KPT."""
    if path.is_dir():
        input_file = path / "INPUT"
        stru = path / "STRU"
        kpt = path / "KPT"
    else:
        input_file = path
        d = path.parent
        stru = d / "STRU"
        kpt = d / "KPT"
    return (
        input_file,
        stru if stru.exists() else None,
        kpt if kpt.exists() else None,
    )


def _convert_abacus(path: Path, date: str, kind_override: str | None) -> dict:
    input_file, stru_file, kpt_file = _resolve_abacus_inputs(path)
    if not input_file.exists():
        raise FileNotFoundError(f"ABACUS INPUT not found at {input_file}")

    params = _parse_abacus_input(input_file)

    calc = params.get("calculation", "scf").lower()
    kind = kind_override or _ABACUS_CALC_TO_KIND.get(calc, "SinglePointCalculation")

    # --- structure ---
    structure: dict[str, Any]
    if stru_file is not None:
        symbols, vectors = _parse_abacus_stru(stru_file)
        if symbols:
            structure = {"formula": _hill_formula(symbols)}
        else:
            structure = {"formula": "[TODO_HUMAN]"}
            structure["_parse_note"] = "STRU parsed but no atoms recognised"
        if vectors:
            structure["lattice_vectors_A"] = vectors
        structure["pbc"] = [True, True, True]
    else:
        structure = {"formula": "[TODO_HUMAN]", "pbc": [True, True, True]}
        structure["_parse_note"] = "no STRU file found beside INPUT"

    # --- level of theory ---
    level: dict[str, Any] = {}
    func = params.get("dft_functional", "PBE")
    level["xc"] = func.upper() if func else "PBE"

    basis_type = params.get("basis_type", "pw").lower()
    basis: dict[str, Any] = {}
    if basis_type in ("pw", "plane_wave", "planewave"):
        basis["kind"] = "plane_waves"
    elif basis_type in ("lcao", "lcao_in_pw"):
        basis["kind"] = "numeric_AOs"
    else:
        basis["kind"] = "plane_waves"
    ecutwfc = _as_float(params.get("ecutwfc"))
    if ecutwfc is not None:
        basis["cutoff_Ry"] = ecutwfc  # ABACUS ecutwfc is in Ry
    level["basis"] = basis

    smearing_method = params.get("smearing_method", "").lower()
    if smearing_method and smearing_method != "fixed":
        sm_kind = _ABACUS_SMEARING_MAP.get(smearing_method, "gaussian")
        smear: dict[str, Any] = {}
        if sm_kind:
            smear["kind"] = sm_kind
        sigma = _as_float(params.get("smearing_sigma"))
        if sigma is not None:
            smear["width_Ry"] = sigma  # ABACUS smearing_sigma is in Ry
        if smear:
            level["smearing"] = smear

    nspin = _as_int(params.get("nspin"))
    if nspin == 4:
        level["spin"] = "non-collinear"
    elif nspin == 2:
        level["spin"] = "collinear"
    else:
        level["spin"] = "none"

    if params.get("ks_solver"):
        level["_ks_solver"] = params["ks_solver"]

    calculation: dict[str, Any] = {"method": "DFT", "level": level}

    kpts = _abacus_kpt(kpt_file) if kpt_file else None
    if kpts is None:
        # ABACUS can also embed gamma_only / kspacing in INPUT
        if params.get("gamma_only", "0") in ("1", "true", "True"):
            kpts = {"mesh": [1, 1, 1], "shift": [0, 0, 0], "mode": "gamma"}
    if kpts:
        calculation["k_points"] = kpts

    scf_thr = _as_float(params.get("scf_thr"))
    if scf_thr is not None:
        calculation["convergence"] = {"scf_thr": scf_thr}

    calculation["code"] = {"name": "ABACUS"}

    doc = _assemble_doc(
        kind=kind,
        structure=structure,
        calculation=calculation,
        date=date,
        source_path=path,
        source_code="ABACUS",
        source_calc=calc,
        nspin=nspin,
        cell_relax_kind=("all" if calc == "cell-relax" else "ions_only"),
    )
    return doc


# --------------------------------------------------------------------------
# common assembly
# --------------------------------------------------------------------------
def _assemble_doc(
    *,
    kind: str,
    structure: dict,
    calculation: dict,
    date: str,
    source_path: Path,
    source_code: str,
    source_calc: str,
    nspin: int | None,
    cell_relax_kind: str,
) -> dict:
    """Fill the required tm-spec/0.3 envelope around a structure + calculation.

    Required top-level fields per schema 0.3: spec, kind, id, structure,
    calculation, sanity, provenance. ``sanity`` may be an empty array (the
    schema only requires the key to exist as an array). Kind-conditional blocks
    (results / workflow / relax_protocol) are added as minimal placeholders so
    the stub validates AND round-trips through `prodromos plan`.
    """
    tag = _mineral_tag(source_path.stem or source_path.name)
    formula = structure.get("formula", "x")
    safe_formula = re.sub(r"[^A-Za-z0-9_.+\-]", "", formula) if isinstance(formula, str) else "x"
    if not safe_formula or safe_formula == "TODO_HUMAN":
        safe_formula = "x"

    doc: dict[str, Any] = {
        "spec": f"tm-spec/{SPEC_VERSION}",
        "kind": kind,
        "id": f"tm.{tag}.{safe_formula}.{date}",
        "schema_url": f"https://exopoiesis.github.io/tm-spec/{SPEC_VERSION}.json",
        "structure": structure,
        "calculation": calculation,
    }

    # Magnetic block when spin-polarised (advisory; state needs a human).
    if nspin == 2:
        doc["magnetic"] = {
            "state": "FM",
            "collinear": True,
            "surrogate_warning": "[TODO_HUMAN] nspin=2 detected; confirm FM/AFM/ferri ordering",
        }
    elif nspin == 4:
        doc["magnetic"] = {
            "state": "FM",
            "collinear": False,
            "surrogate_warning": "[TODO_HUMAN] noncollinear (nspin=4); confirm magnetic state",
        }

    # Kind-conditional required blocks (minimal, valid placeholders).
    if kind == "NEBCalculation":
        doc["workflow"] = {
            "kind": "NEB",
            "stage": "smoke",
            "endpoints": {
                "A": {"ref": "[TODO_HUMAN] endA.extxyz", "geometry_origin": "unknown"},
                "B": {"ref": "[TODO_HUMAN] endB.extxyz", "geometry_origin": "unknown"},
            },
        }
        doc["results"] = {"status": "PRELIMINARY", "paper_quotable": False}
    elif kind == "RelaxCalculation":
        doc["relax_protocol"] = {
            "optimizer": source_calc,  # code-native label (vc-relax / relax)
            "cell_relax": (cell_relax_kind == "all"),
            "cell_relax_kind": cell_relax_kind,
        }
        doc["results"] = {"status": "PRELIMINARY", "paper_quotable": False}
    elif kind == "SinglePointCalculation":
        doc["results"] = {"status": "PRELIMINARY", "paper_quotable": False}
    elif kind == "MDCalculation":
        doc["md_protocol"] = {"ensemble": "NVT", "timestep_fs": 1.0}
        doc["results"] = {"status": "PRELIMINARY", "paper_quotable": False}

    doc["sanity"] = []

    doc["provenance"] = {
        "date": date,
        "author": "you@example.org",
        "import_source": {
            "archive": "other",
            "importer": "prodromos from-inputs@0.1.0",
            "raw_keys": [str(source_path).replace("\\", "/")],
        },
        "compute": {"host": "[TODO_HUMAN]", "cost_usd": 0.0},
    }
    return doc


# --------------------------------------------------------------------------
# format detection + public entry point
# --------------------------------------------------------------------------
def detect_code(path: Path) -> str:
    """Return 'qe' or 'abacus' by sniffing the path. Raises on ambiguity."""
    p = Path(path)
    if p.is_dir():
        if (p / "INPUT").exists() and (p / "STRU").exists():
            return "abacus"
        if (p / "INPUT").exists():
            return "abacus"
        # a directory holding a single .in?
        ins = list(p.glob("*.in"))
        if len(ins) == 1:
            return "qe"
        raise ValueError(
            f"cannot auto-detect code in directory {p}: expected ABACUS INPUT+STRU "
            "or a single QE .in; pass --code qe|abacus"
        )
    name = p.name
    suffix = p.suffix.lower()
    if suffix == ".in":
        return "qe"
    if name.upper() == "INPUT" or name.upper() == "STRU":
        return "abacus"
    # content sniff: QE namelists begin with &control / &system
    try:
        head = p.read_text(encoding="utf-8", errors="ignore")[:4000]
    except OSError:
        head = ""
    if re.search(r"^\s*&(control|system|electrons)", head, re.IGNORECASE | re.MULTILINE):
        return "qe"
    if re.search(r"^\s*INPUT_PARAMETERS", head, re.IGNORECASE | re.MULTILINE):
        return "abacus"
    if re.search(r"ATOMIC_SPECIES|K_POINTS", head):
        return "qe"
    raise ValueError(
        f"cannot auto-detect code for {p}; pass --code qe|abacus"
    )


def convert_to_tmspec(
    path_or_dir: str | Path,
    *,
    code: str = "auto",
    kind: str | None = None,
    date: str = "YYYY-MM-DD",
) -> dict:
    """Convert a QE/ABACUS input (file or directory) into a tm-spec/0.3 doc dict.

    Parameters
    ----------
    path_or_dir : path to a QE ``.in`` file, an ABACUS ``INPUT`` file, or an
        ABACUS run directory (containing INPUT + STRU [+ KPT]).
    code : ``"qe"`` | ``"abacus"`` | ``"auto"`` (default auto-detect).
    kind : optional tm-spec kind override (else derived from ``calculation``).
    date : ISO date string used in ``id`` and ``provenance.date``. Defaults to
        the deterministic placeholder ``"YYYY-MM-DD"`` -- NOT ``datetime.now()``
        -- so output is reproducible. The schema's ``id`` pattern requires a
        real ``\\d{4}-\\d{2}-\\d{2}``; pass ``--date`` for a schema-valid id.
    """
    path = Path(path_or_dir)
    if not path.exists():
        raise FileNotFoundError(f"input path not found: {path}")

    resolved = code.lower()
    if resolved == "auto":
        resolved = detect_code(path)

    if resolved == "qe":
        return _convert_qe(path, date, kind)
    if resolved == "abacus":
        return _convert_abacus(path, date, kind)
    raise ValueError(f"unknown code {code!r}; expected qe | abacus | auto")


def _validate(doc: dict) -> tuple[list[str], list[str]]:
    """Run the tm-spec validator if available. Returns (errors, warnings)."""
    try:
        from tm_spec.validator import validate_doc
    except ModuleNotFoundError:
        return (["tm-spec not installed; skipping validation (pip install -e ../tm-spec)"], [])
    schema_errs, rule_issues = validate_doc(doc)
    errors = [f"{loc}: {msg}" for loc, msg in schema_errs]
    errors += [msg for level, msg in rule_issues if level == "error"]
    warnings = [msg for level, msg in rule_issues if level == "warn"]
    return errors, warnings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prodromos from-inputs",
        description="Convert QE/ABACUS input files into a tm-spec/0.3 starter document.",
    )
    parser.add_argument("path", help="QE .in file, ABACUS INPUT file, or ABACUS run directory")
    parser.add_argument(
        "--code", choices=["qe", "abacus", "auto"], default="auto",
        help="input format (default: auto-detect)",
    )
    parser.add_argument("--kind", help="override the tm-spec kind (else derived from calculation)")
    parser.add_argument(
        "--date", default="YYYY-MM-DD",
        help="ISO date for id/provenance (default: 'YYYY-MM-DD' placeholder; "
        "pass a real date e.g. 2026-06-01 for a schema-valid id)",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of YAML")
    parser.add_argument("--output", help="write to this path instead of stdout")
    args = parser.parse_args(argv)

    try:
        doc = convert_to_tmspec(args.path, code=args.code, kind=args.kind, date=args.date)
    except (FileNotFoundError, ValueError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    errors, warnings = _validate(doc)
    for w in warnings:
        print(f"WARN  {w}", file=sys.stderr)
    if errors:
        print(
            "NOTE: the generated document does not yet validate against tm-spec/"
            f"{SPEC_VERSION}; it is a starter stub -- complete the [TODO_HUMAN] fields:",
            file=sys.stderr,
        )
        for e in errors:
            print(f"  - {e}", file=sys.stderr)

    if args.json:
        import json

        text = json.dumps(doc, indent=2, ensure_ascii=False, default=str)
    else:
        text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=False)

    if args.output:
        Path(args.output).write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
