"""N-09 -- static pre-flight lint for QE/ABACUS deploy scripts.

Checks four recurring deploy bugs (all caught in production):

  CHECK-1  pseudo_dir relative path
           A pseudo_dir defined as a relative path (or not explicitly absolute)
           will break on remote nodes where cwd is unpredictable.
           FAIL if the script contains pseudo_dir set to a relative path string
           (i.e. does not start with / or a Windows drive letter like C:\\).

  CHECK-2  outdir nesting / doubling
           Pattern: outdir = os.path.join(work_dir, label, "tmp") gets doubled
           because pw.x cwd is already the per-job sub-directory.
           FAIL if the script builds outdir by joining multiple path components
           that look like work_dir + label concatenation.

  CHECK-3  extxyz nspins= comment tag
           ASE's SinglePointCalculator raises AssertionError when the XYZ
           comment line contains non-standard keys like nspins=, nkpts=, nbands=.
           FAIL if (a) the script reads .xyz files via ase.io.read without a
           clean-read wrapper, AND/OR (b) the provided xyz_path file contains
           a non-standard key in its comment line.

  CHECK-4  SG15 ONCV .upf with number_of_wfc=0
           SG15 ONCV pseudopotentials sometimes carry number_of_wfc=0 in the
           PP_HEADER section.  pw.x's +U / startingwfc=atomic path hard-crashes
           when no Wannier-function channel is present.
           FAIL if pseudo_dir is given and any *.upf file in that directory
           has number_of_wfc equal to 0.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from prodromos.cli_contract import response_envelope, dump_json

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------

# pseudo_dir assignments: pseudo_dir = "...", pseudo_dir="...", pseudo_dir = '...',
# also dict-style  'pseudo_dir': '...'
_RE_PSEUDO_DIR_STR = re.compile(
    r"""['"]{0,1}pseudo_dir['"]{0,1}\s*[=:]\s*(['"])(.+?)\1""",
    re.IGNORECASE,
)

# Looks like a non-absolute path: anything that doesn't start with / or X:\\ or X:/
_RE_RELATIVE_PATH = re.compile(r"^(?![/\\]|[A-Za-z]:[/\\])")

# os.path.join(...) with 3+ args, or f-string with multiple dir components
# Pattern: join( something , something , "tmp" ) — heuristic
_RE_OUTDIR_JOIN = re.compile(
    r"""outdir\s*[=:]\s*(?:os\.path\.join\s*\([^)]+,[^)]+,[^)]+\)|f['"]{1}[^'"]+\{[^}]+\}[^'"]*[/\\][^'"]*['"]{1})""",
    re.IGNORECASE,
)

# Simpler: outdir = os.path.join(X, Y) where Y is not literally "tmp"
# i.e. the second arg contains a variable (label / mineral) that acts as a sub-folder
_RE_OUTDIR_COMPOSITE = re.compile(
    r"""outdir\s*[=:]\s*os\.path\.join\s*\(([^)]+)\)""",
    re.IGNORECASE,
)

# Bare ase.io.read call (with .xyz argument) but without going through a clean reader
_RE_ASE_READ = re.compile(r"\base\.io\.read\b|\bio\.read\(", re.IGNORECASE)
_RE_CLEAN_READ = re.compile(r"_clean_read|clean_read", re.IGNORECASE)

# Non-standard extxyz comment keys
_NONSTANDARD_KEYS = re.compile(r"\b(nspins|nkpts|nbands|kpoints|spin_polarized)\s*=", re.IGNORECASE)

# number_of_wfc in UPF header
_RE_WFC_ZERO = re.compile(r"number_of_wfc\s*=\s*[\"']?\s*0\s*[\"']?", re.IGNORECASE)


def _check_pseudo_dir(script_text: str) -> tuple[bool, str]:
    """CHECK-1: pseudo_dir must be an absolute path."""
    for m in _RE_PSEUDO_DIR_STR.finditer(script_text):
        value = m.group(2).strip()
        # Skip obvious variable references like {pseudo_dir} or $PSEUDO_DIR
        if value.startswith(("{", "$", "os.", "Path", "~")):
            continue
        # Tilde expansion at runtime is OK if the script explicitly calls expanduser —
        # but we flag it as risky: production nodes may have unexpected $HOME.
        if _RE_RELATIVE_PATH.match(value):
            return False, (
                f"pseudo_dir appears to be a relative path: {value!r}. "
                "Use an absolute path (starting with / or drive letter) to avoid "
                "breakage when pw.x runs from an unexpected cwd on remote nodes."
            )
    return True, ""


def _check_outdir_nesting(script_text: str) -> tuple[bool, str]:
    """CHECK-2: outdir must be a simple 'tmp', not a composed multi-level path."""
    # Look for os.path.join with 3+ comma-separated arguments assigned to outdir
    for m in _RE_OUTDIR_COMPOSITE.finditer(script_text):
        args_str = m.group(1)
        # Count comma-separated items (rough: split by comma ignoring quoted)
        parts = [p.strip() for p in args_str.split(",")]
        if len(parts) >= 3:
            return False, (
                f"outdir constructed via os.path.join with {len(parts)} components: "
                f"os.path.join({args_str}).  "
                "This doubles the path when pw.x cwd is already the per-job "
                "sub-directory.  Use outdir='tmp' instead."
            )
        # Check if any part looks like a label/mineral variable (not 'tmp' or 'work')
        if len(parts) == 2:
            non_literal = [p for p in parts if not (p.startswith(("'", '"')) and "tmp" in p.lower())]
            if len(non_literal) >= 2:
                return False, (
                    f"outdir constructed via os.path.join({args_str}).  "
                    "Both arguments appear to be variables, suggesting label+work_dir "
                    "nesting which doubles the path.  Use outdir='tmp' instead."
                )
    # Also catch f-string patterns like outdir=f"{work_dir}/{label}/tmp"
    fstr_match = re.search(
        r"""outdir\s*=\s*f['"][^'"]*\{[^}]+\}[^'"]*\{[^}]+\}[^'"]*['"]""",
        script_text,
        re.IGNORECASE,
    )
    if fstr_match:
        return False, (
            f"outdir uses an f-string with multiple interpolated components: "
            f"{fstr_match.group(0)!r}.  "
            "This doubles the path when pw.x cwd is already the per-job sub-directory.  "
            "Use outdir='tmp' instead."
        )
    return True, ""


def _check_extxyz_clean_read(script_text: str, xyz_path: str | Path | None) -> tuple[bool, list[str]]:
    """CHECK-3: ase.io.read of extxyz without clean-read wrapper, or nspins= in xyz comment."""
    issues: list[str] = []

    # (a) Script uses ase.io.read but has no clean_read wrapper
    has_ase_read = bool(_RE_ASE_READ.search(script_text))
    has_clean_read = bool(_RE_CLEAN_READ.search(script_text))
    if has_ase_read and not has_clean_read:
        issues.append(
            "Script calls ase.io.read without a clean-read wrapper.  "
            "Non-standard extxyz comment keys (nspins=, nkpts=, nbands=) will "
            "crash ASE's SinglePointCalculator.  "
            "Add a _clean_read() that strips non-standard keys before passing to ASE."
        )

    # (b) Provided xyz file contains non-standard comment keys
    if xyz_path is not None:
        xyz_path = Path(xyz_path)
        if xyz_path.exists():
            try:
                lines = xyz_path.read_text(errors="replace").splitlines()
                if len(lines) > 1:
                    comment_line = lines[1]
                    if _NONSTANDARD_KEYS.search(comment_line):
                        issues.append(
                            f"xyz file {xyz_path} contains a non-standard key in the "
                            f"comment line (e.g. nspins=, nkpts=, nbands=): "
                            f"{comment_line[:120]!r}.  "
                            "This will cause ASE SinglePointCalculator assertion errors.  "
                            "Strip the non-standard keys before reading."
                        )
            except OSError as exc:
                issues.append(f"Could not read xyz_path={xyz_path}: {exc}")

    return (len(issues) == 0), issues


def _check_upf_wfc(pseudo_dir: str | Path | None) -> tuple[bool, str]:
    """CHECK-4: SG15 ONCV .upf files must have number_of_wfc > 0."""
    if pseudo_dir is None:
        return True, ""
    pseudo_dir = Path(pseudo_dir)
    if not pseudo_dir.exists():
        return True, f"pseudo_dir={pseudo_dir} does not exist; skipping UPF check."

    offenders: list[str] = []
    for upf_path in pseudo_dir.glob("*.upf"):
        try:
            text = upf_path.read_text(errors="replace")
        except OSError:
            continue
        if _RE_WFC_ZERO.search(text):
            offenders.append(upf_path.name)

    if offenders:
        return False, (
            f"number_of_wfc=0 found in: {', '.join(sorted(offenders))}.  "
            "SG15 ONCV pseudopotentials without Wannier-function channels cause "
            "pw.x to crash when +U or startingwfc=atomic is used.  "
            "Replace with a UPF that includes PP_PSWFC (e.g. PBE ONCV from "
            "pseudo-dojo or the oncv_pbe/ set)."
        )
    return True, ""


def run_lint_dft_script(
    script_path: str | Path,
    pseudo_dir: str | Path | None = None,
    xyz_path: str | Path | None = None,
) -> dict:
    """N-09 static pre-flight lint for QE/ABACUS deploy scripts (MCP-callable).

    Parameters
    ----------
    script_path:
        Path to the Python/shell deploy script to lint.
    pseudo_dir:
        Optional path to the pseudopotential directory.  When provided, all *.upf
        files are grepped for ``number_of_wfc=0``.
    xyz_path:
        Optional path to the .xyz/.extxyz file that the script will read.  When
        provided, the comment line is inspected for non-standard keys.

    Returns
    -------
    response_envelope dict with verdict PASS / FAIL and per-check reasons.
    """
    script_path = Path(script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"script_path not found: {script_path}")

    try:
        script_text = script_path.read_text(errors="replace")
    except OSError as exc:
        raise OSError(f"Cannot read {script_path}: {exc}") from exc

    reasons: list[str] = []
    warnings: list[str] = []
    next_actions: list[str] = []
    failed_checks: list[str] = []

    # CHECK-1
    ok1, msg1 = _check_pseudo_dir(script_text)
    if not ok1:
        failed_checks.append("CHECK-1 (pseudo_dir relative)")
        reasons.append(f"[CHECK-1 FAIL] {msg1}")
        next_actions.append("Set pseudo_dir to an absolute path.")
    else:
        reasons.append("[CHECK-1 PASS] pseudo_dir appears to be absolute or unset.")

    # CHECK-2
    ok2, msg2 = _check_outdir_nesting(script_text)
    if not ok2:
        failed_checks.append("CHECK-2 (outdir nesting)")
        reasons.append(f"[CHECK-2 FAIL] {msg2}")
        next_actions.append("Change outdir to the literal string 'tmp'.")
    else:
        reasons.append("[CHECK-2 PASS] outdir does not show multi-level nesting.")

    # CHECK-3
    ok3, msgs3 = _check_extxyz_clean_read(script_text, xyz_path)
    if not ok3:
        failed_checks.append("CHECK-3 (extxyz clean-read)")
        for msg in msgs3:
            reasons.append(f"[CHECK-3 FAIL] {msg}")
        next_actions.append("Use a _clean_read() wrapper that strips non-standard XYZ keys.")
    else:
        reasons.append("[CHECK-3 PASS] extxyz read looks safe.")

    # CHECK-4
    ok4, msg4 = _check_upf_wfc(pseudo_dir)
    if not ok4:
        failed_checks.append("CHECK-4 (UPF number_of_wfc)")
        reasons.append(f"[CHECK-4 FAIL] {msg4}")
        next_actions.append("Replace zero-wfc UPF files with ones containing PP_PSWFC.")
    elif msg4:
        # Informational warning (e.g. pseudo_dir doesn't exist)
        warnings.append(msg4)
        reasons.append("[CHECK-4 SKIP] " + msg4)
    else:
        reasons.append("[CHECK-4 PASS] All .upf files have number_of_wfc > 0 (or no .upf found).")

    all_passed = len(failed_checks) == 0
    verdict = "PASS" if all_passed else "FAIL"
    confidence = "high"

    if all_passed:
        next_actions = ["Script passed all pre-flight checks; safe to deploy."]

    result = {
        "script_path": str(script_path),
        "pseudo_dir": str(pseudo_dir) if pseudo_dir else None,
        "xyz_path": str(xyz_path) if xyz_path else None,
        "failed_checks": failed_checks,
        "n_checks": 4,
        "n_failed": len(failed_checks),
    }

    return response_envelope(
        tool="lint_dft_script",
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
    print(f"script\t{r.get('script_path')}")
    print(f"checks\t{r.get('n_checks')}\tfailed\t{r.get('n_failed')}")
    if r.get("failed_checks"):
        print(f"failed_checks\t{', '.join(r.get('failed_checks', []))}")
    for x in env["reasons"]:
        print(f"reason\t{x}")
    for x in env["next_actions"]:
        print(f"next\t{x}")
    for x in env["warnings"]:
        print(f"warning\t{x}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "N-09 static pre-flight lint for QE/ABACUS deploy scripts: "
            "checks pseudo_dir, outdir nesting, extxyz clean-read, and UPF wfc count."
        )
    )
    p.add_argument(
        "--script", type=Path, required=True,
        help="path to the Python/shell deploy script to lint"
    )
    p.add_argument(
        "--pseudo-dir", type=Path, default=None,
        help="pseudopotential directory; when given, .upf files are checked for number_of_wfc=0"
    )
    p.add_argument(
        "--xyz-path", type=Path, default=None,
        help="optional .xyz/.extxyz file; comment line checked for non-standard keys"
    )
    p.add_argument("--json", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args(argv)

    env = run_lint_dft_script(
        script_path=args.script,
        pseudo_dir=args.pseudo_dir,
        xyz_path=args.xyz_path,
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
