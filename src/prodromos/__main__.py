"""Prodromos CLI dispatcher.

Maps a subcommand to a dotted module and runs that module as ``__main__``
via :mod:`runpy`, forwarding the remaining argv. Each target module already
defines its own ``argparse`` interface under ``if __name__ == "__main__"``.
"""
from __future__ import annotations

import runpy
import sys

# subcommand -> dotted module
SUBCOMMANDS: dict[str, str] = {
    "electron-parity": "prodromos.electron_parity_gate",
    "spin-collapse": "prodromos.spin_collapse_verdict",
    "saddle-proximity": "prodromos.saddle_proximity_gate",
    "endpoint-provenance": "prodromos.endpoint_provenance_gate",
    "symmetry-preflight": "prodromos.symmetry_preflight_general",
    "vfe-preflight": "prodromos.vfe_neb_preflight",
    "magnetic-parser": "prodromos.magnetic_output_parser",
    "magnetic-endpoint": "prodromos.magnetic_endpoint_gate",
    "magnetic-band": "prodromos.magnetic_band_gate",
    "magnetic-recommend": "prodromos.magnetic_recommendation",
    "multi-endpoint": "prodromos.multi_endpoint_enumeration",
    "soap-cluster": "prodromos.soap_cluster_minima",
    "adaptive-neb": "prodromos.adaptive_neb_planner",
    "neb-advisor": "prodromos.neb_method_advisor",
    "gp-neb": "prodromos.gp_neb_surrogate",
    "master-equation": "prodromos.master_equation_kinetics",
    "external-reference": "prodromos.external_reference_gate",
    "lint-dft-script": "prodromos.lint_dft_script",
    "h-barrier-readiness": "prodromos.h_barrier_paper_readiness",
    "from-inputs": "prodromos.from_inputs",
    "plan": "prodromos.plan.cli",
}


def _print_list() -> None:
    print("prodromos -- $0 DFT NEB / saddle-search pre-flight diagnostic gates")
    print()
    print("Usage: prodromos <subcommand> [args...]")
    print()
    print("Subcommands:")
    width = max(len(s) for s in SUBCOMMANDS)
    for sub, module in sorted(SUBCOMMANDS.items()):
        print(f"  {sub:<{width}}  {module}")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("--help", "-h", "list"):
        _print_list()
        return 0

    sub = args[0]
    rest = args[1:]
    module = SUBCOMMANDS.get(sub)
    if module is None:
        sys.stderr.write(f"prodromos: unknown subcommand '{sub}'\n")
        sys.stderr.write("Run 'prodromos --help' for the list of subcommands.\n")
        return 2

    sys.argv = [f"prodromos {sub}", *rest]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
