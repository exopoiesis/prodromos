"""Prodromos (πρόδρομος, "the one who runs ahead").

A $0 pre-flight / post-flight diagnostic gate for expensive DFT NEB and
saddle-point searches on Fe-S minerals. It runs ahead of the costly A100
calculation and reports whether the path is worth taking -- catching
same-basin endpoints, spin-sheet discontinuities, MLIP out-of-distribution
artifacts, optimizer/spring failures, and multi-endpoint H landscapes before
any DFT is spent.

See README.md and docs/ for the methodology (Evidence Framework v2, L0->L6).
"""

__version__ = "0.1.0"
