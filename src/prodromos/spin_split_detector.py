"""
$0 two-sheet toy + spin_split detector (Phase 3, task M3.K).

Implements the magnetic-NEB consilium synthesis:
  * CS toy:        two smooth spin sheets E0 (LS, m~1.13), E1 (HS, m~1.67);
                   spin-blind surrogate = lower-envelope leak -> band-collapse.
  * statmech:      per-image magnetization diagnostic (total + abs), threshold
                   Δabs_adjacent > 0.5 μB -> SHEET_CROSSING; endpoint gate.
  * game-theorist: spin_split role discriminator = LOW geom force + HIGH spin
                   incoherence with neighbours (vs `stuck` = HIGH geom force).
  * math:          seam is codim-1 KINK (spin-multiplicity, no SOC coupling);
                   V=min_s E_s -> kink, NOT a conical intersection.

The detector is a GATE: it detects + halts + recommends, it does NOT apply a
geometric escape to a magnetic problem (that would mask it — consilius consensus).

NO DFT. Pure numpy. Calibrated on the marcasite numbers (m 1.67 vs 1.13,
ΔE 174 meV) so the demo is directly relatable.

Reusable entry point: `magnetic_band_diagnostic(...)` -> wire into NEB-AGM.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# statmech thresholds (μB) — calibrated; marc Δabs=0.65, Δtotal=0.54
DELTA_ABS_ADJ = 0.5      # adjacent |Δ abs_mag| -> sheet crossing
DELTA_TOTAL_ENDPOINT = 0.3  # endpoint |Δ total_mag| -> endpoints on different sheets
FGEOM_LOW = 0.10         # geom fmax below this = "geometrically converged"


# ----------------------------------------------------------------------
# Two-sheet toy PES (1D reaction coordinate x in [0,1] = proton hop)
# ----------------------------------------------------------------------
@dataclass
class TwoSheetPES:
    """Two smooth magnetic sheets over a 1D reaction coordinate.

    Sheet 0 (LS): low-spin, m ~ 1.13 μB.   Sheet 1 (HS): high-spin, m ~ 1.67 μB.
    Tuned so HS is the ground state near x=0 and LS near x=1  -> the ground
    sheet SWITCHES along the path = real spin crossover (the marcasite case).
    """
    A0: float = 0.30    # LS barrier amplitude (eV)
    A1: float = 0.30    # HS barrier amplitude
    dE: float = 0.174   # HS-LS offset at x=0 (eV)  ~ marc 174 meV
    tilt: float = 0.40  # linear tilt that makes LS win at large x (eV)
    m_LS: float = 1.13
    m_HS: float = 1.67
    collapse_depth: float = 0.45  # how far the spin-blind surrogate dips below (eV)
    seam_width: float = 0.10

    def E_sheet(self, x, s):
        x = np.asarray(x, float)
        if s == 0:   # LS
            return self.A0 * np.sin(np.pi * x) ** 2 + self.tilt * x
        else:        # HS
            return self.A1 * np.sin(np.pi * x) ** 2 + self.dE

    def active_sheet(self, x):
        """Greedy 'SCF' choice: argmin_s E_s(x) — the lower sheet."""
        return (self.E_sheet(x, 1) < self.E_sheet(x, 0)).astype(int)

    def V_true(self, x):
        """min-envelope (kink at the seam)."""
        return np.minimum(self.E_sheet(x, 0), self.E_sheet(x, 1))

    def seam_x(self):
        """Crossing point E0=E1 -> A*sin²+tilt*x = A*sin²+dE -> tilt*x=dE."""
        xs = np.linspace(0, 1, 4001)
        d = self.E_sheet(xs, 1) - self.E_sheet(xs, 0)
        sign = np.sign(d)
        idx = np.where(np.diff(sign) != 0)[0]
        return float(xs[idx[0]]) if len(idx) else float("nan")

    def V_blind(self, x):
        """Spin-blind smooth surrogate: lower envelope with a Gaussian dip at the
        seam (a smooth regressor fit to BOTH sectors leaks below both branches —
        math consilium: band-collapse is inevitable, not a bug)."""
        x = np.asarray(x, float)
        xs = self.seam_x()
        dip = self.collapse_depth * np.exp(-((x - xs) / self.seam_width) ** 2)
        return self.V_true(x) - dip

    def magmom(self, x):
        """Total magnetization per image from the active (greedy) sheet."""
        s = self.active_sheet(x)
        return np.where(s == 1, self.m_HS, self.m_LS)


# ----------------------------------------------------------------------
# The detector (reusable; wire into NEB-AGM as gate-role `spin_split`)
# ----------------------------------------------------------------------
@dataclass
class BandDiagnostic:
    roles: list
    d_total_adj: list
    d_abs_adj: list
    sheet_crossing: bool
    endpoint_split: bool
    crossing_edge: int           # index i of the i<->i+1 edge with the jump (-1 none)
    recommendation: str
    flags: list = field(default_factory=list)


def magnetic_band_diagnostic(mag_total, mag_abs, geom_fmax, energies,
                             delta_abs=DELTA_ABS_ADJ,
                             delta_total=DELTA_TOTAL_ENDPOINT,
                             fgeom_low=FGEOM_LOW) -> BandDiagnostic:
    """Detect magnetic sheet-splitting along an NEB/string band.

    Inputs (per image, length N):
        mag_total : signed total magnetization
        mag_abs   : absolute magnetization (Σ|m_a|)
        geom_fmax : per-image geometric (perpendicular) force magnitude
        energies  : per-image energy (for sawtooth check)

    GATE semantics: classifies each image, raises band-level flags, and
    RECOMMENDS a protocol. It never prescribes a geometric escape for a
    magnetic discontinuity.
    """
    mag_total = np.asarray(mag_total, float)
    mag_abs = np.asarray(mag_abs, float)
    geom_fmax = np.asarray(geom_fmax, float)
    energies = np.asarray(energies, float)
    N = len(mag_total)

    d_total_adj = np.abs(np.diff(mag_total))
    d_abs_adj = np.abs(np.diff(mag_abs))

    sheet_crossing = bool(np.any(d_abs_adj > delta_abs))
    crossing_edge = int(np.argmax(d_abs_adj)) if sheet_crossing else -1
    endpoint_split = bool(abs(mag_total[-1] - mag_total[0]) > delta_total)

    # sawtooth: an interior image whose energy exceeds BOTH neighbours
    # (kink) co-located with a magnetization jump
    sawtooth = False
    for i in range(1, N - 1):
        if energies[i] > energies[i - 1] and energies[i] > energies[i + 1]:
            local_jump = max(d_abs_adj[i - 1], d_abs_adj[i])
            if local_jump > delta_abs:
                sawtooth = True

    # per-image role
    roles = []
    for i in range(N):
        nb = []
        if i > 0:
            nb.append(d_abs_adj[i - 1])
        if i < N - 1:
            nb.append(d_abs_adj[i])
        spin_incoh = max(nb) if nb else 0.0
        g = geom_fmax[i]
        if spin_incoh > delta_abs and g < fgeom_low:
            roles.append("spin_split")          # magnetic: low geom force + high mag mismatch
        elif g >= fgeom_low and spin_incoh <= delta_abs:
            roles.append("stuck")               # geometric: high geom force, smooth mag
        elif g >= fgeom_low and spin_incoh > delta_abs:
            roles.append("mixed")               # both — resolve magnetism first
        else:
            roles.append("ok")

    flags = []
    if endpoint_split:
        flags.append("ENDPOINT_SPLIT: endpoints on different spin sheets -> single-sheet NEB invalid as-is")
    if sheet_crossing:
        flags.append(f"SHEET_CROSSING at edge {crossing_edge}<->{crossing_edge+1} (Δabs={d_abs_adj[crossing_edge]:.2f} μB)")
    if sawtooth:
        flags.append("SAWTOOTH: energy kink co-located with mag jump (band glued from different sheets)")

    # recommendation (consilium decision logic)
    if not sheet_crossing and not endpoint_split:
        rec = "OK_SINGLE_SHEET: no magnetic discontinuity; spin-blind barrier valid (statmech criterion)"
    elif endpoint_split:
        rec = ("RUN_DFT_DIAGNOSTIC: both endpoints at BOTH M values. "
               "If ΔE<kT & one metastable -> single-sheet constrained-M NEB (spin-IDPP init). "
               "If both true minima -> two-segment + MECP (Bearpark-Robb), decompose barrier geom+spin.")
    else:
        rec = ("TWO_SEGMENT_OR_SINGLE_SHEET: sheet crossing mid-band. Force single magnetic "
               "state (constrained tot_magnetization) OR split at the seam + MECP. Do NOT trust "
               "spin-blind barrier near the seam.")

    return BandDiagnostic(
        roles=roles,
        d_total_adj=d_total_adj.tolist(),
        d_abs_adj=d_abs_adj.tolist(),
        sheet_crossing=sheet_crossing,
        endpoint_split=endpoint_split,
        crossing_edge=crossing_edge,
        recommendation=rec,
        flags=flags,
    )


# ----------------------------------------------------------------------
# Toy band generators (what a converged NEB would look like)
# ----------------------------------------------------------------------
def band_marc_like(n=9):
    """Marcasite-like: endpoints on different sheets, greedy SCF -> mag jump at
    seam, all images geometrically relaxed (low geom force)."""
    pes = TwoSheetPES()
    x = np.linspace(0, 1, n)
    mag_total = pes.magmom(x)                       # jumps at seam
    mag_abs = mag_total.copy()                       # collinear toy: abs≈|total|
    energies = pes.V_true(x)
    geom_fmax = np.full(n, 0.03)                      # all geometrically converged
    geom_fmax[0] = geom_fmax[-1] = 0.0
    return pes, x, mag_total, mag_abs, geom_fmax, energies


def band_pyrite_like(n=9):
    """Pyrite-like (single sheet, but ONE image geometrically stuck on a ridge):
    smooth magnetization, no crossing; the failure is geometric, not magnetic."""
    pes = TwoSheetPES(dE=2.0, tilt=0.0)              # HS far above -> LS everywhere
    x = np.linspace(0, 1, n)
    mag_total = pes.magmom(x)                         # all LS -> smooth
    mag_abs = mag_total.copy()
    energies = pes.E_sheet(x, 0)
    geom_fmax = np.full(n, 0.02)
    geom_fmax[n // 2] = 0.8                           # one image rolling off the ridge
    geom_fmax[0] = geom_fmax[-1] = 0.0
    return pes, x, mag_total, mag_abs, geom_fmax, energies


def band_clean(n=9):
    """Fully converged single sheet: detector must say OK (no false positive)."""
    pes = TwoSheetPES(dE=2.0, tilt=0.0)
    x = np.linspace(0, 1, n)
    mag_total = pes.magmom(x)
    mag_abs = mag_total.copy()
    energies = pes.E_sheet(x, 0)
    geom_fmax = np.full(n, 0.02)
    geom_fmax[0] = geom_fmax[-1] = 0.0
    return pes, x, mag_total, mag_abs, geom_fmax, energies


def _print_case(name, gen):
    pes, x, mt, ma, gf, E = gen()
    diag = magnetic_band_diagnostic(mt, ma, gf, E)
    print(f"\n{'='*68}\n{name}\n{'='*68}")
    print(f"  seam x = {pes.seam_x():.3f}   endpoints total_mag: {mt[0]:.2f} -> {mt[-1]:.2f} μB")
    print(f"  roles: {diag.roles}")
    print(f"  sheet_crossing={diag.sheet_crossing} (edge {diag.crossing_edge})  endpoint_split={diag.endpoint_split}")
    for f in diag.flags:
        print(f"  ⚑ {f}")
    print(f"  -> {diag.recommendation.split(':')[0]}")
    return diag


def main():
    print("Two-sheet toy + spin_split detector (M3.K) — calibrated on marcasite numbers")
    _print_case("CASE A — marcasite-like (real spin crossover)", band_marc_like)
    _print_case("CASE B — pyrite-like (geometric stuck, single sheet)", band_pyrite_like)
    _print_case("CASE C — clean single sheet (negative control)", band_clean)

    # plot (band-collapse visualization for the marc case)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path
        pes, x, *_ = band_marc_like(60)
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        xs = np.linspace(0, 1, 200)
        ax[0].plot(xs, pes.E_sheet(xs, 0), "--", label="sheet LS (m=1.13)")
        ax[0].plot(xs, pes.E_sheet(xs, 1), "--", label="sheet HS (m=1.67)")
        ax[0].plot(xs, pes.V_true(xs), "k-", lw=2, label="V_true = min-envelope (KINK)")
        ax[0].plot(xs, pes.V_blind(xs), "r-", lw=1.5, label="V_blind (spin-blind surrogate)")
        ax[0].axvline(pes.seam_x(), color="gray", ls=":", label="seam (codim-1)")
        ax[0].set(xlabel="reaction coord x", ylabel="E (eV)", title="Two-sheet PES: spin-blind collapse")
        ax[0].legend(fontsize=7)
        xb = np.linspace(0, 1, 9)
        ax[1].step(range(9), pes.magmom(xb), where="mid", marker="o")
        ax[1].axhline(pes.m_LS, color="gray", ls=":")
        ax[1].axhline(pes.m_HS, color="gray", ls=":")
        ax[1].set(xlabel="image", ylabel="total mag (μB)", title="Magnetization jump at seam -> spin_split")
        plt.tight_layout()
        out = Path(__file__).parent / "neb_agm_results" / "spin_split_toy.png"
        out.parent.mkdir(exist_ok=True)
        plt.savefig(out, dpi=130)
        print(f"\nsaved {out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
