"""
MECP finder (Bearpark-Robb-Schlegel 1994 gradient projection) — M3.K core.

For a codim-1 spin-multiplicity crossing (NO derivative coupling without SOC,
per the math consilium MAGNETIC_NEB_CONSILIUM_2026-05-29.md), the relevant
"barrier" when the MEP changes spin sheet is the MINIMUM ENERGY CROSSING POINT,
not an ordinary saddle. Bearpark-Robb projected gradient:

    branching vector  x̂ = ∇(E_A − E_B)/|∇(E_A − E_B)|        (1D for spin crossing)
    f1 = 2 (E_A − E_B) x̂                 # drive onto the seam (close the gap)
    f2 = (I − x̂ x̂ᵀ) ∇E_A                 # minimise energy ON the seam
    g_eff = f1 + f2  ;  x ← x − η g_eff
    converged: E_A = E_B (on seam) AND f2 = 0

This is the rigorous magnetic analogue of the climbing image: where NEB-AGM's
climber detects the max lies on a spin seam (spin_split role), it switches to
this MECP search instead of an ordinary CI ascent.

Validated on a 2D analytic two-sheet toy (seam = a line; MECP = lowest point on it).
NO DFT.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------
# 2D analytic two-sheet PES (spin sheets A=HS, B=LS)
# ---------------------------------------------------------------
@dataclass
class TwoSheet2D:
    """Two paraboloid sheets that cross along a line; A min left, B min right."""
    a: float = 1.0
    xa: float = -1.0     # sheet A (HS) minimum x
    xb: float = +1.0     # sheet B (LS) minimum x
    c: float = 0.5       # B offset (eV)
    m_A: float = 1.67
    m_B: float = 1.13

    def E_A(self, p):
        p = np.atleast_2d(p)
        return self.a * (p[:, 0] - self.xa) ** 2 + p[:, 1] ** 2

    def E_B(self, p):
        p = np.atleast_2d(p)
        return self.a * (p[:, 0] - self.xb) ** 2 + p[:, 1] ** 2 + self.c

    def g_A(self, p):
        p = np.atleast_1d(p).astype(float)
        return np.array([2 * self.a * (p[0] - self.xa), 2 * p[1]])

    def g_B(self, p):
        p = np.atleast_1d(p).astype(float)
        return np.array([2 * self.a * (p[0] - self.xb), 2 * p[1]])

    def _E_A1(self, p):
        return float(self.E_A(np.atleast_2d(p))[0])

    def _E_B1(self, p):
        return float(self.E_B(np.atleast_2d(p))[0])

    def seam_x(self):
        """E_A=E_B: a(x-xa)² = a(x-xb)² + c -> solve linear in x."""
        # a[(x-xa)²-(x-xb)²] = c ; (x-xa)²-(x-xb)² = (xb-xa)(2x-xa-xb)
        # a (xb-xa)(2x-xa-xb) = c
        return ((self.c / (self.a * (self.xb - self.xa))) + self.xa + self.xb) / 2.0

    def mecp_analytic(self):
        """Min-energy crossing point: on seam (x=seam_x), minimise over y -> y=0."""
        xs = self.seam_x()
        p = np.array([xs, 0.0])
        return p, self._E_A1(p)


@dataclass
class MECPResult:
    x: np.ndarray
    energy: float
    gap: float
    converged: bool
    steps: int


def find_mecp(pes, x0, eta=0.05, max_steps=2000, gap_tol=1e-5, fperp_tol=1e-4):
    """Bearpark-Robb projected-gradient MECP search on a two-sheet PES.

    pes must expose _E_A1, _E_B1, g_A, g_B.
    """
    x = np.asarray(x0, float).copy()
    gap = fperp = np.inf
    it = 0
    for it in range(max_steps):
        ea, eb = pes._E_A1(x), pes._E_B1(x)
        ga, gb = pes.g_A(x), pes.g_B(x)
        dg = ga - gb
        nd = np.linalg.norm(dg)
        if nd < 1e-12:
            xhat = np.zeros_like(dg)
        else:
            xhat = dg / nd
        gap = ea - eb
        f1 = 2.0 * gap * xhat                          # close the gap (onto seam)
        f2 = ga - np.dot(ga, xhat) * xhat              # minimise on the seam
        fperp = np.linalg.norm(f2)
        g_eff = f1 + f2
        if abs(gap) < gap_tol and fperp < fperp_tol:
            break
        x = x - eta * g_eff
    converged = abs(gap) < gap_tol and fperp < fperp_tol
    return MECPResult(x=x, energy=0.5 * (pes._E_A1(x) + pes._E_B1(x)),
                      gap=float(gap), converged=converged, steps=it + 1)


def demo():
    pes = TwoSheet2D()
    p_an, E_an = pes.mecp_analytic()
    print(f"Two-sheet 2D toy: seam_x={pes.seam_x():.4f}")
    print(f"  analytic MECP: x={p_an}  E={E_an:.5f}")
    # start from sheet-A minimum
    res = find_mecp(pes, x0=[pes.xa, 0.5])
    print(f"  Bearpark-Robb: x=[{res.x[0]:.4f},{res.x[1]:.4f}]  E={res.energy:.5f}  "
          f"gap={res.gap:.2e}  conv={res.converged}  steps={res.steps}")
    err = np.linalg.norm(res.x - p_an)
    print(f"  |x_MECP - analytic| = {err:.2e}")
    return res


if __name__ == "__main__":
    demo()
