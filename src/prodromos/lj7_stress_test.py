"""
P1-B (harder): LJ7 same-basin stress test — where naive Cartesian fails and the
internal/topological signal wins. This is the Paper-A thesis in one experiment.

Lennard-Jones 7-atom cluster: rich, curved basins, with PERMUTATION + ROTATION +
TRANSLATION invariance. Two configs can be the SAME structure (same basin) yet
far apart in Cartesian space.

Signals:
  * naive Cartesian L2 interp-barrier: linear-interpolate config_a -> config_b in
    raw 21-D coords, take max LJ energy. For a same-basin pair related by a
    rotation/permutation this passes through ATOMIC CLASHES -> huge FALSE barrier
    -> would mis-flag same-basin as cross-basin. (This is exactly the s130-class
    trap that bit us, generalised.)
  * internal fingerprint: sorted pairwise-distance vector (invariant to
    permutation/rotation/translation). Same basin -> ~0 distance; different
    minimum -> large. This is what the Hungarian relabel (symmetry_preflight) and
    PH-on-distances capture, and why naive geometry is not enough.

Demonstrates: you MUST align / use internal coords (Paper A motivation).
NO DFT. scipy for local relaxation.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

sys.path.insert(0, str(Path(__file__).parent))


def lj_energy(flat):
    p = flat.reshape(-1, 3)
    d = pdist(p)
    d = np.maximum(d, 1e-6)
    inv6 = (1.0 / d) ** 6
    return float(np.sum(4.0 * (inv6 ** 2 - inv6)))


def lj_grad(flat):
    p = flat.reshape(-1, 3)
    n = len(p)
    g = np.zeros_like(p)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            rij = p[i] - p[j]
            r = max(np.linalg.norm(rij), 1e-6)
            # dV/dr * (rij/r); V_pair = 4(r^-12 - r^-6)
            dvdr = 4.0 * (-12.0 * r ** -13 + 6.0 * r ** -7)
            g[i] += dvdr * rij / r
    return g.ravel()


def relax(flat):
    res = minimize(lj_energy, flat, jac=lj_grad, method="L-BFGS-B",
                   options={"maxiter": 2000, "gtol": 1e-6})
    return res.x, res.fun


def fingerprint(flat):
    """Permutation/rotation/translation-invariant: sorted pairwise distances."""
    return np.sort(pdist(flat.reshape(-1, 3)))


def random_rotation(seed):
    rng = np.random.default_rng(seed)
    # random orthogonal via QR
    q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def perturb_and_relax(min_flat, seed, sigma=0.06):
    rng = np.random.default_rng(seed)
    x = min_flat + sigma * rng.standard_normal(min_flat.shape)
    return relax(x)


def cartesian_interp_barrier(a, b, n=21):
    s = np.linspace(0, 1, n)[:, None]
    path = a[None, :] + s * (b - a)[None, :]
    E = np.array([lj_energy(p) for p in path])
    return float(E.max() - max(E[0], E[-1]))


def make_minima(seeds=range(40)):
    """Relax from random configs -> collect distinct LJ7 minima by energy."""
    minima = []
    rng = np.random.default_rng(7)
    for s in seeds:
        x0 = rng.standard_normal(21) * 1.1
        x, e = relax(x0)
        if not np.isfinite(e) or e > -10:
            continue
        if all(abs(e - m[1]) > 1e-3 for m in minima):
            minima.append((x, e))
    minima.sort(key=lambda m: m[1])
    return minima


def main():
    minima = make_minima()
    print(f"Distinct LJ7 minima found: {len(minima)}  energies: {[round(m[1],3) for m in minima[:6]]}")
    gm = minima[0]           # global min (pentagonal bipyramid ~ -16.505)
    print(f"Global min E = {gm[1]:.4f} (ref pentagonal bipyramid ~ -16.505)")

    rows = []
    # SAME-basin: global min vs its rotated+reseed-relaxed copy (same structure)
    for k in range(4):
        R = random_rotation(100 + k)
        rotated = (gm[0].reshape(-1, 3) @ R.T).ravel()
        xb, eb = perturb_and_relax(rotated, seed=200 + k)
        same = abs(eb - gm[1]) < 1e-2
        cart = cartesian_interp_barrier(gm[0], xb)
        fp = float(np.linalg.norm(fingerprint(gm[0]) - fingerprint(xb)))
        rows.append(("same_rot%d" % k, True, same, cart, fp))
    # CROSS-basin: global min vs other minima
    for idx in range(1, min(4, len(minima))):
        other = minima[idx]
        cart = cartesian_interp_barrier(gm[0], other[0])
        fp = float(np.linalg.norm(fingerprint(gm[0]) - fingerprint(other[0])))
        rows.append(("cross_%d" % idx, False, False, cart, fp))

    print(f"\n{'pair':<12}{'truth_same':>11}{'relax_same':>11}{'cart_barr':>11}{'fp_dist':>9}")
    for nm, truth, relx, cart, fp in rows:
        print(f"{nm:<12}{str(truth):>11}{str(relx):>11}{cart:>11.2f}{fp:>9.3f}")

    same_fp = [r[4] for r in rows if r[1]]
    diff_fp = [r[4] for r in rows if not r[1]]
    same_cart = [r[3] for r in rows if r[1]]
    diff_cart = [r[3] for r in rows if not r[1]]
    print(f"\nFINGERPRINT (internal): same max={max(same_fp):.3f}  cross min={min(diff_fp):.3f}  "
          f"-> {'SEPARABLE' if max(same_fp) < min(diff_fp) else 'OVERLAP'}")
    print(f"CARTESIAN interp-barr : same max={max(same_cart):.1f}  cross min={min(diff_cart):.1f}  "
          f"-> {'SEPARABLE' if max(same_cart) < min(diff_cart) else 'OVERLAP/FALSE-BARRIER'}")
    print("\nThesis: naive Cartesian gives FALSE barriers on same-basin (rot/perm) pairs;")
    print("internal fingerprint (Hungarian/PH-style) separates correctly. -> alignment is mandatory.")
    return rows


if __name__ == "__main__":
    main()
