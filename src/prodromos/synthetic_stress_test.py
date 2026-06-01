"""
P1-B: Synthetic same-basin stress test with KNOWN ground truth (ROADMAP Phase 1).

The retrospective validation (P0-B) ran on real Fe-S where ground truth is
inferred. Here we CONSTRUCT controlled pairs on Muller-Brown where we KNOW
whether two endpoints share a basin, and measure how well the diagnostics
separate same-basin (artifact) from true-MEP (different-basin) pairs.

Two signals (both from the existing toolkit):
  * L2 interp-barrier: max V along linear A->B interpolation, minus endpoints.
    Same-basin -> ~0 (stays in the bowl); true-MEP -> real barrier.
  * L3 ensemble-PH: persistence diagram of a perturbation cloud around each
    endpoint; same_basin_score via bottleneck distance.

Output: confusion matrix + separation stats -> strengthens Paper A.
NO DFT.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prodromos.admm_neb_prototype import V_MB, MB_MINIMA
from prodromos.ph_neb_diagnostic import persistence_from_cloud, same_basin_score

RNG_SEED_BASE = 12345  # Math.random unavailable in some envs; deterministic seeds


def interp_barrier(a, b, n=21):
    """L2 signal: max V along linear interpolation minus higher endpoint."""
    s = np.linspace(0, 1, n)[:, None]
    path = a[None, :] + s * (b - a)[None, :]
    V = np.array([V_MB(p) for p in path])
    return float(V.max() - max(V[0], V[-1]))


def ensemble_cloud(center, n=120, sigma=0.12, seed=0):
    rng = np.random.default_rng(seed)
    pts = center[None, :] + sigma * rng.standard_normal((n, center.shape[0]))
    vals = np.array([V_MB(p) for p in pts])
    return pts, vals


def ph_same_basin_score(a, b, seed_a, seed_b, tau=10.0):
    """L3 signal: PH bottleneck-based same_basin_score of two endpoint clouds."""
    pa, va = ensemble_cloud(a, seed=seed_a)
    pb, vb = ensemble_cloud(b, seed=seed_b)
    da = persistence_from_cloud(pa, va)
    db = persistence_from_cloud(pb, vb)
    return same_basin_score(da, db, tau=tau)


def build_cases():
    """Controlled pairs with ground-truth same_basin flag."""
    A, B, C = MB_MINIMA["A"], MB_MINIMA["B"], MB_MINIMA["C"]
    cases = []
    # SAME-basin: endpoint + small in-basin displacement (several magnitudes/dirs)
    k = 0
    for base, name in [(A, "A"), (B, "B"), (C, "C")]:
        for j, d in enumerate([0.05, 0.10, 0.15]):
            for sign in (+1, -1):
                k += 1
                delta = np.array([d * sign, d * sign * 0.5])
                cases.append(dict(label=f"same_{name}_{j}_{sign}", a=base, b=base + delta,
                                  same_basin=True, seed_a=RNG_SEED_BASE + k,
                                  seed_b=RNG_SEED_BASE + 1000 + k))
    # TRUE-MEP: different basins
    for (p, q, nm) in [(A, B, "AB"), (A, C, "AC"), (B, C, "BC"),
                       (B, A, "BA"), (C, A, "CA"), (C, B, "CB")]:
        k += 1
        cases.append(dict(label=f"diff_{nm}", a=p, b=q, same_basin=False,
                          seed_a=RNG_SEED_BASE + k, seed_b=RNG_SEED_BASE + 2000 + k))
    return cases


def evaluate(barrier_thresh=5.0, score_thresh=0.5):
    """Classify each case. Predicted same-basin if interp_barrier < thresh.
    (PH score reported alongside as a second, independent signal.)"""
    cases = build_cases()
    rows = []
    for c in cases:
        bar = interp_barrier(c["a"], c["b"])
        score = ph_same_basin_score(c["a"], c["b"], c["seed_a"], c["seed_b"])
        pred_same = bar < barrier_thresh
        rows.append(dict(label=c["label"], truth=c["same_basin"],
                         interp_barrier=bar, ph_score=score, pred_same=pred_same))
    # confusion on the L2 interp-barrier signal
    TP = sum(r["pred_same"] and r["truth"] for r in rows)       # same predicted same
    TN = sum(not r["pred_same"] and not r["truth"] for r in rows)
    FP = sum(r["pred_same"] and not r["truth"] for r in rows)
    FN = sum(not r["pred_same"] and r["truth"] for r in rows)
    return rows, dict(TP=TP, TN=TN, FP=FP, FN=FN)


def main():
    rows, cm = evaluate()
    print(f"{'case':<14}{'truth':>7}{'interp_bar':>12}{'ph_score':>10}{'pred':>7}")
    for r in rows:
        ok = "ok" if (r["pred_same"] == r["truth"]) else "MISS"
        print(f"{r['label']:<14}{str(r['truth']):>7}{r['interp_barrier']:>12.2f}"
              f"{r['ph_score']:>10.3f}{str(r['pred_same']):>7}  {ok}")
    n = len(rows)
    acc = (cm["TP"] + cm["TN"]) / n
    print(f"\nConfusion (L2 interp-barrier, thresh=5.0):  TP={cm['TP']} TN={cm['TN']} "
          f"FP={cm['FP']} FN={cm['FN']}  acc={acc:.2f} (n={n})")
    same = [r["interp_barrier"] for r in rows if r["truth"]]
    diff = [r["interp_barrier"] for r in rows if not r["truth"]]
    print(f"interp_barrier  same-basin: max={max(same):.2f}   true-MEP: min={min(diff):.2f}  "
          f"-> {'SEPARABLE' if max(same) < min(diff) else 'OVERLAP'}")
    sscore = [r["ph_score"] for r in rows if r["truth"]]
    dscore = [r["ph_score"] for r in rows if not r["truth"]]
    print(f"ph_score        same-basin: mean={np.mean(sscore):.3f}   true-MEP: mean={np.mean(dscore):.3f}")
    return rows, cm


if __name__ == "__main__":
    main()
