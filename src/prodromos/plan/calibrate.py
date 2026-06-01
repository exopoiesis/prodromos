"""Beta-Binomial success calibration with hierarchical backoff (consilium C5).

The tree scorer needs a probability that a committed expensive run reaches its
target, given a *chemistry signature* and the production choices (method, nspin)
plus the cheap-gate verdicts already observed. ``confidence`` labels
(low/medium/high) are NOT probabilities and must not be read at face value
("high" != 0.9); they are mapped through the same Beta-Binomial machinery.

Design (consilium CS S7 + info-theory S2):
  * A table of Beta(alpha, beta) counters keyed by
        key = (chemistry_signature, method, nspin, *verdicts)
    where each "success" increments alpha and each "failure" increments beta.
  * ``p_success_lower(key)`` returns the LOWER credible bound (default 5th
    percentile of the Beta posterior) -- conservative, so a cell seen twice with
    2/2 successes does NOT report ~1.0 but a wide-interval lower bound. Scoring
    on the lower bound is the C5 requirement.
  * Hierarchical backoff: an exact key with too little evidence is pooled with
    progressively coarser keys -- drop the verdicts, then drop nspin, then drop
    method, finally a global prior. This is partial pooling by chemistry family
    (the "V_Fe chemistry universal" lesson), not per-mineral overfitting.

CALIBRATION DATA POLICY (no leak):
  The table is SEEDED ONLY with *generic, methodological* priors. The two public
  layer hit-rates documented in docs/EVIDENCE_FRAMEWORK_V2_COMPLETE.md
  (L0 ~30%, L1 ~80%) are methodological summary statistics, not session-specific
  campaign outcomes, and are used purely to shape the global/method-family prior.
  Concrete per-campaign outcomes (which mineral converged at what barrier) are
  NEVER hardcoded here -- that would be a private-data leak. Instead, a user
  calibrates on their own history by passing records to
  :meth:`Calibrator.update_from_outcomes`.

The implementation is dependency-free: the Beta lower bound uses a small,
well-conditioned numerical inverse-CDF (regularised incomplete beta via
bisection on the regularised incomplete beta function) so the package keeps its
"no SciPy required" footprint. If SciPy is importable it is used for accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Generic, methodological priors (public summary stats only -- see module docs)
# --------------------------------------------------------------------------
# Pseudo-counts, not real outcomes. A weak Beta prior (small total) so a handful
# of real ``update_from_outcomes`` records dominate quickly.
#
# Global prior for "a well-posed pre-flight GO leads to a useful result". This is
# anchored to the PUBLIC, methodological L1 (Hungarian symmetry) hit-rate
# ~80% documented in docs/EVIDENCE_FRAMEWORK_V2_COMPLETE.md -- a summary
# statistic about the methodology, NOT any campaign's realised outcome. Encoded
# as a weak Beta so a handful of real update_from_outcomes records dominate.
# Beta(4,1): mean ~0.80, 5th-percentile lower bound ~0.40 (conservative, C5):
# a default GO is plausible but the wide interval keeps the engine honest until
# the user supplies calibration history.
_GLOBAL_PRIOR = (4.0, 1.0)

# Methodological layer hit-rates from docs/EVIDENCE_FRAMEWORK_V2_COMPLETE.md
# ("Hit rate by layer: L0 ~30%, L1 ~80%"). These are GENERIC summary statistics,
# expressed as weak pseudo-counts (total ~5) so they inform the family prior
# without pretending to be a large calibrated sample.
_LAYER_PRIORS: dict[str, tuple[float, float]] = {
    # L0 cubane/pristine triage ~30% -> Beta(1.5, 3.5)
    "L0": (1.5, 3.5),
    # L1 Hungarian symmetry ~80% -> Beta(4.0, 1.0)
    "L1": (4.0, 1.0),
}

# Map a pre-flight gate verdict family to the layer whose prior best describes it
# (methodological, not campaign data). Used only to seed the method-family prior.
_VERDICT_LAYER_HINT: dict[str, str] = {
    # symmetry / distinct-endpoint verdicts are the L1 Hungarian family
    "ASYMMETRIC": "L1",
    "SYMMETRIC": "L1",
    "MARGINAL": "L1",
}

# How wide a confidence label is, expressed as a Beta prior. Calibrated, not
# nominal: "high" maps to a strong-but-not-certain prior, "low" to near-coin.
# These too are generic methodological mappings, refinable via update_from_outcomes.
_CONFIDENCE_PRIOR: dict[str, tuple[float, float]] = {
    "high": (4.0, 1.0),    # ~0.80 mean, lower bound well below
    "medium": (2.0, 1.5),  # ~0.57 mean
    "low": (1.0, 1.5),     # ~0.40 mean
}


# --------------------------------------------------------------------------
# Beta lower credible bound (dependency-free fallback)
# --------------------------------------------------------------------------
def _beta_ppf(q: float, a: float, b: float) -> float:
    """Inverse CDF (quantile) of Beta(a, b) at probability ``q`` in [0, 1].

    Uses SciPy when available; otherwise bisection on the regularised incomplete
    beta function ``I_x(a, b)`` (continued-fraction evaluation). Accurate to
    ~1e-7, which is far finer than the calibration uncertainty itself.
    """
    if not (0.0 < q < 1.0):
        return 0.0 if q <= 0.0 else 1.0
    try:  # prefer SciPy for accuracy if present
        from scipy.stats import beta as _scipy_beta  # type: ignore

        return float(_scipy_beta.ppf(q, a, b))
    except Exception:  # noqa: BLE001 -- fall through to pure-python
        pass

    lo, hi = 0.0, 1.0
    for _ in range(80):  # 80 bisections -> ~2^-80 resolution, well past float eps
        mid = 0.5 * (lo + hi)
        if _reg_inc_beta(mid, a, b) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _reg_inc_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function I_x(a, b) via continued fraction."""
    import math

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta) / a
    # Lentz's algorithm for the continued fraction of the incomplete beta.
    # Use the symmetry I_x(a,b) = 1 - I_{1-x}(b,a) for faster convergence.
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b)
    return 1.0 - front_swapped(x, a, b, ln_beta)


def front_swapped(x: float, a: float, b: float, ln_beta: float) -> float:
    import math

    front = math.exp(b * math.log(1.0 - x) + a * math.log(x) - ln_beta) / b
    return front * _betacf(1.0 - x, b, a)


def _betacf(x: float, a: float, b: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


# --------------------------------------------------------------------------
# key model
# --------------------------------------------------------------------------
def make_key(
    chemistry_signature: str,
    method: str | None,
    nspin: int | None,
    verdicts: tuple[str, ...] = (),
) -> tuple:
    """Build a canonical, hashable calibration key.

    ``verdicts`` are the decision-relevant pre-flight gate verdicts (sorted for
    canonicity). ``method``/``nspin`` may be None at coarser backoff levels.
    """
    return (
        chemistry_signature or "*",
        method or "*",
        int(nspin) if nspin is not None else None,
        tuple(sorted(verdicts)),
    )


def _backoff_chain(key: tuple) -> list[tuple]:
    """Yield progressively coarser keys: drop verdicts -> nspin -> method -> global.

    Order (consilium C5 / CS S7): exact -> drop verdicts -> drop nspin ->
    drop method -> global prior. Each coarser cell pools more evidence.
    """
    sig, method, nspin, verdicts = key
    chain = [key]
    if verdicts:
        chain.append((sig, method, nspin, ()))
    if nspin is not None:
        chain.append((sig, method, None, ()))
    if method != "*":
        chain.append((sig, "*", None, ()))
    chain.append(("*", "*", None, ()))  # global
    # de-duplicate preserving order
    seen: set = set()
    out = []
    for k in chain:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


@dataclass
class _Counter:
    alpha: float = 0.0
    beta: float = 0.0


@dataclass
class Calibrator:
    """Beta-Binomial success-rate table with hierarchical backoff.

    Counters hold ONLY the increments beyond the (generic) prior; the prior is
    added at query time. Seed by methodological priors at construction; refine
    with :meth:`update_from_outcomes` on the user's own campaign history.
    """

    # query parameters
    lower_quantile: float = 0.05      # 5th percentile credible lower bound
    min_evidence: float = 4.0         # alpha+beta increments needed before a cell
    #                                   is trusted on its own (else keep backing off)

    _counts: dict[tuple, _Counter] = field(default_factory=dict)

    # ----------------------------------------------------------------- update
    def update_from_outcomes(self, records: list[dict]) -> int:
        """Increment counters from an EXTERNAL list of outcome records.

        Each record: ``{"key": <key tuple or dict>, "success": bool}`` or the
        flat form ``{"chemistry_signature":..., "method":..., "nspin":...,
        "verdicts": [...], "success": bool}``. Returns the number applied.

        This is the ONLY way concrete campaign outcomes enter the table -- they
        are supplied by the caller from their own history, never hardcoded here.
        """
        n = 0
        for rec in records:
            success = bool(rec.get("success"))
            key = rec.get("key")
            if key is None:
                key = make_key(
                    rec.get("chemistry_signature", "*"),
                    rec.get("method"),
                    rec.get("nspin"),
                    tuple(rec.get("verdicts", ()) or ()),
                )
            elif isinstance(key, dict):
                key = make_key(
                    key.get("chemistry_signature", "*"),
                    key.get("method"),
                    key.get("nspin"),
                    tuple(key.get("verdicts", ()) or ()),
                )
            else:
                key = tuple(key)
            # increment EVERY cell in the backoff chain so coarse cells pool too
            # (partial pooling): a family-level success also nudges the
            # method-family and global priors.
            for ck in _backoff_chain(key):
                c = self._counts.setdefault(ck, _Counter())
                if success:
                    c.alpha += 1.0
                else:
                    c.beta += 1.0
                n += 1
        return n

    # ------------------------------------------------------------ prior model
    def _prior_for(self, key: tuple) -> tuple[float, float]:
        """Generic methodological prior for a key (no campaign data)."""
        sig, method, nspin, verdicts = key
        # If any verdict maps to a known methodological layer, blend that layer's
        # prior; else fall back to the global Laplace prior.
        for v in verdicts:
            layer = _VERDICT_LAYER_HINT.get(v)
            if layer and layer in _LAYER_PRIORS:
                return _LAYER_PRIORS[layer]
        return _GLOBAL_PRIOR

    def _posterior(self, key: tuple) -> tuple[float, float]:
        """Beta(alpha, beta) posterior = prior + accumulated counts for ``key``."""
        a0, b0 = self._prior_for(key)
        c = self._counts.get(key)
        if c is None:
            return a0, b0
        return a0 + c.alpha, b0 + c.beta

    def _evidence(self, key: tuple) -> float:
        c = self._counts.get(key)
        return 0.0 if c is None else (c.alpha + c.beta)

    # ------------------------------------------------------------------ query
    def _resolve(self, key: tuple) -> tuple[float, float, tuple]:
        """Pick the finest backoff cell that has enough evidence; return its posterior."""
        chain = _backoff_chain(key)
        for ck in chain:
            if self._evidence(ck) >= self.min_evidence:
                a, b = self._posterior(ck)
                return a, b, ck
        # nothing meets the bar -> use the coarsest (global) posterior, which at
        # least carries the generic prior + whatever pooled increments exist.
        ck = chain[-1]
        a, b = self._posterior(ck)
        return a, b, ck

    def p_success_lower(self, key: tuple) -> float:
        """Lower credible bound on P(success) for ``key`` (the C5 scoring value)."""
        a, b, _ = self._resolve(key)
        return _beta_ppf(self.lower_quantile, a, b)

    def p_success_mean(self, key: tuple) -> float:
        """Posterior mean P(success) (for diagnostics; scoring uses the lower bound)."""
        a, b, _ = self._resolve(key)
        return a / (a + b)

    def credible_interval(self, key: tuple) -> tuple[float, float]:
        a, b, _ = self._resolve(key)
        lo = _beta_ppf(self.lower_quantile, a, b)
        hi = _beta_ppf(1.0 - self.lower_quantile, a, b)
        return lo, hi

    # ------------------------------------------------- confidence -> P mapping
    def confidence_to_p_lower(self, confidence: str | None) -> float:
        """Map a low/med/high confidence LABEL to a lower-bound probability.

        Through the Beta table, NOT a nominal "high = 0.9". A label's prior may
        also be sharpened by update_from_outcomes via the special signature
        ``conf:<label>``.
        """
        label = (confidence or "low").lower()
        a0, b0 = _CONFIDENCE_PRIOR.get(label, _CONFIDENCE_PRIOR["low"])
        ck = ("conf:" + label, "*", None, ())
        c = self._counts.get(ck)
        if c is not None:
            a0, b0 = a0 + c.alpha, b0 + c.beta
        return _beta_ppf(self.lower_quantile, a0, b0)


def default_calibrator() -> Calibrator:
    """A fresh Calibrator seeded only with generic methodological priors."""
    return Calibrator()
