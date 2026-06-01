"""Beta-Binomial calibration tests (consilium C5)."""
from __future__ import annotations

from pathlib import Path

from prodromos.plan.calibrate import (
    Calibrator,
    default_calibrator,
    make_key,
)


def test_lower_bound_below_mean():
    """The credible lower bound must sit strictly below the posterior mean."""
    calib = default_calibrator()
    key = make_key("Fe-S|Pa-3", "band", 2, ("ASYMMETRIC",))
    lo = calib.p_success_lower(key)
    mean = calib.p_success_mean(key)
    assert 0.0 <= lo < mean <= 1.0


def test_update_from_outcomes_shifts_counters():
    """Successes raise the lower bound; failures lower it."""
    calib = default_calibrator()
    key = make_key("Fe-S|Pa-3", "band", 2, ())
    base = calib.p_success_lower(key)

    # feed several successes -> lower bound rises
    recs = [{"key": key, "success": True} for _ in range(8)]
    n = calib.update_from_outcomes(recs)
    assert n > 0
    after_success = calib.p_success_lower(key)
    assert after_success > base

    # now feed failures -> lower bound drops back down
    calib2 = default_calibrator()
    calib2.update_from_outcomes([{"key": key, "success": False} for _ in range(8)])
    after_fail = calib2.p_success_lower(key)
    assert after_fail < base


def test_hierarchical_backoff():
    """An unseen exact key backs off to a coarser cell that has evidence."""
    calib = default_calibrator()
    # accumulate evidence at the method-family level only (no verdicts, no nspin)
    family_key = make_key("Fe-S|Pa-3", "band", None, ())
    calib.update_from_outcomes([{"key": family_key, "success": True} for _ in range(10)])

    # query a MORE specific key (with nspin + verdicts) that has no direct data;
    # it must back off and benefit from the family evidence.
    specific = make_key("Fe-S|Pa-3", "band", 2, ("ASYMMETRIC",))
    backed_off = calib.p_success_lower(specific)

    fresh = default_calibrator().p_success_lower(specific)
    assert backed_off > fresh  # backoff pooled the family successes


def test_confidence_to_p_is_not_nominal():
    """confidence labels map through the Beta table, NOT high=0.9 nominal."""
    calib = default_calibrator()
    p_high = calib.confidence_to_p_lower("high")
    p_med = calib.confidence_to_p_lower("medium")
    p_low = calib.confidence_to_p_lower("low")
    assert p_high > p_med > p_low
    # high is a LOWER credible bound, so it is well below the nominal 0.9
    assert p_high < 0.9


def test_no_internal_campaign_data_in_source():
    """Guard against private campaign outcomes being hardcoded in the engine.

    The calibration module must not embed session-specific mineral campaign
    identifiers (mack/pent/marc/pyr/greig) -- those are private data and must
    enter only via update_from_outcomes from the caller's own history.
    """
    plan_dir = Path(__file__).resolve().parents[1] / "src" / "prodromos" / "plan"
    # the engine/data modules added in this increment must be campaign-clean
    for name in ("calibrate.py", "score.py", "priors.py"):
        text = (plan_dir / name).read_text(encoding="utf-8").lower()
        for token in ("mack", "marc", "greig", "pentland", " pent", "pyrite"):
            assert token not in text, f"{name} leaks campaign token {token!r}"


def test_lower_quantile_monotone_in_quantile():
    """A higher lower-quantile knob yields a higher (less conservative) bound."""
    a = Calibrator(lower_quantile=0.05)
    b = Calibrator(lower_quantile=0.25)
    key = make_key("X", "band", 2, ())
    assert b.p_success_lower(key) > a.p_success_lower(key)
