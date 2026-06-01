import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prodromos.spin_collapse_verdict import run_spin_collapse_verdict, DEFAULT_THRESHOLD


def test_pyrite_like_collapsed_is_nspin1_ok():
    # pyrite V_Fe+H: tiny residual M_abs ~0.03 uB over 31 Fe -> ~0.001 uB/Fe -> collapsed
    env = run_spin_collapse_verdict(mabs=0.03, n_tm=31)
    assert env["verdict"] == "NSPIN1_OK"
    assert env["confidence"] == "high"
    assert env["result"]["collapsed"] is True
    assert env["result"]["mabs_per_tm"] < DEFAULT_THRESHOLD
    assert env["next_actions"] == ["nspin=1 production OK; nspin=2 control optional"]


def test_pentlandite_like_persists_is_nspin2_required():
    # pentlandite: M_abs=127.5 uB over 71 Fe -> ~1.80 uB/Fe -> persists
    env = run_spin_collapse_verdict(mabs=127.5, n_tm=71)
    assert env["verdict"] == "NSPIN2_REQUIRED"
    assert env["confidence"] == "high"
    assert env["result"]["collapsed"] is False
    assert abs(env["result"]["mabs_per_tm"] - 127.5 / 71) < 1e-9
    assert "nspin=2 production + per-atom starting_magnetization from relaxed AFM" in env["next_actions"]
    assert any("magnetic_endpoint_gate" in a for a in env["next_actions"])


def test_mack_like_collapsed_is_nspin1_ok():
    # mackinawite V_Fe+H: moment collapses, M_abs ~0.5 uB over 32 Fe -> ~0.016 uB/Fe
    env = run_spin_collapse_verdict(mabs=0.5, n_tm=32)
    assert env["verdict"] == "NSPIN1_OK"
    assert env["result"]["collapsed"] is True


def test_exactly_at_threshold_persists():
    # at exactly threshold -> NOT collapsed (criterion is strict <)
    env = run_spin_collapse_verdict(mabs_per_tm=DEFAULT_THRESHOLD)
    assert env["result"]["collapsed"] is False
    assert env["verdict"] == "NSPIN2_REQUIRED"


def test_just_below_threshold_collapsed():
    env = run_spin_collapse_verdict(mabs_per_tm=DEFAULT_THRESHOLD - 1e-6)
    assert env["result"]["collapsed"] is True
    assert env["verdict"] == "NSPIN1_OK"


def test_direct_mabs_per_tm_input():
    env = run_spin_collapse_verdict(mabs_per_tm=1.80)
    assert env["verdict"] == "NSPIN2_REQUIRED"
    assert env["result"]["mabs_per_tm"] == 1.80
    assert env["result"]["mabs"] is None
    assert env["result"]["n_tm"] is None


def test_custom_threshold():
    # 0.5 uB/Fe collapses under default 0.30? no -> persists; but with threshold=1.0 -> collapsed
    env = run_spin_collapse_verdict(mabs_per_tm=0.5, threshold=1.0)
    assert env["verdict"] == "NSPIN1_OK"
    assert env["result"]["threshold"] == 1.0


def test_odd_parity_adds_smearing_caveat_on_collapse():
    env = run_spin_collapse_verdict(mabs=0.03, n_tm=31, parity="odd")
    assert env["verdict"] == "NSPIN1_OK"
    assert any("smearing" in w.lower() for w in env["warnings"])


def test_missing_inputs_raises():
    with pytest.raises(ValueError):
        run_spin_collapse_verdict(mabs=1.0)  # no n_tm, no mabs_per_tm
    with pytest.raises(ValueError):
        run_spin_collapse_verdict()


def test_bad_n_tm_raises():
    with pytest.raises(ValueError):
        run_spin_collapse_verdict(mabs=1.0, n_tm=0)


def test_negative_mabs_uses_abs_with_warning():
    env = run_spin_collapse_verdict(mabs=-127.5, n_tm=71)
    assert env["verdict"] == "NSPIN2_REQUIRED"
    assert env["result"]["mabs"] == 127.5
    assert any("negative" in w.lower() for w in env["warnings"])


def test_mismatch_mabs_per_tm_warns():
    env = run_spin_collapse_verdict(mabs=10.0, n_tm=10, mabs_per_tm=5.0)
    # explicit mabs_per_tm wins (5.0), derived would be 1.0
    assert env["result"]["mabs_per_tm"] == 5.0
    assert any("disagrees" in w for w in env["warnings"])


def test_envelope_has_stable_keys():
    env = run_spin_collapse_verdict(mabs=0.03, n_tm=31)
    assert set(env) == {
        "tool", "version", "status", "verdict", "confidence",
        "reasons", "next_actions", "artifacts", "warnings", "result",
    }
    assert env["tool"] == "spin_collapse_verdict"
    assert env["status"] == "ok"
