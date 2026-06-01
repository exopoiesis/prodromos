"""Tree (SIMULATE) mode integration tests over a tm-spec case."""
from __future__ import annotations

from pathlib import Path

import pytest

from prodromos.plan.calibrate import default_calibrator, make_key
from prodromos.plan.emit import to_envelope, to_preflight_block
from prodromos.plan.interpret import walk
from prodromos.plan.policy import POLICY_GRAPH

_EXAMPLE = (
    Path(__file__).resolve().parents[2] / "tm-spec" / "examples" / "preflight_example.tm.yaml"
)

tm_validator = pytest.importorskip("tm_spec.validator")


def _load_example() -> dict:
    return tm_validator.load_doc(_EXAMPLE)[0]


def test_tree_returns_sorted_nonempty_strategies():
    result = walk(POLICY_GRAPH, _load_example(), mode="tree")
    assert result.mode == "tree"
    strategies = result.strategies
    assert strategies, "strategies[] must be non-empty"
    # sorted by utility descending
    utils = [s.utility for s in strategies]
    assert utils == sorted(utils, reverse=True)
    # every strategy carries the required scored fields
    for s in strategies:
        assert 0.0 <= s.p_success <= 1.0
        assert isinstance(s.cvar_usd, float)
        assert isinstance(s.utility, float)
        assert isinstance(s.paper_grade_reachable, bool)


def test_tree_preflight_block_has_scored_strategies():
    result = walk(POLICY_GRAPH, _load_example(), mode="tree")
    block = to_preflight_block(result)
    strategies = block["plan"]["strategies"]
    assert strategies
    for s in strategies:
        for field in ("label", "method", "expected_cost_usd", "p_success",
                      "cvar_usd", "utility", "paper_grade_reachable"):
            assert field in s, f"strategy missing {field}"


def test_tree_preflight_block_validates_against_tm_spec_0_3():
    """The tree-mode preflight block (with strategies[]) must validate in a doc."""
    doc = _load_example()
    result = walk(POLICY_GRAPH, doc, mode="tree")
    block = to_preflight_block(result)
    doc["preflight"] = block
    schema_errs, rule_issues = tm_validator.validate_doc(doc)
    errors = [f"{loc}: {msg}" for loc, msg in schema_errs]
    errors += [msg for level, msg in rule_issues if level == "error"]
    assert not errors, f"tree preflight block failed 0.3 validation: {errors}"


def test_tree_envelope_surfaces_strategies():
    result = walk(POLICY_GRAPH, _load_example(), mode="tree")
    env = to_envelope(result)
    assert env["tool"] == "plan"
    assert env["result"]["mode"] == "tree"
    assert env["result"]["strategies"]


def test_budget_makes_scoring_cvar_aware():
    """Supplying a tight budget changes the (CVaR-weighted) utilities."""
    doc = _load_example()
    no_budget = walk(POLICY_GRAPH, doc, mode="tree")
    tight = walk(POLICY_GRAPH, doc, mode="tree", budget_usd=300.0)

    def best_run_util(res):
        runs = [s for s in res.strategies if not s.is_stop]
        return max(s.utility for s in runs)

    # the tail penalty under a tight budget lowers the best run's utility.
    assert best_run_util(tight) < best_run_util(no_budget)


def test_top_k_truncates():
    doc = _load_example()
    full = walk(POLICY_GRAPH, doc, mode="tree")
    k2 = walk(POLICY_GRAPH, doc, mode="tree", top_k=2)
    assert len(k2.strategies) <= 2
    assert len(k2.strategies) <= len(full.strategies)


def test_beam_and_pruning_keep_the_best_strategy():
    """Beam/SD pruning must not drop the top-utility strategy."""
    doc = _load_example()
    full = walk(POLICY_GRAPH, doc, mode="tree")
    best_label = full.strategies[0].label
    # re-run with a smaller top_k; the #1 strategy must still be present and first.
    k1 = walk(POLICY_GRAPH, doc, mode="tree", top_k=1)
    assert k1.strategies[0].label == best_label


def test_calibration_outcomes_shift_tree_verdict():
    """Feeding real outcomes (caller history) moves p_success in the strategies."""
    doc = _load_example()
    base = walk(POLICY_GRAPH, doc, mode="tree")
    base_p = max(s.p_success for s in base.strategies if not s.is_stop)

    # The case is Fe-H-S Pa-3, band, nspin-various; feed successes at the family key.
    calib = default_calibrator()
    family = make_key("Fe-H-S|Pa-3", "band", None, ())
    calib.update_from_outcomes([{"key": family, "success": True} for _ in range(12)])
    tuned = walk(POLICY_GRAPH, doc, mode="tree", calib=calib)
    tuned_p = max(s.p_success for s in tuned.strategies if not s.is_stop)
    assert tuned_p > base_p


def test_stop_present_as_reference():
    doc = _load_example()
    result = walk(POLICY_GRAPH, doc, mode="tree")
    assert any(s.is_stop for s in result.strategies), "STOP reference must be listed"
    stop = next(s for s in result.strategies if s.is_stop)
    assert stop.utility == 0.0
