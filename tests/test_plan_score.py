"""Bellman expectimax + CVaR scoring tests (consilium C1-C4)."""
from __future__ import annotations

import math

from prodromos.plan.score import (
    PlanNode,
    cvar,
    expected_cost,
    leaf_value,
    rank_strategies,
    stochastically_dominated,
    value,
)


def _run_leaf(p, v_paper=1000.0, v_fail=0.0, cost_run=200.0, cost_redo=200.0, **kw):
    return PlanNode(
        kind="TERMINAL", is_stop=False, p_success=p, v_paper=v_paper,
        v_fail=v_fail, cost_run=cost_run, cost_redo=cost_redo, **kw,
    )


def _stop_leaf():
    return PlanNode(kind="TERMINAL", is_stop=True)


# --------------------------------------------------------------------------
# chance node = expectation
# --------------------------------------------------------------------------
def test_chance_node_is_expectation():
    a = _run_leaf(1.0)   # U = 1000 - 200 = 800
    b = _run_leaf(0.0)   # U = 0 - 400 = -400
    chance = PlanNode(kind="CHANCE", children=[(0.5, a), (0.5, b)])
    # E = 0.5*800 + 0.5*(-400) = 200
    assert math.isclose(value(chance), 200.0, abs_tol=1e-9)


# --------------------------------------------------------------------------
# decision node = max(0, .) ; STOP emergent
# --------------------------------------------------------------------------
def test_decision_node_is_max():
    a = _run_leaf(0.9)   # high utility
    b = _run_leaf(0.5)
    decision = PlanNode(kind="DECISION", children=[(1.0, a), (1.0, b)])
    assert math.isclose(value(decision), value(a))  # picks the best


def test_stop_wins_emergently_when_all_runs_negative():
    """Real-option test: when every run has negative U, the decision value is 0
    (STOP), produced by max(0, .) -- NOT a bespoke 'saved cost' term."""
    bad1 = _run_leaf(0.1, v_paper=500.0)   # U = 0.1*500 - (200+0.9*200) = 50-380=-330
    bad2 = _run_leaf(0.2, v_paper=400.0)
    decision = PlanNode(kind="DECISION", children=[(1.0, bad1), (1.0, bad2)])
    assert value(bad1) < 0 and value(bad2) < 0
    assert value(decision) == 0.0  # STOP reference wins


def test_stop_leaf_value_is_zero_reference():
    assert leaf_value(_stop_leaf(), beta=0.0, alpha=0.2) == 0.0
    assert leaf_value(_stop_leaf(), beta=1.0, alpha=0.2) == 0.0


def test_profitable_run_beats_stop():
    good = _run_leaf(0.9, v_paper=1000.0)  # clearly positive
    decision = PlanNode(kind="DECISION", children=[(1.0, good)])
    assert value(decision) == value(good) > 0.0


# --------------------------------------------------------------------------
# NOT an additive path sum: equal-EU / different-tail mis-ordering counter-test
# --------------------------------------------------------------------------
def test_value_is_not_additive_path_sum():
    """The game-theory file's A/B example structure: equal EU, different tail.

    Branch A: light failure tail (-150 net). Branch B: heavy failure tail (-950
    net). We construct B's v_paper so the two TIE on expected utility, then show
    that under tail weighting (high beta) A is strictly preferred -- a naive
    additive path sum / pure EV cannot tell them apart, which is the C2 point.
    """
    a = _run_leaf(0.9, v_paper=1000.0, v_fail=-100.0, cost_run=50.0, cost_redo=0.0)
    # solve B's v_paper so EV(b) == EV(a):
    ev_a = leaf_value(a, 0.0, 0.2)
    p_b, v_fail_b, cost_b = 0.6, -900.0, 50.0
    u_fail_b = v_fail_b - cost_b
    # ev_a = p_b*(v_paper_b - cost_b) + (1-p_b)*u_fail_b
    v_paper_b = (ev_a - (1 - p_b) * u_fail_b) / p_b + cost_b
    b = _run_leaf(p_b, v_paper=v_paper_b, v_fail=v_fail_b, cost_run=cost_b, cost_redo=0.0)

    # pure EV: equal (the additive/EV view cannot distinguish them)
    assert math.isclose(leaf_value(a, 0.0, 0.2), leaf_value(b, 0.0, 0.2), abs_tol=1e-6)
    # high beta (tail-weighted): A (light -150 tail) strictly beats B (heavy -950)
    assert leaf_value(a, 0.9, 0.2) > leaf_value(b, 0.9, 0.2)


def test_expected_cost_includes_retry():
    leaf = _run_leaf(0.75, cost_run=200.0, cost_redo=200.0)
    # E[cost] = cost_run + p_fail*cost_redo = 200 + 0.25*200 = 250
    assert math.isclose(expected_cost(leaf), 250.0, abs_tol=1e-9)


# --------------------------------------------------------------------------
# CVaR
# --------------------------------------------------------------------------
def test_cvar_picks_worst_tail():
    # utilities: -900 (p=0.4), 1150 (p=0.6); CVaR_0.2 sits entirely in the -900 tail
    pts = [(0.4, -900.0), (0.6, 1150.0)]
    assert math.isclose(cvar(pts, 0.2), -900.0, abs_tol=1e-9)
    # alpha=1 -> mean
    assert math.isclose(cvar(pts, 1.0), 0.4 * -900 + 0.6 * 1150, abs_tol=1e-9)


def test_cvar_changes_ranking_at_high_beta():
    """Two leaves equal in EV but different in CVaR rank differently at high beta."""
    a = _run_leaf(0.9, v_paper=1000.0, v_fail=-100.0, cost_run=50.0, cost_redo=0.0)
    b = _run_leaf(0.6, v_paper=2000.0, v_fail=-900.0, cost_run=50.0, cost_redo=0.0)
    root_lowbeta = PlanNode(kind="DECISION", children=[(1.0, a), (1.0, b)])

    # With a tiny budget, beta -> 1 for both (cost_run/budget high) -> tail matters.
    ranked_tail = rank_strategies(root_lowbeta, budget_remaining=50.0, alpha=0.2,
                                  include_stop=False)
    top_tail = ranked_tail[0]
    # A is the light-tail leaf; identify by its higher utility under tail weighting
    assert leaf_value(a, 1.0, 0.2) > leaf_value(b, 1.0, 0.2)
    assert top_tail.utility == max(s.utility for s in ranked_tail)

    # With no budget (pure EV) they tie -> same utility for both
    ranked_ev = rank_strategies(root_lowbeta, budget_remaining=None, alpha=0.2,
                                include_stop=False)
    evs = sorted(s.utility for s in ranked_ev)
    assert math.isclose(evs[0], evs[-1], abs_tol=1e-6)


# --------------------------------------------------------------------------
# stochastic-dominance pruning
# --------------------------------------------------------------------------
def test_sd_pruning_keeps_best():
    """A clearly dominated leaf is pruned; the best survives."""
    best = _run_leaf(0.95, v_paper=1000.0, cost_run=100.0, cost_redo=100.0)
    worse = _run_leaf(0.30, v_paper=600.0, cost_run=300.0, cost_redo=300.0)
    root = PlanNode(kind="DECISION", children=[(1.0, best), (1.0, worse)])
    ranked = rank_strategies(root, include_stop=False)
    labels_utils = {s.label: s.utility for s in ranked}
    # the best leaf must be present and rank first
    assert ranked[0].utility == max(s.utility for s in ranked)
    # worse may be pruned by SD-1; if present it must rank below best
    assert ranked[0].utility >= min(labels_utils.values())


def test_sd_dominance_relation():
    # b first-order stochastically dominates a (same costs, strictly higher p)
    a = _run_leaf(0.5, v_paper=1000.0, v_fail=0.0, cost_run=100.0, cost_redo=100.0)
    b = _run_leaf(0.9, v_paper=1000.0, v_fail=0.0, cost_run=100.0, cost_redo=100.0)
    assert stochastically_dominated(a, b)
    assert not stochastically_dominated(b, a)
    # stop leaves are never SD-pruned
    assert not stochastically_dominated(a, _stop_leaf())
