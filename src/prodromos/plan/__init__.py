"""Prodromos ``plan`` engine -- orchestrator over the $0 pre-flight gates.

First increment (route mode + architecture). A single declarative ``PolicyGraph``
(see :mod:`prodromos.plan.policy`) is walked by two interpreters
(:mod:`prodromos.plan.interpret`):

- ``route`` (EXECUTE): on a gate node actually run the gate's ``run_*`` on inputs
  adapted from the tm-spec ``case`` document; follow the edge for the *real*
  verdict. On a choice node pick the myopic-best option. Returns one recommended
  next $0 step plus a trace.
- ``tree`` (SIMULATE): STUB in this increment -- the scoring / Bellman-expectimax /
  calibration layer is the next increment (see :mod:`prodromos.plan.interpret`).

The single source of truth is the ``PolicyGraph``, NOT a pre-built decision tree;
``route`` is an *executor* of that graph, not a greedy walk of a simulated tree
(see PRODROMOS_PLAN_CONSILIUM_cs.md S5).
"""
from __future__ import annotations

PLAN_GRAPH_VERSION = "2026-06-01"
ENGINE_NAME = "prodromos"
ENGINE_VERSION = "0.1.0"

__all__ = ["PLAN_GRAPH_VERSION", "ENGINE_NAME", "ENGINE_VERSION"]
