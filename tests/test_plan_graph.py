"""Tests for the policy-graph data structures and assert_dag."""
from __future__ import annotations

import pytest

from prodromos.plan.graph import (
    Edge,
    GraphError,
    Node,
    PolicyGraph,
    always,
    assert_dag,
    verdict_in,
)
from prodromos.plan.policy import POLICY_GRAPH


def _terminal(node_id: str, verdict: str = "GO") -> Node:
    return Node(id=node_id, kind="terminal", terminal=verdict)


def test_valid_linear_graph_passes():
    nodes = {
        "g": Node(
            id="g", kind="gate", gate="electron-parity",
            edges=(Edge("leaf", always()),),
        ),
        "leaf": _terminal("leaf"),
    }
    graph = PolicyGraph(nodes=nodes, root="g")
    assert_dag(graph)  # no raise


def test_self_loop_is_a_cycle():
    nodes = {
        "g": Node(
            id="g", kind="gate", gate="electron-parity",
            edges=(Edge("g", always()),),  # self-loop
        ),
    }
    graph = PolicyGraph(nodes=nodes, root="g")
    with pytest.raises(GraphError, match="self-loop"):
        assert_dag(graph)


def test_two_node_cycle_detected():
    nodes = {
        "a": Node(id="a", kind="gate", gate="electron-parity",
                  edges=(Edge("b", always()),)),
        "b": Node(id="b", kind="gate", gate="endpoint-provenance",
                  edges=(Edge("a", always()),)),
    }
    graph = PolicyGraph(nodes=nodes, root="a")
    with pytest.raises(GraphError, match="cycle"):
        assert_dag(graph)


def test_dangling_edge_target():
    nodes = {
        "a": Node(id="a", kind="gate", gate="electron-parity",
                  edges=(Edge("missing", always()),)),
    }
    graph = PolicyGraph(nodes=nodes, root="a")
    with pytest.raises(GraphError, match="unknown target"):
        assert_dag(graph)


def test_terminal_with_outgoing_edge_rejected():
    nodes = {
        "a": Node(id="a", kind="gate", gate="electron-parity",
                  edges=(Edge("leaf", always()),)),
        "leaf": Node(id="leaf", kind="terminal", terminal="GO",
                     edges=(Edge("a", always()),)),
    }
    graph = PolicyGraph(nodes=nodes, root="a")
    with pytest.raises(GraphError, match="terminal"):
        assert_dag(graph)


def test_non_terminal_without_edges_rejected():
    nodes = {"a": Node(id="a", kind="gate", gate="electron-parity")}
    graph = PolicyGraph(nodes=nodes, root="a")
    with pytest.raises(GraphError, match="no outgoing edges"):
        assert_dag(graph)


def test_unknown_root_rejected():
    graph = PolicyGraph(nodes={"a": _terminal("a")}, root="nope")
    with pytest.raises(GraphError, match="root"):
        assert_dag(graph)


def test_unreachable_node_rejected():
    nodes = {
        "a": Node(id="a", kind="gate", gate="electron-parity",
                  edges=(Edge("leaf", always()),)),
        "leaf": _terminal("leaf"),
        "orphan": _terminal("orphan", "NO-GO"),
    }
    graph = PolicyGraph(nodes=nodes, root="a")
    with pytest.raises(GraphError, match="unreachable"):
        assert_dag(graph)


def test_bad_terminal_verdict_rejected_at_construction():
    with pytest.raises(ValueError, match="not in"):
        Node(id="x", kind="terminal", terminal="MAYBE")


def test_choice_node_requires_options():
    with pytest.raises(ValueError, match="options"):
        Node(id="c", kind="choice", edges=(Edge("z", always()),))


def test_gate_node_requires_gate_name():
    with pytest.raises(ValueError, match="must name a gate"):
        Node(id="g", kind="gate", edges=(Edge("z", always()),))


def test_predicate_helpers():
    p = verdict_in("A", "B")
    assert p({"verdict": "A"})
    assert not p({"verdict": "C"})


def test_shipped_policy_is_a_valid_dag():
    # The module-level POLICY_GRAPH is validated at import; re-assert explicitly.
    assert_dag(POLICY_GRAPH)
    assert POLICY_GRAPH.root in POLICY_GRAPH.nodes
