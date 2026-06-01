"""Declarative policy-graph data structures + static DAG validation.

The graph is the single source of truth (consilium S1): nodes reference gates or
choice-points, edges carry pure predicates over a gate's verdict / a choice
option. Two interpreters (route=EXECUTE, tree=SIMULATE) share this one graph.

A YAML mini-DSL was rejected: edge predicates are code, so the graph lives as
Python literals in :mod:`prodromos.plan.policy` where mypy/IDE see it and a
static validator can run on the imported object.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

NodeKind = Literal["gate", "choice", "terminal"]

# Terminal verdicts the planner can stop on.
TERMINAL_VERDICTS = frozenset({"GO", "NO-GO", "INVESTIGATE", "NEEDS_DATA"})


@dataclass(frozen=True)
class Edge:
    """A directed transition.

    ``predicate(ctx)`` is a pure boolean over a small context dict built by the
    interpreter: it carries the firing gate's ``verdict`` (gate node) or the
    chosen ``option`` (choice node), plus ``result`` for advisory fields. Keep
    predicates to a single field -- complex logic belongs INSIDE the gate
    (consilium S8.6), the edge only decides *where* to go.
    """

    dst: str
    predicate: Callable[[dict], bool]
    label: str = ""


@dataclass(frozen=True)
class Node:
    """A graph node.

    kind="gate": ``gate`` is a registry subcommand; ``inputs`` lists the
        tm-spec adapter input keys the gate's run_fn consumes (declared so the
        route memo key is a projection, not the whole case).
    kind="choice": ``options`` enumerates the branch labels (nspin {1,2},
        method {band,dimer,string}, ...). route picks the myopic-best option;
        tree branches over all (next increment).
    kind="terminal": a leaf verdict in TERMINAL_VERDICTS.
    """

    id: str
    kind: NodeKind
    gate: str | None = None
    inputs: list[str] = field(default_factory=list)
    options: list[str] | None = None
    edges: tuple[Edge, ...] = ()
    terminal: str | None = None      # for kind="terminal"
    what: str = ""

    def __post_init__(self) -> None:
        if self.kind == "gate" and not self.gate:
            raise ValueError(f"gate node {self.id!r} must name a gate")
        if self.kind == "choice" and not self.options:
            raise ValueError(f"choice node {self.id!r} must list options")
        if self.kind == "terminal":
            if self.terminal not in TERMINAL_VERDICTS:
                raise ValueError(
                    f"terminal node {self.id!r} verdict {self.terminal!r} "
                    f"not in {sorted(TERMINAL_VERDICTS)}"
                )


@dataclass(frozen=True)
class PolicyGraph:
    """A DAG of nodes keyed by id, plus the root entry node."""

    nodes: dict[str, Node]
    root: str

    def node(self, node_id: str) -> Node:
        return self.nodes[node_id]


class GraphError(ValueError):
    """Raised by :func:`assert_dag` on a structurally invalid graph."""


def assert_dag(graph: PolicyGraph) -> None:
    """Validate the graph is a well-formed DAG. Raises :class:`GraphError`.

    Checks (consilium S1, S8.1):
      - root exists;
      - every edge target exists (no dangling ``dst``);
      - no cycles, INCLUDING self-loops (a method-ladder modelled as a self-loop
        is the classic cycle that would make tree-mode expand forever);
      - every non-terminal node has at least one outgoing edge;
      - terminal nodes have no outgoing edges.
    """
    nodes = graph.nodes
    if graph.root not in nodes:
        raise GraphError(f"root {graph.root!r} is not a node")

    for nid, node in nodes.items():
        if node.id != nid:
            raise GraphError(f"node key {nid!r} != node.id {node.id!r}")
        for e in node.edges:
            if e.dst not in nodes:
                raise GraphError(f"node {nid!r} edge -> {e.dst!r}: unknown target")
            if e.dst == nid:
                raise GraphError(f"node {nid!r} has a self-loop edge (cycle)")
        if node.kind == "terminal":
            if node.edges:
                raise GraphError(f"terminal node {nid!r} must have no outgoing edges")
        elif not node.edges:
            raise GraphError(f"non-terminal node {nid!r} has no outgoing edges")

    # Cycle detection via DFS colouring (white/grey/black).
    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = dict.fromkeys(nodes, WHITE)

    def visit(nid: str, stack: list[str]) -> None:
        colour[nid] = GREY
        for e in nodes[nid].edges:
            if colour[e.dst] == GREY:
                cycle = " -> ".join([*stack, nid, e.dst])
                raise GraphError(f"cycle detected: {cycle}")
            if colour[e.dst] == WHITE:
                visit(e.dst, [*stack, nid])
        colour[nid] = BLACK

    for nid in nodes:
        if colour[nid] == WHITE:
            visit(nid, [])

    # Reachability from root (not fatal, but a disconnected node is a design bug).
    seen: set[str] = set()
    frontier = [graph.root]
    while frontier:
        cur = frontier.pop()
        if cur in seen:
            continue
        seen.add(cur)
        frontier.extend(e.dst for e in nodes[cur].edges)
    unreachable = set(nodes) - seen
    if unreachable:
        raise GraphError(f"unreachable nodes from root {graph.root!r}: {sorted(unreachable)}")


# --- small predicate helpers (keep edge predicates one-field, declarative) ---
def verdict_in(*verdicts: str) -> Callable[[dict], bool]:
    allowed = frozenset(verdicts)
    return lambda ctx: ctx.get("verdict") in allowed


def verdict_not_in(*verdicts: str) -> Callable[[dict], bool]:
    blocked = frozenset(verdicts)
    return lambda ctx: ctx.get("verdict") not in blocked


def option_is(option: str) -> Callable[[dict], bool]:
    return lambda ctx: ctx.get("option") == option


def always() -> Callable[[dict], bool]:
    return lambda ctx: True
