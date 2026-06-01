"""Graph interpreters: route (EXECUTE) and tree (SIMULATE stub).

route is NOT a greedy walk of a pre-built tree (consilium D1, the P0 killer):
on a gate node it actually runs the gate's ``run_fn`` on inputs adapted from the
case and follows the edge for the FACTUAL verdict; on a choice node it picks the
myopic-best option. tree (next increment) instead branches over prior outcome
distributions and scores leaves by Bellman expectimax.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from prodromos.plan import PLAN_GRAPH_VERSION
from prodromos.plan.adapters import NeedsData, tm_doc_to_gate_inputs
from prodromos.plan.graph import Node, PolicyGraph
from prodromos.plan.registry import GATE_REGISTRY

MAX_STEPS = 64  # hard cap; the graph is a DAG so this is a belt-and-braces guard.


@dataclass
class GateStep:
    """One executed gate node in a route trace."""

    node_id: str
    gate: str
    sanity_id: str
    verdict: str
    confidence: str | None = None
    reasons: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    needs_data: NeedsData | None = None


@dataclass
class WalkResult:
    """Outcome of a route walk over the policy graph."""

    mode: str
    verdict: str
    confidence: str
    plan_graph_version: str
    steps: list[GateStep] = field(default_factory=list)
    choices: list[dict] = field(default_factory=list)
    terminal_node: str | None = None
    next_action: str | None = None
    next_action_cost_usd: float = 0.0
    reasons: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# myopic choice (first increment honest placeholder)
# --------------------------------------------------------------------------
def _myopic_best_option(node: Node, ctx: dict) -> str:
    """Pick a choice option in route mode.

    HONEST PLACEHOLDER: choose the "cheapest gate that could change the
    recommendation" -- the existing ``cheapest_disambiguating_test`` notion in
    neb_method_advisor -- approximated here as the first declared option (the
    options lists are authored cheapest-first / safest-default-first in
    policy.py). We do NOT fabricate an EVPI from entropy.

    # TODO: upgrade to calibrated greedy-EVPI (decision-loss VoI, $) -- NOT
    #       entropy / not a confidence product (consilium C1).
    """
    upstream = ctx.get("upstream_verdict")
    if node.options is None:
        raise ValueError(f"choice node {node.id!r} has no options")
    # Honest, deterministic default: prefer the safe option implied by upstream.
    if node.id == "nspin_choice":
        # electron-parity already gave the factual verdict upstream; respect it.
        if upstream == "NSPIN2_MANDATORY":
            return "nspin2"
        if upstream == "NSPIN2_RECOMMENDED":
            return "nspin2"
        return "nspin1"
    # method_choice (and any future choice): cheapest/safest-first option.
    return node.options[0]


# --------------------------------------------------------------------------
# route executor
# --------------------------------------------------------------------------
def _project_inputs(node: Node, gate_inputs: dict) -> dict:
    """Project the adapter output onto the node's declared inputs (memo key)."""
    if not node.inputs:
        return dict(gate_inputs)
    return {k: gate_inputs.get(k) for k in node.inputs if k in gate_inputs}


def _memo_key(gate: str, gate_inputs: dict) -> str:
    return gate + ":" + json.dumps(gate_inputs, sort_keys=True, default=str)


def _run_gate(node: Node, doc: dict, memo: dict[str, dict]) -> tuple[dict | None, NeedsData | None]:
    """Adapt + execute one gate node. Returns (envelope, needs_data)."""
    spec = GATE_REGISTRY[node.gate]
    adapted = tm_doc_to_gate_inputs(doc, node.gate)
    if isinstance(adapted, NeedsData):
        return None, adapted
    key = _memo_key(node.gate, adapted)
    if key in memo:
        return memo[key], None
    envelope = spec.run_fn(**adapted)
    memo[key] = envelope
    return envelope, None


def _route(graph: PolicyGraph, doc: dict) -> WalkResult:
    res = WalkResult(
        mode="route",
        verdict="INVESTIGATE",
        confidence="low",
        plan_graph_version=PLAN_GRAPH_VERSION,
    )
    memo: dict[str, dict] = {}
    seen_trace: set[str] = set()
    upstream_verdict: str | None = None

    node = graph.node(graph.root)
    for _ in range(MAX_STEPS):
        # loop_guard: a (node, upstream) pair revisited means a non-progressing
        # walk on a bad case patch -- bail with NEEDS_DATA rather than spin.
        trace_token = f"{node.id}|{upstream_verdict}"
        if trace_token in seen_trace:
            res.verdict = "NEEDS_DATA"
            res.confidence = "low"
            res.warnings.append(
                f"loop_guard: revisited {node.id!r} without progress; "
                "the case is missing data needed to advance the plan"
            )
            res.terminal_node = node.id
            res.next_action = "supply the missing case fields and re-run plan"
            return res
        seen_trace.add(trace_token)

        if node.kind == "terminal":
            res.verdict = node.terminal
            res.terminal_node = node.id
            res.next_action = node.what
            # surface cost of the recommended next action from the last gate's
            # registry cost (all $0 in the pre-flight increment).
            res.confidence = res.confidence if res.steps else "medium"
            if node.terminal == "GO":
                res.confidence = "high" if res.steps else "medium"
                res.reasons.append("all evaluated pre-flight gates passed; path is well-posed")
            return res

        if node.kind == "gate":
            envelope, needs = _run_gate(node, doc, memo)
            spec = GATE_REGISTRY[node.gate]
            if needs is not None:
                step = GateStep(
                    node_id=node.id,
                    gate=node.gate,
                    sanity_id=spec.sanity_id,
                    verdict="NEEDS_DATA",
                    confidence="low",
                    reasons=[f"missing: {', '.join(needs.missing)}"],
                    next_actions=[needs.recommend] if needs.recommend else [],
                    needs_data=needs,
                )
                res.steps.append(step)
                res.verdict = "NEEDS_DATA"
                res.confidence = "low"
                res.terminal_node = node.id
                res.next_action = needs.recommend or f"obtain inputs for {node.gate}"
                res.next_action_cost_usd = spec.cost_usd
                res.reasons.extend(step.reasons)
                res.next_actions.extend(step.next_actions)
                return res
            verdict = envelope.get("verdict")
            step = GateStep(
                node_id=node.id,
                gate=node.gate,
                sanity_id=spec.sanity_id,
                verdict=verdict,
                confidence=envelope.get("confidence"),
                reasons=list(envelope.get("reasons") or []),
                next_actions=list(envelope.get("next_actions") or []),
            )
            res.steps.append(step)
            res.confidence = step.confidence or res.confidence
            upstream_verdict = verdict
            edge = _select_edge(node, {"verdict": verdict, "result": envelope.get("result")})
            if edge is None:
                res.verdict = "INVESTIGATE"
                res.terminal_node = node.id
                res.warnings.append(
                    f"no edge matched verdict {verdict!r} at gate node {node.id!r}"
                )
                res.next_action = (
                    f"verdict {verdict!r} from {node.gate} has no routing edge; "
                    "extend the policy graph"
                )
                return res
            node = graph.node(edge.dst)
            continue

        if node.kind == "choice":
            option = _myopic_best_option(node, {"upstream_verdict": upstream_verdict})
            res.choices.append({
                "node": node.id,
                "chosen": option,
                "options": list(node.options or []),
                "rule": "myopic placeholder (cheapest-disambiguating); TODO greedy-EVPI",
            })
            edge = _select_edge(node, {"option": option})
            if edge is None:
                res.verdict = "INVESTIGATE"
                res.terminal_node = node.id
                res.warnings.append(
                    f"no edge matched option {option!r} at choice node {node.id!r}"
                )
                return res
            node = graph.node(edge.dst)
            continue

        raise ValueError(f"unknown node kind {node.kind!r} at {node.id!r}")

    res.verdict = "NEEDS_DATA"
    res.warnings.append(f"exceeded MAX_STEPS={MAX_STEPS}; aborting walk")
    return res


def _select_edge(node: Node, ctx: dict):
    """First-match edge selection (mirrors the router's first-gate-wins)."""
    for edge in node.edges:
        if edge.predicate(ctx):
            return edge
    return None


# --------------------------------------------------------------------------
# tree simulator (STUB -- next increment)
# --------------------------------------------------------------------------
def _tree(graph: PolicyGraph, doc: dict) -> WalkResult:
    """SIMULATE mode -- scored strategy tree.

    NOT IMPLEMENTED in this increment. The scoring layer is intentionally
    deferred; a minimal skeleton is returned so the CLI/contract surface exists.

    # TODO: Bellman expectimax (max @ decision nodes, E @ chance/gate nodes),
    #       STOP = max(0, E[NPV of best run]), CVaR_alpha tail control,
    #       Beta-Binomial calibration with hierarchical backoff by chemistry
    #       signature, scored on the LOWER credible bound. See design doc
    #       PRODROMOS_PLAN_ENGINE_DESIGN.md S2 corrections C1-C5.
    """
    res = WalkResult(
        mode="tree",
        verdict="INVESTIGATE",
        confidence="low",
        plan_graph_version=PLAN_GRAPH_VERSION,
    )
    res.warnings.append(
        "tree mode is a STUB (first increment): scoring / Bellman-expectimax / "
        "CVaR / Beta-Binomial calibration are the next increment. Use --mode route."
    )
    res.next_action = "run --mode route for an executable next-step recommendation"
    return res


# --------------------------------------------------------------------------
# public entry
# --------------------------------------------------------------------------
def walk(graph: PolicyGraph, doc: dict, mode: str = "route") -> WalkResult:
    """Walk ``graph`` over the tm-spec ``doc`` in route (EXECUTE) or tree (SIMULATE)."""
    if mode == "route":
        return _route(graph, doc)
    if mode == "tree":
        return _tree(graph, doc)
    raise ValueError(f"unknown mode {mode!r} (expected 'route' or 'tree')")
