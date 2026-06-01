"""Render a WalkResult into the two output shapes.

- :func:`to_envelope` -> a ``cli_contract.response_envelope`` (CLI/MCP shape).
- :func:`to_preflight_block` -> a tm-spec 0.3 embeddable ``preflight`` block
  (engine / verdict / confidence / gates[] / plan), suitable to splice into a
  tm-spec document and re-validate with ``tm_spec.validator``.

Both consume one :class:`prodromos.plan.interpret.WalkResult`.
"""
from __future__ import annotations

from prodromos.cli_contract import response_envelope
from prodromos.plan import ENGINE_NAME, ENGINE_VERSION
from prodromos.plan.interpret import WalkResult

TOOL = "plan"


def to_envelope(walk_result: WalkResult) -> dict:
    """Render the response_envelope (CLI/MCP shape)."""
    wr = walk_result
    gate_summaries = [
        {
            "id": step.sanity_id,
            "node": step.node_id,
            "gate": step.gate,
            "verdict": step.verdict,
            "confidence": step.confidence,
            "reasons": step.reasons,
        }
        for step in wr.steps
    ]
    result = {
        "mode": wr.mode,
        "plan_graph_version": wr.plan_graph_version,
        "terminal_node": wr.terminal_node,
        "gate_trace": [f"{s.node_id}:{s.verdict}" for s in wr.steps],
        "gates": gate_summaries,
        "choices": wr.choices,
        "next_action": wr.next_action,
        "next_action_cost_usd": wr.next_action_cost_usd,
    }
    if wr.strategies:
        result["strategies"] = [
            {
                "label": s.label,
                "method": s.method,
                "expected_cost_usd": float(s.expected_cost_usd),
                "p_success": float(s.p_success),
                "cvar_usd": float(s.cvar_usd),
                "utility": float(s.utility),
                "paper_grade_reachable": bool(s.paper_grade_reachable),
                "proposed_workflow_ref": s.proposed_workflow_ref,
                "is_stop": bool(s.is_stop),
            }
            for s in wr.strategies
        ]
    next_actions = list(wr.next_actions)
    if wr.next_action and wr.next_action not in next_actions:
        next_actions.insert(0, wr.next_action)
    return response_envelope(
        tool=TOOL,
        verdict=wr.verdict,
        confidence=wr.confidence,
        reasons=wr.reasons or [r for s in wr.steps for r in s.reasons],
        next_actions=next_actions,
        warnings=wr.warnings,
        result=result,
    )


def to_preflight_block(walk_result: WalkResult) -> dict:
    """Render the embeddable tm-spec 0.3 ``preflight`` block.

    Conforms to schema $defs/preflight: engine{name,version}, verdict,
    confidence, gates[]{id,verdict,...}, plan{next_action,next_action_cost_usd},
    plan_graph_version.
    """
    wr = walk_result
    gates = []
    for step in wr.steps:
        gate_entry: dict = {
            "id": step.sanity_id,
            "verdict": step.verdict,
        }
        if step.confidence:
            gate_entry["confidence"] = step.confidence
        if step.reasons:
            gate_entry["reasons"] = step.reasons
        if step.next_actions:
            gate_entry["next_actions"] = step.next_actions
        gates.append(gate_entry)

    plan_block: dict = {
        "next_action": wr.next_action or "",
        "next_action_cost_usd": float(wr.next_action_cost_usd),
    }

    # tree (SIMULATE) mode: fill plan.strategies[] from the ranked leaves.
    if wr.strategies:
        strategies = []
        for s in wr.strategies:
            entry: dict = {
                "label": s.label,
                "method": s.method,
                "expected_cost_usd": float(s.expected_cost_usd),
                "p_success": float(s.p_success),
                "cvar_usd": float(s.cvar_usd),
                "utility": float(s.utility),
                "paper_grade_reachable": bool(s.paper_grade_reachable),
            }
            if s.proposed_workflow_ref:
                entry["proposed_workflow_ref"] = s.proposed_workflow_ref
            strategies.append(entry)
        plan_block["strategies"] = strategies

    block: dict = {
        "engine": {"name": ENGINE_NAME, "version": ENGINE_VERSION},
        "verdict": wr.verdict,
        "confidence": wr.confidence,
        "plan_graph_version": wr.plan_graph_version,
        "gates": gates,
        "plan": plan_block,
    }
    return block
