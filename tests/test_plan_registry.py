"""Tests for the GATE_REGISTRY crosswalk."""
from __future__ import annotations

import re

from prodromos.__main__ import SUBCOMMANDS
from prodromos.plan.registry import GATE_REGISTRY

_GXX = re.compile(r"^G\d{2}_[A-Za-z0-9_]+$")


def test_every_entry_has_valid_gxx_id():
    for sub, spec in GATE_REGISTRY.items():
        assert _GXX.match(spec.sanity_id), f"{sub}: bad sanity_id {spec.sanity_id!r}"


def test_sanity_ids_are_unique():
    ids = [spec.sanity_id for spec in GATE_REGISTRY.values()]
    assert len(ids) == len(set(ids)), "duplicate sanity_id in GATE_REGISTRY"


def test_run_fn_is_callable():
    for sub, spec in GATE_REGISTRY.items():
        assert callable(spec.run_fn), f"{sub}: run_fn not callable"


def test_subcommand_field_matches_key():
    for key, spec in GATE_REGISTRY.items():
        assert spec.subcommand == key


def test_every_subcommand_is_a_real_prodromos_subcommand():
    # Each registry subcommand must exist in the CLI dispatcher (the registry
    # cannot reference a gate the CLI does not expose).
    for sub in GATE_REGISTRY:
        assert sub in SUBCOMMANDS, f"{sub} not in __main__.SUBCOMMANDS"


def test_cost_is_zero_for_preflight_gates():
    # All pre-flight gates are $0 (predictive, local/cheap-MLIP).
    for sub, spec in GATE_REGISTRY.items():
        assert spec.cost_usd == 0.0, f"{sub}: unexpected non-zero cost"
