"""Tests for N-07 external_reference_gate (all offline / mocked — no real network)."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Import module early so the patch target exists.
import prodromos.external_reference_gate as _erg_mod
from prodromos.external_reference_gate import (
    run_external_reference_gate,
    _parse_nomad_response,
    _parse_optimade_response,
    _build_nomad_query,
    _build_optimade_filter,
)

_PATCH_TARGET = "prodromos.external_reference_gate.httpx.Client"


# ---------------------------------------------------------------------------
# Helpers: fake HTTP responses
# ---------------------------------------------------------------------------

def _make_nomad_response(n_hits: int, formula: str = "FeS2") -> dict:
    """Minimal NOMAD entries/query JSON that _parse_nomad_response can handle."""
    hits = []
    for i in range(n_hits):
        hits.append({
            "entry_id": f"entry_{i}",
            "results": {
                "material": {
                    "chemical_formula_reduced": formula,
                    "elements": ["Fe", "S"],
                },
                "method": {
                    "simulation": {
                        "dft": {"xc_functional_type": "GGA" if i % 2 == 0 else "GGA+U"}
                    }
                },
                "properties": {
                    "structures": {
                        "structure_original": {
                            "lattice_parameters": {"a": 5.416, "b": 5.416, "c": 5.416}
                        }
                    },
                    "magnetic": {"magnetic_ordering": "NM"},
                },
            },
        })
    return {
        "data": hits,
        "pagination": {"total": n_hits + 10},  # pretend there are more
    }


def _make_optimade_response(n_hits: int, formula: str = "FeS2") -> dict:
    """Minimal OPTIMADE /structures JSON."""
    data = []
    for i in range(n_hits):
        data.append({
            "id": f"mp-{i}",
            "attributes": {"chemical_formula_reduced": formula},
        })
    return {
        "data": data,
        "meta": {"data_returned": n_hits},
    }


def _mock_httpx_client(nomad_data=None, optimade_data=None,
                        nomad_exc=None, optimade_exc=None):
    """Return a context manager that mimics httpx.Client with canned responses."""
    client = MagicMock()
    # POST -> NOMAD
    if nomad_exc:
        client.post.side_effect = nomad_exc
    else:
        post_resp = MagicMock()
        post_resp.raise_for_status = MagicMock()
        post_resp.json.return_value = nomad_data or {}
        client.post.return_value = post_resp
    # GET -> OPTIMADE
    if optimade_exc:
        client.get.side_effect = optimade_exc
    else:
        get_resp = MagicMock()
        get_resp.raise_for_status = MagicMock()
        get_resp.json.return_value = optimade_data or {}
        client.get.return_value = get_resp

    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=client)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Tests: live=False (offline mode)
# ---------------------------------------------------------------------------

def test_offline_mode_returns_unknown():
    env = run_external_reference_gate(["Fe", "S"], live=False)
    assert env["verdict"] == "UNKNOWN"
    assert env["status"] == "error"
    assert env["result"]["source"] == "offline"
    assert env["result"]["exists"] is None
    assert "offline" in env["reasons"][0].lower() or "suppressed" in env["reasons"][0].lower()


def test_offline_preserves_inputs():
    env = run_external_reference_gate(
        ["S", "Fe"], reduced_formula="FeS2", space_group="Pa-3", live=False
    )
    r = env["result"]
    assert r["reduced_formula"] == "FeS2"
    assert r["space_group"] == "Pa-3"
    assert "Fe" in r["elements"] and "S" in r["elements"]


# ---------------------------------------------------------------------------
# Tests: REFERENCE_FOUND (NOMAD returns hits)
# ---------------------------------------------------------------------------

def test_reference_found_nomad():
    cm = _mock_httpx_client(nomad_data=_make_nomad_response(15))
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["Fe", "S"], reduced_formula="FeS2", live=True)
    assert env["verdict"] == "REFERENCE_FOUND"
    assert env["status"] == "ok"
    r = env["result"]
    assert r["exists"] is True
    assert r["n_entries"] >= 15
    assert r["source"] == "nomad"
    assert "GGA" in r["functional_histogram"] or "GGA+U" in r["functional_histogram"]
    assert len(r["nearest_stoichiometries"]) > 0


def test_reference_found_has_stable_envelope_keys():
    cm = _mock_httpx_client(nomad_data=_make_nomad_response(5))
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["Fe", "S"], live=True)
    assert set(env) == {
        "tool", "version", "status", "verdict", "confidence",
        "reasons", "next_actions", "artifacts", "warnings", "result",
    }
    assert env["tool"] == "external_reference_gate"


# ---------------------------------------------------------------------------
# Tests: NO_EXTERNAL_REFERENCE (NOMAD returns empty)
# ---------------------------------------------------------------------------

def test_no_external_reference_nomad_empty():
    empty = {"data": [], "pagination": {"total": 0}}
    cm = _mock_httpx_client(nomad_data=empty)
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["Unobtanium", "X"], live=True)
    assert env["verdict"] == "NO_EXTERNAL_REFERENCE"
    assert env["result"]["exists"] is False
    assert env["result"]["n_entries"] == 0


def test_no_external_reference_raises_validation_bar():
    empty = {"data": [], "pagination": {"total": 0}}
    cm = _mock_httpx_client(nomad_data=empty)
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["Fe", "S"], live=True)
    # next_actions should mention validation
    joined = " ".join(env["next_actions"]).lower()
    assert "validation" in joined or "smoke" in joined or "convergence" in joined


# ---------------------------------------------------------------------------
# Tests: NOMAD fails -> OPTIMADE fallback
# ---------------------------------------------------------------------------

def test_nomad_failure_falls_back_to_optimade():
    import httpx as _httpx
    cm = _mock_httpx_client(
        nomad_exc=_httpx.ConnectError("refused"),
        optimade_data=_make_optimade_response(8),
    )
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["Fe", "S"], live=True)
    assert env["verdict"] == "REFERENCE_FOUND"
    assert env["result"]["source"] == "optimade"
    # Warning about NOMAD failure should be present
    assert any("NOMAD" in w for w in env["warnings"])


def test_optimade_fallback_empty_means_no_reference():
    import httpx as _httpx
    cm = _mock_httpx_client(
        nomad_exc=_httpx.ConnectError("refused"),
        optimade_data={"data": [], "meta": {"data_returned": 0}},
    )
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["Fe", "S"], live=True)
    assert env["verdict"] == "NO_EXTERNAL_REFERENCE"
    assert env["result"]["source"] == "optimade"


# ---------------------------------------------------------------------------
# Tests: both sources fail -> UNKNOWN / status=error
# ---------------------------------------------------------------------------

def test_both_sources_fail_returns_unknown():
    import httpx as _httpx
    cm = _mock_httpx_client(
        nomad_exc=_httpx.TimeoutException("timeout"),
        optimade_exc=_httpx.TimeoutException("timeout"),
    )
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["Fe", "S"], live=True)
    assert env["verdict"] == "UNKNOWN"
    assert env["status"] == "error"
    assert env["result"]["exists"] is None
    assert env["result"]["source"] == "none"


def test_network_error_does_not_raise():
    """Gate must not propagate exceptions — soft degradation."""
    import httpx as _httpx
    cm = _mock_httpx_client(
        nomad_exc=_httpx.ConnectError("refused"),
        optimade_exc=_httpx.ConnectError("refused"),
    )
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["Fe", "S"], live=True)
    assert isinstance(env, dict)


# ---------------------------------------------------------------------------
# Tests: internal parser helpers (pure unit tests, no HTTP)
# ---------------------------------------------------------------------------

def test_parse_nomad_response_extracts_functionals():
    data = _make_nomad_response(4)
    parsed = _parse_nomad_response(data)
    assert parsed["n_entries"] >= 4
    assert "GGA" in parsed["functional_histogram"]
    assert "FeS2" in parsed["nearest_stoichiometries"]


def test_parse_nomad_response_empty():
    parsed = _parse_nomad_response({"data": [], "pagination": {"total": 0}})
    assert parsed["n_entries"] == 0
    assert parsed["functional_histogram"] == {}
    assert parsed["nearest_stoichiometries"] == []


def test_parse_optimade_response_extracts_formulas():
    data = _make_optimade_response(3, formula="Fe9S8")
    parsed = _parse_optimade_response(data)
    assert parsed["n_entries"] == 3
    assert "Fe9S8" in parsed["nearest_stoichiometries"]


def test_build_nomad_query_includes_elements():
    body = _build_nomad_query(["S", "Fe"], None)
    assert body["query"]["results.material.elements"] == {"all": ["Fe", "S"]}


def test_build_nomad_query_with_formula():
    body = _build_nomad_query(["Fe", "S"], "FeS2")
    assert body["query"]["results.material.chemical_formula_reduced"] == "FeS2"


def test_build_optimade_filter_no_formula():
    filt = _build_optimade_filter(["S", "Fe"], None)
    assert 'elements HAS ALL' in filt
    assert '"Fe"' in filt and '"S"' in filt


def test_build_optimade_filter_with_formula():
    filt = _build_optimade_filter(["Fe", "S"], "FeS2")
    assert 'chemical_formula_reduced = "FeS2"' in filt


# ---------------------------------------------------------------------------
# Tests: element normalisation
# ---------------------------------------------------------------------------

def test_elements_normalised_and_deduplicated():
    empty = {"data": [], "pagination": {"total": 0}}
    cm = _mock_httpx_client(nomad_data=empty)
    with patch(_PATCH_TARGET, return_value=cm):
        env = run_external_reference_gate(["fe", "s", "Fe"], live=True)
    # Should be deduplicated and capitalised
    assert env["result"]["elements"] == ["Fe", "S"]


def test_empty_elements_raises():
    with pytest.raises(ValueError):
        run_external_reference_gate([], live=False)
