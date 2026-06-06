"""Deterministic reproduction harness for the stdio-transport wedge (roadmap §F).

The production symptom: MCP tool calls wedge the client for minutes-to-hours while
the server-side compute is trivial. Root cause #2 (the reproducible aggravator): in
FastMCP 1.27 a *sync* (``def``) tool is executed INLINE on the asyncio event loop,
which also hosts the stdio reader/writer and (in the lowlevel server) the per-request
dispatch task group. A running sync tool therefore blocks request intake AND response
flushing, serializing every concurrent call.

These tests use the in-memory client/server transport (same ``Server.run`` path, real
``ClientSession``, real event loop -- no OS pipes) to:

1. REPRODUCE the wedge: N concurrent calls to an INLINE sync tool serialize
   (wall ~= N * sleep).
2. PROVE the fix: the same tool wrapped with ``_offload`` runs concurrently
   (wall ~= sleep).
3. Smoke-test the REAL prodromos server: N concurrent fast gate calls all return.

Per the roadmap: "Do NOT claim any fix without this reproducing the wedge."
"""
from __future__ import annotations

import asyncio
import time

import pytest

pytest.importorskip("mcp", reason="requires the optional 'mcp' dependency")

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp.shared.memory import (  # noqa: E402
    create_connected_server_and_client_session as connect,
)

from prodromos import mcp_server  # noqa: E402

_SLEEP = 0.25
_N = 4


def _slow_sync() -> dict:
    """A deliberately blocking sync 'gate' (no real I/O -- pure CPU/sleep stand-in)."""
    time.sleep(_SLEEP)
    return {"tool": "slow", "verdict": "OK"}


async def _fire_concurrent(server: FastMCP, name: str, n: int) -> float:
    """Open an in-memory session and fire ``n`` concurrent calls; return wall seconds."""
    async with connect(server) as client:
        t0 = time.perf_counter()
        results = await asyncio.gather(*[client.call_tool(name, {}) for _ in range(n)])
        elapsed = time.perf_counter() - t0
    assert len(results) == n
    assert all(not r.isError for r in results)
    return elapsed


def test_inline_sync_tool_serializes_concurrent_calls():
    """REPRODUCTION: an inline sync tool blocks the loop -> calls serialize."""
    srv = FastMCP("repro-inline")
    srv.tool(name="slow")(_slow_sync)  # registered RAW (sync, is_async=False)

    elapsed = asyncio.run(_fire_concurrent(srv, "slow", _N))
    # Serialized: ~ N * _SLEEP. Assert it is clearly worse than concurrent.
    assert elapsed > (_N - 1) * _SLEEP, (
        f"expected serialization (> {(_N - 1) * _SLEEP:.2f}s) but got {elapsed:.2f}s"
    )


def test_offloaded_tool_runs_concurrent_calls_in_parallel():
    """FIX: the same tool wrapped with _offload runs off the loop -> concurrent."""
    srv = FastMCP("repro-offload")
    srv.tool(name="slow")(mcp_server._offload("slow", _slow_sync))

    elapsed = asyncio.run(_fire_concurrent(srv, "slow", _N))
    # Concurrent: ~ _SLEEP (one wave). Allow generous headroom for scheduling.
    assert elapsed < (_N - 1) * _SLEEP, (
        f"expected concurrency (< {(_N - 1) * _SLEEP:.2f}s) but got {elapsed:.2f}s "
        "-- offload did not free the event loop"
    )


def test_real_server_concurrent_fast_calls_all_return():
    """SMOKE: the real prodromos server answers N concurrent fast gate calls."""

    async def _run() -> list:
        async with connect(mcp_server.server) as client:
            t0 = time.perf_counter()
            calls = [
                client.call_tool(
                    "electron_parity",
                    {"symbol_counts": {"Fe": 31, "S": 64, "H": 1}},
                )
                for _ in range(6)
            ]
            results = await asyncio.gather(*calls)
            assert time.perf_counter() - t0 < 10.0  # no wedge
            return results

    results = asyncio.run(_run())
    assert len(results) == 6
    assert all(not r.isError for r in results)


def test_real_server_batch_tool_single_round_trip():
    """The batch meta-tool runs several gates server-side in ONE round-trip."""

    async def _run():
        async with connect(mcp_server.server) as client:
            return await client.call_tool(
                "batch",
                {
                    "calls": [
                        {"tool": "electron_parity", "args": {"symbol_counts": {"Fe": 2, "S": 4}}},
                        {"tool": "h_barrier_readiness", "args": {"barrier_eV": 0.3, "has_dft_freq": False}},
                        {"tool": "does_not_exist", "args": {}},
                    ]
                },
            )

    result = asyncio.run(_run())
    assert not result.isError
