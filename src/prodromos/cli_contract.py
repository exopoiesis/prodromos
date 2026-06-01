"""Shared CLI/MCP-shaped response helpers for pre-flight tools."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def response_envelope(
    *,
    tool: str,
    result: Any = None,
    verdict: str | None = None,
    confidence: str | None = None,
    status: str = "ok",
    reasons: list[str] | None = None,
    next_actions: list[str] | None = None,
    artifacts: list[str] | None = None,
    warnings: list[str] | None = None,
    version: str = "0.1",
) -> dict:
    """Return a stable JSON envelope suitable for CLI and future MCP tools."""
    return {
        "tool": tool,
        "version": version,
        "status": status,
        "verdict": verdict,
        "confidence": confidence,
        "reasons": reasons or [],
        "next_actions": next_actions or [],
        "artifacts": artifacts or [],
        "warnings": warnings or [],
        "result": result,
    }


def dump_json(data: Any, path: str | Path | None = None) -> None:
    """Write JSON either to stdout or to a file."""
    text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
    if path:
        Path(path).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
