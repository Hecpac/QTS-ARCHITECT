#!/usr/bin/env python3
"""Scan QTS-ARCHITECT for critical hardening hotspots.

Usage:
  python3 scripts/find_hotspots.py [repo_root]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERNS: dict[str, list[str]] = {
    "qts_core/src/qts_core/execution/oms.py": [
        r"def confirm_execution",
        r"def _settle_buy",
        r"def _settle_sell",
        r"def revert_allocation",
        r"def cancel_order",
        r"def get_open_orders",
        r"def reconcile",
    ],
    "qts_core/src/qts_core/execution/ems.py": [
        r"class CircuitBreaker",
        r"def can_execute",
        r"def record_success",
        r"def record_failure",
        r"class RateLimiter",
        r"async def submit_order",
    ],
    "qts_core/src/qts_core/main_live.py": [
        r"class LiveTrader",
        r"while self\.running",
        r"submit_order",
        r"confirm_execution",
        r"revert_allocation",
        r"asyncio\.wait_for",
    ],
    "qts_core/src/qts_core/backtest/metrics.py": [
        r"def _tail_risk_95",
        r"def var_95",
        r"def cvar_95",
    ],
    "tests/test_execution.py": [
        r"partial",
        r"idempotent",
        r"reconcile",
        r"half_open",
    ],
    "tests/test_backtest.py": [
        r"test_var_95",
        r"test_cvar_95",
    ],
}


def scan_file(path: Path, raw_patterns: list[str]) -> list[str]:
    if not path.exists():
        return [f"[MISSING] {path}"]

    compiled = [re.compile(p) for p in raw_patterns]
    hits: list[str] = []

    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        for pat in compiled:
            if pat.search(line):
                hits.append(f"{path}:{line_no}: {line.strip()}")
                break

    if not hits:
        return [f"[NO_MATCHES] {path}"]
    return hits


def main() -> int:
    repo_root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd().resolve()

    for rel, patterns in PATTERNS.items():
        target = repo_root / rel
        for hit in scan_file(target, patterns):
            print(hit)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
