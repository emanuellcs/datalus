"""Audit report serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_audit_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write the audit report to a JSON file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
