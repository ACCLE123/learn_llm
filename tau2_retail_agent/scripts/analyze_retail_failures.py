#!/usr/bin/env python3
"""Create a cross-task failure report from one Retail baseline artifact directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tau2_retail_agent.failure_analysis import build_failure_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing task_*.json artifacts.")
    args = parser.parse_args()
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.run_dir.glob("task_*.json"))
    ]
    if not records:
        raise SystemExit(f"No task_*.json files found in {args.run_dir}")
    report = build_failure_report(records)
    report_path = args.run_dir / "failure_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
