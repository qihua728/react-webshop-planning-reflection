"""Check committed experiment records without API access or file writes."""

from __future__ import annotations

import json
import math
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    analysis = runpy.run_path(str(ROOT / "analysis" / "analyze_formal.py"))
    try:
        by_method, audit = analysis["load_results"]()
        for method, records in by_method.items():
            details = audit["methods"][method]
            if details["environment_error_rows"] or details["raw_rows"] != len(records):
                raise ValueError(f"{method}: unexpected error or extra rows")
            rows = list(records.values())
            if any(bool(row["success"]) != (float(row["reward"]) == 1.0) for row in rows):
                raise ValueError(f"{method}: success flags disagree with rewards")
            summary_path = ROOT / "results" / "formal" / f"{method}_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("method") != method:
                raise ValueError(f"{method}: summary method mismatch")
            expected = {
                "episodes": len(rows),
                "average_reward": sum(float(r["reward"]) for r in rows) / len(rows),
                "success_rate": sum(bool(r["success"]) for r in rows) / len(rows),
                "average_steps": sum(int(r["steps"]) for r in rows) / len(rows),
                "average_llm_calls": sum(int(r["total_llm_calls"]) for r in rows) / len(rows),
            }
            for key, value in expected.items():
                if not math.isclose(float(summary[key]), value, rel_tol=0, abs_tol=1e-6):
                    raise ValueError(f"{method}: summary mismatch for {key}")
            print(
                f"{method}: {len(rows)} episodes | "
                f"reward {expected['average_reward']:.4f} | "
                f"success {expected['success_rate']:.0%}"
            )
        print(f"PASS: 400 episodes, {audit['paired_task_count']} paired tasks; summaries match.")
        return 0
    except (ValueError, KeyError, TypeError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
