from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def _auxiliary_format_valid(text: str, tag: str) -> bool:
    fields = {
        "plan": ("constraints", "search", "verify", "fallback"),
        "reflection": ("problem", "evidence", "revised_strategy"),
    }[tag]
    if not re.fullmatch(rf"{tag}\[.*]", str(text).strip(), flags=re.IGNORECASE | re.DOTALL):
        return False
    body = str(text).strip()[len(tag) + 1 : -1]
    if re.match(rf"{tag}\s*\[", body, flags=re.IGNORECASE):
        return False
    matches = [
        re.search(rf"(?:^|;\s*){field}\s*:\s*\S", body, flags=re.IGNORECASE)
        for field in fields
    ]
    return all(matches) and [match.start() for match in matches if match] == sorted(
        match.start() for match in matches if match
    )


def _invalid_auxiliary_count(record: dict[str, Any]) -> int:
    if "invalid_auxiliary_outputs" in record:
        return int(record["invalid_auxiliary_outputs"])
    invalid = 0
    if int(record.get("planning_calls", 0)) and not _auxiliary_format_valid(
        str(record.get("plan", "")),
        "plan",
    ):
        invalid += 1
    for reflection in record.get("reflections", []):
        if "format_valid" in reflection:
            invalid += int(not reflection["format_valid"])
        elif not _auxiliary_format_valid(str(reflection.get("text", "")), "reflection"):
            invalid += 1
    return invalid


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return []
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}") from exc
    return records


def summarize(records: Iterable[dict[str, Any]], method: str | None = None) -> dict[str, Any]:
    selected = [r for r in records if method is None or r.get("method") == method]
    completed = [r for r in selected if r.get("status") in {"completed", "max_steps"}]
    n = len(completed)
    if n == 0:
        return {
            "method": method,
            "episodes": 0,
            "average_reward": 0.0,
            "success_rate": 0.0,
            "average_steps": 0.0,
            "average_llm_calls": 0.0,
            "average_action_calls": 0.0,
            "average_planning_calls": 0.0,
            "average_reflection_calls": 0.0,
            "average_input_tokens": 0.0,
            "average_output_tokens": 0.0,
            "average_llm_latency_seconds": 0.0,
            "invalid_action_rate": 0.0,
            "invalid_auxiliary_outputs": 0,
            "auxiliary_format_error_rate": 0.0,
            "max_steps_rate": 0.0,
            "environment_errors": sum(r.get("status") == "environment_error" for r in selected),
        }
    total_calls = sum(int(r.get("llm_calls", 0)) for r in completed)
    action_calls = sum(int(r.get("action_calls", r.get("llm_calls", 0))) for r in completed)
    auxiliary_calls = sum(
        int(r.get("planning_calls", 0)) + int(r.get("reflection_calls", 0))
        for r in completed
    )
    invalid_auxiliary_outputs = sum(
        _invalid_auxiliary_count(r) for r in completed
    )
    return {
        "method": method,
        "episodes": n,
        "average_reward": round(sum(float(r.get("reward", 0.0)) for r in completed) / n, 6),
        "success_rate": round(sum(bool(r.get("success")) for r in completed) / n, 6),
        "average_steps": round(sum(int(r.get("steps", 0)) for r in completed) / n, 4),
        "average_llm_calls": round(total_calls / n, 4),
        "average_action_calls": round(action_calls / n, 4),
        "average_planning_calls": round(
            sum(int(r.get("planning_calls", 0)) for r in completed) / n, 4
        ),
        "average_reflection_calls": round(
            sum(int(r.get("reflection_calls", 0)) for r in completed) / n, 4
        ),
        "average_input_tokens": round(
            sum(int(r.get("input_tokens", 0)) for r in completed) / n, 2
        ),
        "average_output_tokens": round(
            sum(int(r.get("output_tokens", 0)) for r in completed) / n, 2
        ),
        "average_llm_latency_seconds": round(
            sum(float(r.get("llm_latency_seconds", 0.0)) for r in completed) / n, 4
        ),
        "invalid_action_rate": round(
            sum(int(r.get("invalid_actions", 0)) for r in completed) / max(action_calls, 1), 6
        ),
        "invalid_auxiliary_outputs": invalid_auxiliary_outputs,
        "auxiliary_format_error_rate": round(
            invalid_auxiliary_outputs / max(auxiliary_calls, 1), 6
        ),
        "max_steps_rate": round(sum(r.get("status") == "max_steps" for r in completed) / n, 6),
        "environment_errors": sum(r.get("status") == "environment_error" for r in selected),
    }


def write_summary(summary: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
