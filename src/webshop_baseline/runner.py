from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agents import create_agent
from .config import BaselineConfig
from .env import WebShopEnv, WebShopError
from .evaluate import read_jsonl, summarize, write_summary
from .llm import DeepSeekClient


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def _read_optional_prompt(config: BaselineConfig, relative_path: str | None) -> str:
    if not relative_path:
        return ""
    return config.resolve(relative_path).read_text(encoding="utf-8")


def _environment_error(task_id: str, error: str) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "status": "environment_error",
        "termination_reason": "environment_error",
        "reward": 0.0,
        "success": False,
        "steps": 0,
        "action_calls": 0,
        "planning_calls": 0,
        "reflection_calls": 0,
        "llm_calls": 0,
        "total_llm_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "llm_latency_seconds": 0.0,
        "invalid_actions": 0,
        "latency_seconds": 0.0,
        "plan": "",
        "plan_format_valid": True,
        "invalid_auxiliary_outputs": 0,
        "reflections": [],
        "trajectory": [],
        "error": error,
    }


def run_experiment(
    config: BaselineConfig,
    start_index: int | None = None,
    count: int | None = None,
    resume: bool | None = None,
) -> dict[str, Any]:
    experiment = config.experiment
    start = experiment.start_index if start_index is None else start_index
    episode_count = experiment.count if count is None else count
    should_resume = experiment.resume if resume is None else resume
    if start < 0 or episode_count < 1:
        raise ValueError("start_index must be >= 0 and count must be >= 1.")

    demonstration = config.resolve(config.agent.prompt_file).read_text(encoding="utf-8")
    planning_prompt = _read_optional_prompt(config, config.agent.planning_prompt_file)
    reflection_prompt = _read_optional_prompt(config, config.agent.reflection_prompt_file)
    results_path = config.resolve(experiment.results_file)
    summary_path = config.resolve(experiment.summary_file)
    existing = read_jsonl(results_path)
    completed_ids = {
        record.get("task_id")
        for record in existing
        if record.get("method") == config.method
        and record.get("status") in {"completed", "max_steps"}
    }

    env = WebShopEnv(
        base_url=config.webshop.base_url,
        timeout_seconds=config.webshop.request_timeout_seconds,
        retries=config.webshop.request_retries,
    )
    first_task_id = f"{experiment.task_prefix}{start}"
    print(f"[preflight] checking WebShop with {first_task_id}")
    try:
        env.reset(first_task_id)
    except WebShopError as exc:
        raise WebShopError(
            f"WebShop preflight failed before any result was written: {exc}"
        ) from exc
    llm = DeepSeekClient(config.llm)
    agent = create_agent(
        config.method,
        env=env,
        llm=llm,
        demonstration=demonstration,
        max_steps=config.agent.max_steps,
        max_prompt_chars=config.agent.max_prompt_chars,
        planning_prompt=planning_prompt,
        reflection_prompt=reflection_prompt,
        max_reflections=config.agent.max_reflections,
        reflection_midpoint_step=config.agent.reflection_midpoint_step,
        reflection_history_steps=config.agent.reflection_history_steps,
    )

    for index in range(start, start + episode_count):
        task_id = f"{experiment.task_prefix}{index}"
        if should_resume and task_id in completed_ids:
            print(f"[skip] {task_id} already has a completed result")
            continue
        print(f"[run] {task_id}")
        try:
            result = agent.run(task_id)
        except WebShopError as exc:
            result = _environment_error(task_id, str(exc))
        result.update(
            {
                "method": config.method,
                "model": config.llm.model,
                "temperature": config.llm.temperature,
                "max_steps": config.agent.max_steps,
                "dataset_mode": experiment.dataset_mode,
                "dataset_version": experiment.dataset_version,
                "prompt_version": experiment.prompt_version,
                "webshop_git_commit": config.webshop.git_commit,
                "webshop_patch_version": config.webshop.patch_version,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        _append_jsonl(results_path, result)
        print(
            f"[done] {task_id} status={result['status']} "
            f"reward={result['reward']} steps={result['steps']} "
            f"calls={result['total_llm_calls']}"
        )

    all_records = read_jsonl(results_path)
    summary = summarize(all_records, method=config.method)
    write_summary(summary, summary_path)
    return summary


def check_environment(config: BaselineConfig, task_id: str = "fixed_0") -> str:
    env = WebShopEnv(
        base_url=config.webshop.base_url,
        timeout_seconds=config.webshop.request_timeout_seconds,
        retries=config.webshop.request_retries,
    )
    return env.reset(task_id).observation
