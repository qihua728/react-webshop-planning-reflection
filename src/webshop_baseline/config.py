from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class WebShopConfig:
    base_url: str
    request_timeout_seconds: float = 20.0
    request_retries: int = 2
    git_commit: str = ""
    patch_version: str = ""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key_env: str
    base_url: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 100
    auxiliary_max_tokens: int = 220
    request_timeout_seconds: float = 60.0
    max_retries: int = 2
    disable_thinking: bool = True


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 15
    max_prompt_chars: int = 12000
    prompt_file: str = "prompts/react_baseline.txt"
    planning_prompt_file: str | None = None
    reflection_prompt_file: str | None = None
    max_reflections: int = 2
    reflection_midpoint_step: int = 8
    reflection_history_steps: int = 5


@dataclass(frozen=True)
class ExperimentConfig:
    task_prefix: str = "fixed_"
    start_index: int = 0
    count: int = 10
    results_file: str = "results/baseline_results.jsonl"
    summary_file: str = "results/baseline_summary.json"
    resume: bool = True
    dataset_mode: str = "small"
    dataset_version: str = ""
    prompt_version: str = "v1"


@dataclass(frozen=True)
class BaselineConfig:
    method: str
    webshop: WebShopConfig
    llm: LLMConfig
    agent: AgentConfig
    experiment: ExperimentConfig
    project_root: Path

    def resolve(self, relative_path: str) -> Path:
        return (self.project_root / relative_path).resolve()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing or invalid config section: {name}")
    return value


def load_config(path: str | Path) -> BaselineConfig:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    # The standard layout is <project>/configs/<config>.yaml.
    project_root = config_path.parent.parent
    return BaselineConfig(
        method=str(raw.get("method", "react_baseline")),
        webshop=WebShopConfig(**_section(raw, "webshop")),
        llm=LLMConfig(**_section(raw, "llm")),
        agent=AgentConfig(**_section(raw, "agent")),
        experiment=ExperimentConfig(**_section(raw, "experiment")),
        project_root=project_root,
    )
