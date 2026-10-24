from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from .config import LLMConfig
from .env import InvalidAction, normalize_action


@dataclass(frozen=True)
class UsageCounters:
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0

    def __sub__(self, other: "UsageCounters") -> "UsageCounters":
        return UsageCounters(
            input_tokens=self.input_tokens - other.input_tokens,
            output_tokens=self.output_tokens - other.output_tokens,
            latency_seconds=self.latency_seconds - other.latency_seconds,
        )


class ActionGenerator(Protocol):
    def generate_action(self, prompt: str) -> str: ...


class AgentModel(ActionGenerator, Protocol):
    def generate_text(self, prompt: str, output_tag: str) -> str: ...

    def usage_snapshot(self) -> UsageCounters: ...


class DeepSeekClient:
    """OpenAI-compatible DeepSeek client with action and auxiliary calls."""

    def __init__(self, config: LLMConfig) -> None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Environment variable {config.api_key_env} is not set. "
                "Set it before running an experiment."
            )
        self.config = config
        self.client = OpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.request_timeout_seconds,
            max_retries=config.max_retries,
        )
        self._usage = UsageCounters()

    def generate_action(self, prompt: str) -> str:
        raw = self._complete(
            prompt=prompt,
            system_prompt=(
                "You are a ReAct agent in WebShop. Output exactly one action and no other text. "
                "Valid forms are think[...], search[...], and click[...]."
            ),
            stop=["\n"],
        )
        try:
            return normalize_action(raw)
        except InvalidAction:
            match = re.search(r"(?:think|search|click)\[.*?]", raw, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                raise
            return normalize_action(match.group(0))

    def generate_text(self, prompt: str, output_tag: str) -> str:
        return self._complete(
            prompt=prompt,
            system_prompt=(
                f"You are an auxiliary module for a WebShop ReAct agent. Output exactly one "
                f"{output_tag}[...] expression on one line and no other text. "
                f"Write the {output_tag} tag exactly once, keep every required field concise, "
                "and close the final bracket."
            ),
            stop=["\n"],
            max_tokens=self.config.auxiliary_max_tokens,
        )

    def usage_snapshot(self) -> UsageCounters:
        return self._usage

    def _complete(
        self,
        prompt: str,
        system_prompt: str,
        stop: list[str],
        max_tokens: int | None = None,
    ) -> str:
        extra_body = {"thinking": {"type": "disabled"}} if self.config.disable_thinking else None
        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
            stop=stop,
            extra_body=extra_body,
        )
        elapsed = time.perf_counter() - started
        usage = response.usage
        self._usage = UsageCounters(
            input_tokens=self._usage.input_tokens + int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=self._usage.output_tokens + int(getattr(usage, "completion_tokens", 0) or 0),
            latency_seconds=self._usage.latency_seconds + elapsed,
        )
        return response.choices[0].message.content or ""
