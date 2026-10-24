from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..env import InvalidAction, WebShopEnv
from ..llm import AgentModel, UsageCounters


@dataclass
class TrajectoryStep:
    step: int
    action: str
    observation: str
    reward: float
    done: bool
    invalid_action: bool
    latency_seconds: float


@dataclass
class ReflectionRecord:
    step: int
    triggers: list[str]
    text: str
    format_valid: bool


class BaseReActAgent:
    """Shared WebShop loop with optional planning and reflection hooks."""

    planning_enabled = False
    reflection_enabled = False

    def __init__(
        self,
        env: WebShopEnv,
        llm: AgentModel,
        demonstration: str,
        max_steps: int = 15,
        max_prompt_chars: int = 12000,
        planning_prompt: str = "",
        reflection_prompt: str = "",
        max_reflections: int = 2,
        reflection_midpoint_step: int = 8,
        reflection_history_steps: int = 5,
    ) -> None:
        self.env = env
        self.llm = llm
        self.demonstration = demonstration.rstrip() + "\n"
        self.max_steps = max_steps
        self.max_prompt_chars = max_prompt_chars
        self.planning_prompt = planning_prompt.strip()
        self.reflection_prompt = reflection_prompt.strip()
        self.max_reflections = max_reflections
        self.reflection_midpoint_step = reflection_midpoint_step
        self.reflection_history_steps = reflection_history_steps

        if self.planning_enabled and not self.planning_prompt:
            raise ValueError("Planning agent requires a planning prompt template.")
        if self.reflection_enabled and not self.reflection_prompt:
            raise ValueError("Reflection agent requires a reflection prompt template.")

    def run(self, task_id: str) -> dict[str, Any]:
        started = time.perf_counter()
        usage_before = self._usage_snapshot()
        initial = self.env.reset(task_id)
        trajectory: list[TrajectoryStep] = []
        reflections: list[ReflectionRecord] = []
        history = ""
        current_observation = initial.observation
        reward = initial.reward
        status = "max_steps"

        if self.planning_enabled:
            plan, plan_format_valid = self._make_plan(initial.observation)
        else:
            plan, plan_format_valid = "", True
        planning_calls = int(self.planning_enabled)
        midpoint_reflected = False

        for step_number in range(1, self.max_steps + 1):
            triggers: list[str] = []
            if self.reflection_enabled and len(reflections) < self.max_reflections:
                triggers, midpoint_reflected = self._reflection_triggers(
                    trajectory,
                    midpoint_reflected,
                )
            if triggers:
                reflection, reflection_format_valid = self._make_reflection(
                    initial.observation,
                    plan,
                    trajectory,
                    triggers,
                )
                reflections.append(
                    ReflectionRecord(
                        step=len(trajectory),
                        triggers=triggers,
                        text=reflection,
                        format_valid=reflection_format_valid,
                    )
                )
                history += f"\n\nReflection: {reflection}"

            prompt = self._build_action_prompt(
                initial.observation,
                plan,
                history,
                current_observation,
                trajectory,
            )
            step_started = time.perf_counter()
            invalid = False
            action = "<malformed action>"
            try:
                action = self.llm.generate_action(prompt)
                result = self.env.step(task_id, action)
                observation = result.observation
                reward = result.reward
            except InvalidAction as exc:
                observation = f"Invalid action! {exc}"
                result = None
                invalid = True

            done = bool(result and result.done)
            trajectory.append(
                TrajectoryStep(
                    step=step_number,
                    action=action,
                    observation=observation,
                    reward=reward,
                    done=done,
                    invalid_action=invalid,
                    latency_seconds=round(time.perf_counter() - step_started, 4),
                )
            )
            current_observation = observation
            history += f"\n\nAction: {action}\nObservation: {observation}"
            if done:
                status = "completed"
                break

        elapsed = round(time.perf_counter() - started, 4)
        invalid_count = sum(item.invalid_action for item in trajectory)
        action_calls = len(trajectory)
        reflection_calls = len(reflections)
        total_calls = action_calls + planning_calls + reflection_calls
        usage = self._usage_snapshot() - usage_before
        return {
            "task_id": task_id,
            "status": status,
            "termination_reason": "purchased" if status == "completed" else "max_steps",
            "reward": reward,
            "success": reward == 1.0,
            "steps": action_calls,
            "action_calls": action_calls,
            "planning_calls": planning_calls,
            "reflection_calls": reflection_calls,
            "llm_calls": total_calls,
            "total_llm_calls": total_calls,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "llm_latency_seconds": round(usage.latency_seconds, 4),
            "invalid_actions": invalid_count,
            "latency_seconds": elapsed,
            "plan": plan,
            "plan_format_valid": plan_format_valid,
            "invalid_auxiliary_outputs": int(not plan_format_valid)
            + sum(not item.format_valid for item in reflections),
            "reflections": [asdict(item) for item in reflections],
            "trajectory": [asdict(item) for item in trajectory],
        }

    def _make_plan(self, observation: str) -> tuple[str, bool]:
        prompt = self.planning_prompt.format(observation=observation)
        raw = self.llm.generate_text(prompt, output_tag="plan")
        return self._normalize_auxiliary(raw, "plan")

    def _make_reflection(
        self,
        observation: str,
        plan: str,
        trajectory: list[TrajectoryStep],
        triggers: list[str],
    ) -> tuple[str, bool]:
        recent = trajectory[-self.reflection_history_steps :]
        recent_text = "\n".join(
            f"Step {item.step}: {item.action} -> {item.observation}" for item in recent
        )
        prompt = self.reflection_prompt.format(
            observation=observation,
            plan=plan or "(none)",
            triggers=", ".join(triggers),
            recent_trajectory=recent_text,
        )
        raw = self.llm.generate_text(prompt, output_tag="reflection")
        return self._normalize_auxiliary(raw, "reflection")

    def _reflection_triggers(
        self,
        trajectory: list[TrajectoryStep],
        midpoint_reflected: bool,
    ) -> tuple[list[str], bool]:
        triggers: list[str] = []
        if trajectory and trajectory[-1].invalid_action:
            triggers.append("invalid_action")
        if len(trajectory) >= 2 and trajectory[-1].action == trajectory[-2].action:
            triggers.append("repeated_action")
        if len(trajectory) >= self.reflection_midpoint_step and not midpoint_reflected:
            triggers.append("midpoint_not_finished")
            midpoint_reflected = True
        return triggers, midpoint_reflected

    def _build_action_prompt(
        self,
        observation: str,
        plan: str,
        history: str,
        current_observation: str,
        trajectory: list[TrajectoryStep],
    ) -> str:
        pinned = observation
        if plan:
            pinned += f"\n\nPlan: {plan}"
        pinned += self._page_guidance(current_observation, trajectory)
        suffix = "\n\nAction:"
        fixed = self.demonstration + pinned + suffix
        remaining = self.max_prompt_chars - len(fixed)
        if remaining < 0:
            raise ValueError("max_prompt_chars is shorter than the fixed prompt context.")
        retained_history = history[-remaining:] if remaining else ""
        return self.demonstration + pinned + retained_history + suffix

    @staticmethod
    def _page_guidance(
        current_observation: str,
        trajectory: list[TrajectoryStep],
    ) -> str:
        on_initial_page = (
            "[Search]" in current_observation
            and "[Back to Search]" not in current_observation
        )
        search_count = sum(
            item.action.lower().startswith("search[") for item in trajectory
        )
        lines = [
            "\n\nCurrent-page guardrails:",
            (
                "- search[...] is allowed on this initial search page."
                if on_initial_page
                else (
                    "- search[...] is not valid on this page. If a new query is necessary, "
                    "first click an exact visible Back to Search link."
                )
            ),
            (
                "- click[...] only when the exact target is visibly enclosed in brackets "
                "on the current page; otherwise navigate with a visible back link."
            ),
            "- Do not reject a candidate because of an attribute the user did not request.",
            f"- Search actions already used: {search_count}.",
        ]
        if search_count >= 2:
            lines.append(
                "- Prefer inspecting the closest visible candidate instead of repeatedly "
                "rewriting nearly identical searches."
            )
        if trajectory and trajectory[-1].invalid_action:
            lines.append("- The last action was invalid; do not repeat it.")
        recent_clicks = [
            item.action.lower()
            for item in trajectory[-4:]
            if item.action.lower().startswith("click[")
        ]
        repeated_clicks = {action for action in recent_clicks if recent_clicks.count(action) > 1}
        if repeated_clicks:
            lines.append(
                "- A recently clicked target has repeated without progress; do not click "
                "that target again unless the page visibly changed."
            )
        return "\n".join(lines)

    def _usage_snapshot(self) -> UsageCounters:
        snapshot = getattr(self.llm, "usage_snapshot", None)
        return snapshot() if callable(snapshot) else UsageCounters()

    @staticmethod
    def _normalize_auxiliary(raw: str, tag: str) -> tuple[str, bool]:
        text = raw.strip().strip("`").strip()
        opening = re.search(rf"\b{re.escape(tag)}\s*\[", text, flags=re.IGNORECASE)
        body = text[opening.end() :] if opening else text
        if body.endswith("]"):
            body = body[:-1].rstrip()

        nested_opening = re.compile(rf"^{re.escape(tag)}\s*\[", flags=re.IGNORECASE)
        while nested_opening.match(body):
            body = nested_opening.sub("", body, count=1).strip()
            if body.endswith("]"):
                body = body[:-1].rstrip()

        body = " ".join(body.split())
        if not body:
            body = "no usable auxiliary output; continue with the ReAct policy"
        normalized = f"{tag}[{body}]"
        required_fields = {
            "plan": ("constraints", "search", "verify", "fallback"),
            "reflection": ("problem", "evidence", "revised_strategy"),
        }
        fields = required_fields.get(tag.lower(), ())
        positions = [
            re.search(rf"(?:^|;\s*){re.escape(field)}\s*:\s*\S", body, flags=re.IGNORECASE)
            for field in fields
        ]
        format_valid = bool(fields) and all(positions)
        if format_valid:
            starts = [match.start() for match in positions if match is not None]
            format_valid = starts == sorted(starts)
        return normalized, format_valid
