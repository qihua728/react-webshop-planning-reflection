from __future__ import annotations

from typing import Type

from .base import BaseReActAgent
from .baseline import ReActAgent
from .full import PlanReflectAgent
from .planning import PlanningAgent
from .reflection import ReflectionAgent


AGENT_TYPES: dict[str, Type[BaseReActAgent]] = {
    "react_baseline": ReActAgent,
    "react_planning": PlanningAgent,
    "react_reflection": ReflectionAgent,
    "react_plan_reflect": PlanReflectAgent,
}


def create_agent(method: str, **kwargs: object) -> BaseReActAgent:
    try:
        agent_type = AGENT_TYPES[method]
    except KeyError as exc:
        supported = ", ".join(sorted(AGENT_TYPES))
        raise ValueError(f"Unknown method {method!r}. Supported methods: {supported}") from exc
    return agent_type(**kwargs)  # type: ignore[arg-type]
