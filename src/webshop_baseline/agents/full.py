from .base import BaseReActAgent


class PlanReflectAgent(BaseReActAgent):
    """Combined Planning and Reflection method."""

    planning_enabled = True
    reflection_enabled = True
