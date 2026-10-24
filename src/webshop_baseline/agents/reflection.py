from .base import BaseReActAgent


class ReflectionAgent(BaseReActAgent):
    """ReAct with bounded, trigger-based in-episode reflection."""

    reflection_enabled = True
