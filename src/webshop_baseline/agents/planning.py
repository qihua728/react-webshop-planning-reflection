from .base import BaseReActAgent


class PlanningAgent(BaseReActAgent):
    """ReAct with one task-level plan before the first environment action."""

    planning_enabled = True
