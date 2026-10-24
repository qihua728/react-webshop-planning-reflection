"""Backward-compatible imports for the baseline agent.

New code should import from :mod:`webshop_baseline.agents`.
"""

from .agents.base import ReflectionRecord, TrajectoryStep
from .agents.baseline import ReActAgent

__all__ = ["ReActAgent", "ReflectionRecord", "TrajectoryStep"]
