"""
Agent execution phase enums.

Shared enums for agent state management.
"""

from enum import Enum


class PlannerPhase(Enum):
    """Planner Agent execution phases."""
    INITIAL = "initial"           # First run - evaluate context, create plan
    FIX_PLAN = "fix_plan"         # After evaluation rejection - fix the plan
    CLARIFY = "clarify"           # After receiving clarification from Pali


class EvaluationPhase(Enum):
    """Evaluation Agent execution phases."""
    CREATE_PLAN = "create_plan"   # Pre-generation: assess prompt quality
    EXECUTE = "execute"           # Post-generation: assess result quality
