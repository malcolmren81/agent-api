"""
Agent implementations for the Palet8 system.

This module provides the main agents:
- PaliAgent: User-facing orchestrator for requirement gathering
- PlannerAgent: Original planning agent (legacy, ~1700 lines)
- PlannerAgentV2: Thin coordinator that delegates to ReactPromptAgent (~450 lines)
- ReactPromptAgent: ReAct-style prompt building agent
- EvaluatorAgent: Original quality assessment (legacy, ~1070 lines)
- EvaluatorAgentV2: Thin quality gate using tools (~350 lines)
- SafetyAgent: Content safety and IP violation checks
"""

from palet8_agents.agents.pali_agent import PaliAgent
from palet8_agents.agents.planner_agent import PlannerAgent
from palet8_agents.agents.planner_agent_v2 import PlannerAgentV2
from palet8_agents.agents.react_prompt_agent import ReactPromptAgent
from palet8_agents.agents.evaluator_agent import EvaluatorAgent
from palet8_agents.agents.evaluator_agent_v2 import EvaluatorAgentV2
from palet8_agents.agents.safety_agent import SafetyAgent

__all__ = [
    "PaliAgent",
    "PlannerAgent",
    "PlannerAgentV2",
    "ReactPromptAgent",
    "EvaluatorAgent",
    "EvaluatorAgentV2",
    "SafetyAgent",
]
