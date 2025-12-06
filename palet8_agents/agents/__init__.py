"""
Agent implementations for the Palet8 system.

This module provides the main agents:
- PaliAgent: User-facing orchestrator for requirement gathering
- PlannerAgentV2: Thin coordinator that delegates to GenPlan and ReactPrompt
- GenPlanAgent: ReAct-style generation planning agent (complexity, genflow, model)
- ReactPromptAgent: ReAct-style prompt building agent
- EvaluatorAgentV2: Thin quality gate using tools
- SafetyAgent: Content safety and IP violation checks
"""

from palet8_agents.agents.pali_agent import PaliAgent
from palet8_agents.agents.planner_agent_v2 import PlannerAgentV2
from palet8_agents.agents.genplan_agent import GenPlanAgent
from palet8_agents.agents.react_prompt_agent import ReactPromptAgent
from palet8_agents.agents.evaluator_agent_v2 import EvaluatorAgentV2
from palet8_agents.agents.safety_agent import SafetyAgent

# Aliases for backwards compatibility
PlannerAgent = PlannerAgentV2
EvaluatorAgent = EvaluatorAgentV2

__all__ = [
    "PaliAgent",
    "PlannerAgent",
    "PlannerAgentV2",
    "GenPlanAgent",
    "ReactPromptAgent",
    "EvaluatorAgent",
    "EvaluatorAgentV2",
    "SafetyAgent",
]
