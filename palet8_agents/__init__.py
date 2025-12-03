"""
Palet8 Agents - Multi-agent framework for AI-powered design generation.

This package implements the agent system architecture as specified in the
agent-api Development Documentation v0.4.
"""

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult, AgentState
from palet8_agents.core.message import Message, MessageRole, Conversation
from palet8_agents.core.config import AgentConfig, load_config
from palet8_agents.core.exceptions import (
    AgentError,
    AgentConfigError,
    AgentExecutionError,
    LLMClientError,
    ToolError,
)

# Models package - shared data classes and enums
from palet8_agents import models

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "BaseAgent",
    "AgentContext",
    "AgentResult",
    "AgentState",
    # Message classes
    "Message",
    "MessageRole",
    "Conversation",
    # Config
    "AgentConfig",
    "load_config",
    # Exceptions
    "AgentError",
    "AgentConfigError",
    "AgentExecutionError",
    "LLMClientError",
    "ToolError",
    # Models package
    "models",
]
