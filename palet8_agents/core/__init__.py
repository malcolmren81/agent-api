"""
Core components for the Palet8 agent framework.

This module provides the foundational classes and utilities used by all agents:
- BaseAgent: Abstract base class for all agents
- AgentContext: Shared execution context
- AgentResult: Standard result container
- Message: Conversation message definitions
- Config: Configuration loading and management
- Exceptions: Structured error handling
"""

from palet8_agents.core.agent import (
    BaseAgent,
    AgentContext,
    AgentResult,
    AgentState,
)
from palet8_agents.core.message import (
    Message,
    MessageRole,
    Conversation,
    ToolCall,
    ToolResult,
)
from palet8_agents.core.config import (
    AgentConfig,
    ModelProfile,
    load_config,
    get_model_profile,
)
from palet8_agents.core.exceptions import (
    AgentError,
    AgentConfigError,
    AgentExecutionError,
    LLMClientError,
    ToolError,
    RateLimitError,
    SafetyViolationError,
)
from palet8_agents.core.llm_client import LLMClient, LLMResponse

__all__ = [
    # Agent
    "BaseAgent",
    "AgentContext",
    "AgentResult",
    "AgentState",
    # Message
    "Message",
    "MessageRole",
    "Conversation",
    "ToolCall",
    "ToolResult",
    # Config
    "AgentConfig",
    "ModelProfile",
    "load_config",
    "get_model_profile",
    # Exceptions
    "AgentError",
    "AgentConfigError",
    "AgentExecutionError",
    "LLMClientError",
    "ToolError",
    "RateLimitError",
    "SafetyViolationError",
    # LLM Client
    "LLMClient",
    "LLMResponse",
]
