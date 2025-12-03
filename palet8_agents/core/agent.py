"""
Base agent class and related data structures.

This module defines the abstract base class for all agents in the Palet8 system,
along with the shared context and result containers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
import time

if TYPE_CHECKING:
    from palet8_agents.tools.base import BaseTool, ToolResult


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    WAITING_FOR_INPUT = "waiting_for_input"


@dataclass
class AgentContext:
    """
    Shared execution context across agents.

    This context is passed between agents during task execution and contains
    all the information needed to process a user request.
    """
    user_id: str
    job_id: str
    conversation_id: Optional[str] = None

    # Credit and cost tracking
    credit_balance: float = 0.0
    credits_used: float = 0.0

    # Model configuration (TBD - to be aligned during development)
    reasoning_model: str = ""
    image_model: str = ""

    # Task-specific data
    requirements: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)

    # Safety tracking
    safety_score: float = 1.0
    safety_flags: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def deduct_credits(self, amount: float) -> bool:
        """Deduct credits if sufficient balance exists."""
        if self.credit_balance >= amount:
            self.credit_balance -= amount
            self.credits_used += amount
            return True
        return False

    def add_safety_flag(self, flag: str, penalty: float = 0.1) -> None:
        """Add a safety flag and reduce safety score."""
        if flag not in self.safety_flags:
            self.safety_flags.append(flag)
            self.safety_score = max(0.0, self.safety_score - penalty)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "job_id": self.job_id,
            "conversation_id": self.conversation_id,
            "credit_balance": self.credit_balance,
            "credits_used": self.credits_used,
            "reasoning_model": self.reasoning_model,
            "image_model": self.image_model,
            "requirements": self.requirements,
            "plan": self.plan,
            "safety_score": self.safety_score,
            "safety_flags": self.safety_flags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class AgentResult:
    """
    Standard result from agent execution.

    All agents return this structure to provide consistent result handling.
    """
    success: bool
    data: Any
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Cost tracking
    cost_usd: float = 0.0
    credits_used: float = 0.0

    # Performance metrics
    duration_ms: int = 0
    tokens_used: int = 0
    tokens_input: int = 0
    tokens_output: int = 0

    # Agent metadata
    agent_name: str = ""
    model_used: str = ""

    # Follow-up actions
    next_agent: Optional[str] = None
    requires_user_input: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
            "cost_usd": self.cost_usd,
            "credits_used": self.credits_used,
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "agent_name": self.agent_name,
            "model_used": self.model_used,
            "next_agent": self.next_agent,
            "requires_user_input": self.requires_user_input,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the Palet8 system.

    Each agent is responsible for a specific part of the task pipeline:
    - PaliAgent: User-facing orchestrator, requirement gathering
    - PlannerAgent: Planning, RAG, prompt building
    - EvaluatorAgent: Quality assessment
    - SafetyAgent: Content safety and IP checks
    """

    def __init__(
        self,
        name: str,
        description: str,
        tools: Optional[List["BaseTool"]] = None,
    ):
        """
        Initialize the base agent.

        Args:
            name: Unique identifier for the agent
            description: Human-readable description of the agent's purpose
            tools: List of tools available to this agent
        """
        self.name = name
        self.description = description
        self.tools = tools or []
        self.state = AgentState.IDLE

        # System prompt - TBD: To be aligned during Phase 2 development
        self.system_prompt = ""

        # Model configuration - TBD: To be aligned during Phase 2 development
        self.model_profile: Optional[str] = None

        # Execution tracking
        self._start_time: Optional[float] = None
        self._tokens_used: int = 0

    @abstractmethod
    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute the agent's primary task.

        Args:
            context: Shared execution context
            user_input: Optional user input for this agent

        Returns:
            AgentResult containing the outcome of the execution
        """
        pass

    def _start_execution(self) -> None:
        """Mark the start of execution for timing."""
        self.state = AgentState.RUNNING
        self._start_time = time.time()

    def _end_execution(self, success: bool = True) -> int:
        """
        Mark the end of execution and return duration.

        Args:
            success: Whether execution was successful

        Returns:
            Duration in milliseconds
        """
        self.state = AgentState.COMPLETED if success else AgentState.FAILED
        if self._start_time:
            duration_ms = int((time.time() - self._start_time) * 1000)
            self._start_time = None
            return duration_ms
        return 0

    def _create_result(
        self,
        success: bool,
        data: Any,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        **kwargs,
    ) -> AgentResult:
        """
        Create a standardized AgentResult.

        Args:
            success: Whether the execution was successful
            data: Result data
            error: Error message if failed
            error_code: Error code if failed
            **kwargs: Additional fields for AgentResult

        Returns:
            Configured AgentResult instance
        """
        duration_ms = self._end_execution(success)

        return AgentResult(
            success=success,
            data=data,
            error=error,
            error_code=error_code,
            duration_ms=duration_ms,
            tokens_used=kwargs.get("tokens_used", self._tokens_used),
            tokens_input=kwargs.get("tokens_input", 0),
            tokens_output=kwargs.get("tokens_output", 0),
            cost_usd=kwargs.get("cost_usd", 0.0),
            credits_used=kwargs.get("credits_used", 0.0),
            agent_name=self.name,
            model_used=kwargs.get("model_used", ""),
            next_agent=kwargs.get("next_agent"),
            requires_user_input=kwargs.get("requires_user_input", False),
        )

    async def get_tool(self, tool_name: str) -> Optional["BaseTool"]:
        """
        Get a tool by name.

        Args:
            tool_name: Name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    async def call_tool(
        self,
        tool_name: str,
        action: str,
        **kwargs,
    ) -> "ToolResult":
        """
        Call a tool by name with the specified action.

        This is a convenience method for agents to call their tools
        with proper error handling and logging.

        Args:
            tool_name: Name of the tool to call
            action: Action to perform (passed as 'action' parameter)
            **kwargs: Additional parameters for the tool

        Returns:
            ToolResult from the tool execution

        Example:
            result = await self.call_tool("context", "get_user_history", user_id="user-123")
        """
        from palet8_agents.tools.base import ToolResult
        import logging

        logger = logging.getLogger(__name__)

        tool = await self.get_tool(tool_name)
        if tool is None:
            logger.error(f"[{self.name}] Tool not found: {tool_name}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Tool not found: {tool_name}",
                error_code="TOOL_NOT_FOUND",
            )

        try:
            logger.debug(f"[{self.name}] Calling tool: {tool_name}.{action}")
            result = await tool(action=action, **kwargs)
            logger.debug(f"[{self.name}] Tool result: success={result.success}")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Tool call failed: {tool_name}.{action}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                error_code="TOOL_CALL_ERROR",
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, state={self.state.value})"
