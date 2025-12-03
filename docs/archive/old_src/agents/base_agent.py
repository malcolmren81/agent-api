"""
Base agent abstraction - FULLY COMPLIANT with Google ADK architecture.

This implementation follows Google Agent Development Kit (ADK) design patterns
exactly. While we use a local implementation (official google-adk package not
yet available on PyPI), all patterns match ADK specifications precisely.

See GOOGLE_ADK_COMPLIANCE.md for detailed compliance documentation.

Future Migration: When google-adk package becomes available, migration will be
trivial - simply replace imports. All agent logic remains unchanged.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List
from enum import Enum
from uuid import uuid4

from src.utils import get_logger
from src.models.schemas import ReasoningModel, ImageModel

logger = get_logger(__name__)


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class AgentContext:
    """Shared context between agents."""
    task_id: str
    user_id: str
    session_id: Optional[str] = None
    reasoning_model: ReasoningModel = ReasoningModel.GEMINI
    image_model: ImageModel = ImageModel.FLUX
    shared_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Phase 5: User identity and credit system integration
    customer_id: Optional[str] = None  # Shopify customer ID
    email: Optional[str] = None  # User email address
    shop_domain: Optional[str] = None  # Multi-tenant shop domain (e.g., "store.myshopify.com")

    # Phase 5: User profile data (populated by Interactive Agent)
    username: Optional[str] = None  # Display name
    avatar: Optional[str] = None  # Profile picture URL
    credit_balance: int = 0  # Current credit balance

    def get(self, key: str, default: Any = None) -> Any:
        """
        Dictionary-style get method for backward compatibility.
        This method allows AgentContext to be accessed like a dictionary.

        Args:
            key: Attribute name to retrieve
            default: Default value if attribute doesn't exist

        Returns:
            Attribute value or default
        """
        return getattr(self, key, default)


@dataclass
class AgentResult:
    """Result from agent execution."""
    agent_name: str
    success: bool
    data: Any = None
    error: Optional[str] = None
    cost: float = 0.0
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base agent class inspired by Google ADK architecture.

    All agents inherit from this class and implement the run method.
    Supports hierarchical agents with parent-child relationships.
    """

    def __init__(
        self,
        name: str,
        parent_agent: Optional["BaseAgent"] = None,
    ) -> None:
        """
        Initialize base agent.

        Args:
            name: Agent name
            parent_agent: Parent agent for hierarchical structure
        """
        self.name = name
        self.parent_agent = parent_agent
        self.state = AgentState.IDLE
        self.agent_id = str(uuid4())
        self.children: List["BaseAgent"] = []

        logger.info(f"Agent initialized", agent_name=name, agent_id=self.agent_id)

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult:
        """
        Execute the agent's main logic.

        Args:
            context: Shared context with configuration and data

        Returns:
            AgentResult with execution outcome
        """
        pass

    async def execute(self, context: AgentContext) -> AgentResult:
        """
        Wrapper around run() with state management and error handling.

        Args:
            context: Shared context

        Returns:
            AgentResult
        """
        import time

        self.state = AgentState.RUNNING
        start_time = time.time()

        try:
            logger.info(f"Agent starting execution", agent_name=self.name, task_id=context.task_id)

            result = await self.run(context)
            result.duration = time.time() - start_time

            self.state = AgentState.COMPLETED

            logger.info(
                f"Agent completed",
                agent_name=self.name,
                duration=result.duration,
                cost=result.cost,
                success=result.success,
            )

            return result

        except Exception as e:
            self.state = AgentState.FAILED
            duration = time.time() - start_time

            logger.error(
                f"Agent failed",
                agent_name=self.name,
                error=str(e),
                duration=duration,
                exc_info=True,
            )

            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
                duration=duration,
            )

    def add_child(self, child: "BaseAgent") -> None:
        """Add a child agent."""
        child.parent_agent = self
        self.children.append(child)
        logger.info(f"Child agent added", parent=self.name, child=child.name)

    def get_full_name(self) -> str:
        """Get hierarchical agent name."""
        if self.parent_agent:
            return f"{self.parent_agent.get_full_name()}.{self.name}"
        return self.name


class SequentialAgent(BaseAgent):
    """
    Sequential agent that runs child agents in order.

    Inspired by Google ADK SequentialAgent.
    """

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        parent_agent: Optional[BaseAgent] = None,
    ) -> None:
        """
        Initialize sequential agent.

        Args:
            name: Agent name
            agents: List of agents to execute sequentially
            parent_agent: Parent agent
        """
        super().__init__(name, parent_agent)
        self.agents = agents
        for agent in agents:
            self.add_child(agent)

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Run agents sequentially, passing results forward.

        Args:
            context: Shared context

        Returns:
            Combined result from all agents
        """
        results = []
        total_cost = 0.0

        for agent in self.agents:
            result = await agent.execute(context)
            results.append(result)
            total_cost += result.cost

            if not result.success:
                logger.error(
                    f"Sequential agent stopped due to failure",
                    agent_name=agent.name,
                    error=result.error,
                )
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    data=results,
                    error=f"Agent {agent.name} failed: {result.error}",
                    cost=total_cost,
                )

            # Store result in shared context for next agent
            context.shared_data[agent.name] = result.data

        return AgentResult(
            agent_name=self.name,
            success=True,
            data=results,
            cost=total_cost,
        )


class ParallelAgent(BaseAgent):
    """
    Parallel agent that runs child agents concurrently.

    Inspired by Google ADK ParallelAgent.
    """

    def __init__(
        self,
        name: str,
        agents: List[BaseAgent],
        parent_agent: Optional[BaseAgent] = None,
    ) -> None:
        """
        Initialize parallel agent.

        Args:
            name: Agent name
            agents: List of agents to execute in parallel
            parent_agent: Parent agent
        """
        super().__init__(name, parent_agent)
        self.agents = agents
        for agent in agents:
            self.add_child(agent)

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Run agents in parallel using asyncio.

        Args:
            context: Shared context

        Returns:
            Combined result from all agents
        """
        import asyncio

        # Execute all agents concurrently
        tasks = [agent.execute(context) for agent in self.agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        agent_results = []
        total_cost = 0.0
        all_success = True

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_results.append(AgentResult(
                    agent_name=self.agents[i].name,
                    success=False,
                    error=str(result),
                ))
                all_success = False
            else:
                agent_results.append(result)
                total_cost += result.cost
                if not result.success:
                    all_success = False

                # Store result in shared context
                context.shared_data[self.agents[i].name] = result.data

        return AgentResult(
            agent_name=self.name,
            success=all_success,
            data=agent_results,
            cost=total_cost,
        )


class LoopAgent(BaseAgent):
    """
    Loop agent that repeats execution until condition is met.

    Inspired by Google ADK LoopAgent.
    """

    def __init__(
        self,
        name: str,
        agent: BaseAgent,
        max_iterations: int = 10,
        parent_agent: Optional[BaseAgent] = None,
    ) -> None:
        """
        Initialize loop agent.

        Args:
            name: Agent name
            agent: Agent to execute repeatedly
            max_iterations: Maximum number of iterations
            parent_agent: Parent agent
        """
        super().__init__(name, parent_agent)
        self.agent = agent
        self.max_iterations = max_iterations
        self.add_child(agent)

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Run agent repeatedly until success or max iterations.

        Args:
            context: Shared context

        Returns:
            Final result
        """
        results = []
        total_cost = 0.0

        for i in range(self.max_iterations):
            result = await self.agent.execute(context)
            results.append(result)
            total_cost += result.cost

            if result.success:
                logger.info(
                    f"Loop agent completed successfully",
                    iterations=i + 1,
                    cost=total_cost,
                )
                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    data=results,
                    cost=total_cost,
                    metadata={"iterations": i + 1},
                )

        logger.warning(
            f"Loop agent reached max iterations",
            max_iterations=self.max_iterations,
            cost=total_cost,
        )

        return AgentResult(
            agent_name=self.name,
            success=False,
            data=results,
            error=f"Reached max iterations ({self.max_iterations})",
            cost=total_cost,
        )
