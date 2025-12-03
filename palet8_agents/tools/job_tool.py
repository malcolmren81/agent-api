"""
Job Tool - Job state management.

This tool provides agents with the ability to manage job state,
including creating, updating, and transitioning jobs through
the state machine.

Documentation Reference: Section 5.3.3
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from datetime import datetime
import logging

from prisma import Prisma

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult

logger = logging.getLogger(__name__)


class JobState(Enum):
    """Job state enumeration."""
    INIT = "INIT"
    COLLECTING_REQUIREMENTS = "COLLECTING_REQUIREMENTS"
    PLANNING = "PLANNING"
    GENERATING = "GENERATING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    ABANDONED = "ABANDONED"


# Job state machine (from Documentation Section 5.3.3)
# INIT → COLLECTING_REQUIREMENTS → PLANNING → GENERATING → EVALUATING → COMPLETED
#              ↑                       ↓           ↓            ↓
#              └───────────────────────┴───────────┴────────────┘
#                               (loops/retries)
#                                  ↓
#                           REJECTED / FAILED / ABANDONED

JOB_STATES = [state.value for state in JobState]

# Valid state transitions
STATE_TRANSITIONS: Dict[str, List[str]] = {
    "INIT": ["COLLECTING_REQUIREMENTS"],
    "COLLECTING_REQUIREMENTS": ["PLANNING", "COLLECTING_REQUIREMENTS", "ABANDONED"],
    "PLANNING": ["GENERATING", "COLLECTING_REQUIREMENTS", "FAILED"],
    "GENERATING": ["EVALUATING", "PLANNING", "FAILED"],
    "EVALUATING": ["COMPLETED", "PLANNING", "REJECTED", "FAILED"],
    "COMPLETED": [],
    "REJECTED": [],
    "FAILED": [],
    "ABANDONED": [],
}


@dataclass
class Job:
    """Job data model."""
    id: str
    user_id: str
    state: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    plan: Dict[str, Any] = field(default_factory=dict)
    evaluation_result: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "state": self.state,
            "requirements": self.requirements,
            "plan": self.plan,
            "evaluation_result": self.evaluation_result,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata,
        }

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.state in ["COMPLETED", "REJECTED", "FAILED", "ABANDONED"]

    @property
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries and not self.is_terminal

    def get_allowed_transitions(self) -> List[str]:
        """Get list of allowed next states."""
        return STATE_TRANSITIONS.get(self.state, [])


class JobTool(BaseTool):
    """
    Job state management tool.

    Provides job lifecycle management with state machine validation.

    Methods (from Documentation Section 5.3.3):
    - get_job(job_id) -> Job
    - create_job(user_id, requirements) -> Job
    - update_job(job_id, updates) -> Job
    - transition_state(job_id, new_state) -> Job
    - increment_retry(job_id) -> Job
    """

    def __init__(
        self,
        prisma: Optional[Prisma] = None,
    ):
        """
        Initialize the Job Tool.

        Args:
            prisma: Prisma client for database access
        """
        super().__init__(
            name="job",
            description="Manage job state and lifecycle",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform: get, create, update, transition, increment_retry",
                    required=True,
                    enum=["get", "create", "update", "transition", "increment_retry"],
                ),
                ToolParameter(
                    name="job_id",
                    type=ParameterType.STRING,
                    description="Job ID to operate on (required for get, update, transition, increment_retry)",
                    required=False,
                ),
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User ID (required for create)",
                    required=False,
                ),
                ToolParameter(
                    name="updates",
                    type=ParameterType.OBJECT,
                    description="Updates to apply (for update action)",
                    required=False,
                ),
                ToolParameter(
                    name="new_state",
                    type=ParameterType.STRING,
                    description="New state to transition to (for transition action)",
                    required=False,
                    enum=JOB_STATES,
                ),
                ToolParameter(
                    name="requirements",
                    type=ParameterType.OBJECT,
                    description="Initial requirements (for create action)",
                    required=False,
                ),
            ],
        )

        self._prisma = prisma

    async def _get_prisma(self) -> Prisma:
        """Get Prisma client."""
        if self._prisma is None:
            self._prisma = Prisma()
            await self._prisma.connect()
        return self._prisma

    async def close(self) -> None:
        """Close resources."""
        if self._prisma:
            await self._prisma.disconnect()
            self._prisma = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute job management action.

        Args:
            action: The action to perform
            job_id: Job ID
            user_id: User ID (for create)
            updates: Updates to apply
            new_state: New state for transition
            requirements: Initial requirements for create

        Returns:
            ToolResult with job data
        """
        action = kwargs.get("action")
        job_id = kwargs.get("job_id")
        user_id = kwargs.get("user_id")
        updates = kwargs.get("updates", {})
        new_state = kwargs.get("new_state")
        requirements = kwargs.get("requirements", {})

        try:
            if action == "get":
                return await self._get_job(job_id)
            elif action == "create":
                return await self._create_job(user_id, requirements)
            elif action == "update":
                return await self._update_job(job_id, updates)
            elif action == "transition":
                return await self._transition_state(job_id, new_state)
            elif action == "increment_retry":
                return await self._increment_retry(job_id)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except Exception as e:
            logger.error(f"Job tool error: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Job operation failed: {e}",
                error_code="JOB_ERROR",
            )

    async def _get_job(self, job_id: Optional[str]) -> ToolResult:
        """Get job by ID."""
        if not job_id:
            return ToolResult(
                success=False,
                data=None,
                error="job_id is required for get action",
                error_code="MISSING_PARAMETER",
            )

        prisma = await self._get_prisma()

        job_record = await prisma.job.find_unique(
            where={"id": job_id},
            include={"designs": True},
        )

        if not job_record:
            return ToolResult(
                success=False,
                data=None,
                error=f"Job not found: {job_id}",
                error_code="JOB_NOT_FOUND",
            )

        job = Job(
            id=job_record.id,
            user_id=job_record.userId,
            state=job_record.status,
            requirements=job_record.requirements or {},
            plan=job_record.metadata.get("plan", {}) if job_record.metadata else {},
            evaluation_result=job_record.metadata.get("evaluation_result", {}) if job_record.metadata else {},
            retry_count=job_record.metadata.get("retry_count", 0) if job_record.metadata else 0,
            max_retries=job_record.metadata.get("max_retries", 3) if job_record.metadata else 3,
            created_at=job_record.createdAt,
            updated_at=job_record.updatedAt,
            metadata=job_record.metadata or {},
        )

        return ToolResult(
            success=True,
            data={
                "job": job.to_dict(),
                "allowed_transitions": job.get_allowed_transitions(),
                "can_retry": job.can_retry,
            },
        )

    async def _create_job(
        self,
        user_id: Optional[str],
        requirements: Dict[str, Any],
    ) -> ToolResult:
        """Create a new job."""
        if not user_id:
            return ToolResult(
                success=False,
                data=None,
                error="user_id is required for create action",
                error_code="MISSING_PARAMETER",
            )

        prisma = await self._get_prisma()

        job_record = await prisma.job.create(
            data={
                "userId": user_id,
                "status": JobState.INIT.value,
                "requirements": requirements,
                "metadata": {
                    "retry_count": 0,
                    "max_retries": 3,
                },
            },
        )

        job = Job(
            id=job_record.id,
            user_id=job_record.userId,
            state=job_record.status,
            requirements=requirements,
            created_at=job_record.createdAt,
        )

        logger.info(f"Created job {job.id} for user {user_id}")

        return ToolResult(
            success=True,
            data={
                "job": job.to_dict(),
                "allowed_transitions": job.get_allowed_transitions(),
            },
        )

    async def _update_job(
        self,
        job_id: Optional[str],
        updates: Dict[str, Any],
    ) -> ToolResult:
        """Update job with given updates."""
        if not job_id:
            return ToolResult(
                success=False,
                data=None,
                error="job_id is required for update action",
                error_code="MISSING_PARAMETER",
            )

        prisma = await self._get_prisma()

        # Get current job
        current = await prisma.job.find_unique(where={"id": job_id})
        if not current:
            return ToolResult(
                success=False,
                data=None,
                error=f"Job not found: {job_id}",
                error_code="JOB_NOT_FOUND",
            )

        # Build update data
        update_data: Dict[str, Any] = {"updatedAt": datetime.utcnow()}

        # Handle requirements update
        if "requirements" in updates:
            current_reqs = current.requirements or {}
            current_reqs.update(updates["requirements"])
            update_data["requirements"] = current_reqs

        # Handle metadata updates (plan, evaluation_result, etc.)
        current_metadata = current.metadata or {}
        for key in ["plan", "evaluation_result", "retry_count", "max_retries"]:
            if key in updates:
                current_metadata[key] = updates[key]

        if current_metadata != (current.metadata or {}):
            update_data["metadata"] = current_metadata

        # Execute update
        job_record = await prisma.job.update(
            where={"id": job_id},
            data=update_data,
        )

        job = Job(
            id=job_record.id,
            user_id=job_record.userId,
            state=job_record.status,
            requirements=job_record.requirements or {},
            plan=job_record.metadata.get("plan", {}) if job_record.metadata else {},
            evaluation_result=job_record.metadata.get("evaluation_result", {}) if job_record.metadata else {},
            retry_count=job_record.metadata.get("retry_count", 0) if job_record.metadata else 0,
            updated_at=job_record.updatedAt,
            metadata=job_record.metadata or {},
        )

        return ToolResult(
            success=True,
            data={
                "job": job.to_dict(),
                "updated_fields": list(updates.keys()),
            },
        )

    async def _transition_state(
        self,
        job_id: Optional[str],
        new_state: Optional[str],
    ) -> ToolResult:
        """Transition job to new state with validation."""
        if not job_id:
            return ToolResult(
                success=False,
                data=None,
                error="job_id is required for transition action",
                error_code="MISSING_PARAMETER",
            )

        if not new_state:
            return ToolResult(
                success=False,
                data=None,
                error="new_state is required for transition action",
                error_code="MISSING_PARAMETER",
            )

        if new_state not in JOB_STATES:
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid state: {new_state}",
                error_code="INVALID_STATE",
            )

        prisma = await self._get_prisma()

        # Get current job
        current = await prisma.job.find_unique(where={"id": job_id})
        if not current:
            return ToolResult(
                success=False,
                data=None,
                error=f"Job not found: {job_id}",
                error_code="JOB_NOT_FOUND",
            )

        current_state = current.status

        # Validate transition
        if not self.validate_transition(current_state, new_state):
            allowed = STATE_TRANSITIONS.get(current_state, [])
            return ToolResult(
                success=False,
                data=None,
                error=f"Invalid transition from {current_state} to {new_state}. Allowed: {allowed}",
                error_code="INVALID_TRANSITION",
            )

        # Build update data
        update_data: Dict[str, Any] = {
            "status": new_state,
            "updatedAt": datetime.utcnow(),
        }

        # Set completed_at for terminal states
        if new_state in ["COMPLETED", "REJECTED", "FAILED", "ABANDONED"]:
            current_metadata = current.metadata or {}
            current_metadata["completed_at"] = datetime.utcnow().isoformat()
            update_data["metadata"] = current_metadata

        # Execute transition
        job_record = await prisma.job.update(
            where={"id": job_id},
            data=update_data,
        )

        job = Job(
            id=job_record.id,
            user_id=job_record.userId,
            state=job_record.status,
            requirements=job_record.requirements or {},
            updated_at=job_record.updatedAt,
            metadata=job_record.metadata or {},
        )

        logger.info(f"Transitioned job {job_id} from {current_state} to {new_state}")

        return ToolResult(
            success=True,
            data={
                "job": job.to_dict(),
                "previous_state": current_state,
                "new_state": new_state,
                "allowed_transitions": job.get_allowed_transitions(),
            },
        )

    async def _increment_retry(self, job_id: Optional[str]) -> ToolResult:
        """Increment retry count for a job."""
        if not job_id:
            return ToolResult(
                success=False,
                data=None,
                error="job_id is required for increment_retry action",
                error_code="MISSING_PARAMETER",
            )

        prisma = await self._get_prisma()

        # Get current job
        current = await prisma.job.find_unique(where={"id": job_id})
        if not current:
            return ToolResult(
                success=False,
                data=None,
                error=f"Job not found: {job_id}",
                error_code="JOB_NOT_FOUND",
            )

        current_metadata = current.metadata or {}
        retry_count = current_metadata.get("retry_count", 0) + 1
        max_retries = current_metadata.get("max_retries", 3)
        current_metadata["retry_count"] = retry_count

        # Update
        job_record = await prisma.job.update(
            where={"id": job_id},
            data={
                "metadata": current_metadata,
                "updatedAt": datetime.utcnow(),
            },
        )

        can_retry = retry_count < max_retries

        logger.info(f"Incremented retry count for job {job_id} to {retry_count}/{max_retries}")

        return ToolResult(
            success=True,
            data={
                "job_id": job_id,
                "retry_count": retry_count,
                "max_retries": max_retries,
                "can_retry": can_retry,
            },
        )

    @staticmethod
    def validate_transition(current_state: str, new_state: str) -> bool:
        """
        Validate if a state transition is allowed.

        Args:
            current_state: Current job state
            new_state: Desired new state

        Returns:
            True if transition is valid
        """
        if current_state not in STATE_TRANSITIONS:
            return False
        return new_state in STATE_TRANSITIONS[current_state]

    @staticmethod
    def get_allowed_transitions(current_state: str) -> List[str]:
        """
        Get list of allowed transitions from current state.

        Args:
            current_state: Current job state

        Returns:
            List of states that can be transitioned to
        """
        return STATE_TRANSITIONS.get(current_state, [])

    async def __aenter__(self) -> "JobTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
