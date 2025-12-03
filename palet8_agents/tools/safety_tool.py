"""
Safety Tool - Content safety classification.

This tool wraps SafetyClassificationService to provide agents with
the ability to classify content for safety violations.
"""

from typing import Any, Dict, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult
from palet8_agents.services.safety_classification_service import (
    SafetyClassificationService,
    SafetyClassificationError,
)
from palet8_agents.models import SafetySeverity

logger = logging.getLogger(__name__)


class SafetyTool(BaseTool):
    """
    Tool for content safety classification.

    Actions:
    - classify_content: Classify content for safety violations
    - check_ip: Check for IP/trademark issues
    - get_severity_penalty: Get score penalty for a severity level
    - create_classification: Create a SafetyClassification from flags
    """

    def __init__(
        self,
        safety_service: Optional[SafetyClassificationService] = None,
    ):
        """
        Initialize the Safety Tool.

        Args:
            safety_service: Optional service instance for dependency injection
        """
        super().__init__(
            name="safety",
            description="Classify content for safety violations and IP issues",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=[
                        "classify_content",
                        "check_ip",
                        "get_severity_penalty",
                        "create_classification",
                    ],
                ),
                ToolParameter(
                    name="text",
                    type=ParameterType.STRING,
                    description="Content to classify",
                    required=False,
                ),
                ToolParameter(
                    name="source",
                    type=ParameterType.STRING,
                    description="Content source: input, prompt, image",
                    required=False,
                    enum=["input", "prompt", "image"],
                    default="input",
                ),
                ToolParameter(
                    name="severity",
                    type=ParameterType.STRING,
                    description="Severity level for penalty lookup",
                    required=False,
                    enum=["none", "low", "medium", "high", "critical"],
                ),
                ToolParameter(
                    name="flags",
                    type=ParameterType.ARRAY,
                    description="Array of safety flags for create_classification",
                    required=False,
                ),
            ],
        )

        self._safety_service = safety_service
        self._owns_service = safety_service is None

    async def _get_service(self) -> SafetyClassificationService:
        """Get or create safety service."""
        if self._safety_service is None:
            self._safety_service = SafetyClassificationService()
        return self._safety_service

    async def close(self) -> None:
        """Close resources."""
        if self._safety_service and self._owns_service:
            await self._safety_service.close()
            self._safety_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute safety classification action.

        Args:
            action: The action to perform
            text: Content to classify
            source: Content source
            severity: Severity level
            flags: Safety flags

        Returns:
            ToolResult with safety data
        """
        action = kwargs.get("action")
        text = kwargs.get("text", "")
        source = kwargs.get("source", "input")
        severity = kwargs.get("severity")
        flags = kwargs.get("flags", [])

        try:
            if action == "classify_content":
                return await self._classify_content(text, source)
            elif action == "check_ip":
                return await self._check_ip(text, source)
            elif action == "get_severity_penalty":
                return await self._get_severity_penalty(severity)
            elif action == "create_classification":
                return await self._create_classification(flags)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except SafetyClassificationError as e:
            logger.error(f"Safety classification error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Safety classification error: {e}",
                error_code="SAFETY_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in SafetyTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _classify_content(
        self,
        text: str,
        source: str,
    ) -> ToolResult:
        """Classify content for safety violations."""
        service = await self._get_service()
        flag = await service.classify_content(text, source)

        if flag:
            return ToolResult(
                success=True,
                data={
                    "is_safe": False,
                    "flag": {
                        "category": flag.category.value,
                        "severity": flag.severity.value,
                        "score": flag.score,
                        "description": flag.description,
                        "source": flag.source,
                        "metadata": flag.metadata,
                    },
                },
            )

        return ToolResult(
            success=True,
            data={
                "is_safe": True,
                "flag": None,
            },
        )

    async def _check_ip(
        self,
        text: str,
        source: str,
    ) -> ToolResult:
        """Check for IP/trademark issues."""
        service = await self._get_service()
        flag = service._check_ip_trademark(text, source)

        if flag:
            return ToolResult(
                success=True,
                data={
                    "has_ip_issue": True,
                    "category": flag.category.value,
                    "description": flag.description,
                    "visibility_control": flag.metadata.get("visibility_control"),
                },
            )

        return ToolResult(
            success=True,
            data={
                "has_ip_issue": False,
            },
        )

    async def _get_severity_penalty(
        self,
        severity: Optional[str],
    ) -> ToolResult:
        """Get score penalty for a severity level."""
        if not severity:
            return ToolResult(
                success=False,
                data=None,
                error="severity is required for get_severity_penalty",
                error_code="MISSING_PARAMETER",
            )

        service = await self._get_service()
        severity_enum = SafetySeverity[severity.upper()]
        penalty = service.get_severity_penalty(severity_enum)

        return ToolResult(
            success=True,
            data={
                "severity": severity,
                "penalty": penalty,
            },
        )

    async def _create_classification(
        self,
        flags: list,
    ) -> ToolResult:
        """Create a SafetyClassification from flags."""
        from palet8_agents.models import SafetyFlag, SafetyCategory

        service = await self._get_service()

        # Convert flag dicts to SafetyFlag objects
        flag_objects = []
        for f in flags:
            flag_objects.append(
                SafetyFlag(
                    category=SafetyCategory[f.get("category", "NSFW").upper()],
                    severity=SafetySeverity[f.get("severity", "low").upper()],
                    score=f.get("score", 0.5),
                    description=f.get("description", ""),
                    source=f.get("source", "input"),
                    metadata=f.get("metadata", {}),
                )
            )

        classification = service.create_classification(flag_objects)

        return ToolResult(
            success=True,
            data={
                "is_safe": classification.is_safe,
                "requires_review": classification.requires_review,
                "risk_level": classification.risk_level,
                "categories": classification.categories,
                "reason": classification.reason,
            },
        )

    async def __aenter__(self) -> "SafetyTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
