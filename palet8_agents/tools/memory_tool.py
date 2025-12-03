"""
Memory Tool - Long-term memory storage and retrieval.

This tool wraps MemoryService to provide agents with
the ability to store and retrieve design history using RAG.
"""

from typing import Any, Dict, List, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult
from palet8_agents.services.memory_service import (
    MemoryService,
    MemoryServiceError,
)

logger = logging.getLogger(__name__)


class MemoryTool(BaseTool):
    """
    Tool for long-term memory storage and retrieval.

    Actions:
    - store: Store design summary with embedding
    - search_prompts: Search for similar prompts
    - get_history: Get user's design history
    - get_references: Search art library for references
    """

    def __init__(
        self,
        memory_service: Optional[MemoryService] = None,
    ):
        """
        Initialize the Memory Tool.

        Args:
            memory_service: Optional service instance for dependency injection
        """
        super().__init__(
            name="memory",
            description="Store and retrieve design history, prompts, and references",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=[
                        "store",
                        "search_prompts",
                        "get_history",
                        "get_references",
                    ],
                ),
                ToolParameter(
                    name="job_id",
                    type=ParameterType.STRING,
                    description="Job ID for storage",
                    required=False,
                ),
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User ID",
                    required=False,
                ),
                ToolParameter(
                    name="summary",
                    type=ParameterType.STRING,
                    description="Design summary to store",
                    required=False,
                ),
                ToolParameter(
                    name="prompt",
                    type=ParameterType.STRING,
                    description="Prompt text",
                    required=False,
                ),
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query",
                    required=False,
                ),
                ToolParameter(
                    name="product_type",
                    type=ParameterType.STRING,
                    description="Product type filter",
                    required=False,
                ),
                ToolParameter(
                    name="style",
                    type=ParameterType.STRING,
                    description="Style for storage",
                    required=False,
                ),
                ToolParameter(
                    name="image_url",
                    type=ParameterType.STRING,
                    description="Image URL for storage",
                    required=False,
                ),
                ToolParameter(
                    name="category",
                    type=ParameterType.STRING,
                    description="Category filter for art references",
                    required=False,
                ),
                ToolParameter(
                    name="min_score",
                    type=ParameterType.NUMBER,
                    description="Minimum evaluation score for search",
                    required=False,
                    default=0.7,
                ),
                ToolParameter(
                    name="limit",
                    type=ParameterType.INTEGER,
                    description="Max results to return",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="metadata",
                    type=ParameterType.OBJECT,
                    description="Additional metadata for storage",
                    required=False,
                ),
            ],
        )

        self._memory_service = memory_service
        self._owns_service = memory_service is None

    async def _get_service(self) -> MemoryService:
        """Get or create memory service."""
        if self._memory_service is None:
            self._memory_service = MemoryService()
        return self._memory_service

    async def close(self) -> None:
        """Close resources."""
        if self._memory_service and self._owns_service:
            await self._memory_service.close()
            self._memory_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute memory action.

        Args:
            action: The action to perform
            job_id: Job ID
            user_id: User ID
            summary: Design summary
            prompt: Prompt text
            query: Search query
            product_type: Product type
            style: Style
            image_url: Image URL
            category: Category filter
            min_score: Minimum score
            limit: Max results
            metadata: Additional metadata

        Returns:
            ToolResult with memory data
        """
        action = kwargs.get("action")

        try:
            if action == "store":
                return await self._store(
                    job_id=kwargs.get("job_id"),
                    user_id=kwargs.get("user_id"),
                    summary=kwargs.get("summary"),
                    prompt=kwargs.get("prompt"),
                    product_type=kwargs.get("product_type"),
                    style=kwargs.get("style"),
                    image_url=kwargs.get("image_url"),
                    metadata=kwargs.get("metadata"),
                )
            elif action == "search_prompts":
                return await self._search_prompts(
                    query=kwargs.get("query"),
                    user_id=kwargs.get("user_id"),
                    product_type=kwargs.get("product_type"),
                    min_score=kwargs.get("min_score", 0.7),
                    limit=kwargs.get("limit", 10),
                )
            elif action == "get_history":
                return await self._get_history(
                    user_id=kwargs.get("user_id"),
                    limit=kwargs.get("limit", 10),
                )
            elif action == "get_references":
                return await self._get_references(
                    query=kwargs.get("query"),
                    category=kwargs.get("category"),
                    limit=kwargs.get("limit", 10),
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except MemoryServiceError as e:
            logger.error(f"Memory service error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Memory service error: {e}",
                error_code="MEMORY_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in MemoryTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _store(
        self,
        job_id: Optional[str],
        user_id: Optional[str],
        summary: Optional[str],
        prompt: Optional[str],
        product_type: Optional[str],
        style: Optional[str],
        image_url: Optional[str],
        metadata: Optional[Dict[str, Any]],
    ) -> ToolResult:
        """Store design summary with embedding."""
        if not job_id:
            return ToolResult(
                success=False,
                data=None,
                error="job_id is required for store",
                error_code="MISSING_PARAMETER",
            )

        if not user_id:
            return ToolResult(
                success=False,
                data=None,
                error="user_id is required for store",
                error_code="MISSING_PARAMETER",
            )

        if not summary:
            return ToolResult(
                success=False,
                data=None,
                error="summary is required for store",
                error_code="MISSING_PARAMETER",
            )

        service = await self._get_service()
        record_id = await service.store_design_summary(
            job_id=job_id,
            user_id=user_id,
            summary=summary,
            prompt=prompt or "",
            product_type=product_type,
            style=style,
            image_url=image_url,
            metadata=metadata,
        )

        return ToolResult(
            success=True,
            data={
                "id": record_id,
                "job_id": job_id,
                "stored": True,
            },
        )

    async def _search_prompts(
        self,
        query: Optional[str],
        user_id: Optional[str],
        product_type: Optional[str],
        min_score: float,
        limit: int,
    ) -> ToolResult:
        """Search for similar prompts."""
        if not query:
            return ToolResult(
                success=False,
                data=None,
                error="query is required for search_prompts",
                error_code="MISSING_PARAMETER",
            )

        service = await self._get_service()
        results = await service.search_similar_prompts(
            query=query,
            user_id=user_id,
            product_type=product_type,
            min_score=min_score,
            limit=limit,
        )

        return ToolResult(
            success=True,
            data={
                "results": results,
                "count": len(results),
                "query": query,
            },
        )

    async def _get_history(
        self,
        user_id: Optional[str],
        limit: int,
    ) -> ToolResult:
        """Get user's design history."""
        if not user_id:
            return ToolResult(
                success=False,
                data=None,
                error="user_id is required for get_history",
                error_code="MISSING_PARAMETER",
            )

        service = await self._get_service()
        history = await service.get_user_history(user_id, limit)

        return ToolResult(
            success=True,
            data={
                "history": history,
                "count": len(history),
                "user_id": user_id,
            },
        )

    async def _get_references(
        self,
        query: Optional[str],
        category: Optional[str],
        limit: int,
    ) -> ToolResult:
        """Search art library for references."""
        if not query:
            return ToolResult(
                success=False,
                data=None,
                error="query is required for get_references",
                error_code="MISSING_PARAMETER",
            )

        service = await self._get_service()
        references = await service.get_art_references(
            query=query,
            category=category,
            limit=limit,
        )

        return ToolResult(
            success=True,
            data={
                "references": references,
                "count": len(references),
                "query": query,
            },
        )

    async def __aenter__(self) -> "MemoryTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
