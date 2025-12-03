"""
Context Tool - RAG and context retrieval.

This tool provides agents with access to user history, art library,
and similar designs via RAG retrieval.

Documentation Reference: Section 5.3.1
"""

from typing import Any, Dict, List, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult

from palet8_agents.services.context_service import ContextService, ContextServiceError

logger = logging.getLogger(__name__)


class ContextTool(BaseTool):
    """
    Context retrieval tool for RAG operations.

    Provides access to:
    - User History (Relational DB)
    - Art Library RAG (Vector DB)
    - Similar Designs (Vector DB)

    Methods (from Documentation Section 5.3.1):
    - fetch_user_history(user_id, limit) -> List[Dict]
    - search_art_library(query, limit) -> List[Dict]
    - search_similar_designs(query, limit) -> List[Dict]
    - build_context(user_id, requirements) -> Context
    """

    def __init__(
        self,
        context_service: Optional[ContextService] = None,
    ):
        """
        Initialize the Context Tool.

        Args:
            context_service: Optional ContextService instance for dependency injection
        """
        super().__init__(
            name="context",
            description="Retrieve context via RAG - user history, art library, and similar designs",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform: fetch_user_history, search_art_library, search_similar_designs, build_context",
                    required=True,
                    enum=["fetch_user_history", "search_art_library", "search_similar_designs", "build_context"],
                ),
                ToolParameter(
                    name="user_id",
                    type=ParameterType.STRING,
                    description="User ID for user history retrieval",
                    required=False,
                ),
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query for art library or similar designs",
                    required=False,
                ),
                ToolParameter(
                    name="limit",
                    type=ParameterType.INTEGER,
                    description="Maximum number of results to return",
                    required=False,
                    default=10,
                ),
                ToolParameter(
                    name="category",
                    type=ParameterType.STRING,
                    description="Category filter for art library search",
                    required=False,
                ),
                ToolParameter(
                    name="min_score",
                    type=ParameterType.NUMBER,
                    description="Minimum evaluation score for similar prompts",
                    required=False,
                    default=0.0,
                ),
                ToolParameter(
                    name="requirements",
                    type=ParameterType.OBJECT,
                    description="Requirements dict for build_context action",
                    required=False,
                ),
            ],
        )

        self._context_service = context_service
        self._owns_service = context_service is None

    async def _get_context_service(self) -> ContextService:
        """Get or create context service."""
        if self._context_service is None:
            self._context_service = ContextService()
        return self._context_service

    async def close(self) -> None:
        """Close resources."""
        if self._context_service and self._owns_service:
            await self._context_service.close()
            self._context_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute context retrieval action.

        Args:
            action: The action to perform
            user_id: User ID for history retrieval
            query: Search query
            limit: Maximum results
            category: Category filter
            min_score: Minimum evaluation score
            requirements: Requirements dict for build_context

        Returns:
            ToolResult with retrieved context
        """
        action = kwargs.get("action")
        user_id = kwargs.get("user_id")
        query = kwargs.get("query")
        limit = kwargs.get("limit", 10)
        category = kwargs.get("category")
        min_score = kwargs.get("min_score", 0.0)
        requirements = kwargs.get("requirements", {})

        try:
            if action == "fetch_user_history":
                return await self._fetch_user_history(user_id, limit)
            elif action == "search_art_library":
                return await self._search_art_library(query, limit, category)
            elif action == "search_similar_designs":
                return await self._search_similar_designs(query, limit, min_score)
            elif action == "build_context":
                return await self._build_context(user_id, requirements, query)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except ContextServiceError as e:
            logger.error(f"Context service error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Context service error: {e}",
                error_code="CONTEXT_SERVICE_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in ContextTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _fetch_user_history(
        self,
        user_id: Optional[str],
        limit: int,
    ) -> ToolResult:
        """Fetch user's design history."""
        if not user_id:
            return ToolResult(
                success=False,
                data=None,
                error="user_id is required for fetch_user_history",
                error_code="MISSING_PARAMETER",
            )

        context_service = await self._get_context_service()
        history = await context_service.get_user_history(user_id, limit)

        return ToolResult(
            success=True,
            data={
                "user_id": user_id,
                "history": [h.to_dict() for h in history],
                "count": len(history),
            },
        )

    async def _search_art_library(
        self,
        query: Optional[str],
        limit: int,
        category: Optional[str] = None,
    ) -> ToolResult:
        """Search art library via vector similarity."""
        if not query:
            return ToolResult(
                success=False,
                data=None,
                error="query is required for search_art_library",
                error_code="MISSING_PARAMETER",
            )

        context_service = await self._get_context_service()
        results = await context_service.search_art_library(query, limit, category)

        return ToolResult(
            success=True,
            data={
                "query": query,
                "category": category,
                "results": [r.to_dict() for r in results],
                "count": len(results),
            },
        )

    async def _search_similar_designs(
        self,
        query: Optional[str],
        limit: int,
        min_score: float = 0.0,
    ) -> ToolResult:
        """Search for similar designs via vector similarity."""
        if not query:
            return ToolResult(
                success=False,
                data=None,
                error="query is required for search_similar_designs",
                error_code="MISSING_PARAMETER",
            )

        context_service = await self._get_context_service()
        results = await context_service.search_similar_prompts(query, limit, min_score)

        return ToolResult(
            success=True,
            data={
                "query": query,
                "min_score": min_score,
                "results": [r.to_dict() for r in results],
                "count": len(results),
            },
        )

    async def _build_context(
        self,
        user_id: Optional[str],
        requirements: Dict[str, Any],
        query: Optional[str] = None,
    ) -> ToolResult:
        """Build aggregated context for generation."""
        if not user_id:
            return ToolResult(
                success=False,
                data=None,
                error="user_id is required for build_context",
                error_code="MISSING_PARAMETER",
            )

        context_service = await self._get_context_service()
        context = await context_service.build_context(
            user_id=user_id,
            requirements=requirements,
            query=query,
        )

        return ToolResult(
            success=True,
            data={
                "context": context.to_dict(),
                "has_context": context.has_context,
                "prompt_context": context.to_prompt_context(),
            },
        )

    async def __aenter__(self) -> "ContextTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
