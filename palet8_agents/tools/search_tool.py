"""
Search Tool - Web search for agents.

This tool provides agents with web search capabilities
for retrieving real-time online information.

Documentation Reference: Section 5.3.2
"""

from typing import Any, Dict, List, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult

from palet8_agents.services.web_search_service import WebSearchService, WebSearchServiceError

logger = logging.getLogger(__name__)


class SearchTool(BaseTool):
    """
    Web search tool for agents.

    Provides access to online information via:
    - OpenAI GPT-4o-mini Search Preview (primary)
    - Tavily API (fallback)
    - Serper API (fallback)

    Methods:
    - search(query, max_results) -> Search results
    - search_for_reference(topic, context) -> Formatted reference
    """

    def __init__(
        self,
        search_service: Optional[WebSearchService] = None,
    ):
        """
        Initialize the Search Tool.

        Args:
            search_service: Optional WebSearchService instance for dependency injection
        """
        super().__init__(
            name="search",
            description="Search the web for real-time information, trends, and references",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform: search, search_for_reference",
                    required=True,
                    enum=["search", "search_for_reference"],
                ),
                ToolParameter(
                    name="query",
                    type=ParameterType.STRING,
                    description="Search query or topic",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type=ParameterType.INTEGER,
                    description="Maximum number of results to return (default: 5)",
                    required=False,
                    default=5,
                ),
                ToolParameter(
                    name="context",
                    type=ParameterType.STRING,
                    description="Additional context for better search results",
                    required=False,
                ),
                ToolParameter(
                    name="include_answer",
                    type=ParameterType.BOOLEAN,
                    description="Include AI-generated answer summary (default: true)",
                    required=False,
                    default=True,
                ),
            ],
        )

        self._search_service = search_service
        self._owns_service = search_service is None

    async def _get_search_service(self) -> WebSearchService:
        """Get or create search service."""
        if self._search_service is None:
            self._search_service = WebSearchService()
        return self._search_service

    async def close(self) -> None:
        """Close resources."""
        if self._search_service and self._owns_service:
            await self._search_service.close()
            self._search_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute search action.

        Args:
            action: The action to perform (search, search_for_reference)
            query: Search query or topic
            max_results: Maximum number of results
            context: Additional context
            include_answer: Whether to include AI answer

        Returns:
            ToolResult with search results
        """
        action = kwargs.get("action")
        query = kwargs.get("query")
        max_results = kwargs.get("max_results", 5)
        context = kwargs.get("context")
        include_answer = kwargs.get("include_answer", True)

        if not query:
            return ToolResult(
                success=False,
                data=None,
                error="query is required",
                error_code="MISSING_PARAMETER",
            )

        try:
            if action == "search":
                return await self._search(query, max_results, include_answer)
            elif action == "search_for_reference":
                return await self._search_for_reference(query, context)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except WebSearchServiceError as e:
            logger.error(f"Search service error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Search failed: {e}",
                error_code="SEARCH_SERVICE_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in SearchTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _search(
        self,
        query: str,
        max_results: int,
        include_answer: bool,
    ) -> ToolResult:
        """Perform web search."""
        search_service = await self._get_search_service()
        response = await search_service.search(
            query=query,
            max_results=max_results,
            include_answer=include_answer,
        )

        return ToolResult(
            success=True,
            data={
                "query": response.query,
                "results": [r.to_dict() for r in response.results],
                "answer": response.answer,
                "provider": response.provider,
                "total_results": response.total_results,
            },
        )

    async def _search_for_reference(
        self,
        topic: str,
        context: Optional[str],
    ) -> ToolResult:
        """Search and return formatted reference."""
        search_service = await self._get_search_service()
        reference = await search_service.search_for_reference(
            topic=topic,
            context=context,
        )

        return ToolResult(
            success=True,
            data={
                "topic": topic,
                "reference": reference,
            },
        )

    async def __aenter__(self) -> "SearchTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
