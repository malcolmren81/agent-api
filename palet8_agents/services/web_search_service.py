"""
Web Search Service - Online information retrieval.

This service provides web search capabilities using:
- Primary: OpenAI GPT-4o-mini Search Preview
- Fallback: Tavily API / Serper API

Documentation Reference: Section 6.4
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import os
import httpx

from palet8_agents.core.config import get_config
from palet8_agents.core.llm_client import LLMClient
from palet8_agents.core.exceptions import LLMClientError

logger = logging.getLogger(__name__)


class WebSearchServiceError(Exception):
    """Base exception for WebSearchService errors."""
    pass


class SearchProviderError(WebSearchServiceError):
    """Raised when search provider fails."""
    pass


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class WebSearchResponse:
    """Response from web search."""
    query: str
    results: List[SearchResult]
    answer: Optional[str] = None  # AI-generated answer summary
    provider: str = ""
    total_results: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "answer": self.answer,
            "provider": self.provider,
            "total_results": self.total_results,
            "metadata": self.metadata,
        }


class WebSearchService:
    """
    Service for web search and online information retrieval.

    Features:
    - Primary: OpenAI GPT-4o-mini Search Preview (grounded search)
    - Fallback: Tavily API for reliable search results
    - Optional: Serper API as additional fallback
    - AI-generated answer summaries
    - Result ranking and filtering

    Usage:
        service = WebSearchService()
        response = await service.search("latest trends in streetwear design")
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        tavily_api_key: Optional[str] = None,
        serper_api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize WebSearchService.

        Args:
            llm_client: Optional LLMClient for OpenAI search
            tavily_api_key: Tavily API key (reads from env if None)
            serper_api_key: Serper API key (reads from env if None)
            timeout: Request timeout in seconds
        """
        self._llm_client = llm_client
        self._owns_client = llm_client is None
        self.tavily_api_key = tavily_api_key or os.environ.get("TAVILY_API_KEY", "")
        self.serper_api_key = serper_api_key or os.environ.get("SERPER_API_KEY", "")
        self.timeout = timeout
        self._http_client: Optional[httpx.AsyncClient] = None
        self._config = get_config()

    async def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
            )
        return self._http_client

    async def close(self) -> None:
        """Close the service and release resources."""
        if self._llm_client and self._owns_client:
            await self._llm_client.close()
            self._llm_client = None
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        include_answer: bool = True,
    ) -> WebSearchResponse:
        """
        Perform web search.

        Tries OpenAI search first, falls back to Tavily/Serper on failure.

        Args:
            query: Search query
            max_results: Maximum number of results to return
            include_answer: Whether to include AI-generated answer

        Returns:
            WebSearchResponse with results and optional answer

        Raises:
            WebSearchServiceError: If all providers fail
        """
        # Try OpenAI GPT-4o-mini Search Preview first
        try:
            return await self._search_openai(query, max_results, include_answer)
        except Exception as e:
            logger.warning(f"OpenAI search failed: {e}, trying fallback")

        # Fallback to Tavily
        if self.tavily_api_key:
            try:
                return await self._search_tavily(query, max_results, include_answer)
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}, trying Serper")

        # Fallback to Serper
        if self.serper_api_key:
            try:
                return await self._search_serper(query, max_results)
            except Exception as e:
                logger.error(f"Serper search failed: {e}")

        raise WebSearchServiceError(f"All search providers failed for query: {query}")

    async def _search_openai(
        self,
        query: str,
        max_results: int,
        include_answer: bool,
    ) -> WebSearchResponse:
        """Search using OpenAI GPT-4o-mini Search Preview."""
        client = await self._get_llm_client()
        profile = self._config.get_model_profile("search")
        model = profile.primary_model if profile else "openai/gpt-4o-mini-search-preview"

        messages = [
            {
                "role": "system",
                "content": "You are a helpful search assistant. Search the web and provide accurate, up-to-date information with source citations.",
            },
            {
                "role": "user",
                "content": f"Search for: {query}\n\nProvide {max_results} relevant results with titles, URLs, and brief descriptions.",
            },
        ]

        try:
            response = await client.chat(
                messages=messages,
                model=model,
                temperature=0.0,
                max_tokens=2000,
            )

            # Parse the response - OpenAI search returns structured results
            results = self._parse_openai_response(response.content, max_results)

            return WebSearchResponse(
                query=query,
                results=results,
                answer=response.content if include_answer else None,
                provider="openai",
                total_results=len(results),
                metadata={"model": model},
            )

        except LLMClientError as e:
            raise SearchProviderError(f"OpenAI search error: {e}") from e

    async def _search_tavily(
        self,
        query: str,
        max_results: int,
        include_answer: bool,
    ) -> WebSearchResponse:
        """Search using Tavily API."""
        client = await self._get_http_client()

        payload = {
            "api_key": self.tavily_api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": include_answer,
            "search_depth": "basic",
        }

        try:
            response = await client.post(
                "https://api.tavily.com/search",
                json=payload,
            )

            if response.status_code != 200:
                raise SearchProviderError(f"Tavily API error ({response.status_code}): {response.text}")

            data = response.json()

            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    snippet=r.get("content", ""),
                    score=r.get("score", 0.0),
                )
                for r in data.get("results", [])
            ]

            return WebSearchResponse(
                query=query,
                results=results,
                answer=data.get("answer") if include_answer else None,
                provider="tavily",
                total_results=len(results),
                metadata={"response_time": data.get("response_time")},
            )

        except httpx.TimeoutException:
            raise SearchProviderError(f"Tavily request timed out after {self.timeout}s")

    async def _search_serper(
        self,
        query: str,
        max_results: int,
    ) -> WebSearchResponse:
        """Search using Serper API."""
        client = await self._get_http_client()

        try:
            response = await client.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": self.serper_api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "num": max_results,
                },
            )

            if response.status_code != 200:
                raise SearchProviderError(f"Serper API error ({response.status_code}): {response.text}")

            data = response.json()

            results = [
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("link", ""),
                    snippet=r.get("snippet", ""),
                    score=float(r.get("position", 0)) / 10,  # Convert position to score
                )
                for r in data.get("organic", [])
            ]

            return WebSearchResponse(
                query=query,
                results=results,
                answer=data.get("answerBox", {}).get("snippet"),
                provider="serper",
                total_results=len(results),
                metadata={"search_time": data.get("searchTime")},
            )

        except httpx.TimeoutException:
            raise SearchProviderError(f"Serper request timed out after {self.timeout}s")

    def _parse_openai_response(
        self,
        content: str,
        max_results: int,
    ) -> List[SearchResult]:
        """Parse OpenAI search response into structured results."""
        # OpenAI search typically returns markdown-formatted results
        # This is a simple parser - can be enhanced based on actual response format
        results = []
        lines = content.split("\n")

        current_title = ""
        current_url = ""
        current_snippet = ""

        for line in lines:
            line = line.strip()
            if not line:
                if current_title and current_snippet:
                    results.append(SearchResult(
                        title=current_title,
                        url=current_url,
                        snippet=current_snippet,
                    ))
                    current_title = ""
                    current_url = ""
                    current_snippet = ""
                continue

            # Parse markdown links [title](url)
            if line.startswith("[") and "](" in line:
                try:
                    title_end = line.index("](")
                    url_end = line.index(")", title_end)
                    current_title = line[1:title_end]
                    current_url = line[title_end + 2:url_end]
                except ValueError:
                    current_snippet += line + " "
            elif line.startswith("http"):
                current_url = line
            elif line.startswith("#") or line.startswith("**"):
                current_title = line.lstrip("#* ").rstrip("*")
            else:
                current_snippet += line + " "

        # Don't forget the last result
        if current_title and current_snippet:
            results.append(SearchResult(
                title=current_title,
                url=current_url,
                snippet=current_snippet.strip(),
            ))

        return results[:max_results]

    async def search_for_reference(
        self,
        topic: str,
        context: Optional[str] = None,
    ) -> str:
        """
        Search and return a formatted reference for use in prompts.

        This is a convenience method for getting search results
        formatted for inclusion in generation prompts.

        Args:
            topic: Topic to search for
            context: Optional context for better search

        Returns:
            Formatted string with search results
        """
        query = f"{topic} {context}" if context else topic
        response = await self.search(query, max_results=3, include_answer=True)

        if response.answer:
            return f"Reference information about {topic}:\n{response.answer}"

        if response.results:
            snippets = [f"- {r.snippet}" for r in response.results[:3]]
            return f"Reference information about {topic}:\n" + "\n".join(snippets)

        return f"No reference information found for: {topic}"

    async def __aenter__(self) -> "WebSearchService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
