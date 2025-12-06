"""
Selector Service - Caches selector data from Admin API on startup.

Provides cached access to UI selector options (aesthetics, characters, categories)
so Pali can reference them without making live API calls during conversation.

The frontend (customer-app) generates the actual HTML based on selector_id.
This service only validates that selectors exist and provides metadata.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

import httpx

from src.utils.logger import get_logger

logger = get_logger(__name__)


class SelectorServiceError(Exception):
    """Base exception for selector service errors."""
    pass


class CacheRefreshError(SelectorServiceError):
    """Error refreshing selector cache."""
    pass


class SelectorService:
    """
    Caches selector data from Admin API.

    Singleton service that loads selector options on startup and provides
    quick validation for selector_id references.
    """

    ADMIN_API_URL = "https://api.palet8.biz"
    CACHE_TTL = 3600  # 1 hour in seconds

    _instance: Optional["SelectorService"] = None
    _cache: Dict[str, Any] = {}
    _last_refresh: float = 0
    _initialized: bool = False

    def __new__(cls) -> "SelectorService":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "SelectorService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None
        cls._cache = {}
        cls._last_refresh = 0
        cls._initialized = False

    async def initialize(self) -> None:
        """Load all selector data on startup."""
        if self._initialized:
            logger.info("selector_service.already_initialized")
            return

        logger.info("selector_service.initializing")
        await self._refresh_cache()
        self._initialized = True

    async def _refresh_cache(self) -> None:
        """Fetch fresh data from Admin API."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                # Fetch all selector data in parallel
                results = await asyncio.gather(
                    client.get(f"{self.ADMIN_API_URL}/api/aesthetics/light"),
                    client.get(f"{self.ADMIN_API_URL}/api/characters/light"),
                    client.get(f"{self.ADMIN_API_URL}/api/categories"),
                    return_exceptions=True
                )

                # Parse responses
                aesthetics = self._parse_response(results[0], "aesthetics")
                characters = self._parse_response(results[1], "characters")
                categories = self._parse_response(results[2], "categories")

                self._cache = {
                    "aesthetic_style": {
                        "type": "image_selector",
                        "options": aesthetics,
                        "count": len(aesthetics),
                    },
                    "system_character": {
                        "type": "image_selector",
                        "options": characters,
                        "count": len(characters),
                    },
                    "product_category": {
                        "type": "grid_selector",
                        "options": categories,
                        "count": len(categories),
                    },
                    "aspect_ratio": self._get_static_ratios(),
                    "task_complexity": self._get_static_complexity(),
                    "reference_image": {
                        "type": "upload",
                        "options": [],
                        "count": 0,
                    },
                    "text_in_image": {
                        "type": "text_input",
                        "options": [],
                        "count": 0,
                    },
                }

                self._last_refresh = time.time()
                logger.info(
                    "selector_service.cache_refreshed",
                    selectors=list(self._cache.keys()),
                    aesthetics_count=len(aesthetics),
                    characters_count=len(characters),
                    categories_count=len(categories),
                )

            except Exception as e:
                logger.error("selector_service.refresh_failed", error_detail=str(e))
                # Initialize with static selectors only if API fails
                self._cache = {
                    "aesthetic_style": {"type": "image_selector", "options": [], "count": 0},
                    "system_character": {"type": "image_selector", "options": [], "count": 0},
                    "product_category": {"type": "grid_selector", "options": [], "count": 0},
                    "aspect_ratio": self._get_static_ratios(),
                    "task_complexity": self._get_static_complexity(),
                    "reference_image": {"type": "upload", "options": [], "count": 0},
                    "text_in_image": {"type": "text_input", "options": [], "count": 0},
                }
                self._last_refresh = time.time()

    def _parse_response(
        self, response: Any, selector_name: str
    ) -> List[Dict[str, Any]]:
        """Parse API response, handling errors gracefully."""
        if isinstance(response, Exception):
            logger.warning(
                "selector_service.api_error",
                selector=selector_name,
                error=str(response),
            )
            return []

        if response.status_code != 200:
            logger.warning(
                "selector_service.api_status_error",
                selector=selector_name,
                status=response.status_code,
            )
            return []

        try:
            data = response.json()
            # Handle both array and object responses
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                # Check common response patterns
                if "data" in data:
                    return data["data"]
                if "items" in data:
                    return data["items"]
                if selector_name in data:
                    return data[selector_name]
            return []
        except Exception as e:
            logger.warning(
                "selector_service.parse_error",
                selector=selector_name,
                error_detail=str(e),
            )
            return []

    def _get_static_ratios(self) -> Dict[str, Any]:
        """Get static aspect ratio options."""
        return {
            "type": "ratio_selector",
            "options": [
                {"value": "1024x1024", "label": "1:1 Square"},
                {"value": "1024x1792", "label": "9:16 Portrait"},
                {"value": "1792x1024", "label": "16:9 Landscape"},
                {"value": "768x1024", "label": "3:4 Portrait"},
                {"value": "1024x768", "label": "4:3 Landscape"},
                {"value": "683x1024", "label": "2:3 Portrait"},
                {"value": "1024x683", "label": "3:2 Landscape"},
            ],
            "count": 7,
        }

    def _get_static_complexity(self) -> Dict[str, Any]:
        """Get static task complexity options."""
        return {
            "type": "complexity_selector",
            "options": [
                {
                    "value": "relax",
                    "label": "Relax",
                    "description": "Simple & quick generation",
                },
                {
                    "value": "standard",
                    "label": "Standard",
                    "description": "Balanced quality & speed",
                },
                {
                    "value": "complex",
                    "label": "Complex",
                    "description": "Maximum quality & detail",
                },
            ],
            "count": 3,
        }

    def get_selector(self, selector_id: str) -> Optional[Dict[str, Any]]:
        """Get cached selector data by ID."""
        return self._cache.get(selector_id)

    def selector_exists(self, selector_id: str) -> bool:
        """Check if selector is available in cache."""
        return selector_id in self._cache

    def get_all_selector_ids(self) -> List[str]:
        """Get list of all available selector IDs."""
        return list(self._cache.keys())

    def get_selector_type(self, selector_id: str) -> Optional[str]:
        """Get the type of a selector (image_selector, grid_selector, etc.)."""
        selector = self._cache.get(selector_id)
        return selector.get("type") if selector else None

    def is_cache_stale(self) -> bool:
        """Check if cache needs refresh."""
        if not self._last_refresh:
            return True
        return (time.time() - self._last_refresh) > self.CACHE_TTL

    async def refresh_if_stale(self) -> None:
        """Refresh cache if TTL expired."""
        if self.is_cache_stale():
            logger.info("selector_service.refreshing_stale_cache")
            await self._refresh_cache()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        return {
            "initialized": self._initialized,
            "selector_count": len(self._cache),
            "last_refresh": self._last_refresh,
            "cache_age_seconds": time.time() - self._last_refresh if self._last_refresh else None,
            "is_stale": self.is_cache_stale(),
            "selectors": {
                selector_id: {
                    "type": data.get("type"),
                    "count": data.get("count", 0),
                }
                for selector_id, data in self._cache.items()
            },
        }
