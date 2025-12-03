"""
Aesthetic Reference Service
HTTP client for fetching aesthetic references from admin-api.

Phase 7.3: Aesthetic References Management
This service connects to the admin-api to fetch aesthetic references
that users can select during image generation to guide visual style.
"""

import httpx
import logging
from typing import List, Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class AestheticServiceError(Exception):
    """Base exception for AestheticService errors"""
    pass


class AestheticNotFoundError(AestheticServiceError):
    """Raised when aesthetic reference is not found"""
    pass


class AestheticService:
    """
    Client for admin-api aesthetic reference endpoints.

    This service fetches aesthetic references (mood boards, visual styles) from the
    admin-api, which stores aesthetics created via the admin frontend.
    """

    def __init__(self, admin_api_url: Optional[str] = None):
        """
        Initialize AestheticService with admin-api URL.

        Args:
            admin_api_url: Base URL for admin-api. If not provided, uses settings.
        """
        self.base_url = admin_api_url or getattr(settings, 'admin_api_url', None)

        if not self.base_url:
            logger.warning(
                "ADMIN_API_URL not configured. Aesthetic fetching will fail. "
                "Set ADMIN_API_URL environment variable."
            )
            self.base_url = "https://palet8-admin-api-702210710671.us-central1.run.app"

        self.client = httpx.AsyncClient(timeout=10.0)
        logger.info(f"AestheticService initialized with base_url: {self.base_url}")

    async def list_aesthetics(
        self,
        category_id: Optional[str] = None,
        is_active: bool = True,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch list of aesthetic references from admin-api.

        Args:
            category_id: Filter by category (2-digit ID like "01", "02")
            is_active: Only fetch active aesthetics
            page_size: Number of aesthetics to fetch (max 100)

        Returns:
            List of aesthetic dictionaries with:
            - id: UUID
            - displayId: Human-readable ID (aes-01-001)
            - name: Aesthetic name
            - mainImage: Main image URL
            - categoryRel: {id, name}

        Raises:
            AestheticServiceError: If request fails
        """
        params = {
            "isActive": str(is_active).lower(),
            "pageSize": min(page_size, 100)  # Cap at 100
        }

        if category_id:
            params["category_id"] = category_id

        logger.info(f"Fetching aesthetic references with params: {params}")

        try:
            response = await self.client.get(
                f"{self.base_url}/api/aesthetics/light",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                logger.error(f"Admin API returned error: {error_msg}")
                raise AestheticServiceError(f"API error: {error_msg}")

            aesthetics = data.get("data", [])

            logger.info(f"✓ Fetched {len(aesthetics)} aesthetic references")
            return aesthetics

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching aesthetics: {e.response.status_code} - {e.response.text}"
            )
            raise AestheticServiceError(
                f"Failed to fetch aesthetics: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Network error fetching aesthetics: {e}")
            raise AestheticServiceError(
                f"Network error: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching aesthetics: {e}", exc_info=True)
            raise AestheticServiceError(
                f"Unexpected error: {str(e)}"
            ) from e

    async def get_aesthetic(self, aesthetic_id: str) -> Dict[str, Any]:
        """
        Fetch single aesthetic reference by ID from admin-api.

        Args:
            aesthetic_id: UUID of aesthetic reference

        Returns:
            Aesthetic dictionary with full details including:
            - id, displayId, name, description
            - categoryId, categoryRel {id, name}
            - mainImage (GCS URL)
            - moodBoardImages (array of {url, position})
            - isActive
            - createdAt, updatedAt

        Raises:
            AestheticNotFoundError: If aesthetic not found (404)
            AestheticServiceError: If request fails
        """
        logger.info(f"Fetching aesthetic reference: {aesthetic_id}")

        try:
            response = await self.client.get(
                f"{self.base_url}/api/aesthetics/{aesthetic_id}"
            )
            response.raise_for_status()

            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                logger.error(f"Admin API returned error: {error_msg}")
                raise AestheticServiceError(f"API error: {error_msg}")

            aesthetic = data.get("data", {})

            if not aesthetic:
                raise AestheticNotFoundError(f"Aesthetic {aesthetic_id} not found")

            logger.info(
                f"✓ Fetched aesthetic: {aesthetic.get('name', 'Unknown')} "
                f"({aesthetic.get('displayId', 'Unknown')})"
            )

            # Log mood board for debugging Phase 7.3
            if "moodBoardImages" in aesthetic:
                logger.debug(f"Mood Board Images: {len(aesthetic['moodBoardImages'])} images")

            return aesthetic

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Aesthetic {aesthetic_id} not found (404)")
                raise AestheticNotFoundError(
                    f"Aesthetic {aesthetic_id} not found"
                ) from e

            logger.error(
                f"HTTP error fetching aesthetic: {e.response.status_code} - {e.response.text}"
            )
            raise AestheticServiceError(
                f"Failed to fetch aesthetic: HTTP {e.response.status_code}"
            ) from e
        except AestheticNotFoundError:
            # Re-raise as-is
            raise
        except httpx.RequestError as e:
            logger.error(f"Network error fetching aesthetic: {e}")
            raise AestheticServiceError(
                f"Network error: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching aesthetic: {e}", exc_info=True)
            raise AestheticServiceError(
                f"Unexpected error: {str(e)}"
            ) from e

    async def close(self):
        """Close the HTTP client connection."""
        await self.client.aclose()
        logger.debug("AestheticService HTTP client closed")


# Singleton instance for convenience
_aesthetic_service: Optional[AestheticService] = None


def get_aesthetic_service() -> AestheticService:
    """
    Get or create singleton AestheticService instance.

    Returns:
        AestheticService instance
    """
    global _aesthetic_service

    if _aesthetic_service is None:
        _aesthetic_service = AestheticService()

    return _aesthetic_service


# Export for convenience
aesthetic_service = get_aesthetic_service()


# Example usage:
"""
from src.services.aesthetic_service import aesthetic_service

# List all active aesthetics
aesthetics = await aesthetic_service.list_aesthetics()
print(f"Found {len(aesthetics)} aesthetics")

# List aesthetics by category
minimalist = await aesthetic_service.list_aesthetics(category_id="01")
print(f"Found {len(minimalist)} minimalist aesthetics")

# Get specific aesthetic
aesthetic = await aesthetic_service.get_aesthetic("0ff594eb-5aa7-4ea4-8bb4-34e529069388")
print(f"Aesthetic: {aesthetic['name']}")
print(f"Main Image: {aesthetic['mainImage']}")
print(f"Mood Board: {len(aesthetic['moodBoardImages'])} images")

# Clean up
await aesthetic_service.close()
"""
