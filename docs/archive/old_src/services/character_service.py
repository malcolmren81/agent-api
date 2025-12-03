"""
Character Reference Service
HTTP client for fetching character references from admin-api.

Phase 7.4: Character References Management
This service connects to the admin-api to fetch character references
that users can select during image generation to include specific characters.
"""

import httpx
import logging
from typing import List, Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class CharacterServiceError(Exception):
    """Base exception for CharacterService errors"""
    pass


class CharacterNotFoundError(CharacterServiceError):
    """Raised when character reference is not found"""
    pass


class CharacterService:
    """
    Client for admin-api character reference endpoints.

    This service fetches character references (mascots, brand characters, influencers)
    from the admin-api, which stores characters created via the admin frontend.
    """

    def __init__(self, admin_api_url: Optional[str] = None):
        """
        Initialize CharacterService with admin-api URL.

        Args:
            admin_api_url: Base URL for admin-api. If not provided, uses settings.
        """
        self.base_url = admin_api_url or getattr(settings, 'admin_api_url', None)

        if not self.base_url:
            logger.warning(
                "ADMIN_API_URL not configured. Character fetching will fail. "
                "Set ADMIN_API_URL environment variable."
            )
            self.base_url = "https://palet8-admin-api-702210710671.us-central1.run.app"

        self.client = httpx.AsyncClient(timeout=10.0)
        logger.info(f"CharacterService initialized with base_url: {self.base_url}")

    async def list_characters(
        self,
        category_id: Optional[str] = None,
        is_active: bool = True,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch list of character references from admin-api.

        Args:
            category_id: Filter by category (2-digit ID like "01", "02")
            is_active: Only fetch active characters
            page_size: Number of characters to fetch (max 100)

        Returns:
            List of character dictionaries with:
            - id: UUID
            - displayId: Human-readable ID (chr-01-001)
            - name: Character name
            - mainImage: Main image URL
            - categoryRel: {id, name}

        Raises:
            CharacterServiceError: If request fails
        """
        params = {
            "isActive": str(is_active).lower(),
            "pageSize": min(page_size, 100)  # Cap at 100
        }

        if category_id:
            params["category_id"] = category_id

        logger.info(f"Fetching character references with params: {params}")

        try:
            response = await self.client.get(
                f"{self.base_url}/api/characters/light",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                logger.error(f"Admin API returned error: {error_msg}")
                raise CharacterServiceError(f"API error: {error_msg}")

            characters = data.get("data", [])

            logger.info(f"✓ Fetched {len(characters)} character references")
            return characters

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching characters: {e.response.status_code} - {e.response.text}"
            )
            raise CharacterServiceError(
                f"Failed to fetch characters: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Network error fetching characters: {e}")
            raise CharacterServiceError(
                f"Network error: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching characters: {e}", exc_info=True)
            raise CharacterServiceError(
                f"Unexpected error: {str(e)}"
            ) from e

    async def get_character(self, character_id: str) -> Dict[str, Any]:
        """
        Fetch single character reference by ID from admin-api.

        Args:
            character_id: UUID of character reference

        Returns:
            Character dictionary with full details including:
            - id, displayId, name, description
            - categoryId, categoryRel {id, name}
            - mainImage (GCS URL)
            - creativeAssets (array of {url, position})
            - isActive
            - createdAt, updatedAt

        Raises:
            CharacterNotFoundError: If character not found (404)
            CharacterServiceError: If request fails
        """
        logger.info(f"Fetching character reference: {character_id}")

        try:
            response = await self.client.get(
                f"{self.base_url}/api/characters/{character_id}"
            )
            response.raise_for_status()

            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                logger.error(f"Admin API returned error: {error_msg}")
                raise CharacterServiceError(f"API error: {error_msg}")

            character = data.get("data", {})

            if not character:
                raise CharacterNotFoundError(f"Character {character_id} not found")

            logger.info(
                f"✓ Fetched character: {character.get('name', 'Unknown')} "
                f"({character.get('displayId', 'Unknown')})"
            )

            # Log creative assets for debugging Phase 7.4
            if "creativeAssets" in character:
                logger.debug(f"Creative Assets: {len(character['creativeAssets'])} assets")

            return character

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Character {character_id} not found (404)")
                raise CharacterNotFoundError(
                    f"Character {character_id} not found"
                ) from e

            logger.error(
                f"HTTP error fetching character: {e.response.status_code} - {e.response.text}"
            )
            raise CharacterServiceError(
                f"Failed to fetch character: HTTP {e.response.status_code}"
            ) from e
        except CharacterNotFoundError:
            # Re-raise as-is
            raise
        except httpx.RequestError as e:
            logger.error(f"Network error fetching character: {e}")
            raise CharacterServiceError(
                f"Network error: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching character: {e}", exc_info=True)
            raise CharacterServiceError(
                f"Unexpected error: {str(e)}"
            ) from e

    async def close(self):
        """Close the HTTP client connection."""
        await self.client.aclose()
        logger.debug("CharacterService HTTP client closed")


# Singleton instance for convenience
_character_service: Optional[CharacterService] = None


def get_character_service() -> CharacterService:
    """
    Get or create singleton CharacterService instance.

    Returns:
        CharacterService instance
    """
    global _character_service

    if _character_service is None:
        _character_service = CharacterService()

    return _character_service


# Export for convenience
character_service = get_character_service()


# Example usage:
"""
from src.services.character_service import character_service

# List all active characters
characters = await character_service.list_characters()
print(f"Found {len(characters)} characters")

# List characters by category
mascots = await character_service.list_characters(category_id="01")
print(f"Found {len(mascots)} mascot characters")

# Get specific character
character = await character_service.get_character("da84e044-2a1f-49e6-bedd-f5c2699d567b")
print(f"Character: {character['name']}")
print(f"Main Image: {character['mainImage']}")
print(f"Creative Assets: {len(character['creativeAssets'])} assets")

# Clean up
await character_service.close()
"""
