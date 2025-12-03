"""
Product Template Service
HTTP client for fetching product templates from admin-api.

Phase 7.1.2: Generator Integration
This service connects to the admin-api to fetch product templates
that users can select during image generation.
"""

import httpx
import logging
from typing import List, Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)


class ProductTemplateServiceError(Exception):
    """Base exception for ProductTemplateService errors"""
    pass


class ProductTemplateNotFoundError(ProductTemplateServiceError):
    """Raised when product template is not found"""
    pass


class ProductTemplateService:
    """
    Client for admin-api product template endpoints.

    This service fetches product templates (t-shirts, mugs, etc.) from the
    admin-api, which stores templates created via the admin frontend.
    """

    def __init__(self, admin_api_url: Optional[str] = None):
        """
        Initialize ProductTemplateService with admin-api URL.

        Args:
            admin_api_url: Base URL for admin-api. If not provided, uses settings.
        """
        self.base_url = admin_api_url or getattr(settings, 'admin_api_url', None)

        if not self.base_url:
            logger.warning(
                "ADMIN_API_URL not configured. Product template fetching will fail. "
                "Set ADMIN_API_URL environment variable."
            )
            self.base_url = "https://palet8-admin-api-702210710671.us-central1.run.app"

        self.client = httpx.AsyncClient(timeout=10.0)
        logger.info(f"ProductTemplateService initialized with base_url: {self.base_url}")

    async def list_templates(
        self,
        category: Optional[str] = None,
        is_active: bool = True,
        is_suspended: bool = False,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Fetch list of product templates from admin-api.

        Args:
            category: Filter by category (apparel, drinkware, wall-art, accessories)
            is_active: Only fetch active templates
            is_suspended: Exclude suspended templates
            page_size: Number of templates to fetch (max 100)

        Returns:
            List of template dictionaries

        Raises:
            ProductTemplateServiceError: If request fails
        """
        params = {
            "isActive": str(is_active).lower(),
            "isSuspended": str(is_suspended).lower(),
            "pageSize": min(page_size, 100)  # Cap at 100
        }

        if category:
            params["category"] = category

        logger.info(f"Fetching product templates with params: {params}")

        try:
            response = await self.client.get(
                f"{self.base_url}/api/product-templates",
                params=params
            )
            response.raise_for_status()

            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                logger.error(f"Admin API returned error: {error_msg}")
                raise ProductTemplateServiceError(f"API error: {error_msg}")

            templates = data.get("data", [])

            logger.info(f"✓ Fetched {len(templates)} product templates")
            return templates

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error fetching templates: {e.response.status_code} - {e.response.text}"
            )
            raise ProductTemplateServiceError(
                f"Failed to fetch templates: HTTP {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            logger.error(f"Network error fetching templates: {e}")
            raise ProductTemplateServiceError(
                f"Network error: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching templates: {e}", exc_info=True)
            raise ProductTemplateServiceError(
                f"Unexpected error: {str(e)}"
            ) from e

    async def get_template(self, template_id: str) -> Dict[str, Any]:
        """
        Fetch single product template by ID from admin-api.

        Args:
            template_id: UUID of product template

        Returns:
            Template dictionary with full details including:
            - id, name, description, category
            - images (list with urls)
            - designArea (x, y, width, height, unit)
            - printSpecifications
            - price, cost
            - isActive, isSuspended

        Raises:
            ProductTemplateNotFoundError: If template not found (404)
            ProductTemplateServiceError: If request fails
        """
        logger.info(f"Fetching product template: {template_id}")

        try:
            response = await self.client.get(
                f"{self.base_url}/api/product-templates/{template_id}"
            )
            response.raise_for_status()

            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", "Unknown error")
                logger.error(f"Admin API returned error: {error_msg}")
                raise ProductTemplateServiceError(f"API error: {error_msg}")

            template = data.get("data", {})

            if not template:
                raise ProductTemplateNotFoundError(f"Template {template_id} not found")

            logger.info(
                f"✓ Fetched template: {template.get('name', 'Unknown')} "
                f"({template.get('category', 'Unknown')})"
            )

            # Log design area for debugging Phase 7.2
            if "designArea" in template:
                logger.debug(f"Design Area: {template['designArea']}")

            return template

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.warning(f"Template {template_id} not found (404)")
                raise ProductTemplateNotFoundError(
                    f"Template {template_id} not found"
                ) from e

            logger.error(
                f"HTTP error fetching template: {e.response.status_code} - {e.response.text}"
            )
            raise ProductTemplateServiceError(
                f"Failed to fetch template: HTTP {e.response.status_code}"
            ) from e
        except ProductTemplateNotFoundError:
            # Re-raise as-is
            raise
        except httpx.RequestError as e:
            logger.error(f"Network error fetching template: {e}")
            raise ProductTemplateServiceError(
                f"Network error: {str(e)}"
            ) from e
        except Exception as e:
            logger.error(f"Unexpected error fetching template: {e}", exc_info=True)
            raise ProductTemplateServiceError(
                f"Unexpected error: {str(e)}"
            ) from e

    async def close(self):
        """Close the HTTP client connection."""
        await self.client.aclose()
        logger.debug("ProductTemplateService HTTP client closed")


# Singleton instance for convenience
_product_template_service: Optional[ProductTemplateService] = None


def get_product_template_service() -> ProductTemplateService:
    """
    Get or create singleton ProductTemplateService instance.

    Returns:
        ProductTemplateService instance
    """
    global _product_template_service

    if _product_template_service is None:
        _product_template_service = ProductTemplateService()

    return _product_template_service


# Export for convenience
product_template_service = get_product_template_service()


# Example usage:
"""
from src.services.product_template_service import product_template_service

# List templates by category
templates = await product_template_service.list_templates(category="apparel")
print(f"Found {len(templates)} apparel templates")

# Get specific template
template = await product_template_service.get_template("12b02b0b-ea91-45fa-9ee0-c8b137888cc8")
print(f"Template: {template['name']}")
print(f"Design Area: {template['designArea']}")
print(f"Price: ${template['price']}")

# Clean up
await product_template_service.close()
"""
