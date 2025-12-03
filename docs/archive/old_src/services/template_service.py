"""
Template Service

Manages template operations including CRUD operations, validation,
and template-based asset generation workflows.
"""

from prisma import Prisma
from prisma.models import Template
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class TemplateServiceError(Exception):
    """Base exception for TemplateService errors"""
    pass


class TemplateNotFoundError(TemplateServiceError):
    """Raised when template is not found"""
    pass


class TemplateService:
    """
    Service for managing Template operations.

    Templates define reusable configurations for image generation,
    including style presets, dimensions, and generation parameters.
    """

    def __init__(self, prisma: Prisma):
        """
        Initialize TemplateService with Prisma client.

        Args:
            prisma: Connected Prisma client instance
        """
        self.prisma = prisma

    async def create_template(
        self,
        shop: str,
        name: str,
        config: Dict[str, Any],
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new template.

        Args:
            shop: Shop domain
            name: Template name
            config: Template configuration (JSON)
            description: Optional template description

        Returns:
            Dictionary representation of created template

        Raises:
            TemplateServiceError: If template creation fails
        """
        try:
            logger.info(f"Creating template: shop={shop}, name={name}")

            template = await self.prisma.template.create(
                data={
                    "shop": shop,
                    "name": name,
                    "config": config,
                    "description": description or ""
                }
            )

            logger.info(f"Template created: id={template.id}")
            return template.model_dump()

        except Exception as e:
            logger.error(f"Failed to create template: {e}", exc_info=True)
            raise TemplateServiceError(f"Failed to create template: {str(e)}") from e

    async def get_template(self, template_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve template by ID.

        Args:
            template_id: Template identifier

        Returns:
            Template dictionary or None if not found
        """
        try:
            template = await self.prisma.template.find_unique(
                where={"id": template_id}
            )

            if template:
                return template.model_dump()
            return None

        except Exception as e:
            logger.error(f"Failed to get template: {e}", exc_info=True)
            raise TemplateServiceError(f"Failed to retrieve template: {str(e)}") from e

    async def get_shop_templates(
        self,
        shop: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get all templates for a shop.

        Args:
            shop: Shop domain
            limit: Maximum number of templates to return

        Returns:
            List of template dictionaries ordered by creation date
        """
        try:
            templates = await self.prisma.template.find_many(
                where={"shop": shop},
                order_by={"createdAt": "desc"},
                take=limit
            )

            logger.info(f"Retrieved {len(templates)} templates for shop={shop}")
            return [t.model_dump() for t in templates]

        except Exception as e:
            logger.error(f"Failed to get shop templates: {e}", exc_info=True)
            raise TemplateServiceError(f"Failed to retrieve shop templates: {str(e)}") from e

    async def update_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update template fields.

        Args:
            template_id: Template identifier
            name: New template name (optional)
            config: New configuration (optional)
            description: New description (optional)

        Returns:
            Dictionary representation of updated template

        Raises:
            TemplateNotFoundError: If template doesn't exist
            TemplateServiceError: If no fields provided to update
        """
        try:
            # Build update data
            update_data = {}
            if name is not None:
                update_data["name"] = name
            if config is not None:
                update_data["config"] = config
            if description is not None:
                update_data["description"] = description

            if not update_data:
                raise TemplateServiceError("No fields to update")

            logger.info(f"Updating template: id={template_id}")

            template = await self.prisma.template.update(
                where={"id": template_id},
                data=update_data
            )

            logger.info(f"Template updated: id={template.id}")
            return template.model_dump()

        except TemplateServiceError:
            raise
        except Exception as e:
            logger.error(f"Failed to update template: {e}", exc_info=True)
            if "Record to update not found" in str(e):
                raise TemplateNotFoundError(f"Template {template_id} not found") from e
            raise TemplateServiceError(f"Failed to update template: {str(e)}") from e

    async def delete_template(self, template_id: str) -> bool:
        """
        Delete template by ID.

        Args:
            template_id: Template identifier

        Returns:
            True if deleted successfully

        Raises:
            TemplateNotFoundError: If template doesn't exist
        """
        try:
            logger.info(f"Deleting template: id={template_id}")

            await self.prisma.template.delete(
                where={"id": template_id}
            )

            logger.info(f"Template deleted: id={template_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete template: {e}", exc_info=True)
            if "Record to delete does not exist" in str(e):
                raise TemplateNotFoundError(f"Template {template_id} not found") from e
            raise TemplateServiceError(f"Failed to delete template: {str(e)}") from e


# Example usage
"""
from services.template_service import TemplateService

# Initialize service
template_service = TemplateService(prisma)

# Create template
template = await template_service.create_template(
    shop="myshop.myshopify.com",
    name="Product Photography",
    config={
        "model": "stable-diffusion-xl",
        "style": "photorealistic",
        "width": 1024,
        "height": 1024,
        "steps": 50,
        "guidance_scale": 7.5
    },
    description="Professional product photos"
)

# Get all templates for shop
templates = await template_service.get_shop_templates(shop="myshop.myshopify.com")

# Update template
updated = await template_service.update_template(
    template_id="template-123",
    name="Updated Name",
    config={"model": "new-model"}
)

# Delete template
await template_service.delete_template(template_id="template-123")
"""
