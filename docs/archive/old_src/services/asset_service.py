"""
Asset Service

Manages the lifecycle of generated image assets including creation, updates,
retrieval, and cleanup operations with comprehensive error handling.
"""

from prisma import Prisma
from prisma.models import Asset
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class AssetServiceError(Exception):
    """Base exception for AssetService errors"""
    pass


class AssetNotFoundError(AssetServiceError):
    """Raised when asset is not found"""
    pass


class AssetCreationError(AssetServiceError):
    """Raised when asset creation fails"""
    pass


class AssetService:
    """
    Service for managing Asset lifecycle operations.

    This service provides centralized business logic for asset management,
    ensuring consistent error handling and validation across the application.
    """

    def __init__(self, prisma: Prisma):
        """
        Initialize AssetService with Prisma client.

        Args:
            prisma: Connected Prisma client instance
        """
        self.prisma = prisma

    async def create_asset(
        self,
        shop: str,
        task_id: str,
        prompt: str,
        cost: int,
        status: str = "processing"
    ) -> Dict[str, Any]:
        """
        Create a new asset record.

        Args:
            shop: Shop domain (e.g., "myshop.myshopify.com")
            task_id: Unique task identifier
            prompt: User's generation prompt
            cost: Credit cost for this generation
            status: Initial status (default: "processing")

        Returns:
            Dictionary representation of created asset

        Raises:
            AssetCreationError: If asset creation fails
        """
        try:
            logger.info(f"Creating asset for shop={shop}, task_id={task_id}, cost={cost}")

            asset = await self.prisma.asset.create(
                data={
                    "shop": shop,
                    "taskId": task_id,
                    "prompt": prompt,
                    "status": status,
                    "cost": cost,
                }
            )

            logger.info(f"Asset created successfully: id={asset.id}")
            return asset.model_dump()

        except Exception as e:
            logger.error(f"Failed to create asset: {e}", exc_info=True)
            raise AssetCreationError(f"Failed to create asset: {str(e)}") from e

    async def update_asset_completed(
        self,
        task_id: str,
        image_url: str = None
    ) -> Dict[str, Any]:
        """
        Mark asset as completed with optional image URL.

        Args:
            task_id: Task identifier
            image_url: URL of generated image (optional, defaults to None)

        Returns:
            Dictionary representation of updated asset

        Raises:
            AssetNotFoundError: If asset with task_id doesn't exist
        """
        try:
            logger.info(f"Updating asset to completed: task_id={task_id}, has_url={bool(image_url)}")

            # Build update data - only include imageUrl if provided
            update_data = {"status": "completed"}
            if image_url:
                update_data["imageUrl"] = image_url

            asset = await self.prisma.asset.update(
                where={"taskId": task_id},
                data=update_data
            )

            logger.info(f"Asset marked as completed: id={asset.id}")
            return asset.model_dump()

        except Exception as e:
            logger.error(f"Failed to update asset: {e}", exc_info=True)
            if "Record to update not found" in str(e):
                raise AssetNotFoundError(f"Asset with task_id={task_id} not found") from e
            raise AssetServiceError(f"Failed to update asset: {str(e)}") from e

    async def update_asset_failed(
        self,
        task_id: str,
        error_message: str
    ) -> Dict[str, Any]:
        """
        Mark asset as failed with error message.

        Args:
            task_id: Task identifier
            error_message: Error description

        Returns:
            Dictionary representation of updated asset

        Raises:
            AssetNotFoundError: If asset with task_id doesn't exist
        """
        try:
            logger.warning(f"Marking asset as failed: task_id={task_id}, error={error_message}")

            asset = await self.prisma.asset.update(
                where={"taskId": task_id},
                data={
                    "status": "failed"
                    # Note: Add error_message field to schema if needed
                }
            )

            logger.info(f"Asset marked as failed: id={asset.id}")
            return asset.model_dump()

        except Exception as e:
            logger.error(f"Failed to update asset: {e}", exc_info=True)
            if "Record to update not found" in str(e):
                raise AssetNotFoundError(f"Asset with task_id={task_id} not found") from e
            raise AssetServiceError(f"Failed to update asset: {str(e)}") from e

    async def get_asset_by_task_id(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve asset by task ID.

        Args:
            task_id: Task identifier

        Returns:
            Asset dictionary or None if not found
        """
        try:
            asset = await self.prisma.asset.find_unique(
                where={"taskId": task_id}
            )

            if asset:
                return asset.model_dump()
            return None

        except Exception as e:
            logger.error(f"Failed to get asset by task_id: {e}", exc_info=True)
            raise AssetServiceError(f"Failed to retrieve asset: {str(e)}") from e

    async def get_shop_assets(
        self,
        shop: str,
        limit: int = 50,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent assets for a shop.

        Args:
            shop: Shop domain
            limit: Maximum number of assets to return (default: 50)
            status: Optional status filter ("completed", "processing", "failed")

        Returns:
            List of asset dictionaries ordered by creation date (newest first)
        """
        try:
            where_clause = {"shop": shop}
            if status:
                where_clause["status"] = status

            assets = await self.prisma.asset.find_many(
                where=where_clause,
                order_by={"createdAt": "desc"},
                take=limit
            )

            logger.info(f"Retrieved {len(assets)} assets for shop={shop}")
            return [a.model_dump() for a in assets]

        except Exception as e:
            logger.error(f"Failed to get shop assets: {e}", exc_info=True)
            raise AssetServiceError(f"Failed to retrieve shop assets: {str(e)}") from e

    async def delete_old_assets(
        self,
        shop: str,
        days: int = 90
    ) -> int:
        """
        Delete assets older than specified number of days for a shop.

        Args:
            shop: Shop domain
            days: Delete assets older than this many days (default: 90)

        Returns:
            Number of assets deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            logger.info(f"Deleting assets older than {days} days for shop={shop}")

            result = await self.prisma.asset.delete_many(
                where={
                    "shop": shop,
                    "createdAt": {
                        "lt": cutoff_date
                    }
                }
            )

            deleted_count = result  # Prisma returns count directly
            logger.info(f"Deleted {deleted_count} old assets for shop={shop}")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to delete old assets: {e}", exc_info=True)
            raise AssetServiceError(f"Failed to delete old assets: {str(e)}") from e

    async def get_asset_stats(self, shop: str) -> Dict[str, int]:
        """
        Get asset statistics for a shop.

        Args:
            shop: Shop domain

        Returns:
            Dictionary with counts by status:
            {
                "total": 100,
                "completed": 95,
                "processing": 3,
                "failed": 2
            }
        """
        try:
            # Get all assets for shop
            assets = await self.prisma.asset.find_many(
                where={"shop": shop},
                select={"status": True}
            )

            # Count by status
            stats = {
                "total": len(assets),
                "completed": sum(1 for a in assets if a.status == "completed"),
                "processing": sum(1 for a in assets if a.status == "processing"),
                "failed": sum(1 for a in assets if a.status == "failed")
            }

            logger.info(f"Asset stats for shop={shop}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get asset stats: {e}", exc_info=True)
            raise AssetServiceError(f"Failed to get asset statistics: {str(e)}") from e


# Example usage in orchestrator.py
"""
from services.asset_service import AssetService

# In orchestrator initialization
asset_service = AssetService(prisma)

# When creating asset (replace direct Prisma call)
asset = await asset_service.create_asset(
    shop=shop_domain,
    task_id=task_id,
    prompt=prompt,
    cost=credit_cost
)

# When generation completes
await asset_service.update_asset_completed(
    task_id=task_id,
    image_url=generated_image_url
)

# When generation fails
await asset_service.update_asset_failed(
    task_id=task_id,
    error_message=str(error)
)
"""
