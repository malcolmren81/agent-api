"""
Unit tests for AssetService

Tests all CRUD operations, error handling, and edge cases for asset management.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
from services.asset_service import (
    AssetService,
    AssetNotFoundError,
    AssetCreationError,
    AssetServiceError
)


@pytest.fixture
def mock_prisma():
    """Mock Prisma client"""
    prisma = Mock()
    prisma.asset = Mock()
    return prisma


@pytest.fixture
def asset_service(mock_prisma):
    """AssetService instance with mocked Prisma"""
    return AssetService(mock_prisma)


@pytest.fixture
def sample_asset_data():
    """Sample asset data for testing"""
    return {
        "id": "asset-123",
        "shop": "test.myshopify.com",
        "taskId": "task-456",
        "prompt": "A red apple on a table",
        "imageUrl": None,
        "status": "processing",
        "cost": 10,
        "createdAt": datetime.now()
    }


class TestCreateAsset:
    """Tests for create_asset method"""

    @pytest.mark.asyncio
    async def test_create_asset_success(self, asset_service, mock_prisma, sample_asset_data):
        """Test successful asset creation"""
        # Arrange
        mock_asset = Mock()
        mock_asset.id = sample_asset_data["id"]
        mock_asset.model_dump.return_value = sample_asset_data
        mock_prisma.asset.create = AsyncMock(return_value=mock_asset)

        # Act
        result = await asset_service.create_asset(
            shop="test.myshopify.com",
            task_id="task-456",
            prompt="A red apple on a table",
            cost=10
        )

        # Assert
        assert result["id"] == "asset-123"
        assert result["status"] == "processing"
        assert result["shop"] == "test.myshopify.com"
        mock_prisma.asset.create.assert_called_once()

        # Verify the data passed to create
        call_args = mock_prisma.asset.create.call_args
        assert call_args[1]["data"]["shop"] == "test.myshopify.com"
        assert call_args[1]["data"]["taskId"] == "task-456"
        assert call_args[1]["data"]["cost"] == 10

    @pytest.mark.asyncio
    async def test_create_asset_with_custom_status(self, asset_service, mock_prisma):
        """Test creating asset with custom status"""
        # Arrange
        mock_asset = Mock()
        mock_asset.model_dump.return_value = {"id": "123", "status": "pending"}
        mock_prisma.asset.create = AsyncMock(return_value=mock_asset)

        # Act
        result = await asset_service.create_asset(
            shop="test.myshopify.com",
            task_id="task-789",
            prompt="test",
            cost=5,
            status="pending"
        )

        # Assert
        assert result["status"] == "pending"
        call_args = mock_prisma.asset.create.call_args
        assert call_args[1]["data"]["status"] == "pending"

    @pytest.mark.asyncio
    async def test_create_asset_failure(self, asset_service, mock_prisma):
        """Test asset creation failure"""
        # Arrange
        mock_prisma.asset.create = AsyncMock(side_effect=Exception("Database error"))

        # Act & Assert
        with pytest.raises(AssetCreationError) as exc_info:
            await asset_service.create_asset(
                shop="test.myshopify.com",
                task_id="task-456",
                prompt="test prompt",
                cost=10
            )

        assert "Database error" in str(exc_info.value)


class TestUpdateAssetCompleted:
    """Tests for update_asset_completed method"""

    @pytest.mark.asyncio
    async def test_update_completed_success(self, asset_service, mock_prisma):
        """Test successfully marking asset as completed"""
        # Arrange
        mock_asset = Mock()
        mock_asset.id = "asset-123"
        mock_asset.model_dump.return_value = {
            "id": "asset-123",
            "status": "completed",
            "imageUrl": "https://example.com/image.jpg"
        }
        mock_prisma.asset.update = AsyncMock(return_value=mock_asset)

        # Act
        result = await asset_service.update_asset_completed(
            task_id="task-456",
            image_url="https://example.com/image.jpg"
        )

        # Assert
        assert result["status"] == "completed"
        assert result["imageUrl"] == "https://example.com/image.jpg"
        mock_prisma.asset.update.assert_called_once()

        call_args = mock_prisma.asset.update.call_args
        assert call_args[1]["where"]["taskId"] == "task-456"
        assert call_args[1]["data"]["status"] == "completed"
        assert call_args[1]["data"]["imageUrl"] == "https://example.com/image.jpg"

    @pytest.mark.asyncio
    async def test_update_completed_not_found(self, asset_service, mock_prisma):
        """Test updating non-existent asset"""
        # Arrange
        mock_prisma.asset.update = AsyncMock(
            side_effect=Exception("Record to update not found")
        )

        # Act & Assert
        with pytest.raises(AssetNotFoundError) as exc_info:
            await asset_service.update_asset_completed(
                task_id="nonexistent-task",
                image_url="https://example.com/image.jpg"
            )

        assert "nonexistent-task" in str(exc_info.value)


class TestUpdateAssetFailed:
    """Tests for update_asset_failed method"""

    @pytest.mark.asyncio
    async def test_update_failed_success(self, asset_service, mock_prisma):
        """Test successfully marking asset as failed"""
        # Arrange
        mock_asset = Mock()
        mock_asset.id = "asset-123"
        mock_asset.model_dump.return_value = {
            "id": "asset-123",
            "status": "failed"
        }
        mock_prisma.asset.update = AsyncMock(return_value=mock_asset)

        # Act
        result = await asset_service.update_asset_failed(
            task_id="task-456",
            error_message="Model timeout"
        )

        # Assert
        assert result["status"] == "failed"
        mock_prisma.asset.update.assert_called_once()


class TestGetAssetByTaskId:
    """Tests for get_asset_by_task_id method"""

    @pytest.mark.asyncio
    async def test_get_asset_found(self, asset_service, mock_prisma, sample_asset_data):
        """Test retrieving existing asset"""
        # Arrange
        mock_asset = Mock()
        mock_asset.model_dump.return_value = sample_asset_data
        mock_prisma.asset.find_unique = AsyncMock(return_value=mock_asset)

        # Act
        result = await asset_service.get_asset_by_task_id("task-456")

        # Assert
        assert result is not None
        assert result["id"] == "asset-123"
        assert result["taskId"] == "task-456"
        mock_prisma.asset.find_unique.assert_called_once_with(
            where={"taskId": "task-456"}
        )

    @pytest.mark.asyncio
    async def test_get_asset_not_found(self, asset_service, mock_prisma):
        """Test retrieving non-existent asset"""
        # Arrange
        mock_prisma.asset.find_unique = AsyncMock(return_value=None)

        # Act
        result = await asset_service.get_asset_by_task_id("nonexistent-task")

        # Assert
        assert result is None


class TestGetShopAssets:
    """Tests for get_shop_assets method"""

    @pytest.mark.asyncio
    async def test_get_shop_assets_success(self, asset_service, mock_prisma):
        """Test retrieving assets for a shop"""
        # Arrange
        mock_assets = [
            Mock(model_dump=Mock(return_value={"id": "1", "shop": "test.myshopify.com"})),
            Mock(model_dump=Mock(return_value={"id": "2", "shop": "test.myshopify.com"}))
        ]
        mock_prisma.asset.find_many = AsyncMock(return_value=mock_assets)

        # Act
        result = await asset_service.get_shop_assets(
            shop="test.myshopify.com",
            limit=10
        )

        # Assert
        assert len(result) == 2
        assert result[0]["id"] == "1"
        assert result[1]["id"] == "2"

        call_args = mock_prisma.asset.find_many.call_args
        assert call_args[1]["where"]["shop"] == "test.myshopify.com"
        assert call_args[1]["take"] == 10
        assert call_args[1]["order_by"]["createdAt"] == "desc"

    @pytest.mark.asyncio
    async def test_get_shop_assets_with_status_filter(self, asset_service, mock_prisma):
        """Test retrieving assets with status filter"""
        # Arrange
        mock_assets = [
            Mock(model_dump=Mock(return_value={"id": "1", "status": "completed"}))
        ]
        mock_prisma.asset.find_many = AsyncMock(return_value=mock_assets)

        # Act
        result = await asset_service.get_shop_assets(
            shop="test.myshopify.com",
            limit=50,
            status="completed"
        )

        # Assert
        assert len(result) == 1
        call_args = mock_prisma.asset.find_many.call_args
        assert call_args[1]["where"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_get_shop_assets_empty(self, asset_service, mock_prisma):
        """Test retrieving assets when none exist"""
        # Arrange
        mock_prisma.asset.find_many = AsyncMock(return_value=[])

        # Act
        result = await asset_service.get_shop_assets(shop="empty.myshopify.com")

        # Assert
        assert len(result) == 0


class TestDeleteOldAssets:
    """Tests for delete_old_assets method"""

    @pytest.mark.asyncio
    async def test_delete_old_assets_success(self, asset_service, mock_prisma):
        """Test deleting old assets"""
        # Arrange
        mock_prisma.asset.delete_many = AsyncMock(return_value=5)

        # Act
        result = await asset_service.delete_old_assets(
            shop="test.myshopify.com",
            days=90
        )

        # Assert
        assert result == 5
        mock_prisma.asset.delete_many.assert_called_once()

        call_args = mock_prisma.asset.delete_many.call_args
        assert call_args[1]["where"]["shop"] == "test.myshopify.com"
        assert "createdAt" in call_args[1]["where"]
        assert "lt" in call_args[1]["where"]["createdAt"]

    @pytest.mark.asyncio
    async def test_delete_old_assets_custom_days(self, asset_service, mock_prisma):
        """Test deleting assets with custom days threshold"""
        # Arrange
        mock_prisma.asset.delete_many = AsyncMock(return_value=10)

        # Act
        result = await asset_service.delete_old_assets(
            shop="test.myshopify.com",
            days=30
        )

        # Assert
        assert result == 10

    @pytest.mark.asyncio
    async def test_delete_old_assets_none_found(self, asset_service, mock_prisma):
        """Test deleting when no old assets exist"""
        # Arrange
        mock_prisma.asset.delete_many = AsyncMock(return_value=0)

        # Act
        result = await asset_service.delete_old_assets(shop="test.myshopify.com")

        # Assert
        assert result == 0


class TestGetAssetStats:
    """Tests for get_asset_stats method"""

    @pytest.mark.asyncio
    async def test_get_asset_stats_success(self, asset_service, mock_prisma):
        """Test getting asset statistics"""
        # Arrange
        mock_assets = [
            Mock(status="completed"),
            Mock(status="completed"),
            Mock(status="completed"),
            Mock(status="processing"),
            Mock(status="failed")
        ]
        mock_prisma.asset.find_many = AsyncMock(return_value=mock_assets)

        # Act
        result = await asset_service.get_asset_stats(shop="test.myshopify.com")

        # Assert
        assert result["total"] == 5
        assert result["completed"] == 3
        assert result["processing"] == 1
        assert result["failed"] == 1

    @pytest.mark.asyncio
    async def test_get_asset_stats_empty(self, asset_service, mock_prisma):
        """Test getting stats when no assets exist"""
        # Arrange
        mock_prisma.asset.find_many = AsyncMock(return_value=[])

        # Act
        result = await asset_service.get_asset_stats(shop="empty.myshopify.com")

        # Assert
        assert result["total"] == 0
        assert result["completed"] == 0
        assert result["processing"] == 0
        assert result["failed"] == 0


class TestErrorHandling:
    """Tests for error handling scenarios"""

    @pytest.mark.asyncio
    async def test_generic_database_error(self, asset_service, mock_prisma):
        """Test handling of generic database errors"""
        # Arrange
        mock_prisma.asset.find_many = AsyncMock(
            side_effect=Exception("Connection timeout")
        )

        # Act & Assert
        with pytest.raises(AssetServiceError) as exc_info:
            await asset_service.get_shop_assets(shop="test.myshopify.com")

        assert "Connection timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_with_generic_error(self, asset_service, mock_prisma):
        """Test update failure with generic error"""
        # Arrange
        mock_prisma.asset.update = AsyncMock(
            side_effect=Exception("Constraint violation")
        )

        # Act & Assert
        with pytest.raises(AssetServiceError) as exc_info:
            await asset_service.update_asset_completed(
                task_id="task-123",
                image_url="https://example.com/image.jpg"
            )

        assert "Constraint violation" in str(exc_info.value)
