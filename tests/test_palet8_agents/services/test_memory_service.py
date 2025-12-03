"""Tests for palet8_agents.services.memory_service module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from palet8_agents.services.memory_service import (
    MemoryService,
    MemoryServiceError,
    DatabaseConnectionError,
    StorageError,
    RetrievalError,
    MemorySearchResult,
)


class TestMemorySearchResult:
    """Tests for MemorySearchResult dataclass."""

    def test_init(self):
        """Test initialization."""
        result = MemorySearchResult(
            id="123",
            content="Test content",
            similarity=0.95,
            metadata={"key": "value"},
        )
        assert result.id == "123"
        assert result.content == "Test content"
        assert result.similarity == 0.95
        assert result.metadata["key"] == "value"

    def test_init_defaults(self):
        """Test default initialization."""
        result = MemorySearchResult(
            id="456",
            content="Another content",
            similarity=0.8,
        )
        assert result.metadata == {}

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = MemorySearchResult(
            id="789",
            content="Dict test",
            similarity=0.75,
            metadata={"source": "test"},
        )
        data = result.to_dict()

        assert data["id"] == "789"
        assert data["content"] == "Dict test"
        assert data["similarity"] == 0.75
        assert data["metadata"]["source"] == "test"


class TestMemoryServiceInit:
    """Tests for MemoryService initialization."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        service = MemoryService()
        assert service._embedding_service is None
        assert service._owns_embedding_service is True
        assert service._pool is None

    def test_init_with_services(self):
        """Test initialization with provided services."""
        mock_embedding = MagicMock()
        service = MemoryService(
            embedding_service=mock_embedding,
            db_url="postgresql://test:test@localhost/test",
        )
        assert service._embedding_service == mock_embedding
        assert service._owns_embedding_service is False
        assert service._db_url == "postgresql://test:test@localhost/test"


class TestMemoryServiceExceptions:
    """Tests for MemoryService exceptions."""

    def test_memory_service_error(self):
        """Test base exception."""
        error = MemoryServiceError("Test error")
        assert str(error) == "Test error"

    def test_database_connection_error(self):
        """Test database connection error."""
        error = DatabaseConnectionError("Connection failed")
        assert isinstance(error, MemoryServiceError)

    def test_storage_error(self):
        """Test storage error."""
        error = StorageError("Storage failed")
        assert isinstance(error, MemoryServiceError)

    def test_retrieval_error(self):
        """Test retrieval error."""
        error = RetrievalError("Retrieval failed")
        assert isinstance(error, MemoryServiceError)


class TestMemoryServiceAsync:
    """Async tests for MemoryService."""

    @pytest.mark.asyncio
    async def test_close_pool(self):
        """Test closing service releases pool."""
        service = MemoryService()

        # Mock pool
        mock_pool = AsyncMock()
        service._pool = mock_pool

        await service.close()

        mock_pool.close.assert_called_once()
        assert service._pool is None

    @pytest.mark.asyncio
    async def test_close_embedding_service(self):
        """Test closing service releases embedding service."""
        service = MemoryService()

        # Mock embedding service
        mock_embedding = AsyncMock()
        service._embedding_service = mock_embedding
        service._owns_embedding_service = True

        await service.close()

        mock_embedding.close.assert_called_once()
        assert service._embedding_service is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with MemoryService() as service:
            assert service is not None

    @pytest.mark.asyncio
    async def test_get_pool_no_asyncpg(self):
        """Test error when asyncpg not installed."""
        service = MemoryService(db_url="postgresql://test")

        # Mock import error
        with patch.dict("sys.modules", {"asyncpg": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                with pytest.raises(DatabaseConnectionError):
                    await service._get_pool()


class MockAsyncContextManager:
    """Helper class to mock async context managers."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


class TestMemoryServiceMocked:
    """Tests for MemoryService with mocked dependencies."""

    @pytest.fixture
    def mock_embedding_service(self):
        """Create mock embedding service."""
        mock = AsyncMock()
        mock.embed_for_storage = AsyncMock(return_value=[0.1] * 768)
        mock.embed_for_search = AsyncMock(return_value=[0.1] * 768)
        return mock

    @pytest.fixture
    def mock_conn(self):
        """Create mock database connection."""
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(return_value={"id": "test-id"})
        conn.fetch = AsyncMock(return_value=[])
        return conn

    @pytest.fixture
    def mock_pool(self, mock_conn):
        """Create mock connection pool."""
        pool = MagicMock()
        pool.close = AsyncMock()
        # acquire() returns an async context manager that yields mock_conn
        pool.acquire = MagicMock(return_value=MockAsyncContextManager(mock_conn))
        return pool

    @pytest.mark.asyncio
    async def test_store_design_summary(self, mock_embedding_service, mock_pool):
        """Test storing design summary."""
        service = MemoryService(embedding_service=mock_embedding_service)
        service._pool = mock_pool

        result = await service.store_design_summary(
            job_id="job-123",
            user_id="user-456",
            summary="A beautiful sunset design",
            prompt="sunset over ocean",
            product_type="poster",
        )

        assert result == "test-id"
        mock_embedding_service.embed_for_storage.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_prompt(self, mock_embedding_service, mock_pool):
        """Test storing prompt."""
        service = MemoryService(embedding_service=mock_embedding_service)
        service._pool = mock_pool

        result = await service.store_prompt(
            job_id="job-123",
            user_id="user-456",
            prompt="A cat sitting on a windowsill",
            evaluation_score=0.85,
        )

        assert result == "test-id"
        mock_embedding_service.embed_for_storage.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_history(self, mock_conn, mock_pool):
        """Test getting user history."""
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "1",
                    "job_id": "job-1",
                    "summary": "Test summary",
                    "title": "Test",
                    "product_type": "poster",
                    "style": "realistic",
                    "image_url": "http://example.com/img.jpg",
                    "thumbnail_url": None,
                    "tags": ["art"],
                    "created_at": None,
                }
            ]
        )

        service = MemoryService()
        service._pool = mock_pool

        results = await service.get_user_history("user-123", limit=10)

        assert len(results) == 1
        assert results[0]["summary"] == "Test summary"

    @pytest.mark.asyncio
    async def test_search_user_history(self, mock_embedding_service, mock_conn, mock_pool):
        """Test searching user history."""
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "1",
                    "summary": "Sunset design",
                    "title": "Sunset",
                    "product_type": "poster",
                    "style": "realistic",
                    "image_url": "http://example.com/img.jpg",
                    "created_at": None,
                    "similarity": 0.92,
                }
            ]
        )

        service = MemoryService(embedding_service=mock_embedding_service)
        service._pool = mock_pool

        results = await service.search_user_history(
            query="sunset landscape",
            user_id="user-123",
        )

        assert len(results) == 1
        assert isinstance(results[0], MemorySearchResult)
        assert results[0].similarity == 0.92

    @pytest.mark.asyncio
    async def test_search_similar_prompts(self, mock_embedding_service, mock_conn, mock_pool):
        """Test searching similar prompts."""
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "1",
                    "prompt": "sunset over ocean",
                    "negative_prompt": "blurry",
                    "style": "realistic",
                    "product_type": "poster",
                    "evaluation_score": 0.88,
                    "similarity": 0.95,
                }
            ]
        )

        service = MemoryService(embedding_service=mock_embedding_service)
        service._pool = mock_pool

        results = await service.search_similar_prompts(
            query="sunset beach",
            min_score=0.7,
        )

        assert len(results) == 1
        assert results[0].content == "sunset over ocean"
        assert results[0].metadata["evaluation_score"] == 0.88

    @pytest.mark.asyncio
    async def test_get_art_references(self, mock_embedding_service, mock_conn, mock_pool):
        """Test getting art references."""
        mock_conn.fetch = AsyncMock(
            return_value=[
                {
                    "id": "1",
                    "name": "Sunset Reference",
                    "description": "Beautiful sunset over mountains",
                    "category": "landscapes",
                    "tags": ["sunset", "mountains"],
                    "image_url": "http://example.com/art.jpg",
                    "thumbnail_url": "http://example.com/art_thumb.jpg",
                    "similarity": 0.88,
                }
            ]
        )

        service = MemoryService(embedding_service=mock_embedding_service)
        service._pool = mock_pool

        results = await service.get_art_references(query="sunset mountains")

        assert len(results) == 1
        assert results[0].metadata["name"] == "Sunset Reference"
        assert results[0].metadata["category"] == "landscapes"

    @pytest.mark.asyncio
    async def test_search_all(self, mock_embedding_service, mock_conn, mock_pool):
        """Test searching all memory types."""
        # Mock fetch for all queries
        mock_conn.fetch = AsyncMock(return_value=[])

        service = MemoryService(embedding_service=mock_embedding_service)
        service._pool = mock_pool

        results = await service.search_all(
            query="sunset",
            user_id="user-123",
            limit_per_type=3,
        )

        assert "prompts" in results
        assert "history" in results
        assert "art" in results
