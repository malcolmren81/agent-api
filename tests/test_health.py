"""
Health Check Tests
Tests for health check endpoints and service dependency checks
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock, MagicMock
import httpx

from src.api.main import app
from src.utils.health_check import (
    check_gemini_api,
    check_openai_api,
    check_flux_api,
    check_product_generator,
    check_database,
    perform_health_checks,
)


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_endpoint_exists(self, client):
        """Test that health endpoint is accessible."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_endpoint_returns_json(self, client):
        """Test that health endpoint returns JSON."""
        response = client.get("/health")
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert "status" in data
        assert "services" in data

    def test_health_endpoint_structure(self, client):
        """Test that health endpoint has correct structure."""
        response = client.get("/health")
        data = response.json()

        # Check status field
        assert data["status"] in ["healthy", "degraded"]

        # Check services field
        assert isinstance(data["services"], dict)
        assert "gemini" in data["services"]
        assert "openai" in data["services"]
        assert "flux" in data["services"]


class TestGeminiHealthCheck:
    """Tests for Gemini API health checks."""

    @pytest.mark.asyncio
    async def test_gemini_check_with_valid_key(self, mock_settings_with_keys):
        """Test Gemini health check with valid API key."""
        result = await check_gemini_api()

        assert result["status"] is True
        assert "message" in result
        assert result["message"] == "API key configured"

    @pytest.mark.asyncio
    async def test_gemini_check_without_key(self, mock_settings_no_keys):
        """Test Gemini health check without API key."""
        result = await check_gemini_api()

        assert result["status"] is False
        assert "error" in result
        assert "not configured" in result["error"]

    @pytest.mark.asyncio
    async def test_gemini_check_with_invalid_key(self):
        """Test Gemini health check with invalid API key."""
        with patch("config.settings") as mock_settings:
            mock_settings.gemini_api_key = "invalid"

            result = await check_gemini_api()

            assert result["status"] is False
            assert "error" in result
            assert "appears invalid" in result["error"]


class TestOpenAIHealthCheck:
    """Tests for OpenAI API health checks."""

    @pytest.mark.asyncio
    async def test_openai_check_success(self, mock_settings_with_keys):
        """Test successful OpenAI API check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await check_openai_api()

            assert result["status"] is True
            assert result["message"] == "API accessible"

    @pytest.mark.asyncio
    async def test_openai_check_invalid_key(self, mock_settings_with_keys):
        """Test OpenAI check with invalid API key."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await check_openai_api()

            assert result["status"] is False
            assert "Invalid API key" in result["error"]

    @pytest.mark.asyncio
    async def test_openai_check_timeout(self, mock_settings_with_keys):
        """Test OpenAI check with timeout."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            result = await check_openai_api()

            assert result["status"] is False
            assert "timeout" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_openai_check_without_key(self, mock_settings_no_keys):
        """Test OpenAI check without API key."""
        result = await check_openai_api()

        assert result["status"] is False
        assert "not configured" in result["error"]


class TestFluxHealthCheck:
    """Tests for Flux API health checks."""

    @pytest.mark.asyncio
    async def test_flux_check_success(self, mock_settings_with_keys):
        """Test successful Flux API check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await check_flux_api()

            assert result["status"] is True
            assert result["message"] == "API accessible"

    @pytest.mark.asyncio
    async def test_flux_check_unauthorized(self, mock_settings_with_keys):
        """Test Flux check with unauthorized access."""
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await check_flux_api()

            assert result["status"] is False
            assert "401" in result["error"]

    @pytest.mark.asyncio
    async def test_flux_check_without_key(self, mock_settings_no_keys):
        """Test Flux check without API key."""
        result = await check_flux_api()

        assert result["status"] is False
        assert "not configured" in result["error"]


class TestProductGeneratorHealthCheck:
    """Tests for Product Generator A2A health checks."""

    @pytest.mark.asyncio
    async def test_product_generator_disabled(self):
        """Test when product generator is disabled."""
        with patch("config.settings") as mock_settings:
            mock_settings.use_product_a2a = False

            result = await check_product_generator()

            assert result["status"] is True
            assert "disabled" in result["message"]

    @pytest.mark.asyncio
    async def test_product_generator_success(self):
        """Test successful product generator check."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("config.settings") as mock_settings:
            mock_settings.use_product_a2a = True
            mock_settings.product_generator_url = "http://localhost:8081"

            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    return_value=mock_response
                )

                result = await check_product_generator()

                assert result["status"] is True
                assert "accessible" in result["message"]

    @pytest.mark.asyncio
    async def test_product_generator_unreachable(self):
        """Test when product generator is unreachable."""
        with patch("config.settings") as mock_settings:
            mock_settings.use_product_a2a = True
            mock_settings.product_generator_url = "http://localhost:8081"

            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                    side_effect=httpx.ConnectError("Connection refused")
                )

                result = await check_product_generator()

                assert result["status"] is False
                assert "unreachable" in result["error"].lower()


class TestDatabaseHealthCheck:
    """Tests for database health checks."""

    @pytest.mark.asyncio
    async def test_database_check_placeholder(self):
        """Test database health check (Phase 5 placeholder)."""
        result = await check_database()

        assert result["status"] is True
        assert "Phase 5" in result["message"]


class TestPerformHealthChecks:
    """Tests for combined health checks."""

    @pytest.mark.asyncio
    async def test_all_checks_run_concurrently(self):
        """Test that all health checks run concurrently."""
        result = await perform_health_checks()

        # Should have all service checks
        assert "gemini" in result
        assert "openai" in result
        assert "flux" in result
        assert "product_generator" in result
        assert "database" in result

        # Each check should have status and either message or error
        for service, check_result in result.items():
            assert "status" in check_result
            assert "message" in check_result or "error" in check_result

    @pytest.mark.asyncio
    async def test_health_checks_handle_exceptions(self):
        """Test that health checks handle exceptions gracefully."""
        with patch(
            "src.utils.health_check.check_gemini_api",
            side_effect=Exception("Test error"),
        ):
            result = await perform_health_checks()

            # Should still return results for other services
            assert "openai" in result
            assert "flux" in result


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    def test_metrics_endpoint_with_enabled_metrics(self, client):
        """Test metrics endpoint when metrics are enabled."""
        with patch("config.settings") as mock_settings:
            mock_settings.enable_metrics = True

            response = client.get("/metrics")

            # Should return Prometheus metrics
            assert response.status_code == 200

    def test_metrics_endpoint_with_disabled_metrics(self, client):
        """Test metrics endpoint when metrics are disabled."""
        with patch("config.settings") as mock_settings:
            mock_settings.enable_metrics = False

            response = client.get("/metrics")

            assert response.status_code == 404
            data = response.json()
            assert "error" in data
            assert "disabled" in data["error"]


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Phase 2 Agent Backend"
        assert "version" in data
        assert data["status"] == "running"
