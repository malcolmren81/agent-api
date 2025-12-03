"""
Unit tests for ProductTemplateService

Tests the HTTP client that fetches product templates from admin-api.
Phase 7.1.2: Generator Integration
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock
from src.services.product_template_service import (
    ProductTemplateService,
    ProductTemplateServiceError,
    ProductTemplateNotFoundError,
    get_product_template_service
)


@pytest.fixture
def mock_admin_api_url():
    """Mock admin API URL for testing"""
    return "https://test-admin-api.example.com"


@pytest.fixture
def product_template_service(mock_admin_api_url):
    """Create ProductTemplateService instance for testing"""
    return ProductTemplateService(admin_api_url=mock_admin_api_url)


@pytest.fixture
def mock_templates():
    """Mock product templates data"""
    return [
        {
            "id": "template-1",
            "name": "White Plain T-Shirt",
            "description": "100% cotton unisex t-shirt",
            "category": "apparel",
            "price": "29.99",
            "cost": "15.99",
            "isActive": True,
            "isSuspended": False,
            "images": [
                {"url": "https://storage.example.com/tshirt.png", "isMain": True, "position": 1}
            ],
            "designArea": {"x": 150, "y": 200, "width": 800, "height": 1000, "unit": "pixels"}
        },
        {
            "id": "template-2",
            "name": "Black Hoodie",
            "description": "Cozy fleece hoodie",
            "category": "apparel",
            "price": "49.99",
            "cost": "25.99",
            "isActive": True,
            "isSuspended": False,
            "images": [
                {"url": "https://storage.example.com/hoodie.png", "isMain": True, "position": 1}
            ],
            "designArea": {"x": 200, "y": 250, "width": 700, "height": 900, "unit": "pixels"}
        }
    ]


@pytest.fixture
def mock_template_detail():
    """Mock single template detail"""
    return {
        "id": "template-1",
        "name": "White Plain T-Shirt",
        "description": "100% cotton unisex t-shirt",
        "category": "apparel",
        "price": "29.99",
        "cost": "15.99",
        "templateId": "printful-3001",
        "supplierSku": "3001-white-m",
        "supplier": "printful",
        "isActive": True,
        "isSuspended": False,
        "images": [
            {"url": "https://storage.example.com/tshirt-front.png", "isMain": True, "position": 1},
            {"url": "https://storage.example.com/tshirt-back.png", "isMain": False, "position": 2}
        ],
        "designArea": {
            "x": 150,
            "y": 200,
            "width": 800,
            "height": 1000,
            "unit": "pixels"
        },
        "printSpecifications": {
            "dpi": 300,
            "colorMode": "RGB",
            "maxFileSizeMB": 10,
            "fileFormats": ["PNG", "JPG"]
        },
        "createdAt": "2025-11-01T10:00:00Z",
        "updatedAt": "2025-11-10T15:30:00Z"
    }


# =============================================================================
# Test: list_templates() - Success Cases
# =============================================================================

@pytest.mark.asyncio
async def test_list_templates_success(product_template_service, mock_templates):
    """Test successful template listing"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": True,
            "data": mock_templates,
            "pagination": {"page": 1, "pageSize": 50, "total": 2}
        }
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        result = await product_template_service.list_templates(category="apparel")

        assert len(result) == 2
        assert result[0]["name"] == "White Plain T-Shirt"
        assert result[1]["name"] == "Black Hoodie"
        assert all(t["category"] == "apparel" for t in result)


@pytest.mark.asyncio
async def test_list_templates_with_filters(product_template_service):
    """Test template listing with various filters"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"success": True, "data": []}
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        await product_template_service.list_templates(
            category="drinkware",
            is_active=True,
            is_suspended=False,
            page_size=10
        )

        # Verify correct parameters passed
        call_args = mock_get.call_args
        assert call_args[1]["params"]["category"] == "drinkware"
        assert call_args[1]["params"]["isActive"] == "true"
        assert call_args[1]["params"]["isSuspended"] == "false"
        assert call_args[1]["params"]["pageSize"] == 10


@pytest.mark.asyncio
async def test_list_templates_empty_result(product_template_service):
    """Test handling of empty template list"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"success": True, "data": []}
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        result = await product_template_service.list_templates(category="accessories")

        assert result == []
        assert isinstance(result, list)


@pytest.mark.asyncio
async def test_list_templates_page_size_capped(product_template_service):
    """Test that page_size is capped at 100"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {"success": True, "data": []}
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        await product_template_service.list_templates(page_size=500)

        # Verify page_size capped at 100
        call_args = mock_get.call_args
        assert call_args[1]["params"]["pageSize"] == 100


# =============================================================================
# Test: list_templates() - Error Cases
# =============================================================================

@pytest.mark.asyncio
async def test_list_templates_api_error(product_template_service):
    """Test handling of API error response"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": False,
            "error": "Database connection failed"
        }
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        with pytest.raises(ProductTemplateServiceError) as exc_info:
            await product_template_service.list_templates()

        assert "API error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_templates_http_error(product_template_service):
    """Test handling of HTTP error (500)"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(ProductTemplateServiceError) as exc_info:
            await product_template_service.list_templates()

        assert "HTTP 500" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_templates_network_error(product_template_service):
    """Test handling of network error"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")

        with pytest.raises(ProductTemplateServiceError) as exc_info:
            await product_template_service.list_templates()

        assert "Network error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_templates_timeout(product_template_service):
    """Test handling of request timeout"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_get.side_effect = httpx.TimeoutException("Request timeout")

        with pytest.raises(ProductTemplateServiceError) as exc_info:
            await product_template_service.list_templates()

        assert "Network error" in str(exc_info.value)


# =============================================================================
# Test: get_template() - Success Cases
# =============================================================================

@pytest.mark.asyncio
async def test_get_template_success(product_template_service, mock_template_detail):
    """Test fetching single template successfully"""
    template_id = "template-1"

    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": True,
            "data": mock_template_detail
        }
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        result = await product_template_service.get_template(template_id)

        assert result["id"] == template_id
        assert result["name"] == "White Plain T-Shirt"
        assert "designArea" in result
        assert result["designArea"]["width"] == 800
        assert len(result["images"]) == 2


@pytest.mark.asyncio
async def test_get_template_with_design_area(product_template_service, mock_template_detail):
    """Test that design area is correctly parsed"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": True,
            "data": mock_template_detail
        }
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        result = await product_template_service.get_template("template-1")

        design_area = result["designArea"]
        assert design_area["x"] == 150
        assert design_area["y"] == 200
        assert design_area["width"] == 800
        assert design_area["height"] == 1000
        assert design_area["unit"] == "pixels"


# =============================================================================
# Test: get_template() - Error Cases
# =============================================================================

@pytest.mark.asyncio
async def test_get_template_not_found(product_template_service):
    """Test 404 error handling"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found",
            request=MagicMock(),
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(ProductTemplateNotFoundError) as exc_info:
            await product_template_service.get_template("invalid-id")

        assert "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_get_template_empty_data(product_template_service):
    """Test handling of empty data in successful response"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": True,
            "data": {}
        }
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        with pytest.raises(ProductTemplateNotFoundError):
            await product_template_service.get_template("template-1")


@pytest.mark.asyncio
async def test_get_template_api_error(product_template_service):
    """Test handling of API error in response"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.json.return_value = {
            "success": False,
            "error": "Template query failed"
        }
        mock_response.raise_for_status = AsyncMock()
        mock_get.return_value = mock_response

        with pytest.raises(ProductTemplateServiceError) as exc_info:
            await product_template_service.get_template("template-1")

        assert "API error" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_template_http_500(product_template_service):
    """Test handling of HTTP 500 error"""
    with patch.object(product_template_service.client, 'get') as mock_get:
        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error",
            request=MagicMock(),
            response=mock_response
        )
        mock_get.return_value = mock_response

        with pytest.raises(ProductTemplateServiceError) as exc_info:
            await product_template_service.get_template("template-1")

        assert "HTTP 500" in str(exc_info.value)


# =============================================================================
# Test: Singleton Instance
# =============================================================================

def test_get_product_template_service_singleton():
    """Test that get_product_template_service returns singleton"""
    service1 = get_product_template_service()
    service2 = get_product_template_service()

    assert service1 is service2


# =============================================================================
# Test: Initialization
# =============================================================================

def test_initialization_with_custom_url():
    """Test initialization with custom admin API URL"""
    custom_url = "https://custom-api.example.com"
    service = ProductTemplateService(admin_api_url=custom_url)

    assert service.base_url == custom_url


def test_initialization_without_url():
    """Test initialization falls back to default URL"""
    service = ProductTemplateService()

    assert service.base_url is not None
    assert "palet8-admin-api" in service.base_url


# =============================================================================
# Test: Cleanup
# =============================================================================

@pytest.mark.asyncio
async def test_close_client(product_template_service):
    """Test closing HTTP client"""
    with patch.object(product_template_service.client, 'aclose') as mock_close:
        mock_close.return_value = AsyncMock()

        await product_template_service.close()

        mock_close.assert_called_once()
