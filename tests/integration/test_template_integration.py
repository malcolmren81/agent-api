"""
Integration tests for Phase 7.1.2 - Generator Template Integration

Tests the end-to-end flow of template selection through the agent pipeline.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.agents.interactive_agent import InteractiveAgent
from src.agents.product_generator_agent import ProductGeneratorAgent
from src.agents.base_agent import AgentContext


@pytest.fixture
def mock_product_template():
    """Mock product template from admin-api"""
    return {
        "id": "12b02b0b-ea91-45fa-9ee0-c8b137888cc8",
        "name": "White Plain T-Shirt",
        "description": "100% cotton unisex t-shirt",
        "category": "apparel",
        "price": "29.99",
        "cost": "15.99",
        "isActive": True,
        "isSuspended": False,
        "images": [
            {
                "url": "https://storage.googleapis.com/palet8-product-templates/apparel/tshirt.png",
                "isMain": True,
                "position": 1
            }
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
        }
    }


@pytest.mark.asyncio
@pytest.mark.integration
async def test_interactive_agent_with_template_id():
    """Test that Interactive Agent accepts and stores template_id"""
    agent = InteractiveAgent()

    # Create context with template selection
    context = AgentContext(
        task_id="test-task-123",
        user_id="test-user",
        customer_id="test-customer",
        email="test@example.com",
        shop_domain="test.myshopify.com"
    )

    context.shared_data = {
        "prompt": "A cool abstract design for a t-shirt",
        "template_id": "12b02b0b-ea91-45fa-9ee0-c8b137888cc8",
        "template_category": "apparel"
    }

    # Run agent
    result = await agent.run(context)

    # Verify agent succeeded
    assert result.success is True

    # Verify template_id is in shared_data
    assert context.shared_data.get("template_id") == "12b02b0b-ea91-45fa-9ee0-c8b137888cc8"
    assert context.shared_data.get("template_category") == "apparel"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_interactive_agent_without_template_id():
    """Test that Interactive Agent works without template_id"""
    agent = InteractiveAgent()

    # Create context without template
    context = AgentContext(
        task_id="test-task-456",
        user_id="test-user",
        customer_id="test-customer",
        email="test@example.com"
    )

    context.shared_data = {
        "prompt": "A beautiful landscape painting"
    }

    # Run agent
    result = await agent.run(context)

    # Verify agent succeeded
    assert result.success is True

    # Verify no template_id in shared_data
    assert context.shared_data.get("template_id") is None


@pytest.mark.asyncio
@pytest.mark.integration
async def test_product_generator_fetches_template(mock_product_template):
    """Test that Product Generator Agent fetches template from admin-api"""
    from src.services.product_template_service import product_template_service

    # Mock the template service
    with patch.object(product_template_service, 'get_template', return_value=mock_product_template):
        agent = ProductGeneratorAgent()

        # Create input with template_id in context
        input_data = {
            "best_image": {
                "image_id": "image-123",
                "base64_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="  # 1x1 red pixel
            },
            "context": {
                "template_id": "12b02b0b-ea91-45fa-9ee0-c8b137888cc8",
                "template_category": "apparel",
                "product_types": ["tshirt"]
            }
        }

        # Run agent
        result = await agent.run(input_data)

        # Verify agent succeeded
        assert result.get("success") is True

        # Verify template was fetched and stored
        context = result.get("context", {})
        assert "product_template" in context
        assert context["product_template"]["id"] == "12b02b0b-ea91-45fa-9ee0-c8b137888cc8"
        assert context["product_template"]["name"] == "White Plain T-Shirt"

        # Verify design_area stored
        assert "design_area" in context
        assert context["design_area"]["width"] == 800
        assert context["design_area"]["height"] == 1000

        # Verify template service was called
        product_template_service.get_template.assert_called_once_with("12b02b0b-ea91-45fa-9ee0-c8b137888cc8")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_product_generator_without_template():
    """Test that Product Generator Agent works without template_id"""
    agent = ProductGeneratorAgent()

    # Create input without template_id
    input_data = {
        "best_image": {
            "image_id": "image-456",
            "base64_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        },
        "context": {
            "product_types": ["mug"]
        }
    }

    # Run agent
    result = await agent.run(input_data)

    # Verify agent succeeded
    assert result.get("success") is True

    # Verify no template in context
    context = result.get("context", {})
    assert "product_template" not in context
    assert "template_id" not in context


@pytest.mark.asyncio
@pytest.mark.integration
async def test_product_generator_handles_template_not_found():
    """Test that Product Generator Agent handles template not found gracefully"""
    from src.services.product_template_service import product_template_service, ProductTemplateNotFoundError

    # Mock the template service to raise not found error
    with patch.object(product_template_service, 'get_template', side_effect=ProductTemplateNotFoundError("Template not found")):
        agent = ProductGeneratorAgent()

        input_data = {
            "best_image": {
                "image_id": "image-789",
                "base64_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            },
            "context": {
                "template_id": "invalid-template-id",
                "product_types": ["tshirt"]
            }
        }

        # Run agent
        result = await agent.run(input_data)

        # Agent should not fail the entire workflow
        assert result.get("success") is True

        # Error should be logged in context
        context = result.get("context", {})
        assert "template_error" in context
        assert "not found" in context["template_error"].lower()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline_with_template(mock_product_template):
    """Test full agent pipeline with template selection"""
    from src.services.product_template_service import product_template_service

    with patch.object(product_template_service, 'get_template', return_value=mock_product_template):
        # Step 1: Interactive Agent
        interactive_agent = InteractiveAgent()

        context = AgentContext(
            task_id="test-pipeline-123",
            user_id="test-user",
            customer_id="test-customer",
            email="test@example.com"
        )

        context.shared_data = {
            "prompt": "A vibrant tropical sunset",
            "template_id": "12b02b0b-ea91-45fa-9ee0-c8b137888cc8",
            "template_category": "apparel"
        }

        interactive_result = await interactive_agent.run(context)
        assert interactive_result.success is True
        assert context.shared_data.get("template_id") is not None

        # Step 2: Product Generator Agent
        product_generator = ProductGeneratorAgent()

        product_input = {
            "best_image": {
                "image_id": "generated-image-123",
                "base64_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            },
            "context": {
                "template_id": context.shared_data.get("template_id"),
                "template_category": context.shared_data.get("template_category"),
                "product_types": ["tshirt"]
            }
        }

        product_result = await product_generator.run(product_input)

        # Verify full pipeline succeeded
        assert product_result.get("success") is True

        # Verify template flowed through pipeline
        final_context = product_result.get("context", {})
        assert "product_template" in final_context
        assert "design_area" in final_context
        assert final_context["product_template"]["name"] == "White Plain T-Shirt"

        # Verify products were generated
        assert "products" in product_result
        assert len(product_result["products"]) > 0


# Run tests with:
# cd services/agents-api
# pytest tests/integration/test_template_integration.py -v -m integration
