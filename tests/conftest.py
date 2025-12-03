"""
Pytest Configuration and Fixtures
Global test configuration and reusable test fixtures
"""

import os
import sys

# Load .env.test before any other imports
from dotenv import load_dotenv
load_dotenv('.env.test')

import pytest
from unittest.mock import MagicMock, patch
from typing import Generator

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Override environment for tests
os.environ["ENVIRONMENT"] = "development"
os.environ["LOG_LEVEL"] = "ERROR"


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    test_env = {
        "ENVIRONMENT": "development",
        "LOG_LEVEL": "ERROR",
        "SERVER_HOST": "0.0.0.0",
        "SERVER_PORT": "8000",
        "GEMINI_API_KEY": "test-gemini-key-12345",
        "OPENAI_API_KEY": "test-openai-key-67890",
        "FLUX_API_KEY": "test-flux-key-abcdef",
        "GCP_PROJECT_ID": "test-project-id",
        "DEFAULT_REASONING_MODEL": "gemini",
        "DEFAULT_IMAGE_MODEL": "flux",
        "USE_PRODUCT_A2A": "false",
        "ENABLE_METRICS": "true",
        "ENABLE_STREAMING": "true",
        "GPU_ENABLED": "false",
    }

    with patch.dict(os.environ, test_env):
        yield


@pytest.fixture
def mock_settings_with_keys():
    """Mock settings with all API keys configured."""
    from config import Settings

    mock_settings = Settings()
    mock_settings.gemini_api_key = "test-gemini-key-12345"
    mock_settings.openai_api_key = "test-openai-key-67890"
    mock_settings.flux_api_key = "test-flux-key-abcdef"

    with patch("config.settings", mock_settings):
        yield mock_settings


@pytest.fixture
def mock_settings_no_keys():
    """Mock settings with no API keys configured."""
    from config import Settings

    mock_settings = Settings()
    mock_settings.gemini_api_key = ""
    mock_settings.openai_api_key = ""
    mock_settings.flux_api_key = ""

    with patch("config.settings", mock_settings):
        yield mock_settings


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini API client."""
    mock_client = MagicMock()
    mock_client.generate_content = MagicMock(return_value=MagicMock(text="Test response"))

    with patch("google.generativeai.GenerativeModel", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI API client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Test response from OpenAI"))
    ]
    mock_client.chat.completions.create = MagicMock(return_value=mock_response)

    with patch("openai.OpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_flux_client():
    """Mock Flux API client."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "test-image-id",
        "result": {"sample": "base64-encoded-image-data"},
    }

    with patch("httpx.AsyncClient") as mock_client:
        mock_client.return_value.__aenter__.return_value.post.return_value = (
            mock_response
        )
        yield mock_client


@pytest.fixture
def sample_user_message():
    """Sample user message for testing."""
    return {
        "role": "user",
        "content": "Create a product description for a blue cotton t-shirt",
    }


@pytest.fixture
def sample_interactive_request():
    """Sample interactive agent request."""
    return {
        "session_id": "test-session-123",
        "message": "I need help creating a product description",
        "context": {},
    }


@pytest.fixture
def sample_planner_request():
    """Sample planner request."""
    return {
        "session_id": "test-session-123",
        "user_input": "Create product descriptions for summer collection",
        "context": {"category": "apparel", "season": "summer"},
    }


@pytest.fixture
def sample_generation_request():
    """Sample generation request."""
    return {
        "session_id": "test-session-123",
        "prompt": "Generate a compelling product description",
        "model": "gemini",
        "parameters": {"temperature": 0.7, "max_tokens": 500},
    }


@pytest.fixture
def sample_evaluation_request():
    """Sample evaluation request."""
    return {
        "session_id": "test-session-123",
        "content": "This is a test product description",
        "criteria": ["clarity", "persuasiveness", "seo_optimization"],
    }


@pytest.fixture
def sample_product_request():
    """Sample product generator request."""
    return {
        "session_id": "test-session-123",
        "product_info": {
            "name": "Blue Cotton T-Shirt",
            "category": "Apparel",
            "attributes": {
                "color": "blue",
                "material": "100% cotton",
                "size": "M",
            },
        },
    }


@pytest.fixture
def mock_database():
    """Mock database connection (for Phase 5)."""
    mock_db = MagicMock()
    mock_db.is_connected.return_value = True
    mock_db.execute.return_value = []

    with patch("src.utils.database.get_db", return_value=mock_db):
        yield mock_db


# ============================================================================
# Phase 3 Hybrid Routing Fixtures
# ============================================================================

@pytest.fixture
def mock_prisma_client():
    """Mock Prisma client for Phase 3 hybrid routing tests."""
    mock_prisma = MagicMock()

    # Mock ModelStats table
    mock_prisma.modelstats = MagicMock()
    mock_prisma.modelstats.find_unique = MagicMock()
    mock_prisma.modelstats.find_many = MagicMock()
    mock_prisma.modelstats.create = MagicMock()
    mock_prisma.modelstats.update = MagicMock()
    mock_prisma.modelstats.upsert = MagicMock()

    # Mock AgentLog table
    mock_prisma.agentlog = MagicMock()
    mock_prisma.agentlog.create = MagicMock()
    mock_prisma.agentlog.find_many = MagicMock()
    mock_prisma.agentlog.find_first = MagicMock()

    # Mock Template table
    mock_prisma.template = MagicMock()
    mock_prisma.template.find_many = MagicMock()
    mock_prisma.template.find_first = MagicMock()
    mock_prisma.template.update = MagicMock()
    mock_prisma.template.create = MagicMock()

    # Mock connection methods
    mock_prisma.connect = MagicMock()
    mock_prisma.disconnect = MagicMock()

    return mock_prisma


@pytest.fixture
def sample_model_stats():
    """Sample ModelStats data for testing UCB1 bandit."""
    return [
        {
            "id": "stat-1",
            "modelName": "flux-pro",
            "bucket": "product:realistic:high-detail",
            "impressions": 50,
            "rewardMean": 0.82,
            "rewardVar": 0.05,
            "lastUpdated": "2025-10-23T10:00:00Z",
        },
        {
            "id": "stat-2",
            "modelName": "flux-dev",
            "bucket": "product:realistic:high-detail",
            "impressions": 30,
            "rewardMean": 0.75,
            "rewardVar": 0.08,
            "lastUpdated": "2025-10-23T09:30:00Z",
        },
        {
            "id": "stat-3",
            "modelName": "dall-e-3",
            "bucket": "product:realistic:high-detail",
            "impressions": 2,
            "rewardMean": 0.50,
            "rewardVar": 0.10,
            "lastUpdated": "2025-10-23T08:00:00Z",
        },
    ]


@pytest.fixture
def sample_templates():
    """Sample Template data for testing Prompt Manager."""
    return [
        {
            "id": "template-1",
            "name": "Product Photo - White Background",
            "category": "product",
            "tags": ["product", "white background", "high-detail", "professional"],
            "templateText": "Professional product photo of {product}, on a pure white background, studio lighting, high detail, commercial photography",
            "isActive": True,
            "usageCount": 150,
            "acceptRate": 0.88,
            "avgScore": 0.85,
            "createdAt": "2025-09-01T00:00:00Z",
            "updatedAt": "2025-10-22T15:30:00Z",
        },
        {
            "id": "template-2",
            "name": "Product Photo - Lifestyle",
            "category": "product",
            "tags": ["product", "lifestyle", "natural", "context"],
            "templateText": "Lifestyle photo of {product} in natural setting, soft lighting, authentic feel",
            "isActive": True,
            "usageCount": 75,
            "acceptRate": 0.72,
            "avgScore": 0.78,
            "createdAt": "2025-09-15T00:00:00Z",
            "updatedAt": "2025-10-20T10:00:00Z",
        },
        {
            "id": "template-3",
            "name": "Generic Product",
            "category": "product",
            "tags": ["product"],
            "templateText": "Photo of {product}",
            "isActive": True,
            "usageCount": 10,
            "acceptRate": 0.50,
            "avgScore": 0.60,
            "createdAt": "2025-08-01T00:00:00Z",
            "updatedAt": "2025-10-10T08:00:00Z",
        },
    ]


@pytest.fixture
def sample_agent_logs():
    """Sample AgentLog data for testing routing analytics."""
    return [
        {
            "id": "log-1",
            "sessionId": "session-123",
            "agentType": "planner",
            "action": "plan",
            "input": "Product photo of blue t-shirt on white background",
            "output": '{"steps": ["analyze", "template", "generate", "evaluate"]}',
            "status": "success",
            "routingMode": "rule",
            "routingConfidence": 0.95,
            "usedLLM": False,
            "llmCost": 0.0,
            "processingTime": 5,
            "createdAt": "2025-10-23T10:00:00Z",
        },
        {
            "id": "log-2",
            "sessionId": "session-124",
            "agentType": "planner",
            "action": "plan",
            "input": "Complex multi-object scene with intricate lighting",
            "output": '{"steps": ["analyze", "template", "generate", "evaluate"]}',
            "status": "success",
            "routingMode": "llm",
            "routingConfidence": 0.40,
            "usedLLM": True,
            "llmCost": 0.0012,
            "processingTime": 235,
            "createdAt": "2025-10-23T10:05:00Z",
        },
    ]


@pytest.fixture
def sample_policy_config():
    """Sample policy configuration for testing policy loader."""
    return {
        "planner": {
            "mode": "hybrid",
            "rule_conditions": {
                "min_word_count": 8,
                "max_objects": 1,
                "max_composition_elements": 1,
                "novelty_threshold": 0.35,
                "required_keywords": ["product", "white background"],
            },
            "llm_fallback": {
                "provider": "gemini",
                "model": "gemini-2.0-flash-thinking-exp-01-21",
                "temperature": 0.3,
                "max_tokens": 500,
            },
        },
        "prompt_manager": {
            "mode": "db_first",
            "template_scoring": {
                "tag_coverage_weight": 0.70,
                "usage_bonus_weight": 0.20,
                "accept_bonus_weight": 0.10,
                "min_tag_coverage": 0.50,
            },
            "fallback_llm": {
                "enabled": True,
                "provider": "gemini",
            },
        },
        "model_selection": {
            "strategy": "ucb1",
            "exploration": {
                "min_trials_per_model": 1,
                "decay_alpha": 0.031,
                "exploration_constant": 2.0,
            },
            "buckets": {
                "enabled": True,
                "dimensions": ["category", "style", "context"],
            },
        },
        "evaluation": {
            "mode": "hybrid",
            "acceptance_threshold": 0.75,
            "objective_checks": {
                "coverage_min": 0.05,
                "coverage_max": 0.90,
                "whiteness_min": 0.85,
                "whiteness_max": 1.0,
            },
            "vision_llm": {
                "provider": "gemini",
                "model": "gemini-2.0-flash-thinking-exp-01-21",
            },
            "combined_weights": {
                "coverage": 0.35,
                "aesthetics": 0.40,
                "suitability": 0.25,
            },
        },
    }


@pytest.fixture
def sample_image_base64():
    """Sample base64-encoded test image (1x1 white pixel PNG)."""
    # 1x1 white pixel PNG
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="


@pytest.fixture
def sample_prompts():
    """Sample prompts for testing routing logic."""
    return {
        "simple_product": "Product photo of blue t-shirt on white background",
        "complex_scene": "Surreal landscape with floating islands, bioluminescent plants, multiple light sources, and intricate reflections",
        "medium_complexity": "Product photo of smartphone with accessories on wooden table",
        "novel_request": "Quantum computing visualization with holographic data streams",
        "minimal_prompt": "blue shirt",
        "template_match": "Professional product photo of red shoes on pure white background with studio lighting",
    }


@pytest.fixture
def mock_policy_loader():
    """Mock PolicyLoader for testing."""
    from src.config.policy_loader import AgentPolicyConfig

    mock_loader = MagicMock(spec=AgentPolicyConfig)
    mock_loader.get = MagicMock()

    # Default return values for common config paths
    def get_config(path, default=None):
        config_map = {
            "planner.mode": "hybrid",
            "planner.rule_conditions.min_word_count": 8,
            "planner.rule_conditions.novelty_threshold": 0.35,
            "model_selection.strategy": "ucb1",
            "model_selection.exploration.min_trials_per_model": 1,
            "model_selection.exploration.decay_alpha": 0.031,
            "evaluation.mode": "hybrid",
            "evaluation.acceptance_threshold": 0.75,
            "evaluation.combined_weights.coverage": 0.35,
            "evaluation.combined_weights.aesthetics": 0.40,
            "evaluation.combined_weights.suitability": 0.25,
        }
        return config_map.get(path, default)

    mock_loader.get.side_effect = get_config

    return mock_loader


@pytest.fixture
def mock_secret_manager():
    """Mock Google Cloud Secret Manager."""
    mock_client = MagicMock()

    def mock_access_secret(name):
        """Mock secret access."""
        secrets = {
            "gemini-api-key": "test-gemini-secret",
            "openai-api-key": "test-openai-secret",
            "flux-api-key": "test-flux-secret",
        }

        secret_name = name.split("/")[-2]
        if secret_name in secrets:
            return [MagicMock(payload=MagicMock(data=secrets[secret_name].encode()))]

        raise Exception(f"Secret {secret_name} not found")

    mock_client.access_secret_version = mock_access_secret

    with patch(
        "google.cloud.secretmanager.SecretManagerServiceClient",
        return_value=mock_client,
    ):
        yield mock_client


@pytest.fixture
def mock_logger():
    """Mock logger to reduce test output noise."""
    with patch("src.utils.logger.get_logger") as mock_log:
        mock_logger = MagicMock()
        mock_log.return_value = mock_logger
        yield mock_logger


@pytest.fixture
def mock_prometheus_metrics():
    """Mock Prometheus metrics."""
    with patch("prometheus_client.Counter") as mock_counter, patch(
        "prometheus_client.Histogram"
    ) as mock_histogram, patch("prometheus_client.Gauge") as mock_gauge:
        yield {
            "counter": mock_counter,
            "histogram": mock_histogram,
            "gauge": mock_gauge,
        }


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset Prometheus metrics between tests."""
    from prometheus_client import REGISTRY

    # Clear all collectors
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass

    yield

    # Clean up after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Test utilities
class TestUtils:
    """Utility functions for testing."""

    @staticmethod
    def create_mock_response(content: str, status_code: int = 200):
        """Create a mock HTTP response."""
        response = MagicMock()
        response.status_code = status_code
        response.text = content
        response.json.return_value = {"content": content}
        return response

    @staticmethod
    def assert_valid_session_id(session_id: str):
        """Assert that a session ID is valid."""
        assert isinstance(session_id, str)
        assert len(session_id) > 0

    @staticmethod
    def assert_valid_response_structure(response: dict):
        """Assert that a response has valid structure."""
        assert isinstance(response, dict)
        assert "status" in response or "success" in response


@pytest.fixture
def test_utils():
    """Provide test utilities."""
    return TestUtils()


# Pytest configuration hooks
def pytest_configure(config):
    """Pytest configuration hook."""
    # Register custom markers
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "slow: slow-running tests")
    config.addinivalue_line("markers", "requires_api_key: tests that require real API keys")


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Add 'unit' marker to all tests by default
        if "integration" not in item.keywords and "slow" not in item.keywords:
            item.add_marker(pytest.mark.unit)


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up resources after each test."""
    yield

    # Clean up any test files or resources
    import gc

    gc.collect()
