"""
Configuration Tests
Tests for application configuration and settings management
"""

import pytest
import os
from unittest.mock import patch
from pydantic import ValidationError

from config import Settings


class TestSettingsBasic:
    """Tests for basic settings configuration."""

    def test_settings_can_be_created(self):
        """Test that settings can be created with defaults."""
        settings = Settings()

        assert settings is not None
        assert hasattr(settings, "server_host")
        assert hasattr(settings, "server_port")
        assert hasattr(settings, "environment")

    def test_server_configuration(self):
        """Test server configuration defaults."""
        settings = Settings()

        assert settings.server_host == "0.0.0.0"
        assert isinstance(settings.server_port, int)
        assert 1000 <= settings.server_port <= 65535

    def test_environment_values(self):
        """Test that environment can only be specific values."""
        valid_environments = ["development", "staging", "production"]

        settings = Settings()
        assert settings.environment in valid_environments


class TestAPIKeys:
    """Tests for API key configuration."""

    def test_gemini_api_key_configuration(self):
        """Test Gemini API key configuration."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-gemini-key"}):
            settings = Settings()
            assert settings.gemini_api_key == "test-gemini-key"

    def test_openai_api_key_configuration(self):
        """Test OpenAI API key configuration."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            settings = Settings()
            assert settings.openai_api_key == "test-openai-key"

    def test_flux_api_key_configuration(self):
        """Test Flux API key configuration."""
        with patch.dict(os.environ, {"FLUX_API_KEY": "test-flux-key"}):
            settings = Settings()
            assert settings.flux_api_key == "test-flux-key"

    def test_api_keys_default_to_empty_string(self):
        """Test that API keys default to empty string."""
        settings = Settings()

        # Should not raise an error even if keys are not set
        assert isinstance(settings.gemini_api_key, str)
        assert isinstance(settings.openai_api_key, str)
        assert isinstance(settings.flux_api_key, str)


class TestModelConfiguration:
    """Tests for model configuration settings."""

    def test_gemini_model_configuration(self):
        """Test Gemini model settings."""
        settings = Settings()

        assert hasattr(settings, "gemini_model_base")
        assert hasattr(settings, "gemini_model_reasoning")
        assert hasattr(settings, "gemini_model_image")
        assert settings.gemini_thinking_budget in ["default", "low", "medium", "high"]

    def test_openai_model_configuration(self):
        """Test OpenAI model settings."""
        settings = Settings()

        assert hasattr(settings, "openai_model")
        assert isinstance(settings.openai_model, str)
        assert len(settings.openai_model) > 0

    def test_flux_model_configuration(self):
        """Test Flux model settings."""
        settings = Settings()

        assert hasattr(settings, "flux_model_pro")
        assert hasattr(settings, "flux_model_max")
        assert hasattr(settings, "flux_default_model")

    def test_default_model_selection(self):
        """Test default model selection settings."""
        settings = Settings()

        assert settings.default_reasoning_model in ["gemini", "chatgpt"]
        assert settings.default_image_model in ["flux", "gemini"]


class TestCostConfiguration:
    """Tests for cost configuration settings."""

    def test_cost_values_are_positive(self):
        """Test that all cost values are positive."""
        settings = Settings()

        assert settings.gemini_base_cost > 0
        assert settings.gemini_reasoning_cost > 0
        assert settings.gemini_image_cost > 0
        assert settings.chatgpt_reasoning_cost > 0
        assert settings.flux_pro_cost > 0
        assert settings.flux_max_cost > 0

    def test_cost_configuration_can_be_overridden(self):
        """Test that costs can be overridden via environment."""
        with patch.dict(
            os.environ, {"GEMINI_BASE_COST": "0.001", "FLUX_PRO_COST": "0.05"}
        ):
            settings = Settings()

            assert settings.gemini_base_cost == 0.001
            assert settings.flux_pro_cost == 0.05


class TestPerformanceSettings:
    """Tests for performance-related settings."""

    def test_timeout_configuration(self):
        """Test request timeout configuration."""
        settings = Settings()

        assert hasattr(settings, "max_request_timeout")
        assert settings.max_request_timeout > 0
        assert settings.max_request_timeout <= 600  # Max 10 minutes

    def test_gpu_enabled_flag(self):
        """Test GPU enabled flag."""
        settings = Settings()

        assert hasattr(settings, "gpu_enabled")
        assert isinstance(settings.gpu_enabled, bool)

    def test_streaming_enabled_flag(self):
        """Test streaming enabled flag."""
        settings = Settings()

        assert hasattr(settings, "enable_streaming")
        assert isinstance(settings.enable_streaming, bool)


class TestA2AConfiguration:
    """Tests for Agent-to-Agent (A2A) communication settings."""

    def test_a2a_disabled_by_default(self):
        """Test that A2A is disabled by default."""
        settings = Settings()

        assert settings.use_product_a2a is False

    def test_a2a_url_configuration(self):
        """Test A2A URL configuration."""
        with patch.dict(
            os.environ,
            {
                "USE_PRODUCT_A2A": "true",
                "PRODUCT_GENERATOR_URL": "http://product-service:8081",
            },
        ):
            settings = Settings()

            assert settings.use_product_a2a is True
            assert settings.product_generator_url == "http://product-service:8081"

    def test_a2a_auth_token_configuration(self):
        """Test A2A authentication token configuration."""
        with patch.dict(os.environ, {"A2A_AUTH_TOKEN": "test-auth-token"}):
            settings = Settings()

            assert settings.a2a_auth_token == "test-auth-token"

    def test_a2a_tls_configuration(self):
        """Test A2A TLS configuration."""
        settings = Settings()

        assert hasattr(settings, "a2a_enable_tls")
        assert isinstance(settings.a2a_enable_tls, bool)


class TestMonitoringConfiguration:
    """Tests for monitoring and observability settings."""

    def test_metrics_enabled_by_default(self):
        """Test that metrics are enabled by default."""
        settings = Settings()

        assert settings.enable_metrics is True

    def test_metrics_port_configuration(self):
        """Test metrics port configuration."""
        settings = Settings()

        assert hasattr(settings, "metrics_port")
        assert isinstance(settings.metrics_port, int)
        assert 1000 <= settings.metrics_port <= 65535

    def test_structured_logging_configuration(self):
        """Test structured logging configuration."""
        settings = Settings()

        assert hasattr(settings, "enable_structured_logging")
        assert isinstance(settings.enable_structured_logging, bool)


class TestLanguageConfiguration:
    """Tests for language configuration settings."""

    def test_allowed_languages_configuration(self):
        """Test allowed languages configuration."""
        settings = Settings()

        assert hasattr(settings, "allowed_languages")
        assert isinstance(settings.allowed_languages, list)
        assert len(settings.allowed_languages) > 0
        assert "en" in settings.allowed_languages

    def test_default_language_configuration(self):
        """Test default language configuration."""
        settings = Settings()

        assert hasattr(settings, "default_language")
        assert settings.default_language in settings.allowed_languages


class TestGCPConfiguration:
    """Tests for Google Cloud Platform settings."""

    def test_gcp_project_id_configuration(self):
        """Test GCP project ID configuration."""
        with patch.dict(os.environ, {"GCP_PROJECT_ID": "test-project-123"}):
            settings = Settings()

            assert settings.gcp_project_id == "test-project-123"

    def test_google_cloud_project_configuration(self):
        """Test GOOGLE_CLOUD_PROJECT configuration."""
        with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "test-gcp-project"}):
            settings = Settings()

            assert settings.google_cloud_project == "test-gcp-project"


class TestEnvironmentOverrides:
    """Tests for environment variable overrides."""

    def test_server_port_override(self):
        """Test server port can be overridden."""
        with patch.dict(os.environ, {"SERVER_PORT": "9000"}):
            settings = Settings()

            assert settings.server_port == 9000

    def test_environment_override(self):
        """Test environment can be overridden."""
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            settings = Settings()

            assert settings.environment == "production"

    def test_log_level_override(self):
        """Test log level can be overridden."""
        with patch.dict(os.environ, {"LOG_LEVEL": "DEBUG"}):
            settings = Settings()

            assert settings.log_level == "DEBUG"


class TestSettingsValidation:
    """Tests for settings validation."""

    def test_invalid_environment_raises_error(self):
        """Test that invalid environment raises validation error."""
        with patch.dict(os.environ, {"ENVIRONMENT": "invalid_env"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_port_raises_error(self):
        """Test that invalid port raises validation error."""
        with patch.dict(os.environ, {"SERVER_PORT": "70000"}):  # Too high
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_reasoning_model_raises_error(self):
        """Test that invalid reasoning model raises error."""
        with patch.dict(os.environ, {"DEFAULT_REASONING_MODEL": "invalid_model"}):
            with pytest.raises(ValidationError):
                Settings()

    def test_invalid_image_model_raises_error(self):
        """Test that invalid image model raises error."""
        with patch.dict(os.environ, {"DEFAULT_IMAGE_MODEL": "invalid_image_model"}):
            with pytest.raises(ValidationError):
                Settings()


class TestSettingsSingleton:
    """Tests for settings singleton behavior."""

    def test_settings_can_be_imported(self):
        """Test that settings can be imported from config."""
        from config import settings

        assert settings is not None
        assert isinstance(settings, Settings)

    def test_settings_are_consistent(self):
        """Test that settings remain consistent across imports."""
        from config import settings as settings1
        from config import settings as settings2

        assert settings1.server_port == settings2.server_port
        assert settings1.environment == settings2.environment
