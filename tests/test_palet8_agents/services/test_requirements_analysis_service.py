"""Tests for palet8_agents.services.requirements_analysis_service module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from palet8_agents.services.requirements_analysis_service import (
    RequirementsAnalysisService,
    RequirementsConfig,
)
from palet8_agents.models import RequirementsStatus


class TestRequirementsConfig:
    """Tests for RequirementsConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = RequirementsConfig()
        assert config.completeness_threshold == 0.5
        assert config.completeness_weights["subject"] == 0.50
        assert config.required_fields == ["subject"]
        assert config.extraction_temperature == 0.2
        assert config.min_input_length == 3
        assert config.max_input_length == 10000

    def test_custom_values(self):
        """Test custom configuration values."""
        config = RequirementsConfig(
            completeness_threshold=0.7,
            completeness_weights={"subject": 0.6, "style": 0.4},
            required_fields=["subject", "style"],
        )
        assert config.completeness_threshold == 0.7
        assert config.completeness_weights["subject"] == 0.6
        assert "style" in config.required_fields


class TestRequirementsAnalysisService:
    """Tests for RequirementsAnalysisService."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        service = RequirementsAnalysisService()
        assert service._text_service is None
        assert service._owns_service is True
        assert service._config is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = RequirementsConfig(completeness_threshold=0.8)
        service = RequirementsAnalysisService(config=config)
        assert service._config.completeness_threshold == 0.8

    def test_calculate_completeness_full(self):
        """Test completeness calculation with all fields."""
        service = RequirementsAnalysisService()
        requirements = RequirementsStatus()
        requirements.subject = "A cat"
        requirements.style = "cartoon"
        requirements.colors = ["blue", "green"]
        requirements.mood = "playful"
        score = service.calculate_completeness(requirements)
        assert score == 1.0  # 0.5 + 0.2 + 0.15 + 0.15 = 1.0

    def test_calculate_completeness_partial(self):
        """Test completeness calculation with some fields."""
        service = RequirementsAnalysisService()
        requirements = RequirementsStatus()
        requirements.subject = "A dog"
        score = service.calculate_completeness(requirements)
        assert score == 0.5  # Only subject

    def test_calculate_completeness_empty(self):
        """Test completeness calculation with no fields."""
        service = RequirementsAnalysisService()
        requirements = RequirementsStatus()
        score = service.calculate_completeness(requirements)
        assert score == 0.0

    def test_get_missing_fields_all_missing(self):
        """Test getting missing fields when all are missing."""
        service = RequirementsAnalysisService()
        requirements = RequirementsStatus()
        missing = service.get_missing_fields(requirements)
        assert "subject" in missing
        assert "style" in missing
        assert "colors" in missing

    def test_get_missing_fields_some_present(self):
        """Test getting missing fields when some are present."""
        service = RequirementsAnalysisService()
        requirements = RequirementsStatus()
        requirements.subject = "A mountain"
        requirements.style = "realistic"
        missing = service.get_missing_fields(requirements)
        assert "subject" not in missing
        assert "style" not in missing
        assert "colors" in missing

    def test_get_missing_fields_required_only(self):
        """Test getting only required missing fields."""
        service = RequirementsAnalysisService()
        requirements = RequirementsStatus()
        missing = service.get_missing_fields(requirements, include_recommended=False)
        assert "subject" in missing
        assert "style" not in missing  # style is recommended, not required

    def test_is_complete_true(self):
        """Test is_complete when subject is present."""
        service = RequirementsAnalysisService()
        requirements = RequirementsStatus()
        requirements.subject = "A sunset"
        assert service.is_complete(requirements) is True

    def test_is_complete_false(self):
        """Test is_complete when subject is missing."""
        service = RequirementsAnalysisService()
        requirements = RequirementsStatus()
        requirements.style = "watercolor"
        assert service.is_complete(requirements) is False

    def test_validate_input_valid(self):
        """Test validation of valid input."""
        service = RequirementsAnalysisService()
        result = service.validate_input("Create a beautiful sunset image")
        assert result["is_valid"] is True
        assert result["issues"] == []

    def test_validate_input_empty(self):
        """Test validation of empty input."""
        service = RequirementsAnalysisService()
        result = service.validate_input("")
        assert result["is_valid"] is False
        assert "Input cannot be empty" in result["issues"]

    def test_validate_input_too_short(self):
        """Test validation of too short input."""
        service = RequirementsAnalysisService()
        result = service.validate_input("ab")
        assert result["is_valid"] is False
        assert "Input is too short" in result["issues"]

    def test_validate_input_too_long(self):
        """Test validation of too long input."""
        service = RequirementsAnalysisService()
        result = service.validate_input("a" * 10001)
        assert result["is_valid"] is False
        assert any("too long" in issue for issue in result["issues"])


class TestRequirementsAnalysisServiceAsync:
    """Async tests for RequirementsAnalysisService."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing service releases resources."""
        service = RequirementsAnalysisService()
        # Create a mock text service
        mock_service = AsyncMock()
        service._text_service = mock_service
        service._owns_service = True

        await service.close()

        mock_service.close.assert_called_once()
        assert service._text_service is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with RequirementsAnalysisService() as service:
            assert service is not None

    @pytest.mark.asyncio
    async def test_analyze_conversation_empty(self):
        """Test analyzing empty conversation."""
        service = RequirementsAnalysisService()

        # Create mock conversation
        mock_conversation = MagicMock()
        mock_conversation.messages = []

        result = await service.analyze_conversation(mock_conversation)

        assert isinstance(result, RequirementsStatus)
        assert result.subject is None

    @pytest.mark.asyncio
    async def test_analyze_conversation_with_existing(self):
        """Test analyzing conversation with existing requirements."""
        service = RequirementsAnalysisService()

        mock_conversation = MagicMock()
        mock_conversation.messages = []

        existing = {
            "subject": "A landscape",
            "style": "oil painting",
            "colors": ["blue", "green"],
        }

        result = await service.analyze_conversation(
            mock_conversation,
            existing_requirements=existing,
        )

        assert result.subject == "A landscape"
        assert result.style == "oil painting"
        assert result.colors == ["blue", "green"]
