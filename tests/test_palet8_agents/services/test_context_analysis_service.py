"""Tests for palet8_agents.services.context_analysis_service module."""

import pytest
from palet8_agents.services.context_analysis_service import (
    ContextAnalysisService,
    ContextConfig,
)
from palet8_agents.models import ContextCompleteness


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = ContextConfig()
        assert config.required_fields == ["subject"]
        assert "style" in config.important_fields
        assert "mood" in config.optional_fields
        assert config.min_completeness == 0.5
        assert "subject" in config.clarifying_questions

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ContextConfig(
            required_fields=["subject", "style"],
            min_completeness=0.7,
        )
        assert "style" in config.required_fields
        assert config.min_completeness == 0.7


class TestContextAnalysisService:
    """Tests for ContextAnalysisService."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        service = ContextAnalysisService()
        assert service._config is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ContextConfig(min_completeness=0.8)
        service = ContextAnalysisService(config=config)
        assert service._config.min_completeness == 0.8

    def test_evaluate_completeness_sufficient(self):
        """Test completeness evaluation when sufficient."""
        service = ContextAnalysisService()
        requirements = {
            "subject": "A mountain landscape",
            "style": "realistic",
            "aesthetic": "dramatic",
            "colors": ["blue", "white"],
            "product_type": "poster",
        }
        result = service.evaluate_completeness(requirements)

        assert isinstance(result, ContextCompleteness)
        assert result.is_sufficient is True
        assert result.score >= 0.5
        assert len(result.missing_fields) < 5

    def test_evaluate_completeness_insufficient(self):
        """Test completeness evaluation when insufficient."""
        service = ContextAnalysisService()
        requirements = {}  # Empty requirements
        result = service.evaluate_completeness(requirements)

        assert result.is_sufficient is False
        assert result.score == 0.0
        assert "subject" in result.missing_fields
        assert len(result.clarifying_questions) > 0

    def test_evaluate_completeness_missing_required(self):
        """Test that missing required fields makes it insufficient."""
        service = ContextAnalysisService()
        requirements = {
            "style": "realistic",
            "colors": ["red", "blue"],
            # Missing subject (required)
        }
        result = service.evaluate_completeness(requirements)

        assert result.is_sufficient is False
        assert "subject" in result.missing_fields
        assert "subject" in result.metadata["required_missing"]

    def test_evaluate_completeness_with_threshold(self):
        """Test completeness with custom threshold."""
        service = ContextAnalysisService()
        requirements = {
            "subject": "A cat",
            "style": "cartoon",
        }
        # With low threshold, should be sufficient
        result = service.evaluate_completeness(requirements, threshold=0.3)
        assert result.is_sufficient is True

        # With high threshold, should be insufficient
        result = service.evaluate_completeness(requirements, threshold=0.9)
        assert result.is_sufficient is False

    def test_has_value_string(self):
        """Test _has_value with strings."""
        service = ContextAnalysisService()
        assert service._has_value("hello") is True
        assert service._has_value("") is False
        assert service._has_value("  ") is False
        assert service._has_value(None) is False

    def test_has_value_list(self):
        """Test _has_value with lists."""
        service = ContextAnalysisService()
        assert service._has_value(["a", "b"]) is True
        assert service._has_value([]) is False
        assert service._has_value(None) is False

    def test_has_value_dict(self):
        """Test _has_value with dicts."""
        service = ContextAnalysisService()
        assert service._has_value({"key": "value"}) is True
        assert service._has_value({}) is False

    def test_generate_question(self):
        """Test question generation for missing fields."""
        service = ContextAnalysisService()
        question = service._generate_question("subject")
        assert question is not None
        assert "subject" in question.lower() or "image" in question.lower()

        # Unknown field should return None
        question = service._generate_question("unknown_field")
        assert question is None

    def test_get_priority_missing_fields(self):
        """Test getting priority missing fields."""
        service = ContextAnalysisService()
        requirements = {
            "style": "cartoon",  # important field present
            # subject (required) missing
            # aesthetic (important) missing
        }
        missing = service.get_priority_missing_fields(requirements, max_fields=3)

        # Required fields should come first
        assert "subject" in missing
        assert missing[0] == "subject"  # First missing field is required

    def test_get_priority_questions(self):
        """Test getting priority questions."""
        service = ContextAnalysisService()
        requirements = {}  # All fields missing
        questions = service.get_priority_questions(requirements, max_questions=2)

        assert len(questions) <= 2
        assert all(isinstance(q, str) for q in questions)

    def test_calculate_field_score(self):
        """Test field score calculation."""
        service = ContextAnalysisService()

        # Present field should get weight
        score = service.calculate_field_score("subject", "A cat")
        assert score == service._config.completeness_weights.get("subject", 0.0)

        # Missing field should get 0
        score = service.calculate_field_score("subject", None)
        assert score == 0.0

        # Empty field should get 0
        score = service.calculate_field_score("subject", "")
        assert score == 0.0

    def test_get_all_fields(self):
        """Test getting all field categories."""
        service = ContextAnalysisService()
        fields = service.get_all_fields()

        assert "required" in fields
        assert "important" in fields
        assert "optional" in fields
        assert "subject" in fields["required"]

    def test_merge_requirements(self):
        """Test merging requirements."""
        service = ContextAnalysisService()
        base = {
            "subject": "A dog",
            "style": "realistic",
            "colors": ["brown"],
        }
        updates = {
            "style": "cartoon",  # Override
            "mood": "playful",  # Add new
            "colors": [],  # Empty, should not override
        }
        result = service.merge_requirements(base, updates)

        assert result["subject"] == "A dog"  # Unchanged
        assert result["style"] == "cartoon"  # Updated
        assert result["mood"] == "playful"  # Added
        assert result["colors"] == ["brown"]  # Not overridden by empty

    def test_metadata_categorization(self):
        """Test that metadata properly categorizes missing fields."""
        service = ContextAnalysisService()
        requirements = {
            "style": "cartoon",  # important present
            # subject (required) missing
            # aesthetic (important) missing
            # mood (optional) missing
        }
        result = service.evaluate_completeness(requirements)

        assert "subject" in result.metadata["required_missing"]
        assert "aesthetic" in result.metadata["important_missing"]
        assert "mood" in result.metadata["optional_missing"]
        assert "style" not in result.metadata["important_missing"]
