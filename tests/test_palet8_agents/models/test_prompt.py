"""Tests for palet8_agents.models.prompt module."""

import pytest
from palet8_agents.models.prompt import (
    PromptQualityDimension,
    PromptDimensions,
    PromptQualityResult,
)


class TestPromptQualityDimension:
    """Tests for PromptQualityDimension enum."""

    def test_values(self):
        """Test enum values exist."""
        assert PromptQualityDimension.COVERAGE.value == "coverage"
        assert PromptQualityDimension.CLARITY.value == "clarity"
        assert PromptQualityDimension.PRODUCT_CONSTRAINTS.value == "product_constraints"
        assert PromptQualityDimension.STYLE_ALIGNMENT.value == "style_alignment"
        assert PromptQualityDimension.CONTROL_SURFACE.value == "control_surface"

    def test_from_string(self):
        """Test creating from string value."""
        assert PromptQualityDimension("coverage") == PromptQualityDimension.COVERAGE
        assert PromptQualityDimension("clarity") == PromptQualityDimension.CLARITY


class TestPromptDimensions:
    """Tests for PromptDimensions dataclass."""

    def test_init_defaults(self):
        """Test default initialization."""
        dims = PromptDimensions()
        assert dims.subject is None
        assert dims.aesthetic is None
        assert dims.color is None
        assert dims.composition is None
        assert dims.background is None
        assert dims.lighting is None
        assert dims.texture is None
        assert dims.detail_level is None
        assert dims.mood is None
        assert dims.expression is None
        assert dims.pose is None
        assert dims.art_movement is None
        assert dims.reference_style is None
        assert dims.technical == {}

    def test_init_partial(self):
        """Test partial initialization."""
        dims = PromptDimensions(
            subject="A majestic lion",
            aesthetic="realistic",
            color="golden, brown",
            mood="powerful",
        )
        assert dims.subject == "A majestic lion"
        assert dims.aesthetic == "realistic"
        assert dims.color == "golden, brown"
        assert dims.mood == "powerful"
        assert dims.background is None

    def test_init_with_technical(self):
        """Test initialization with technical specs."""
        dims = PromptDimensions(
            subject="Logo design",
            technical={
                "dpi": "300",
                "color_separation": "spot colors",
                "bleed": "0.125 inch",
            },
        )
        assert dims.technical["dpi"] == "300"
        assert dims.technical["color_separation"] == "spot colors"

    def test_to_dict_excludes_none(self):
        """Test to_dict excludes None values."""
        dims = PromptDimensions(
            subject="A cat",
            aesthetic="cartoon",
        )
        data = dims.to_dict()

        assert data["subject"] == "A cat"
        assert data["aesthetic"] == "cartoon"
        assert "color" not in data or data.get("color") is not None
        # Only non-None values should be present

    def test_to_dict_excludes_empty_dict(self):
        """Test to_dict excludes empty technical dict."""
        dims = PromptDimensions(subject="Test")
        data = dims.to_dict()

        # Empty technical dict should be excluded
        assert "technical" not in data or data.get("technical") == {}

    def test_to_dict_includes_non_empty_dict(self):
        """Test to_dict includes non-empty technical dict."""
        dims = PromptDimensions(
            subject="Test",
            technical={"dpi": "300"},
        )
        data = dims.to_dict()

        assert data["technical"] == {"dpi": "300"}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "subject": "A dog playing",
            "aesthetic": "illustration",
            "color": "warm tones",
            "background": "park",
            "mood": "playful",
            "technical": {"size": "11x17 inch"},
        }
        dims = PromptDimensions.from_dict(data)

        assert dims.subject == "A dog playing"
        assert dims.aesthetic == "illustration"
        assert dims.background == "park"
        assert dims.technical["size"] == "11x17 inch"

    def test_from_dict_defaults(self):
        """Test from_dict with missing fields uses None."""
        data = {"subject": "Simple test"}
        dims = PromptDimensions.from_dict(data)

        assert dims.subject == "Simple test"
        assert dims.aesthetic is None
        assert dims.technical == {}

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = PromptDimensions(
            subject="Mountain landscape",
            aesthetic="impressionist",
            color="blues and greens",
            composition="rule of thirds",
            background="cloudy sky",
            lighting="golden hour",
            mood="serene",
            art_movement="impressionism",
        )

        data = original.to_dict()
        restored = PromptDimensions.from_dict(data)

        assert restored.subject == original.subject
        assert restored.aesthetic == original.aesthetic
        assert restored.color == original.color
        assert restored.mood == original.mood


class TestPromptQualityResult:
    """Tests for PromptQualityResult dataclass."""

    def test_init_defaults(self):
        """Test default initialization."""
        result = PromptQualityResult(overall=0.7)
        assert result.overall == 0.7
        assert result.dimensions == {}
        assert result.mode == "STANDARD"
        assert result.threshold == 0.70
        assert result.decision == "PASS"
        assert result.feedback == []
        assert result.failed_dimensions == []

    def test_init_full(self):
        """Test full initialization."""
        result = PromptQualityResult(
            overall=0.85,
            dimensions={
                "coverage": 0.9,
                "clarity": 0.8,
                "product_constraints": 0.85,
                "style_alignment": 0.8,
                "control_surface": 0.9,
            },
            mode="COMPLEX",
            threshold=0.85,
            decision="PASS",
            feedback=["Good coverage", "Clear prompt"],
            failed_dimensions=[],
        )
        assert result.overall == 0.85
        assert result.dimensions["coverage"] == 0.9
        assert result.mode == "COMPLEX"

    def test_is_acceptable_true(self):
        """Test is_acceptable when prompt passes."""
        result = PromptQualityResult(
            overall=0.8,
            threshold=0.7,
            decision="PASS",
        )
        assert result.is_acceptable is True

    def test_is_acceptable_false_low_score(self):
        """Test is_acceptable when score is below threshold."""
        result = PromptQualityResult(
            overall=0.6,
            threshold=0.7,
            decision="PASS",
        )
        assert result.is_acceptable is False

    def test_is_acceptable_false_fix_required(self):
        """Test is_acceptable when decision is FIX_REQUIRED."""
        result = PromptQualityResult(
            overall=0.8,
            threshold=0.7,
            decision="FIX_REQUIRED",
        )
        assert result.is_acceptable is False

    def test_to_dict(self):
        """Test to_dict serialization."""
        result = PromptQualityResult(
            overall=0.75,
            dimensions={"clarity": 0.8, "coverage": 0.7},
            mode="RELAX",
            threshold=0.5,
            decision="PASS",
            feedback=["Good job"],
            failed_dimensions=[],
        )
        data = result.to_dict()

        assert data["overall"] == 0.75
        assert data["dimensions"]["clarity"] == 0.8
        assert data["mode"] == "RELAX"
        assert data["decision"] == "PASS"
        assert data["is_acceptable"] is True

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "overall": 0.65,
            "dimensions": {"coverage": 0.6, "clarity": 0.7},
            "mode": "STANDARD",
            "threshold": 0.70,
            "decision": "FIX_REQUIRED",
            "feedback": ["Coverage too low"],
            "failed_dimensions": ["coverage"],
        }
        result = PromptQualityResult.from_dict(data)

        assert result.overall == 0.65
        assert result.decision == "FIX_REQUIRED"
        assert "coverage" in result.failed_dimensions

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = PromptQualityResult(
            overall=0.82,
            dimensions={
                "coverage": 0.85,
                "clarity": 0.80,
                "product_constraints": 0.75,
            },
            mode="COMPLEX",
            threshold=0.85,
            decision="FIX_REQUIRED",
            feedback=["Product constraints need improvement"],
            failed_dimensions=["product_constraints"],
        )

        data = original.to_dict()
        restored = PromptQualityResult.from_dict(data)

        assert restored.overall == original.overall
        assert restored.dimensions == original.dimensions
        assert restored.mode == original.mode
        assert restored.decision == original.decision
        assert restored.failed_dimensions == original.failed_dimensions
