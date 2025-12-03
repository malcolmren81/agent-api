"""Tests for palet8_agents.services.prompt_evaluation_service module."""

import pytest
from palet8_agents.services.prompt_evaluation_service import (
    PromptEvaluationService,
    PromptEvaluationError,
    PromptEvaluationConfig,
)


class TestPromptEvaluationConfig:
    """Tests for PromptEvaluationConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = PromptEvaluationConfig()
        assert "RELAX" in config.weights
        assert "STANDARD" in config.weights
        assert "COMPLEX" in config.weights
        assert "coverage" in config.weights["STANDARD"]
        assert len(config.contradiction_pairs) > 0
        assert len(config.vague_terms) > 0

    def test_custom_values(self):
        """Test custom configuration values."""
        config = PromptEvaluationConfig(
            vague_terms=["custom", "terms"],
            contradiction_penalty=-0.5,
        )
        assert "custom" in config.vague_terms
        assert config.contradiction_penalty == -0.5


class TestPromptEvaluationService:
    """Tests for PromptEvaluationService."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        service = PromptEvaluationService()
        assert service._reasoning_service is None
        assert service._owns_service is True
        assert service._config is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = PromptEvaluationConfig(contradiction_penalty=-0.5)
        service = PromptEvaluationService(config=config)
        assert service._config.contradiction_penalty == -0.5

    def test_get_weights(self):
        """Test getting weights for a mode."""
        service = PromptEvaluationService()

        relax_weights = service.get_weights("RELAX")
        standard_weights = service.get_weights("STANDARD")

        assert relax_weights["coverage"] == 0.30
        assert standard_weights["coverage"] == 0.25

    def test_get_thresholds(self):
        """Test getting thresholds for a mode."""
        service = PromptEvaluationService()

        relax_thresholds = service.get_thresholds("RELAX")
        complex_thresholds = service.get_thresholds("COMPLEX")

        assert relax_thresholds["overall"] == 0.50
        assert complex_thresholds["overall"] == 0.85


class TestCoverageScoring:
    """Tests for coverage dimension scoring."""

    def test_coverage_full(self):
        """Test coverage with all required dimensions."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A sunset landscape over mountains",
            mode="RELAX",
            dimensions={"subject": "sunset"},
        )

        assert result.dimensions["coverage"] == 1.0

    def test_coverage_missing_required(self):
        """Test coverage with missing required dimensions."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="Some colors and shapes",
            mode="STANDARD",
            dimensions={},  # Missing subject, aesthetic, background
        )

        assert result.dimensions["coverage"] < 1.0
        assert "coverage" in result.failed_dimensions
        assert any("Missing" in f for f in result.feedback)

    def test_coverage_apparel_product(self):
        """Test coverage feedback for apparel product."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A cat design",
            mode="STANDARD",
            product_type="apparel",
            dimensions={"subject": "cat"},
        )

        # Should suggest print area for apparel
        assert any("apparel" in f.lower() for f in result.feedback)


class TestClarityScoring:
    """Tests for clarity dimension scoring."""

    def test_clarity_clean_prompt(self):
        """Test clarity with clean prompt."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A detailed mountain landscape at sunset with golden light",
            mode="STANDARD",
            dimensions={"subject": "mountain"},
        )

        assert result.dimensions["clarity"] >= 0.7

    def test_clarity_contradiction_detected(self):
        """Test clarity with contradiction."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A dark scene with bright white background",
            mode="STANDARD",
            dimensions={"subject": "scene"},
        )

        assert result.dimensions["clarity"] < 1.0
        assert any("Contradiction" in f for f in result.feedback)

    def test_clarity_vague_terms(self):
        """Test clarity with vague terms."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A nice cool awesome design that looks great",
            mode="STANDARD",
            dimensions={"subject": "design"},
        )

        assert result.dimensions["clarity"] < 1.0
        assert any("Vague" in f for f in result.feedback)

    def test_clarity_negative_contradiction(self):
        """Test clarity with negative prompt contradiction."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A beautiful sunset over mountains",
            negative_prompt="mountains sunset blurry",
            mode="STANDARD",
            dimensions={"subject": "sunset"},
        )

        assert result.dimensions["clarity"] < 1.0
        assert any("contradict" in f.lower() for f in result.feedback)


class TestProductConstraintsScoring:
    """Tests for product constraints scoring."""

    def test_constraints_apparel_with_placement(self):
        """Test constraints for apparel with placement info."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A centered design for t-shirt print area",
            mode="STANDARD",
            product_type="apparel",
            dimensions={"subject": "design"},
        )

        assert result.dimensions["product_constraints"] > 0.7

    def test_constraints_screen_print_gradient(self):
        """Test constraints warning for gradient in screen print."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A gradient design with smooth color transitions",
            mode="STANDARD",
            product_type="apparel",
            print_method="screen_print",
            dimensions={"subject": "gradient"},
        )

        assert any("halftone" in f.lower() for f in result.feedback)

    def test_constraints_embroidery_fine_detail(self):
        """Test constraints warning for fine detail in embroidery."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A photorealistic portrait with fine detail",
            mode="STANDARD",
            product_type="apparel",
            print_method="embroidery",
            dimensions={"subject": "portrait"},
        )

        assert any("embroidery" in f.lower() for f in result.feedback)


class TestStyleAlignmentScoring:
    """Tests for style alignment scoring."""

    def test_style_relax_simple(self):
        """Test style for RELAX mode with simple prompt."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A cat sitting on a mat",
            mode="RELAX",
            dimensions={"subject": "cat", "aesthetic": "cartoon"},
        )

        assert result.dimensions["style_alignment"] >= 0.7

    def test_style_relax_too_complex(self):
        """Test style for RELAX mode with too complex prompt."""
        service = PromptEvaluationService()
        # Create a prompt with more than 50 words
        long_prompt = " ".join(["word"] * 60)
        result = service.assess_quality(
            prompt=long_prompt,
            mode="RELAX",
            dimensions={"subject": "test"},
        )

        assert any("complex" in f.lower() for f in result.feedback)

    def test_style_complex_too_simple(self):
        """Test style for COMPLEX mode with too simple prompt."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A cat",  # Very short
            mode="COMPLEX",
            dimensions={"subject": "cat"},
        )

        assert any("more detail" in f.lower() for f in result.feedback)


class TestControlSurfaceScoring:
    """Tests for control surface (negative prompt) scoring."""

    def test_control_with_useful_negatives(self):
        """Test control with useful negative prompts."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A beautiful landscape",
            negative_prompt="blurry, low quality, distorted, bad anatomy",
            mode="STANDARD",
            dimensions={"subject": "landscape"},
        )

        assert result.dimensions["control_surface"] >= 0.7

    def test_control_no_negative(self):
        """Test control without negative prompt."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A beautiful landscape",
            negative_prompt="",
            mode="STANDARD",
            dimensions={"subject": "landscape"},
        )

        assert result.dimensions["control_surface"] < 0.7
        assert any("Add negative" in f for f in result.feedback)

    def test_control_weak_negative(self):
        """Test control with weak negative prompt."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A beautiful landscape",
            negative_prompt="stuff things",  # Not useful
            mode="STANDARD",
            dimensions={"subject": "landscape"},
        )

        assert any("more specific" in f.lower() for f in result.feedback)


class TestOverallAssessment:
    """Tests for overall quality assessment."""

    def test_pass_decision(self):
        """Test PASS decision for good prompt."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A detailed mountain landscape at golden hour with dramatic lighting, "
                   "realistic style, centered composition for poster print",
            negative_prompt="blurry, low quality, distorted",
            mode="STANDARD",
            dimensions={
                "subject": "mountain landscape",
                "aesthetic": "realistic",
                "background": "sky",
            },
        )

        assert result.decision == "PASS"
        assert service.is_acceptable(result) is True

    def test_fix_required_decision(self):
        """Test FIX_REQUIRED decision for poor prompt."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="nice cool stuff",
            mode="COMPLEX",  # High thresholds
            dimensions={},
        )

        assert result.decision == "FIX_REQUIRED"
        assert service.is_acceptable(result) is False
        assert len(result.failed_dimensions) > 0

    def test_mode_thresholds(self):
        """Test that mode affects thresholds."""
        service = PromptEvaluationService()

        # Same prompt, different modes
        relax_result = service.assess_quality(
            prompt="A cat",
            mode="RELAX",
            dimensions={"subject": "cat"},
        )
        complex_result = service.assess_quality(
            prompt="A cat",
            mode="COMPLEX",
            dimensions={"subject": "cat"},
        )

        # RELAX should be more lenient
        assert relax_result.threshold < complex_result.threshold


class TestAsyncOperations:
    """Async tests for PromptEvaluationService."""

    @pytest.mark.asyncio
    async def test_propose_revision_no_service(self):
        """Test revision proposal without reasoning service."""
        service = PromptEvaluationService()
        result = service.assess_quality(
            prompt="A nice cat",
            mode="STANDARD",
            dimensions={"subject": "cat"},
        )

        revised = await service.propose_revision("A nice cat", result)

        assert "Consider" in revised

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with PromptEvaluationService() as service:
            assert service is not None

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing service."""
        service = PromptEvaluationService()
        await service.close()  # Should not raise
