"""Tests for palet8_agents.services.result_evaluation_service module."""

import pytest
from palet8_agents.services.result_evaluation_service import (
    ResultEvaluationService,
    ResultEvaluationError,
    ResultEvaluationConfig,
)
from palet8_agents.models import EvaluationPlan, ResultQualityResult


class TestResultEvaluationConfig:
    """Tests for ResultEvaluationConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = ResultEvaluationConfig()
        assert "RELAX" in config.weights
        assert "STANDARD" in config.weights
        assert "COMPLEX" in config.weights
        assert config.min_resolution == 1024
        assert config.sharpness_threshold == 0.5
        assert config.retry_enabled is True
        assert "COMPLEX" in config.retry_applicable_modes

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ResultEvaluationConfig(
            min_resolution=2048,
            max_retry_attempts=5,
        )
        assert config.min_resolution == 2048
        assert config.max_retry_attempts == 5


class TestResultEvaluationService:
    """Tests for ResultEvaluationService."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        service = ResultEvaluationService()
        assert service._reasoning_service is None
        assert service._owns_service is True
        assert service._config is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ResultEvaluationConfig(min_resolution=2048)
        service = ResultEvaluationService(config=config)
        assert service._config.min_resolution == 2048

    def test_get_weights(self):
        """Test getting weights for a mode."""
        service = ResultEvaluationService()

        relax_weights = service.get_weights("RELAX")
        standard_weights = service.get_weights("STANDARD")

        assert relax_weights["product_readiness"] == 0.30
        assert standard_weights["product_readiness"] == 0.20

    def test_get_thresholds(self):
        """Test getting thresholds for a mode."""
        service = ResultEvaluationService()

        relax_thresholds = service.get_thresholds("RELAX")
        complex_thresholds = service.get_thresholds("COMPLEX")

        assert relax_thresholds["overall"] == 0.70
        assert complex_thresholds["overall"] == 0.85


class TestProductReadinessScoring:
    """Tests for product readiness scoring."""

    def test_readiness_good_resolution(self):
        """Test readiness with good resolution."""
        service = ResultEvaluationService()
        score, feedback = service._score_product_readiness(
            image_data={"width": 2048, "height": 2048},
            plan=EvaluationPlan(job_id="test", prompt="test"),
        )

        assert score >= 0.7
        assert not any("Resolution" in f for f in feedback)

    def test_readiness_low_resolution(self):
        """Test readiness with low resolution."""
        service = ResultEvaluationService()
        score, feedback = service._score_product_readiness(
            image_data={"width": 512, "height": 512},
            plan=EvaluationPlan(job_id="test", prompt="test"),
        )

        assert score < 0.7
        assert any("Resolution" in f for f in feedback)

    def test_readiness_cropping_issues(self):
        """Test readiness with cropping issues."""
        service = ResultEvaluationService()
        score, feedback = service._score_product_readiness(
            image_data={
                "width": 1024,
                "height": 1024,
                "has_cropping_issues": True,
            },
            plan=EvaluationPlan(job_id="test", prompt="test"),
        )

        assert any("cropped" in f.lower() for f in feedback)

    def test_readiness_low_coverage(self):
        """Test readiness with low subject coverage."""
        service = ResultEvaluationService()
        score, feedback = service._score_product_readiness(
            image_data={
                "width": 1024,
                "height": 1024,
                "coverage_percent": 50,  # Below 80%
            },
            plan=EvaluationPlan(job_id="test", prompt="test"),
        )

        assert any("coverage" in f.lower() for f in feedback)


class TestTechnicalQualityScoring:
    """Tests for technical quality scoring."""

    def test_quality_good_image(self):
        """Test quality with good image."""
        service = ResultEvaluationService()
        score, feedback = service._score_technical_quality(
            image_data={
                "sharpness_score": 0.9,
                "noise_level": 0.1,
                "detected_defects": [],
            }
        )

        assert score >= 0.8
        assert len(feedback) == 0

    def test_quality_blurry_image(self):
        """Test quality with blurry image."""
        service = ResultEvaluationService()
        score, feedback = service._score_technical_quality(
            image_data={"sharpness_score": 0.3}
        )

        assert score < 0.8
        assert any("blurry" in f.lower() for f in feedback)

    def test_quality_noisy_image(self):
        """Test quality with noisy image."""
        service = ResultEvaluationService()
        score, feedback = service._score_technical_quality(
            image_data={"noise_level": 0.5}
        )

        assert any("noise" in f.lower() for f in feedback)

    def test_quality_defects(self):
        """Test quality with detected defects."""
        service = ResultEvaluationService()
        score, feedback = service._score_technical_quality(
            image_data={"detected_defects": ["watermark", "artifacts"]}
        )

        assert score < 0.8
        assert any("watermark" in f for f in feedback)


class TestBackgroundCompositionScoring:
    """Tests for background and composition scoring."""

    def test_background_match(self):
        """Test background matching."""
        service = ResultEvaluationService()
        score, feedback = service._score_background_composition(
            image_data={"detected_background": "solid white"},
            plan=EvaluationPlan(
                job_id="test",
                prompt="test",
                dimensions_requested={"background": "solid white"},
            ),
        )

        assert score >= 0.7
        assert not any("mismatch" in f.lower() for f in feedback)

    def test_background_mismatch(self):
        """Test background mismatch."""
        service = ResultEvaluationService()
        score, feedback = service._score_background_composition(
            image_data={"detected_background": "gradient blue"},
            plan=EvaluationPlan(
                job_id="test",
                prompt="test",
                dimensions_requested={"background": "solid white"},
            ),
        )

        assert any("mismatch" in f.lower() for f in feedback)


class TestTextLegibilityScoring:
    """Tests for text legibility scoring."""

    def test_text_good_ocr(self):
        """Test text with good OCR confidence."""
        service = ResultEvaluationService()
        score, feedback = service._score_text_legibility(
            image_data={"ocr_confidence": 0.95}
        )

        assert score >= 0.9
        assert len(feedback) == 0

    def test_text_poor_ocr(self):
        """Test text with poor OCR confidence."""
        service = ResultEvaluationService()
        score, feedback = service._score_text_legibility(
            image_data={"ocr_confidence": 0.4}
        )

        assert score < 0.7
        assert any("difficult to read" in f.lower() for f in feedback)

    def test_text_no_ocr_data(self):
        """Test text without OCR data."""
        service = ResultEvaluationService()
        score, feedback = service._score_text_legibility(image_data={})

        assert score == 0.6
        assert any("could not be verified" in f.lower() for f in feedback)


class TestTextContentDetection:
    """Tests for text content detection."""

    def test_has_text_explicit(self):
        """Test explicit text flag."""
        service = ResultEvaluationService()
        has_text = service._has_text_content(
            image_data={"has_text": True},
            plan=EvaluationPlan(job_id="test", prompt="A landscape"),
        )

        assert has_text is True

    def test_has_text_from_prompt(self):
        """Test text detection from prompt keywords."""
        service = ResultEvaluationService()
        has_text = service._has_text_content(
            image_data={},
            plan=EvaluationPlan(
                job_id="test",
                prompt="A logo with typography and headline",
            ),
        )

        assert has_text is True

    def test_no_text_content(self):
        """Test no text content."""
        service = ResultEvaluationService()
        has_text = service._has_text_content(
            image_data={},
            plan=EvaluationPlan(job_id="test", prompt="A simple landscape"),
        )

        assert has_text is False


class TestRetryLogic:
    """Tests for retry decision logic."""

    def test_should_retry_rejected_complex(self):
        """Test retry for rejected COMPLEX mode."""
        service = ResultEvaluationService()
        result = ResultQualityResult(
            overall=0.6,
            dimensions={},
            mode="COMPLEX",
            threshold=0.85,
            decision="REJECT",
            feedback=[],
            failed_dimensions=["prompt_fidelity"],
            retry_suggestions=[],
        )

        assert service.should_retry(result, attempt_count=1) is True

    def test_should_not_retry_approved(self):
        """Test no retry for approved result."""
        service = ResultEvaluationService()
        result = ResultQualityResult(
            overall=0.9,
            dimensions={},
            mode="COMPLEX",
            threshold=0.85,
            decision="APPROVE",
            feedback=[],
            failed_dimensions=[],
            retry_suggestions=[],
        )

        assert service.should_retry(result, attempt_count=1) is False

    def test_should_not_retry_max_attempts(self):
        """Test no retry after max attempts."""
        service = ResultEvaluationService()
        result = ResultQualityResult(
            overall=0.6,
            dimensions={},
            mode="COMPLEX",
            threshold=0.85,
            decision="REJECT",
            feedback=[],
            failed_dimensions=["prompt_fidelity"],
            retry_suggestions=[],
        )

        assert service.should_retry(result, attempt_count=3) is False

    def test_should_not_retry_standard_mode(self):
        """Test no retry for STANDARD mode (not in retry modes)."""
        service = ResultEvaluationService()
        result = ResultQualityResult(
            overall=0.6,
            dimensions={},
            mode="STANDARD",
            threshold=0.80,
            decision="REJECT",
            feedback=[],
            failed_dimensions=["prompt_fidelity"],
            retry_suggestions=[],
        )

        assert service.should_retry(result, attempt_count=1) is False


class TestOverallEvaluation:
    """Tests for overall image evaluation."""

    @pytest.mark.asyncio
    async def test_evaluate_good_image(self):
        """Test evaluation of good image."""
        service = ResultEvaluationService()
        plan = EvaluationPlan(
            job_id="test-123",
            prompt="A mountain landscape",
            mode="STANDARD",
        )
        image_data = {
            "width": 2048,
            "height": 2048,
            "sharpness_score": 0.9,
            "noise_level": 0.1,
            "aesthetic_score": 0.85,
            "color_harmony_score": 0.8,
        }

        result = await service.evaluate_image(image_data, plan)

        assert result.overall >= 0.7
        assert result.decision in ["APPROVE", "REJECT"]

    @pytest.mark.asyncio
    async def test_evaluate_poor_image(self):
        """Test evaluation of poor image."""
        service = ResultEvaluationService()
        plan = EvaluationPlan(
            job_id="test-123",
            prompt="A detailed portrait",
            mode="COMPLEX",
        )
        image_data = {
            "width": 512,
            "height": 512,
            "sharpness_score": 0.2,
            "noise_level": 0.6,
            "detected_defects": ["artifacts", "blur"],
        }

        result = await service.evaluate_image(image_data, plan)

        assert result.decision == "REJECT"
        assert len(result.failed_dimensions) > 0
        assert len(result.retry_suggestions) > 0

    @pytest.mark.asyncio
    async def test_evaluate_image_set(self):
        """Test evaluation of image set."""
        service = ResultEvaluationService()
        plan = EvaluationPlan(
            job_id="test-123",
            prompt="A cat",
            mode="STANDARD",
        )
        images = [
            {"width": 1024, "height": 1024, "sharpness_score": 0.8},
            {"width": 1024, "height": 1024, "sharpness_score": 0.7},
        ]

        results = await service.evaluate_image_set(images, plan)

        assert len(results) == 2
        assert all(isinstance(r, ResultQualityResult) for r in results)


class TestAsyncOperations:
    """Async tests for ResultEvaluationService."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with ResultEvaluationService() as service:
            assert service is not None

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing service."""
        service = ResultEvaluationService()
        await service.close()  # Should not raise
