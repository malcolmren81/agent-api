"""Tests for palet8_agents.models.evaluation module."""

import pytest
from palet8_agents.models.evaluation import (
    EvaluationDecision,
    ResultQualityDimension,
    RetrySuggestion,
    ResultQualityResult,
    EvaluationPlan,
    EvaluationFeedback,
)
from palet8_agents.models.prompt import PromptQualityResult


class TestEvaluationDecision:
    """Tests for EvaluationDecision enum."""

    def test_values(self):
        """Test enum values exist."""
        assert EvaluationDecision.PASS.value == "PASS"
        assert EvaluationDecision.FIX_REQUIRED.value == "FIX_REQUIRED"
        assert EvaluationDecision.APPROVE.value == "APPROVE"
        assert EvaluationDecision.REJECT.value == "REJECT"
        assert EvaluationDecision.POLICY_FAIL.value == "POLICY_FAIL"


class TestResultQualityDimension:
    """Tests for ResultQualityDimension enum."""

    def test_values(self):
        """Test enum values exist."""
        assert ResultQualityDimension.PROMPT_FIDELITY.value == "prompt_fidelity"
        assert ResultQualityDimension.PRODUCT_READINESS.value == "product_readiness"
        assert ResultQualityDimension.TECHNICAL_QUALITY.value == "technical_quality"
        assert ResultQualityDimension.BACKGROUND_COMPOSITION.value == "background_composition"
        assert ResultQualityDimension.AESTHETIC.value == "aesthetic"
        assert ResultQualityDimension.TEXT_LEGIBILITY.value == "text_legibility"
        assert ResultQualityDimension.SET_CONSISTENCY.value == "set_consistency"


class TestRetrySuggestion:
    """Tests for RetrySuggestion dataclass."""

    def test_init(self):
        """Test initialization."""
        suggestion = RetrySuggestion(
            dimension="prompt_fidelity",
            suggested_changes=["Add more detail to subject"],
        )
        assert suggestion.dimension == "prompt_fidelity"
        assert suggestion.suggested_changes == ["Add more detail to subject"]

    def test_init_defaults(self):
        """Test default initialization."""
        suggestion = RetrySuggestion(dimension="technical_quality")
        assert suggestion.suggested_changes == []

    def test_to_dict(self):
        """Test to_dict serialization."""
        suggestion = RetrySuggestion(
            dimension="aesthetic",
            suggested_changes=["Improve lighting", "Adjust colors"],
        )
        data = suggestion.to_dict()

        assert data["dimension"] == "aesthetic"
        assert data["suggested_changes"] == ["Improve lighting", "Adjust colors"]

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "dimension": "composition",
            "suggested_changes": ["Center the subject"],
        }
        suggestion = RetrySuggestion.from_dict(data)

        assert suggestion.dimension == "composition"
        assert suggestion.suggested_changes == ["Center the subject"]


class TestResultQualityResult:
    """Tests for ResultQualityResult dataclass."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        result = ResultQualityResult(overall=0.85)
        assert result.overall == 0.85
        assert result.dimensions == {}
        assert result.mode == "STANDARD"
        assert result.threshold == 0.80
        assert result.decision == "APPROVE"
        assert result.feedback == []
        assert result.failed_dimensions == []
        assert result.retry_suggestions == []

    def test_is_acceptable_true(self):
        """Test is_acceptable when result passes."""
        result = ResultQualityResult(
            overall=0.9,
            threshold=0.8,
            decision="APPROVE",
        )
        assert result.is_acceptable is True

    def test_is_acceptable_false_low_score(self):
        """Test is_acceptable when score is below threshold."""
        result = ResultQualityResult(
            overall=0.7,
            threshold=0.8,
            decision="APPROVE",
        )
        assert result.is_acceptable is False

    def test_is_acceptable_false_rejected(self):
        """Test is_acceptable when decision is REJECT."""
        result = ResultQualityResult(
            overall=0.9,
            threshold=0.8,
            decision="REJECT",
        )
        assert result.is_acceptable is False

    def test_to_dict(self):
        """Test to_dict serialization."""
        suggestion = RetrySuggestion(
            dimension="lighting",
            suggested_changes=["Add more contrast"],
        )
        result = ResultQualityResult(
            overall=0.75,
            dimensions={"prompt_fidelity": 0.8, "aesthetic": 0.7},
            mode="COMPLEX",
            threshold=0.85,
            decision="REJECT",
            feedback=["Image too dark"],
            failed_dimensions=["aesthetic"],
            retry_suggestions=[suggestion],
        )
        data = result.to_dict()

        assert data["overall"] == 0.75
        assert data["dimensions"]["prompt_fidelity"] == 0.8
        assert data["mode"] == "COMPLEX"
        assert data["decision"] == "REJECT"
        assert len(data["retry_suggestions"]) == 1
        assert data["is_acceptable"] is False

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "overall": 0.9,
            "dimensions": {"prompt_fidelity": 0.95},
            "mode": "RELAX",
            "threshold": 0.70,
            "decision": "APPROVE",
            "feedback": [],
            "failed_dimensions": [],
            "retry_suggestions": [],
        }
        result = ResultQualityResult.from_dict(data)

        assert result.overall == 0.9
        assert result.mode == "RELAX"
        assert result.is_acceptable is True


class TestEvaluationPlan:
    """Tests for EvaluationPlan dataclass."""

    def test_init_minimal(self):
        """Test minimal initialization."""
        plan = EvaluationPlan(
            job_id="job-123",
            prompt="A beautiful sunset",
        )
        assert plan.job_id == "job-123"
        assert plan.prompt == "A beautiful sunset"
        assert plan.negative_prompt == ""
        assert plan.mode == "STANDARD"
        assert plan.product_type is None
        assert plan.prompt_quality is None

    def test_init_full(self):
        """Test full initialization."""
        prompt_quality = PromptQualityResult(overall=0.8)
        plan = EvaluationPlan(
            job_id="job-456",
            prompt="A cat on a couch",
            negative_prompt="blurry, low quality",
            mode="COMPLEX",
            product_type="poster",
            print_method="digital",
            dimensions_requested={"subject": "cat", "background": "living room"},
            prompt_quality=prompt_quality,
            result_weights={"prompt_fidelity": 0.25},
            result_thresholds={"overall": 0.85},
        )
        assert plan.mode == "COMPLEX"
        assert plan.product_type == "poster"
        assert plan.prompt_quality.overall == 0.8

    def test_to_dict(self):
        """Test to_dict serialization."""
        prompt_quality = PromptQualityResult(overall=0.85, decision="PASS")
        plan = EvaluationPlan(
            job_id="job-789",
            prompt="Test prompt",
            prompt_quality=prompt_quality,
        )
        data = plan.to_dict()

        assert data["job_id"] == "job-789"
        assert data["prompt_quality"]["overall"] == 0.85

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "job_id": "job-abc",
            "prompt": "A dog",
            "negative_prompt": "cat",
            "mode": "RELAX",
            "product_type": "t-shirt",
            "print_method": None,
            "dimensions_requested": {},
            "prompt_quality": {
                "overall": 0.7,
                "dimensions": {},
                "mode": "RELAX",
                "threshold": 0.5,
                "decision": "PASS",
                "feedback": [],
                "failed_dimensions": [],
            },
            "result_weights": {},
            "result_thresholds": {},
        }
        plan = EvaluationPlan.from_dict(data)

        assert plan.job_id == "job-abc"
        assert plan.mode == "RELAX"
        assert plan.prompt_quality.overall == 0.7


class TestEvaluationFeedback:
    """Tests for EvaluationFeedback dataclass."""

    def test_init(self):
        """Test initialization."""
        feedback = EvaluationFeedback(
            passed=False,
            overall_score=0.65,
            issues=["Subject unclear", "Background too busy"],
            retry_suggestions=["Add more detail to subject"],
        )
        assert feedback.passed is False
        assert feedback.overall_score == 0.65
        assert len(feedback.issues) == 2
        assert len(feedback.retry_suggestions) == 1

    def test_init_defaults(self):
        """Test default initialization."""
        feedback = EvaluationFeedback(
            passed=True,
            overall_score=0.9,
            issues=[],
            retry_suggestions=[],
        )
        assert feedback.dimension_scores == {}
        assert feedback.metadata == {}

    def test_to_dict(self):
        """Test to_dict serialization."""
        feedback = EvaluationFeedback(
            passed=True,
            overall_score=0.85,
            issues=[],
            retry_suggestions=[],
            dimension_scores={"prompt_fidelity": 0.9, "aesthetic": 0.8},
            metadata={"revision": 1},
        )
        data = feedback.to_dict()

        assert data["passed"] is True
        assert data["overall_score"] == 0.85
        assert data["dimension_scores"]["prompt_fidelity"] == 0.9
        assert data["metadata"]["revision"] == 1

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "passed": False,
            "overall_score": 0.6,
            "issues": ["Too dark"],
            "retry_suggestions": ["Increase brightness"],
            "dimension_scores": {"lighting": 0.3},
            "metadata": {},
        }
        feedback = EvaluationFeedback.from_dict(data)

        assert feedback.passed is False
        assert feedback.overall_score == 0.6
        assert feedback.issues == ["Too dark"]
        assert feedback.dimension_scores["lighting"] == 0.3

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        original = EvaluationFeedback(
            passed=True,
            overall_score=0.88,
            issues=[],
            retry_suggestions=[],
            dimension_scores={"prompt_fidelity": 0.9, "aesthetic": 0.85},
        )

        data = original.to_dict()
        restored = EvaluationFeedback.from_dict(data)

        assert restored.passed == original.passed
        assert restored.overall_score == original.overall_score
        assert restored.dimension_scores == original.dimension_scores
