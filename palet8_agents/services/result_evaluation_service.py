"""
Result Evaluation Service.

Handles assessment of generated image quality across multiple dimensions
and provides retry suggestions when quality is insufficient.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from palet8_agents.models import (
    EvaluationPlan,
    ResultQualityResult,
    RetrySuggestion,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ResultEvaluationConfig:
    """Configuration for result evaluation service."""

    # Weights by mode
    weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "RELAX": {
            "prompt_fidelity": 0.25,
            "product_readiness": 0.30,
            "technical_quality": 0.20,
            "background_composition": 0.10,
            "aesthetic": 0.10,
            "text_legibility": 0.05,
            "set_consistency": 0.00,
        },
        "STANDARD": {
            "prompt_fidelity": 0.25,
            "product_readiness": 0.20,
            "technical_quality": 0.20,
            "background_composition": 0.15,
            "aesthetic": 0.15,
            "text_legibility": 0.05,
            "set_consistency": 0.00,
        },
        "COMPLEX": {
            "prompt_fidelity": 0.22,
            "product_readiness": 0.18,
            "technical_quality": 0.18,
            "background_composition": 0.17,
            "aesthetic": 0.17,
            "text_legibility": 0.05,
            "set_consistency": 0.03,
        },
    })

    # Thresholds by mode
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "RELAX": {
            "overall": 0.70,
            "prompt_fidelity": 0.50,
            "product_readiness": 0.60,
            "technical_quality": 0.50,
            "background_composition": 0.40,
            "aesthetic": 0.40,
            "text_legibility": 0.50,
            "set_consistency": 0.0,
        },
        "STANDARD": {
            "overall": 0.80,
            "prompt_fidelity": 0.70,
            "product_readiness": 0.70,
            "technical_quality": 0.60,
            "background_composition": 0.60,
            "aesthetic": 0.60,
            "text_legibility": 0.60,
            "set_consistency": 0.0,
        },
        "COMPLEX": {
            "overall": 0.85,
            "prompt_fidelity": 0.80,
            "product_readiness": 0.75,
            "technical_quality": 0.75,
            "background_composition": 0.70,
            "aesthetic": 0.70,
            "text_legibility": 0.70,
            "set_consistency": 0.60,
        },
    })

    # Quality checks
    min_resolution: int = 1024
    min_coverage_percent: float = 0.8
    sharpness_threshold: float = 0.5
    noise_threshold: float = 0.3

    # Text detection keywords
    text_detection_keywords: List[str] = field(default_factory=lambda: [
        "text", "logo", "typography", "lettering", "words",
        "title", "headline", "quote", "caption", "label",
    ])

    # Retry settings
    retry_enabled: bool = True
    max_retry_attempts: int = 3
    retry_applicable_modes: List[str] = field(default_factory=lambda: ["COMPLEX"])


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ResultEvaluationError(Exception):
    """Base exception for result evaluation errors."""
    pass


# =============================================================================
# SERVICE
# =============================================================================


class ResultEvaluationService:
    """
    Service for evaluating generated image quality.

    This service handles:
    - Scoring images across multiple quality dimensions
    - Checking technical quality (resolution, sharpness, noise)
    - Validating product readiness
    - Providing retry suggestions for failed dimensions
    """

    def __init__(
        self,
        reasoning_service: Optional[Any] = None,
        config: Optional[ResultEvaluationConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize result evaluation service.

        Args:
            reasoning_service: Optional ReasoningService for fidelity scoring
            config: Optional custom configuration
            config_path: Optional path to config file
        """
        self._reasoning_service = reasoning_service
        self._owns_service = reasoning_service is None
        self._config = config or ResultEvaluationConfig()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            if data.get("weights"):
                self._config.weights = data["weights"]
            if data.get("thresholds"):
                self._config.thresholds = data["thresholds"]

            # Quality checks
            qc = data.get("quality_checks", {})
            if "min_resolution" in qc:
                self._config.min_resolution = qc["min_resolution"]
            if "min_coverage_percent" in qc:
                self._config.min_coverage_percent = qc["min_coverage_percent"]
            if "sharpness_threshold" in qc:
                self._config.sharpness_threshold = qc["sharpness_threshold"]
            if "noise_threshold" in qc:
                self._config.noise_threshold = qc["noise_threshold"]

            if data.get("text_detection_keywords"):
                self._config.text_detection_keywords = data["text_detection_keywords"]

            # Retry settings
            retry = data.get("retry", {})
            if "enabled" in retry:
                self._config.retry_enabled = retry["enabled"]
            if "max_attempts" in retry:
                self._config.max_retry_attempts = retry["max_attempts"]
            if "applicable_modes" in retry:
                self._config.retry_applicable_modes = retry["applicable_modes"]

        except Exception as e:
            logger.warning(f"Failed to load result evaluation config: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def evaluate_image(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> ResultQualityResult:
        """
        Evaluate generated image against quality standards.

        Args:
            image_data: Dict with image URL, metadata, analysis results
            plan: EvaluationPlan with prompt, mode, product context

        Returns:
            ResultQualityResult with scores and decision
        """
        mode = plan.mode.upper()
        weights = plan.result_weights or self._config.weights.get(
            mode, self._config.weights["STANDARD"]
        )
        thresholds = plan.result_thresholds or self._config.thresholds.get(
            mode, self._config.thresholds["STANDARD"]
        )

        scores = {}
        feedback = []
        failed_dimensions = []
        retry_suggestions = []

        # 1. Prompt Fidelity
        fidelity_score, fidelity_feedback = await self._score_prompt_fidelity(
            image_data, plan
        )
        scores["prompt_fidelity"] = fidelity_score
        if fidelity_score < thresholds.get("prompt_fidelity", 0.5):
            failed_dimensions.append("prompt_fidelity")
            feedback.extend(fidelity_feedback)
            retry_suggestions.append(RetrySuggestion(
                dimension="prompt_fidelity",
                suggested_changes=[
                    "Make prompt more specific about key elements",
                    "Add emphasis to important subjects",
                ]
            ))

        # 2. Product Readiness
        readiness_score, readiness_feedback = self._score_product_readiness(
            image_data, plan
        )
        scores["product_readiness"] = readiness_score
        if readiness_score < thresholds.get("product_readiness", 0.5):
            failed_dimensions.append("product_readiness")
            feedback.extend(readiness_feedback)
            retry_suggestions.append(RetrySuggestion(
                dimension="product_readiness",
                suggested_changes=[
                    "Ensure subject is centered with safe margins",
                    "Check framing and crop for product fit",
                ]
            ))

        # 3. Technical Quality
        technical_score, technical_feedback = self._score_technical_quality(image_data)
        scores["technical_quality"] = technical_score
        if technical_score < thresholds.get("technical_quality", 0.5):
            failed_dimensions.append("technical_quality")
            feedback.extend(technical_feedback)
            retry_suggestions.append(RetrySuggestion(
                dimension="technical_quality",
                suggested_changes=[
                    "Increase inference steps",
                    "Request higher resolution",
                    "Add 'sharp, detailed' to prompt",
                ]
            ))

        # 4. Background & Composition
        bg_score, bg_feedback = self._score_background_composition(image_data, plan)
        scores["background_composition"] = bg_score
        if bg_score < thresholds.get("background_composition", 0.5):
            failed_dimensions.append("background_composition")
            feedback.extend(bg_feedback)
            retry_suggestions.append(RetrySuggestion(
                dimension="background_composition",
                suggested_changes=[
                    "Specify background type explicitly",
                    "Add composition guidance (centered, rule of thirds)",
                ]
            ))

        # 5. Aesthetic Quality
        aesthetic_score, aesthetic_feedback = self._score_aesthetic(image_data, plan)
        scores["aesthetic"] = aesthetic_score
        if aesthetic_score < thresholds.get("aesthetic", 0.5):
            failed_dimensions.append("aesthetic")
            feedback.extend(aesthetic_feedback)
            retry_suggestions.append(RetrySuggestion(
                dimension="aesthetic",
                suggested_changes=[
                    "Add style keywords (e.g., 'professional, polished')",
                    "Specify color palette more precisely",
                ]
            ))

        # 6. Text Legibility (if applicable)
        if self._has_text_content(image_data, plan):
            text_score, text_feedback = self._score_text_legibility(image_data)
            scores["text_legibility"] = text_score
            if text_score < thresholds.get("text_legibility", 0.5):
                failed_dimensions.append("text_legibility")
                feedback.extend(text_feedback)
                retry_suggestions.append(RetrySuggestion(
                    dimension="text_legibility",
                    suggested_changes=[
                        "Use simpler fonts",
                        "Increase text size",
                        "Improve contrast with background",
                    ]
                ))
        else:
            scores["text_legibility"] = 1.0  # N/A

        # 7. Set Consistency (if multiple images)
        if self._is_multi_image(image_data):
            consistency_score, consistency_feedback = self._score_set_consistency(
                image_data
            )
            scores["set_consistency"] = consistency_score
            if consistency_score < thresholds.get("set_consistency", 0.5):
                failed_dimensions.append("set_consistency")
                feedback.extend(consistency_feedback)
        else:
            scores["set_consistency"] = 1.0  # N/A for single image

        # Calculate weighted overall score
        overall = sum(scores[d] * weights.get(d, 0.1) for d in scores)

        # Determine decision
        if overall < thresholds.get("overall", 0.7) or len(failed_dimensions) > 0:
            decision = "REJECT"
        else:
            decision = "APPROVE"

        return ResultQualityResult(
            overall=overall,
            dimensions=scores,
            mode=mode,
            threshold=thresholds.get("overall", 0.7),
            decision=decision,
            feedback=feedback,
            failed_dimensions=failed_dimensions,
            retry_suggestions=retry_suggestions,
            metadata={
                "job_id": plan.job_id,
                "product_type": plan.product_type,
            },
        )

    async def evaluate_image_set(
        self,
        images: List[Dict[str, Any]],
        plan: EvaluationPlan,
    ) -> List[ResultQualityResult]:
        """
        Evaluate a set of generated images.

        Args:
            images: List of image data dicts
            plan: Shared evaluation plan

        Returns:
            List of ResultQualityResult for each image
        """
        results = []
        for img in images:
            result = await self.evaluate_image(img, plan)
            results.append(result)
        return results

    def should_retry(
        self,
        result: ResultQualityResult,
        attempt_count: int,
    ) -> bool:
        """
        Determine if generation should be retried.

        Args:
            result: Quality assessment result
            attempt_count: Current attempt number

        Returns:
            True if retry should be attempted
        """
        if not self._config.retry_enabled:
            return False

        if result.decision == "APPROVE":
            return False

        if attempt_count >= self._config.max_retry_attempts:
            return False

        # Only retry for applicable modes
        if result.mode not in self._config.retry_applicable_modes:
            return False

        return True

    def get_weights(self, mode: str) -> Dict[str, float]:
        """Get weights for a mode."""
        return self._config.weights.get(
            mode.upper(),
            self._config.weights["STANDARD"]
        )

    def get_thresholds(self, mode: str) -> Dict[str, float]:
        """Get thresholds for a mode."""
        return self._config.thresholds.get(
            mode.upper(),
            self._config.thresholds["STANDARD"]
        )

    # =========================================================================
    # SCORING METHODS
    # =========================================================================

    async def _score_prompt_fidelity(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> Tuple[float, List[str]]:
        """Score how well image matches prompt."""
        feedback = []
        description = image_data.get("description", "")

        if not description:
            return 0.6, ["No image description available for fidelity check"]

        # Use reasoning service for alignment scoring if available
        if self._reasoning_service:
            try:
                alignment = await self._reasoning_service.assess_design_alignment(
                    prompt=plan.prompt,
                    description=description,
                    product_type=plan.product_type,
                )
                score = alignment.prompt_adherence
                if alignment.issues:
                    feedback.extend(alignment.issues)
                return score, feedback
            except Exception as e:
                logger.warning(f"Fidelity scoring failed: {e}")

        # Fallback: simple keyword overlap check
        prompt_words = set(plan.prompt.lower().split())
        desc_words = set(description.lower().split())
        overlap = len(prompt_words & desc_words)
        score = min(1.0, overlap / max(len(prompt_words), 1) + 0.3)

        return score, feedback

    def _score_product_readiness(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> Tuple[float, List[str]]:
        """Score product readiness (commercial asset quality)."""
        feedback = []
        score = 0.7

        # Check resolution
        width = image_data.get("width", 0)
        height = image_data.get("height", 0)

        if width < self._config.min_resolution or height < self._config.min_resolution:
            score -= 0.3
            feedback.append(
                f"Resolution too low ({width}x{height}), "
                f"need {self._config.min_resolution}+"
            )

        # Check coverage/framing
        coverage = image_data.get("coverage_percent", 100)
        if coverage < self._config.min_coverage_percent * 100:
            score -= 0.2
            feedback.append(
                f"Subject coverage too low ({coverage}%), "
                f"need {self._config.min_coverage_percent * 100}%+"
            )

        # Check for cropping issues
        if image_data.get("has_cropping_issues", False):
            score -= 0.3
            feedback.append("Subject may be cropped at edges")

        return max(0.0, score), feedback

    def _score_technical_quality(
        self,
        image_data: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Score technical quality (sharpness, artifacts)."""
        feedback = []
        score = 0.8

        # Check for known defects
        defects = image_data.get("detected_defects", [])
        for defect in defects:
            score -= 0.15
            feedback.append(f"Detected defect: {defect}")

        # Check quality metrics
        sharpness = image_data.get("sharpness_score", 1.0)
        if sharpness < self._config.sharpness_threshold:
            score -= 0.2
            feedback.append("Image appears soft/blurry")

        noise_level = image_data.get("noise_level", 0.0)
        if noise_level > self._config.noise_threshold:
            score -= 0.2
            feedback.append("High noise/artifact level detected")

        return max(0.0, score), feedback

    def _score_background_composition(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> Tuple[float, List[str]]:
        """Score background and composition."""
        feedback = []
        score = 0.7

        # Check background type matches request
        dimensions_requested = plan.dimensions_requested or {}
        expected_bg = dimensions_requested.get("background", "")
        actual_bg = image_data.get("detected_background", "")

        if expected_bg and actual_bg:
            if expected_bg.lower() not in actual_bg.lower():
                score -= 0.3
                feedback.append(
                    f"Background mismatch: expected '{expected_bg}', got '{actual_bg}'"
                )

        # Check composition quality
        composition_score = image_data.get("composition_score", 0.7)
        if composition_score < 0.5:
            score -= 0.2
            feedback.append("Composition could be improved")

        return max(0.0, min(1.0, score)), feedback

    def _score_aesthetic(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> Tuple[float, List[str]]:
        """Score aesthetic quality."""
        feedback = []

        # Use aesthetic score if available from image analysis
        aesthetic_score = image_data.get("aesthetic_score", 0.7)

        # Check color harmony if available
        color_harmony = image_data.get("color_harmony_score", 0.7)
        if color_harmony < 0.5:
            feedback.append("Color harmony could be improved")

        # Average with fallback
        score = (aesthetic_score + color_harmony) / 2

        return max(0.0, min(1.0, score)), feedback

    def _score_text_legibility(
        self,
        image_data: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Score text legibility in image."""
        feedback = []
        score = 0.7

        # Check OCR confidence if available
        ocr_confidence = image_data.get("ocr_confidence", 0.0)
        if ocr_confidence > 0:
            score = ocr_confidence
            if ocr_confidence < 0.7:
                feedback.append("Text may be difficult to read")
        else:
            # No OCR data available
            score = 0.6
            feedback.append("Text legibility could not be verified")

        return max(0.0, min(1.0, score)), feedback

    def _score_set_consistency(
        self,
        image_data: Dict[str, Any],
    ) -> Tuple[float, List[str]]:
        """Score consistency across multiple images."""
        feedback = []

        # Check consistency metrics
        style_consistency = image_data.get("style_consistency", 0.8)
        color_consistency = image_data.get("color_consistency", 0.8)

        score = (style_consistency + color_consistency) / 2

        if score < 0.6:
            feedback.append("Images in set have inconsistent style/colors")

        return score, feedback

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _has_text_content(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> bool:
        """Check if image should contain text."""
        # Check explicit flag
        if image_data.get("has_text", False):
            return True

        # Check prompt for text keywords
        prompt_lower = plan.prompt.lower()
        return any(
            keyword in prompt_lower
            for keyword in self._config.text_detection_keywords
        )

    def _is_multi_image(self, image_data: Dict[str, Any]) -> bool:
        """Check if this is a multi-image evaluation."""
        return image_data.get("is_set", False) or "images" in image_data

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    async def close(self) -> None:
        """Close service and release resources."""
        if self._reasoning_service and self._owns_service:
            await self._reasoning_service.close()
            self._reasoning_service = None

    async def __aenter__(self) -> "ResultEvaluationService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
