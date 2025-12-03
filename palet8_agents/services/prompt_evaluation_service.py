"""
Prompt Evaluation Service.

Handles assessment of prompt quality across multiple dimensions
and proposes revisions when needed.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from palet8_agents.models import PromptQualityResult

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class PromptEvaluationConfig:
    """Configuration for prompt evaluation service."""

    # Weights by mode
    weights: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "RELAX": {
            "coverage": 0.30,
            "clarity": 0.25,
            "product_constraints": 0.20,
            "style_alignment": 0.10,
            "control_surface": 0.15,
        },
        "STANDARD": {
            "coverage": 0.25,
            "clarity": 0.25,
            "product_constraints": 0.20,
            "style_alignment": 0.15,
            "control_surface": 0.15,
        },
        "COMPLEX": {
            "coverage": 0.22,
            "clarity": 0.17,
            "product_constraints": 0.22,
            "style_alignment": 0.22,
            "control_surface": 0.17,
        },
    })

    # Thresholds by mode
    thresholds: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "RELAX": {
            "overall": 0.50,
            "coverage": 0.50,
            "clarity": 0.40,
            "product_constraints": 0.40,
            "style_alignment": 0.30,
            "control_surface": 0.30,
        },
        "STANDARD": {
            "overall": 0.70,
            "coverage": 0.70,
            "clarity": 0.60,
            "product_constraints": 0.60,
            "style_alignment": 0.50,
            "control_surface": 0.50,
        },
        "COMPLEX": {
            "overall": 0.85,
            "coverage": 0.85,
            "clarity": 0.75,
            "product_constraints": 0.80,
            "style_alignment": 0.70,
            "control_surface": 0.70,
        },
    })

    # Required fields by mode
    required_fields_by_mode: Dict[str, List[str]] = field(default_factory=lambda: {
        "RELAX": ["subject"],
        "STANDARD": ["subject", "aesthetic", "background"],
        "COMPLEX": ["subject", "aesthetic", "background", "composition", "lighting"],
    })

    # Clarity checks
    contradiction_pairs: List[List[str]] = field(default_factory=lambda: [
        ["dark", "bright white background"],
        ["minimalist", "highly detailed intricate"],
        ["simple", "complex elaborate"],
        ["muted colors", "vibrant saturated"],
        ["vintage", "futuristic modern"],
    ])

    vague_terms: List[str] = field(default_factory=lambda: [
        "nice", "cool", "good looking", "awesome", "great",
        "pretty", "beautiful", "amazing",
    ])

    # Penalties
    contradiction_penalty: float = -0.3
    vague_term_penalty: float = -0.1
    negative_contradiction_penalty: float = -0.2

    # Useful negative prompt terms
    useful_negatives: List[str] = field(default_factory=lambda: [
        "blurry", "low quality", "distorted", "extra", "bad anatomy",
        "watermark", "text", "ugly", "deformed", "noisy",
    ])


# =============================================================================
# EXCEPTIONS
# =============================================================================


class PromptEvaluationError(Exception):
    """Base exception for prompt evaluation errors."""
    pass


# =============================================================================
# SERVICE
# =============================================================================


class PromptEvaluationService:
    """
    Service for assessing prompt quality and proposing revisions.

    This service handles:
    - Scoring prompts across multiple quality dimensions
    - Checking for contradictions and vague terms
    - Validating product and print method constraints
    - Proposing prompt revisions when quality is insufficient
    """

    def __init__(
        self,
        reasoning_service: Optional[Any] = None,
        config: Optional[PromptEvaluationConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize prompt evaluation service.

        Args:
            reasoning_service: Optional ReasoningService for revision proposals
            config: Optional custom configuration
            config_path: Optional path to config file
        """
        self._reasoning_service = reasoning_service
        self._owns_service = reasoning_service is None
        self._config = config or PromptEvaluationConfig()

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
            if data.get("required_fields_by_mode"):
                self._config.required_fields_by_mode = data["required_fields_by_mode"]

            # Clarity checks
            clarity = data.get("clarity_checks", {})
            if clarity.get("contradiction_pairs"):
                self._config.contradiction_pairs = clarity["contradiction_pairs"]
            if clarity.get("vague_terms"):
                self._config.vague_terms = clarity["vague_terms"]

            # Penalties
            penalties = clarity.get("penalties", {})
            if "contradiction" in penalties:
                self._config.contradiction_penalty = penalties["contradiction"]
            if "vague_term" in penalties:
                self._config.vague_term_penalty = penalties["vague_term"]
            if "negative_contradiction" in penalties:
                self._config.negative_contradiction_penalty = penalties["negative_contradiction"]

        except Exception as e:
            logger.warning(f"Failed to load prompt evaluation config: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def assess_quality(
        self,
        prompt: str,
        negative_prompt: str = "",
        mode: str = "STANDARD",
        product_type: Optional[str] = None,
        print_method: Optional[str] = None,
        dimensions: Optional[Dict[str, Any]] = None,
    ) -> PromptQualityResult:
        """
        Assess prompt quality across multiple dimensions.

        Args:
            prompt: Positive prompt text
            negative_prompt: Negative prompt text
            mode: RELAX, STANDARD, or COMPLEX
            product_type: Target product for constraint checking
            print_method: Print method (screen_print, DTG, etc.)
            dimensions: Requested dimensions for coverage check

        Returns:
            PromptQualityResult with scores, decision, and feedback
        """
        mode_upper = mode.upper()
        weights = self._config.weights.get(mode_upper, self._config.weights["STANDARD"])
        thresholds = self._config.thresholds.get(mode_upper, self._config.thresholds["STANDARD"])
        dimensions = dimensions or {}

        # Score each dimension
        scores = {}
        feedback = []
        failed_dimensions = []

        # 1. Coverage - Required dimensions present
        coverage_score, coverage_feedback = self._score_coverage(
            prompt, dimensions, mode_upper, product_type
        )
        scores["coverage"] = coverage_score
        feedback.extend(coverage_feedback)  # Always include feedback
        if coverage_score < thresholds.get("coverage", 0.5):
            failed_dimensions.append("coverage")

        # 2. Clarity - No contradictions/ambiguity
        clarity_score, clarity_feedback = self._score_clarity(prompt, negative_prompt)
        scores["clarity"] = clarity_score
        feedback.extend(clarity_feedback)  # Always include feedback
        if clarity_score < thresholds.get("clarity", 0.5):
            failed_dimensions.append("clarity")

        # 3. Product Constraints - Print method alignment
        constraints_score, constraints_feedback = self._score_product_constraints(
            prompt, product_type, print_method
        )
        scores["product_constraints"] = constraints_score
        feedback.extend(constraints_feedback)  # Always include feedback
        if constraints_score < thresholds.get("product_constraints", 0.5):
            failed_dimensions.append("product_constraints")

        # 4. Style Alignment - Mode style match
        style_score, style_feedback = self._score_style_alignment(
            prompt, dimensions, mode_upper
        )
        scores["style_alignment"] = style_score
        feedback.extend(style_feedback)  # Always include feedback
        if style_score < thresholds.get("style_alignment", 0.5):
            failed_dimensions.append("style_alignment")

        # 5. Control Surface - Negative prompt quality
        control_score, control_feedback = self._score_control_surface(
            prompt, negative_prompt
        )
        scores["control_surface"] = control_score
        feedback.extend(control_feedback)  # Always include feedback
        if control_score < thresholds.get("control_surface", 0.5):
            failed_dimensions.append("control_surface")

        # Calculate weighted overall score
        overall = sum(scores[d] * weights[d] for d in scores)

        # Determine decision
        if overall < thresholds.get("overall", 0.7) or len(failed_dimensions) > 0:
            decision = "FIX_REQUIRED"
        else:
            decision = "PASS"

        return PromptQualityResult(
            overall=overall,
            dimensions=scores,
            mode=mode_upper,
            threshold=thresholds.get("overall", 0.7),
            decision=decision,
            feedback=feedback,
            failed_dimensions=failed_dimensions,
            metadata={
                "product_type": product_type,
                "print_method": print_method,
            },
        )

    async def propose_revision(
        self,
        prompt: str,
        quality_result: PromptQualityResult,
    ) -> str:
        """
        Use reasoning service to propose improved prompt.

        Args:
            prompt: Original prompt
            quality_result: Quality assessment result

        Returns:
            Revised prompt text
        """
        if self._reasoning_service is None:
            # Return original with suggestions appended
            suggestions = ", ".join(quality_result.feedback[:3])
            return f"{prompt}. Consider: {suggestions}"

        try:
            revised = await self._reasoning_service.propose_prompt_revision(
                prompt=prompt,
                feedback=quality_result.feedback,
                failed_dimensions=quality_result.failed_dimensions,
            )
            return revised
        except Exception as e:
            logger.warning(f"Prompt revision failed: {e}")
            return prompt

    def is_acceptable(self, quality_result: PromptQualityResult) -> bool:
        """
        Check if quality result is acceptable.

        Args:
            quality_result: Quality assessment result

        Returns:
            True if prompt quality is acceptable
        """
        return quality_result.decision == "PASS"

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

    def _score_coverage(
        self,
        prompt: str,
        dimensions: Dict[str, Any],
        mode: str,
        product_type: Optional[str],
    ) -> Tuple[float, List[str]]:
        """Score coverage of required dimensions."""
        feedback = []
        prompt_lower = prompt.lower()

        required = self._config.required_fields_by_mode.get(
            mode,
            self._config.required_fields_by_mode["STANDARD"]
        )

        present = 0
        for field in required:
            if dimensions.get(field) or field in prompt_lower:
                present += 1
            else:
                feedback.append(f"Missing required dimension: {field}")

        # Check for product-specific requirements
        if product_type == "apparel" and "print" not in prompt_lower:
            feedback.append("For apparel, specify print area or placement")

        score = present / len(required) if required else 1.0
        return score, feedback

    def _score_clarity(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> Tuple[float, List[str]]:
        """Score clarity and check for contradictions."""
        feedback = []
        score = 1.0
        prompt_lower = prompt.lower()

        # Check for contradictions
        for pair in self._config.contradiction_pairs:
            word1, word2 = pair[0], pair[1]
            if word1 in prompt_lower and word2 in prompt_lower:
                score += self._config.contradiction_penalty
                feedback.append(f"Contradiction detected: '{word1}' vs '{word2}'")

        # Check for vague terms
        for term in self._config.vague_terms:
            if term in prompt_lower:
                score += self._config.vague_term_penalty
                feedback.append(f"Vague term '{term}' - replace with specific descriptor")

        # Check if negative prompt contradicts positive
        if negative_prompt:
            neg_lower = negative_prompt.lower()
            # Simple check - are key positive terms in negative?
            positive_keywords = [w for w in prompt_lower.split() if len(w) > 4]
            for kw in positive_keywords[:5]:
                if kw in neg_lower:
                    score += self._config.negative_contradiction_penalty
                    feedback.append(f"Negative prompt may contradict positive ('{kw}')")
                    break

        return max(0.0, min(1.0, score)), feedback

    def _score_product_constraints(
        self,
        prompt: str,
        product_type: Optional[str],
        print_method: Optional[str],
    ) -> Tuple[float, List[str]]:
        """Score alignment with product and print constraints."""
        feedback = []
        score = 0.7  # Base score
        prompt_lower = prompt.lower()

        # Product-specific checks
        if product_type == "apparel":
            if any(term in prompt_lower for term in ["centered", "placement", "print area"]):
                score += 0.2
            else:
                feedback.append("Add placement info for apparel (e.g., 'centered design')")

        elif product_type in ["mug", "phone_case"]:
            if "edges" in prompt_lower or "margin" in prompt_lower:
                score += 0.2
            else:
                feedback.append("Mention edge safety for this product type")

        elif product_type in ["poster", "canvas"]:
            if "aspect" in prompt_lower or "bleed" in prompt_lower:
                score += 0.2

        # Print method checks
        if print_method == "screen_print":
            if "gradient" in prompt_lower and "halftone" not in prompt_lower:
                score -= 0.2
                feedback.append("Screen print: use halftone instead of smooth gradients")
            if any(term in prompt_lower for term in ["limited colors", "spot color"]):
                score += 0.1

        elif print_method == "embroidery":
            if "fine detail" in prompt_lower or "photorealistic" in prompt_lower:
                score -= 0.3
                feedback.append("Embroidery: avoid fine details, use bold shapes")

        return min(1.0, max(0.0, score)), feedback

    def _score_style_alignment(
        self,
        prompt: str,
        dimensions: Dict[str, Any],
        mode: str,
    ) -> Tuple[float, List[str]]:
        """Score style alignment with mode."""
        feedback = []
        score = 0.7
        aesthetic = dimensions.get("aesthetic", "")

        if mode == "RELAX":
            # RELAX should be simple
            if len(prompt.split()) > 50:
                score -= 0.2
                feedback.append("RELAX mode: prompt too complex, simplify")
        elif mode == "COMPLEX":
            # COMPLEX should be detailed
            if len(prompt.split()) < 30:
                score -= 0.2
                feedback.append("COMPLEX mode: add more detail to prompt")

        # Check if aesthetic is specified
        if aesthetic:
            score += 0.2
        else:
            if mode != "RELAX":
                feedback.append("Consider specifying an aesthetic style")

        return min(1.0, max(0.0, score)), feedback

    def _score_control_surface(
        self,
        prompt: str,
        negative_prompt: str,
    ) -> Tuple[float, List[str]]:
        """Score quality of negative prompt and control parameters."""
        feedback = []
        score = 0.7

        if not negative_prompt:
            score -= 0.2
            feedback.append("Add negative prompt to avoid common defects")
        else:
            # Check for useful negative prompts
            has_useful = any(
                term in negative_prompt.lower()
                for term in self._config.useful_negatives
            )
            if has_useful:
                score += 0.2
            else:
                feedback.append(
                    "Negative prompt could be more specific (e.g., 'blurry, distorted')"
                )

        return min(1.0, max(0.0, score)), feedback

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    async def close(self) -> None:
        """Close service and release resources."""
        if self._reasoning_service and self._owns_service:
            await self._reasoning_service.close()
            self._reasoning_service = None

    async def __aenter__(self) -> "PromptEvaluationService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
