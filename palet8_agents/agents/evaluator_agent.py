"""
Evaluator Agent - Quality assessment and evaluation.

This agent handles quality evaluation using a two-phase approach:
1. PROMPT QUALITY (create_plan) - Evaluate prompt BEFORE generation
2. RESULT QUALITY (execute) - Evaluate image AFTER generation

Prompt Quality Dimensions:
- coverage: Required dimensions present for product/mode
- clarity: Self-consistent, no contradictions, no vague terms
- product_constraints: Aligned with print method and product type
- style_alignment: Matches brand style and mode requirements
- control_surface: Negative prompt and references are usable

Result Quality Dimensions:
- prompt_fidelity: Image matches what was requested
- product_readiness: Usable as commercial product asset
- technical_quality: Resolution, sharpness, no artifacts
- background_composition: Background type and composition correct
- aesthetic: Visual appeal and style consistency
- text_legibility: Text/logos readable (if applicable)
- set_consistency: Consistency across multiple images (if applicable)

Note: Safety checks are handled by SafetyAgent, not this evaluator.

Documentation Reference: Section 5.2.3
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import logging
import json

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult
from palet8_agents.core.config import get_config
from palet8_agents.tools.base import BaseTool

from palet8_agents.services.text_llm_service import TextLLMService
from palet8_agents.services.reasoning_service import ReasoningService

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class EvaluationPhase(Enum):
    """Evaluation phases."""
    CREATE_PLAN = "create_plan"  # Prompt quality - BEFORE generation
    EXECUTE = "execute"          # Result quality - AFTER generation


class EvaluationDecision(Enum):
    """Evaluation decisions."""
    PASS = "PASS"
    FIX_REQUIRED = "FIX_REQUIRED"
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    POLICY_FAIL = "POLICY_FAIL"  # Hard block for safety violations


class PromptQualityDimension(Enum):
    """Prompt quality dimensions (Phase 1: create_plan)."""
    COVERAGE = "coverage"                    # Required dimensions present
    CLARITY = "clarity"                      # No contradictions/ambiguity
    PRODUCT_CONSTRAINTS = "product_constraints"  # Print method alignment
    STYLE_ALIGNMENT = "style_alignment"      # Brand/mode style match
    CONTROL_SURFACE = "control_surface"      # Negative prompt quality


class ResultQualityDimension(Enum):
    """Result quality dimensions (Phase 2: execute)."""
    PROMPT_FIDELITY = "prompt_fidelity"      # Image matches prompt
    PRODUCT_READINESS = "product_readiness"  # Commercial asset ready
    TECHNICAL_QUALITY = "technical_quality"  # Resolution, sharpness
    BACKGROUND_COMPOSITION = "background_composition"  # Background/layout
    AESTHETIC = "aesthetic"                  # Visual appeal
    TEXT_LEGIBILITY = "text_legibility"      # Text/logos readable
    SET_CONSISTENCY = "set_consistency"      # Multi-image consistency


# =============================================================================
# DIMENSION WEIGHTS BY MODE
# =============================================================================

# Prompt Quality weights per mode
PROMPT_QUALITY_WEIGHTS = {
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
}

# Prompt Quality thresholds per mode
PROMPT_QUALITY_THRESHOLDS = {
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
}

# Result Quality weights per mode
RESULT_QUALITY_WEIGHTS = {
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
}

# Result Quality thresholds per mode
RESULT_QUALITY_THRESHOLDS = {
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
}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

EVALUATOR_SYSTEM_PROMPT = """You are the Evaluator for Palet8's image pipeline.

YOUR ROLE
Quality gate for generation. Aim to evaluate quickly and efficiently so you don't slow down normal flows. You don't talk to users.

PRE-GENERATION (Prompt/Plan Review)
Assess the plan before images are created:
- Complete enough for this product?
- Clear and self-consistent?
- Respects product/print constraints?
- Stylistically coherent?
- Safe and policy-compliant?

Decisions:
- PASS → proceed to generation
- FIX_REQUIRED → recommend stopping and return feedback to Planner
- POLICY_FAIL → recommend hard stop

POST-GENERATION (Result Review)
Evaluate when image arrives:
- Does image match request?
- Usable for target product? (margins, orientation, composition)
- Technical quality acceptable?
- Artifacts or defects?
- Aesthetically appropriate?
- Safe?

Decisions:
- APPROVE → return to user via Pali
- REJECT → return to Planner with retry suggestions
- POLICY_FAIL → recommend blocking delivery

MULTI-IMAGE
Evaluate all images in a job:
- Score each image independently
- Rank candidates from strongest to weakest
- Note consistency issues across the set

EVALUATION ORDER (prioritize)
1. Obvious technical defects
2. Fit to product and composition
3. Fine-grained aesthetic scoring
(Safety handled by SafetyAgent)

FEEDBACK STYLE
Be specific, concise, and actionable:
- Not: "composition is bad"
- Instead: "subject cut off at top, add 20% headroom"

Catch problems early without slowing the happy path."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PromptQualityResult:
    """Result from prompt quality evaluation (Phase 1)."""
    overall: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    mode: str = "STANDARD"
    threshold: float = 0.70
    decision: str = "PASS"  # PASS, FIX_REQUIRED, POLICY_FAIL
    feedback: List[str] = field(default_factory=list)
    failed_dimensions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "dimensions": self.dimensions,
            "mode": self.mode,
            "threshold": self.threshold,
            "decision": self.decision,
            "feedback": self.feedback,
            "failed_dimensions": self.failed_dimensions,
            "metadata": self.metadata,
        }


@dataclass
class RetrySuggestion:
    """Suggestion for retrying a failed dimension."""
    dimension: str
    suggested_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "suggested_changes": self.suggested_changes,
        }


@dataclass
class ResultQualityResult:
    """Result from result quality evaluation (Phase 2)."""
    overall: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    mode: str = "STANDARD"
    threshold: float = 0.80
    decision: str = "APPROVE"  # APPROVE, REJECT, POLICY_FAIL
    feedback: List[str] = field(default_factory=list)
    failed_dimensions: List[str] = field(default_factory=list)
    retry_suggestions: List[RetrySuggestion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "dimensions": self.dimensions,
            "mode": self.mode,
            "threshold": self.threshold,
            "decision": self.decision,
            "feedback": self.feedback,
            "failed_dimensions": self.failed_dimensions,
            "retry_suggestions": [s.to_dict() for s in self.retry_suggestions],
            "metadata": self.metadata,
        }


@dataclass
class EvaluationPlan:
    """Evaluation plan created in Phase 1."""
    job_id: str
    prompt: str
    negative_prompt: str = ""
    mode: str = "STANDARD"
    product_type: Optional[str] = None
    print_method: Optional[str] = None
    dimensions_requested: Dict[str, Any] = field(default_factory=dict)
    prompt_quality: Optional[PromptQualityResult] = None
    result_weights: Dict[str, float] = field(default_factory=dict)
    result_thresholds: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "mode": self.mode,
            "product_type": self.product_type,
            "print_method": self.print_method,
            "dimensions_requested": self.dimensions_requested,
            "prompt_quality": self.prompt_quality.to_dict() if self.prompt_quality else None,
            "result_weights": self.result_weights,
            "result_thresholds": self.result_thresholds,
            "metadata": self.metadata,
        }


# =============================================================================
# EVALUATOR AGENT
# =============================================================================

class EvaluatorAgent(BaseAgent):
    """
    Quality evaluation agent with two-phase execution.

    Phase 1: create_plan (BEFORE generation)
    - Evaluates prompt quality
    - Creates evaluation plan for Phase 2
    - Returns PASS/FIX_REQUIRED/POLICY_FAIL

    Phase 2: execute (AFTER generation)
    - Evaluates result quality against plan
    - Returns APPROVE/REJECT/POLICY_FAIL
    - Provides retry suggestions for rejected images
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        text_service: Optional[TextLLMService] = None,
        reasoning_service: Optional[ReasoningService] = None,
    ):
        """Initialize the Evaluator Agent."""
        super().__init__(
            name="evaluator",
            description="Quality evaluation agent for prompts and generated images",
            tools=tools,
        )

        self._text_service = text_service
        self._reasoning_service = reasoning_service
        self._owns_services = {
            "text": text_service is None,
            "reasoning": reasoning_service is None,
        }

        self.system_prompt = EVALUATOR_SYSTEM_PROMPT
        self.model_profile = "evaluator"

        # Load configuration
        self.config = get_config()
        self.max_retries = self.config.evaluation.max_retries

    async def _get_text_service(self) -> TextLLMService:
        """Get or create text LLM service."""
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name=self.model_profile)
        return self._text_service

    async def _get_reasoning_service(self) -> ReasoningService:
        """Get or create reasoning service."""
        if self._reasoning_service is None:
            self._reasoning_service = ReasoningService()
        return self._reasoning_service

    async def close(self) -> None:
        """Close resources."""
        if self._text_service and self._owns_services["text"]:
            await self._text_service.close()
        if self._reasoning_service and self._owns_services["reasoning"]:
            await self._reasoning_service.close()

    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
        phase: str = "create_plan",
        image_data: Optional[Dict[str, Any]] = None,
        evaluation_plan: Optional[EvaluationPlan] = None,
    ) -> AgentResult:
        """
        Execute the Evaluator Agent.

        Args:
            context: Shared execution context
            user_input: Optional additional input
            phase: "create_plan" or "execute"
            image_data: Generated image data (for execute phase)
            evaluation_plan: Plan to execute (for execute phase)

        Returns:
            AgentResult with evaluation results
        """
        self._start_execution()

        try:
            if phase == "create_plan":
                return await self._handle_create_plan(context)
            elif phase == "execute":
                return await self._handle_execute(context, image_data, evaluation_plan)
            else:
                return self._create_result(
                    success=False,
                    data=None,
                    error=f"Unknown phase: {phase}",
                    error_code="INVALID_PHASE",
                )

        except Exception as e:
            logger.error(f"Evaluator Agent error: {e}", exc_info=True)
            return self._create_result(
                success=False,
                data=None,
                error=f"Evaluation failed: {e}",
                error_code="EVALUATION_ERROR",
            )

    # =========================================================================
    # PHASE 1: CREATE PLAN (Prompt Quality)
    # =========================================================================

    async def _handle_create_plan(self, context: AgentContext) -> AgentResult:
        """
        Phase 1: Evaluate prompt quality and create evaluation plan.

        This runs BEFORE image generation.
        """
        plan_data = context.plan or {}
        requirements = context.requirements or {}

        # Extract inputs
        prompt = plan_data.get("prompt", "")
        negative_prompt = plan_data.get("negative_prompt", "")
        mode = plan_data.get("mode", "STANDARD")
        product_type = requirements.get("product_type", "general")
        print_method = requirements.get("print_method")
        dimensions = plan_data.get("dimensions", {})

        # Evaluate prompt quality
        prompt_quality = await self._evaluate_prompt_quality(
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            product_type=product_type,
            print_method=print_method,
            dimensions=dimensions,
        )

        # Create evaluation plan for Phase 2
        evaluation_plan = EvaluationPlan(
            job_id=context.job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            product_type=product_type,
            print_method=print_method,
            dimensions_requested=dimensions,
            prompt_quality=prompt_quality,
            result_weights=RESULT_QUALITY_WEIGHTS.get(mode, RESULT_QUALITY_WEIGHTS["STANDARD"]),
            result_thresholds=RESULT_QUALITY_THRESHOLDS.get(mode, RESULT_QUALITY_THRESHOLDS["STANDARD"]),
            metadata={
                "created_for": context.job_id,
                "max_retries": self.max_retries,
            },
        )

        # Determine next action based on prompt quality
        if prompt_quality.decision == "POLICY_FAIL":
            # Hard block - handled by SafetyAgent, this is a fallback
            return self._create_result(
                success=False,
                data={
                    "evaluation_plan": evaluation_plan.to_dict(),
                    "action": "policy_blocked",
                    "reason": "Policy violation detected",
                },
                error="Policy violation",
                error_code="POLICY_VIOLATION",
                next_agent=None,  # Stop processing
            )

        elif prompt_quality.decision == "FIX_REQUIRED":
            # Prompt needs improvement - return to Planner
            return self._create_result(
                success=True,
                data={
                    "evaluation_plan": evaluation_plan.to_dict(),
                    "action": "fix_required",
                    "prompt_quality": prompt_quality.to_dict(),
                    "feedback": prompt_quality.feedback,
                },
                next_agent="planner",  # Fix Plan loop
            )

        else:
            # Prompt quality passed - proceed to generation
            return self._create_result(
                success=True,
                data={
                    "evaluation_plan": evaluation_plan.to_dict(),
                    "action": "proceed_to_generation",
                    "prompt_quality": prompt_quality.to_dict(),
                },
                next_agent=None,  # Proceed to image generation
            )

    async def _evaluate_prompt_quality(
        self,
        prompt: str,
        negative_prompt: str,
        mode: str,
        product_type: str,
        print_method: Optional[str],
        dimensions: Dict[str, Any],
    ) -> PromptQualityResult:
        """Evaluate prompt quality across all dimensions."""

        weights = PROMPT_QUALITY_WEIGHTS.get(mode, PROMPT_QUALITY_WEIGHTS["STANDARD"])
        thresholds = PROMPT_QUALITY_THRESHOLDS.get(mode, PROMPT_QUALITY_THRESHOLDS["STANDARD"])

        # Score each dimension
        scores = {}
        feedback = []
        failed_dimensions = []

        # 1. Coverage - Required dimensions present
        coverage_score, coverage_feedback = self._score_coverage(
            prompt, dimensions, mode, product_type
        )
        scores["coverage"] = coverage_score
        if coverage_score < thresholds.get("coverage", 0.5):
            failed_dimensions.append("coverage")
            feedback.extend(coverage_feedback)

        # 2. Clarity - No contradictions/ambiguity
        clarity_score, clarity_feedback = self._score_clarity(prompt, negative_prompt)
        scores["clarity"] = clarity_score
        if clarity_score < thresholds.get("clarity", 0.5):
            failed_dimensions.append("clarity")
            feedback.extend(clarity_feedback)

        # 3. Product Constraints - Print method alignment
        constraints_score, constraints_feedback = self._score_product_constraints(
            prompt, product_type, print_method
        )
        scores["product_constraints"] = constraints_score
        if constraints_score < thresholds.get("product_constraints", 0.5):
            failed_dimensions.append("product_constraints")
            feedback.extend(constraints_feedback)

        # 4. Style Alignment - Brand/mode style match
        style_score, style_feedback = self._score_style_alignment(
            prompt, dimensions, mode
        )
        scores["style_alignment"] = style_score
        if style_score < thresholds.get("style_alignment", 0.5):
            failed_dimensions.append("style_alignment")
            feedback.extend(style_feedback)

        # 5. Control Surface - Negative prompt quality
        control_score, control_feedback = self._score_control_surface(
            prompt, negative_prompt
        )
        scores["control_surface"] = control_score
        if control_score < thresholds.get("control_surface", 0.5):
            failed_dimensions.append("control_surface")
            feedback.extend(control_feedback)

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
            mode=mode,
            threshold=thresholds.get("overall", 0.7),
            decision=decision,
            feedback=feedback,
            failed_dimensions=failed_dimensions,
            metadata={
                "product_type": product_type,
                "print_method": print_method,
            },
        )

    def _score_coverage(
        self,
        prompt: str,
        dimensions: Dict[str, Any],
        mode: str,
        product_type: str,
    ) -> tuple[float, List[str]]:
        """Score coverage of required dimensions."""
        feedback = []
        prompt_lower = prompt.lower()

        # Required fields by mode
        required_by_mode = {
            "RELAX": ["subject"],
            "STANDARD": ["subject", "aesthetic", "background"],
            "COMPLEX": ["subject", "aesthetic", "background", "composition", "lighting"],
        }

        required = required_by_mode.get(mode, required_by_mode["STANDARD"])
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
    ) -> tuple[float, List[str]]:
        """Score clarity and check for contradictions."""
        feedback = []
        score = 1.0

        prompt_lower = prompt.lower()

        # Check for contradictions
        contradictions = [
            ("dark", "bright white background"),
            ("minimalist", "highly detailed intricate"),
            ("simple", "complex elaborate"),
        ]

        for word1, word2 in contradictions:
            if word1 in prompt_lower and word2 in prompt_lower:
                score -= 0.3
                feedback.append(f"Contradiction detected: '{word1}' vs '{word2}'")

        # Check for vague terms
        vague_terms = ["nice", "cool", "good looking", "awesome", "great"]
        for term in vague_terms:
            if term in prompt_lower:
                score -= 0.1
                feedback.append(f"Vague term '{term}' - replace with specific descriptor")

        # Check if negative prompt contradicts positive
        if negative_prompt:
            neg_lower = negative_prompt.lower()
            # Simple check - are key positive terms in negative?
            positive_keywords = [w for w in prompt_lower.split() if len(w) > 4]
            for kw in positive_keywords[:5]:
                if kw in neg_lower:
                    score -= 0.2
                    feedback.append(f"Negative prompt may contradict positive ('{kw}')")
                    break

        return max(0.0, score), feedback

    def _score_product_constraints(
        self,
        prompt: str,
        product_type: str,
        print_method: Optional[str],
    ) -> tuple[float, List[str]]:
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

        elif product_type == "mug" or product_type == "phone_case":
            if "edges" in prompt_lower or "margin" in prompt_lower:
                score += 0.2
            else:
                feedback.append("Mention edge safety for this product type")

        elif product_type == "poster" or product_type == "canvas":
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
    ) -> tuple[float, List[str]]:
        """Score style alignment with mode and brand."""
        feedback = []
        score = 0.7

        # Mode-specific style expectations
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
    ) -> tuple[float, List[str]]:
        """Score quality of negative prompt and control parameters."""
        feedback = []
        score = 0.7

        if not negative_prompt:
            score -= 0.2
            feedback.append("Add negative prompt to avoid common defects")
        else:
            # Check for useful negative prompts
            useful_negatives = ["blurry", "low quality", "distorted", "extra", "bad anatomy"]
            has_useful = any(term in negative_prompt.lower() for term in useful_negatives)
            if has_useful:
                score += 0.2
            else:
                feedback.append("Negative prompt could be more specific (e.g., 'blurry, distorted')")

        return min(1.0, max(0.0, score)), feedback

    # =========================================================================
    # PHASE 2: EXECUTE (Result Quality)
    # =========================================================================

    async def _handle_execute(
        self,
        context: AgentContext,
        image_data: Optional[Dict[str, Any]],
        evaluation_plan: Optional[EvaluationPlan],
    ) -> AgentResult:
        """
        Phase 2: Evaluate result quality after generation.
        """
        if not image_data:
            return self._create_result(
                success=False,
                data=None,
                error="No image data provided for evaluation",
                error_code="MISSING_IMAGE_DATA",
            )

        # Create plan if not provided
        if not evaluation_plan:
            plan_data = context.plan or {}
            evaluation_plan = EvaluationPlan(
                job_id=context.job_id,
                prompt=plan_data.get("prompt", ""),
                negative_prompt=plan_data.get("negative_prompt", ""),
                mode=plan_data.get("mode", "STANDARD"),
                result_weights=RESULT_QUALITY_WEIGHTS.get(
                    plan_data.get("mode", "STANDARD"),
                    RESULT_QUALITY_WEIGHTS["STANDARD"]
                ),
                result_thresholds=RESULT_QUALITY_THRESHOLDS.get(
                    plan_data.get("mode", "STANDARD"),
                    RESULT_QUALITY_THRESHOLDS["STANDARD"]
                ),
            )

        # Evaluate result quality
        result_quality = await self._evaluate_result_quality(
            image_data=image_data,
            plan=evaluation_plan,
        )

        # Determine next action
        if result_quality.decision == "POLICY_FAIL":
            # Fallback - safety handled by SafetyAgent
            return self._create_result(
                success=False,
                data={
                    "result_quality": result_quality.to_dict(),
                    "action": "policy_blocked",
                },
                error="Policy violation detected",
                error_code="POLICY_VIOLATION",
                next_agent=None,
            )

        elif result_quality.decision == "REJECT":
            return self._create_result(
                success=True,
                data={
                    "result_quality": result_quality.to_dict(),
                    "action": "rejected",
                    "feedback": result_quality.feedback,
                    "retry_suggestions": [s.to_dict() for s in result_quality.retry_suggestions],
                },
                next_agent="planner",  # Fix Plan loop
            )

        else:
            return self._create_result(
                success=True,
                data={
                    "result_quality": result_quality.to_dict(),
                    "action": "approved",
                },
                next_agent="pali",  # Return to user
            )

    async def _evaluate_result_quality(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> ResultQualityResult:
        """Evaluate result quality across all dimensions."""

        weights = plan.result_weights or RESULT_QUALITY_WEIGHTS["STANDARD"]
        thresholds = plan.result_thresholds or RESULT_QUALITY_THRESHOLDS["STANDARD"]

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
        else:
            scores["text_legibility"] = 1.0  # N/A

        # 7. Set Consistency (if multiple images)
        if self._is_multi_image(image_data):
            consistency_score, consistency_feedback = self._score_set_consistency(image_data)
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
            mode=plan.mode,
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

    async def _score_prompt_fidelity(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> tuple[float, List[str]]:
        """Score how well image matches prompt."""
        feedback = []

        # Get image description (from vision model or metadata)
        description = image_data.get("description", "")

        if not description:
            # No description available - use placeholder
            return 0.6, ["No image description available for fidelity check"]

        # Use reasoning service for alignment scoring
        try:
            reasoning_service = await self._get_reasoning_service()
            alignment = await reasoning_service.assess_design_alignment(
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
            return 0.6, ["Fidelity check incomplete"]

    def _score_product_readiness(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> tuple[float, List[str]]:
        """Score product readiness (commercial asset quality)."""
        feedback = []
        score = 0.7

        # Check resolution
        width = image_data.get("width", 0)
        height = image_data.get("height", 0)
        min_resolution = self.config.evaluation.min_resolution

        if width < min_resolution or height < min_resolution:
            score -= 0.3
            feedback.append(f"Resolution too low ({width}x{height}), need {min_resolution}+")

        # Check coverage/framing (if available)
        coverage = image_data.get("coverage_percent", 100)
        min_coverage = self.config.evaluation.min_coverage_percent

        if coverage < min_coverage:
            score -= 0.2
            feedback.append(f"Subject coverage too low ({coverage}%), need {min_coverage}%+")

        # Check for cropping issues
        if image_data.get("has_cropping_issues", False):
            score -= 0.3
            feedback.append("Subject may be cropped at edges")

        return max(0.0, score), feedback

    def _score_technical_quality(
        self,
        image_data: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """Score technical quality (sharpness, artifacts)."""
        feedback = []
        score = 0.8

        # Check for known defects
        defects = image_data.get("detected_defects", [])
        for defect in defects:
            score -= 0.15
            feedback.append(f"Detected defect: {defect}")

        # Check quality metrics if available
        sharpness = image_data.get("sharpness_score", 1.0)
        if sharpness < 0.5:
            score -= 0.2
            feedback.append("Image appears soft/blurry")

        noise_level = image_data.get("noise_level", 0.0)
        if noise_level > 0.3:
            score -= 0.2
            feedback.append("High noise/artifact level detected")

        return max(0.0, score), feedback

    def _score_background_composition(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> tuple[float, List[str]]:
        """Score background and composition."""
        feedback = []
        score = 0.7

        # Check background type matches request
        expected_bg = plan.dimensions_requested.get("background", "")
        actual_bg = image_data.get("detected_background", "")

        if expected_bg and actual_bg:
            if expected_bg.lower() not in actual_bg.lower():
                score -= 0.3
                feedback.append(f"Background mismatch: expected '{expected_bg}', got '{actual_bg}'")

        # Check composition
        composition_score = image_data.get("composition_score", 0.7)
        if composition_score < 0.5:
            score -= 0.2
            feedback.append("Composition could be improved (centering, balance)")

        return max(0.0, score), feedback

    def _score_aesthetic(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> tuple[float, List[str]]:
        """Score aesthetic quality."""
        feedback = []

        # Use aesthetic score if available from image analysis
        aesthetic_score = image_data.get("aesthetic_score", 0.7)

        if aesthetic_score < 0.5:
            feedback.append("Image aesthetic quality below threshold")

        # Check style match
        expected_style = plan.dimensions_requested.get("aesthetic", "")
        detected_style = image_data.get("detected_style", "")

        if expected_style and detected_style:
            if expected_style.lower() not in detected_style.lower():
                aesthetic_score -= 0.1
                feedback.append(f"Style may not match: expected '{expected_style}'")

        return max(0.0, aesthetic_score), feedback

    def _has_text_content(
        self,
        image_data: Dict[str, Any],
        plan: EvaluationPlan,
    ) -> bool:
        """Check if image should have text content."""
        prompt = plan.prompt.lower()
        return any(term in prompt for term in ["text", "logo", "typography", "lettering", "words"])

    def _score_text_legibility(
        self,
        image_data: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """Score text/logo legibility."""
        feedback = []

        # Check OCR results if available
        ocr_confidence = image_data.get("ocr_confidence", 0.7)
        if ocr_confidence < 0.5:
            feedback.append("Text may not be legible")
            return ocr_confidence, feedback

        # Check for text warping
        if image_data.get("text_warped", False):
            feedback.append("Text appears warped/distorted")
            return 0.4, feedback

        return ocr_confidence, feedback

    def _is_multi_image(self, image_data: Dict[str, Any]) -> bool:
        """Check if this is a multi-image generation."""
        return image_data.get("num_images", 1) > 1 or "images" in image_data

    def _score_set_consistency(
        self,
        image_data: Dict[str, Any],
    ) -> tuple[float, List[str]]:
        """Score consistency across multiple images."""
        feedback = []

        consistency_score = image_data.get("set_consistency_score", 0.7)
        if consistency_score < 0.6:
            feedback.append("Images in set lack visual consistency")

        return consistency_score, feedback

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def __aenter__(self) -> "EvaluatorAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
