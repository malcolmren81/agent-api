"""
Evaluation Agent - Quality assessment with hybrid evaluation.

Uses Google ADK BaseAgent.
Implements hybrid routing: objective checks first, vision LLM for subjective scoring.
"""
from typing import Any, Dict, List, Optional, Tuple
import base64
from io import BytesIO
from PIL import Image
import numpy as np
from src.agents.base_agent import BaseAgent, ParallelAgent
from src.connectors.gemini_reasoning import GeminiReasoningEngine
from src.connectors.chatgpt_reasoning import ChatGPTReasoningEngine
from src.config.policy_loader import policy
from src.utils import get_logger
from src.models.schemas import ReasoningModel

logger = get_logger(__name__)


class EvaluationAgent(BaseAgent):
    """
    Evaluation Agent scores and approves generated images.

    Responsibilities:
    - Evaluate images on multiple criteria
    - Run sub-agents in parallel for efficiency
    - Provide detailed feedback
    - Approve/reject images
    """

    def __init__(self, name: str = "EvaluationAgent") -> None:
        """Initialize Evaluation Agent."""
        super().__init__(name=name)

        # Initialize reasoning engines
        self.gemini_engine = GeminiReasoningEngine()
        self.chatgpt_engine = ChatGPTReasoningEngine()

        # Initialize sub-agents
        self.sub_agents = self._create_sub_agents()

        logger.info("Evaluation Agent initialized with sub-agents")

    def _create_sub_agents(self) -> List[BaseAgent]:
        """Create evaluation sub-agents."""
        return [
            PromptAdherenceAgent(),
            AestheticsAgent(),
            ProductSuitabilityAgent(),
            SafetyAgent(),
        ]

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate generated images.

        Args:
            input_data: Contains:
                - images: Generated images to evaluate
                - context: User context
                - prompts: Original prompts

        Returns:
            Evaluation scores and approval status
        """
        images = input_data.get("images", [])
        context = input_data.get("context", {})
        prompts = input_data.get("prompts", {})
        reasoning_model = context.get("reasoning_model", ReasoningModel.GEMINI)

        logger.info(
            "Evaluation Agent starting",
            num_images=len(images),
            reasoning_model=reasoning_model.value,
        )

        try:
            # Evaluate each image
            evaluations = []
            total_tokens = 0
            used_vision = False
            model_name = None

            for idx, image_data in enumerate(images):
                evaluation = await self._evaluate_image(
                    image_data, prompts, context, reasoning_model, image_index=idx
                )
                evaluations.append(evaluation)

                # Accumulate tokens from each image evaluation
                if evaluation.get("routing_metadata"):
                    eval_tokens = evaluation["routing_metadata"].get("tokens", 0)
                    if eval_tokens:
                        total_tokens += eval_tokens
                    if evaluation["routing_metadata"].get("used_vision"):
                        used_vision = True
                    # Capture model_name from first evaluation that has it
                    if not model_name and evaluation["routing_metadata"].get("model_name"):
                        model_name = evaluation["routing_metadata"]["model_name"]

            # Select best image
            best_image = self._select_best_image(evaluations)

            logger.info(
                "Evaluation complete",
                num_evaluated=len(evaluations),
                best_image_id=best_image.get("image_id") if best_image else None,
                total_tokens=total_tokens,
            )

            return {
                "success": True,
                "evaluations": evaluations,
                "best_image": best_image,
                "approved_images": [e for e in evaluations if e.get("approved")],
                "context": context,
                "routing_metadata": {
                    "tokens": total_tokens,
                    "used_llm": used_vision,
                    "used_vision": used_vision,
                    "num_images_evaluated": len(evaluations),
                    "model_name": model_name,  # Add aggregated model name
                }
            }

        except Exception as e:
            logger.error("Evaluation failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": f"Evaluation failed: {str(e)}",
                "context": context,
            }

    async def _evaluate_image(
        self,
        image_data: Dict[str, Any],
        prompts: Dict[str, Any],
        context: Dict[str, Any],
        reasoning_model: ReasoningModel,
        image_index: int = 0,
    ) -> Dict[str, Any]:
        """
        Evaluate a single image using hybrid approach.

        Hybrid Routing:
        1. Objective checks (resolution, coverage, background)
        2. Vision LLM for subjective scoring (if enabled and needed)

        Args:
            image_data: Image to evaluate
            prompts: Original prompts
            context: Execution context
            reasoning_model: Model for evaluation
            image_index: Index of image in batch

        Returns:
            Evaluation result with routing metadata
        """
        # Check routing mode from policy
        mode = policy.get("evaluation.mode", "hybrid")
        vision_enabled = policy.get_feature_flag("evaluation_vision_enabled")

        # Run objective checks first (fast, free, deterministic)
        objective_scores, objective_passed = await self._objective_checks(image_data)

        # Initialize routing metadata
        routing_metadata = {
            "mode": mode,
            "used_vision": False,
            "objective_passed": objective_passed
        }

        # Determine if we need vision LLM
        use_vision = False
        subjective_scores = {}
        vision_feedback = ""

        if mode == "hybrid":
            # Use vision only if objective checks passed and vision is enabled
            use_vision = objective_passed and vision_enabled
        elif mode == "vision":
            # Force vision LLM
            use_vision = vision_enabled
        # else mode == "objective": skip vision

        # Run vision LLM if needed
        if use_vision:
            try:
                subjective_scores, vision_feedback, vision_tokens, model_name = await self._vision_evaluation(
                    image_data, prompts, reasoning_model
                )
                routing_metadata["used_vision"] = True
                routing_metadata["tokens"] = vision_tokens
                routing_metadata["used_llm"] = True  # Vision models count as LLM usage
                routing_metadata["model_name"] = model_name  # Add model name for tracking
            except Exception as e:
                logger.warning(f"Vision evaluation failed: {e}, using objective scores only")
                routing_metadata["vision_error"] = str(e)

        # Combine objective and subjective scores
        combined_scores = self._combine_scores(objective_scores, subjective_scores)

        # Calculate overall score using policy weights
        overall_score = float(self._calculate_overall_score_weighted(combined_scores))

        # Get acceptance threshold from policy
        acceptance_threshold = policy.get_evaluation_threshold()

        # Determine approval
        approved = bool(overall_score >= acceptance_threshold)

        # Build feedback
        feedback = self._build_feedback(
            objective_scores, subjective_scores, vision_feedback, routing_metadata
        )

        return {
            "image_id": image_data.get("image_id"),
            "image_index": image_index,
            "scores": combined_scores,
            "objective_scores": objective_scores,
            "subjective_scores": subjective_scores,
            "overall_score": overall_score,
            "total_score": overall_score * 100,  # Percentage for backward compatibility
            "score": overall_score * 100,
            "feedback": feedback,
            "approved": approved,
            "accepted": approved,  # For bandit reward calculation
            "model_used": image_data.get("model_used"),
            "routing_metadata": routing_metadata,
            # Include original image data for downstream agents
            "base64_data": image_data.get("base64_data"),
        }

    async def _objective_checks(
        self, image_data: Dict[str, Any]
    ) -> Tuple[Dict[str, float], bool]:
        """
        Run objective (deterministic) quality checks.

        Checks:
        1. Resolution check (min 1024x1024 from policy)
        2. Coverage check (foreground coverage ratio)
        3. Background whiteness check (for white-bg products)

        Args:
            image_data: Image data with base64

        Returns:
            Tuple of (scores_dict, passed_all_checks)
        """
        scores = {
            "resolution": 0.0,
            "coverage": 0.0,
            "background": 0.0,
        }

        try:
            # Decode base64 image
            base64_data = image_data.get("base64_data", "")
            if not base64_data:
                logger.warning("No base64_data in image_data for objective checks")
                return scores, False

            # Remove data URL prefix if present
            if "base64," in base64_data:
                base64_data = base64_data.split("base64,")[1]

            image_bytes = base64.b64decode(base64_data)
            image = Image.open(BytesIO(image_bytes))

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Get numpy array
            img_array = np.array(image)
            height, width = img_array.shape[:2]

            # 1. Resolution check
            min_resolution = policy.get("evaluation.objective_checks.min_resolution", [1024, 1024])
            if width >= min_resolution[0] and height >= min_resolution[1]:
                scores["resolution"] = 1.0
            else:
                # Partial credit for close resolutions
                ratio = float((width * height) / (min_resolution[0] * min_resolution[1]))
                scores["resolution"] = float(min(ratio, 1.0))

            # 2. Coverage check (simplified: detect foreground vs background)
            coverage_score = self._calculate_coverage(img_array)
            scores["coverage"] = float(coverage_score)

            # 3. Background whiteness check
            background_score = self._calculate_background_whiteness(img_array)
            scores["background"] = float(background_score)

            # Determine if passed all checks
            min_coverage = policy.get("evaluation.objective_checks.min_coverage", 0.70)
            min_bg_whiteness = policy.get("evaluation.objective_checks.background_whiteness", 0.92)

            passed = bool(
                scores["resolution"] >= 1.0 and
                scores["coverage"] >= min_coverage and
                scores["background"] >= min_bg_whiteness
            )

            logger.debug(
                f"Objective checks",
                resolution=round(scores["resolution"], 3),
                coverage=round(scores["coverage"], 3),
                background=round(scores["background"], 3),
                passed=passed
            )

            return scores, passed

        except Exception as e:
            logger.error(f"Objective checks failed: {e}", exc_info=True)
            return scores, False

    def _calculate_coverage(self, img_array: np.ndarray) -> float:
        """
        Calculate foreground coverage ratio.

        Simple heuristic: pixels that are NOT very bright (not background).

        Args:
            img_array: RGB image array

        Returns:
            Coverage score (0.0-1.0)
        """
        # Convert to grayscale
        gray = np.mean(img_array, axis=2)

        # Threshold for background (very bright pixels)
        background_threshold = 240  # Pixels brighter than this are background

        # Count non-background pixels
        foreground_pixels = int(np.sum(gray < background_threshold))
        total_pixels = int(gray.size)

        coverage = float(foreground_pixels / total_pixels) if total_pixels > 0 else 0.0

        return float(min(coverage, 1.0))

    def _calculate_background_whiteness(self, img_array: np.ndarray) -> float:
        """
        Calculate background whiteness score.

        For product photos on white background, we expect high percentage of
        very bright pixels.

        Args:
            img_array: RGB image array

        Returns:
            Background whiteness score (0.0-1.0)
        """
        # Convert to grayscale
        gray = np.mean(img_array, axis=2)

        # Threshold for white background
        white_threshold = 250  # Very bright pixels

        # Count white pixels
        white_pixels = int(np.sum(gray >= white_threshold))
        total_pixels = int(gray.size)

        whiteness = float(white_pixels / total_pixels) if total_pixels > 0 else 0.0

        return float(min(whiteness, 1.0))

    async def _vision_evaluation(
        self, image_data: Dict[str, Any], prompts: Dict[str, Any], reasoning_model: ReasoningModel
    ) -> Tuple[Dict[str, float], str]:
        """
        Vision LLM-based subjective evaluation.

        Uses Gemini Vision to analyze the actual image content.

        Args:
            image_data: Image data with base64
            prompts: Original prompts
            reasoning_model: Model for evaluation

        Returns:
            Tuple of (subjective_scores, feedback_text)
        """
        # Get vision model from policy
        vision_model = policy.get_vision_model()

        # Build vision prompt
        original_prompt = prompts.get("primary", "")
        vision_prompt = f"""
Analyze this generated image and score it on the following criteria (0.0-1.0 scale):

ORIGINAL PROMPT: {original_prompt}

Criteria:
1. AESTHETICS: Visual appeal, composition, lighting, color harmony
2. PROMPT_ADHERENCE: How well does the image match the original prompt?
3. PRODUCT_SUITABILITY: Suitable for product mockups (clear, professional, usable)?

Provide scores in this exact format:
AESTHETICS: 0.X
PROMPT_ADHERENCE: 0.X
PRODUCT_SUITABILITY: 0.X

Then provide brief feedback explaining your scores.
"""

        # Use Gemini engine for vision (it supports multimodal)
        try:
            # Build multimodal input with image
            base64_data = image_data.get("base64_data", "")
            if "base64," in base64_data:
                base64_data = base64_data.split("base64,")[1]

            # Call Gemini Vision (gemini-2.0-flash-exp supports vision)
            llm_response = await self.gemini_engine.generate_with_image(
                prompt=vision_prompt,
                image_base64=base64_data,
                temperature=0.2,
            )

            # Parse subjective scores (extract text from LLMResponse)
            subjective_scores = self._parse_vision_scores(llm_response.text)

            logger.debug(
                f"Vision evaluation complete",
                aesthetics=round(subjective_scores.get("aesthetics", 0.0), 3),
                adherence=round(subjective_scores.get("prompt_adherence", 0.0), 3),
                suitability=round(subjective_scores.get("product_suitability", 0.0), 3),
                tokens_used=llm_response.tokens_used,
                model_name=llm_response.model_name,
            )

            return subjective_scores, llm_response.text, llm_response.tokens_used, llm_response.model_name

        except Exception as e:
            logger.error(f"Vision evaluation failed: {e}", exc_info=True)
            # Return default scores with zero tokens and no model
            return {
                "aesthetics": 0.5,
                "prompt_adherence": 0.5,
                "product_suitability": 0.5,
            }, f"Vision evaluation error: {str(e)}", 0, None

    def _parse_vision_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse scores from vision evaluation text."""
        scores = {
            "aesthetics": 0.5,
            "prompt_adherence": 0.5,
            "product_suitability": 0.5,
        }

        lines = evaluation_text.split("\n")
        for line in lines:
            line_upper = line.strip().upper()

            if "AESTHETICS:" in line_upper:
                try:
                    score = float(line_upper.split(":")[-1].strip())
                    scores["aesthetics"] = max(0.0, min(1.0, score))
                except:
                    pass

            elif "PROMPT_ADHERENCE:" in line_upper or "PROMPT ADHERENCE:" in line_upper:
                try:
                    score = float(line_upper.split(":")[-1].strip())
                    scores["prompt_adherence"] = max(0.0, min(1.0, score))
                except:
                    pass

            elif "PRODUCT_SUITABILITY:" in line_upper or "PRODUCT SUITABILITY:" in line_upper:
                try:
                    score = float(line_upper.split(":")[-1].strip())
                    scores["product_suitability"] = max(0.0, min(1.0, score))
                except:
                    pass

        return scores

    def _combine_scores(
        self, objective_scores: Dict[str, float], subjective_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Combine objective and subjective scores.

        Args:
            objective_scores: Scores from objective checks
            subjective_scores: Scores from vision LLM

        Returns:
            Combined scores dictionary
        """
        combined = {}

        # Add all objective scores
        combined.update(objective_scores)

        # Add all subjective scores
        combined.update(subjective_scores)

        # Add safety default (no safety check implemented yet)
        if "safety" not in combined:
            combined["safety"] = 0.95  # Default high

        return combined

    def _calculate_overall_score_weighted(self, scores: Dict[str, float]) -> float:
        """
        Calculate weighted overall score using policy weights.

        Args:
            scores: Combined scores dictionary

        Returns:
            Overall score (0.0-1.0)
        """
        # Get weights from policy
        weights = policy.get_evaluation_weights()

        # Calculate weighted sum
        overall = 0.0
        total_weight = 0.0

        # Coverage weight (from objective)
        if "coverage" in scores and "coverage" in weights:
            overall += scores["coverage"] * weights["coverage"]
            total_weight += weights["coverage"]

        # Aesthetics weight (from subjective)
        if "aesthetics" in scores and "aesthetics" in weights:
            overall += scores["aesthetics"] * weights["aesthetics"]
            total_weight += weights["aesthetics"]

        # Suitability weight (from subjective)
        if "product_suitability" in scores and "suitability" in weights:
            overall += scores["product_suitability"] * weights["suitability"]
            total_weight += weights["suitability"]

        # Normalize if weights don't sum to 1.0
        if total_weight > 0:
            overall /= total_weight

        return round(overall, 2)

    def _build_feedback(
        self,
        objective_scores: Dict[str, float],
        subjective_scores: Dict[str, float],
        vision_feedback: str,
        routing_metadata: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive feedback from all evaluation sources.

        Args:
            objective_scores: Objective check scores
            subjective_scores: Vision LLM scores
            vision_feedback: Vision LLM feedback text
            routing_metadata: Routing information

        Returns:
            Combined feedback string
        """
        feedback_parts = []

        # Objective scores
        feedback_parts.append("=== Objective Checks ===")
        feedback_parts.append(f"Resolution: {objective_scores.get('resolution', 0.0):.2f}")
        feedback_parts.append(f"Coverage: {objective_scores.get('coverage', 0.0):.2f}")
        feedback_parts.append(f"Background: {objective_scores.get('background', 0.0):.2f}")

        # Vision scores if used
        if routing_metadata.get("used_vision"):
            feedback_parts.append("\n=== Vision LLM Evaluation ===")
            if subjective_scores:
                feedback_parts.append(f"Aesthetics: {subjective_scores.get('aesthetics', 0.0):.2f}")
                feedback_parts.append(f"Prompt Adherence: {subjective_scores.get('prompt_adherence', 0.0):.2f}")
                feedback_parts.append(f"Product Suitability: {subjective_scores.get('product_suitability', 0.0):.2f}")
            if vision_feedback:
                feedback_parts.append(f"\n{vision_feedback}")
        else:
            feedback_parts.append("\n(Vision LLM not used)")

        return "\n".join(feedback_parts)

    def _build_evaluation_prompt(
        self, image_data: Dict[str, Any], prompts: Dict[str, Any]
    ) -> str:
        """Build evaluation prompt."""
        original_prompt = prompts.get("primary", "")

        return f"""
Evaluate the following generated image based on these criteria:

ORIGINAL PROMPT:
{original_prompt}

IMAGE METADATA:
- Model: {image_data.get('model_used')}
- Size: {image_data.get('size_bytes')} bytes

Please score the image on a scale of 0.0 to 1.0 for each criterion:

1. PROMPT ADHERENCE (0.0-1.0): Does the image match the prompt description?
2. AESTHETICS (0.0-1.0): Is the image visually appealing with good composition?
3. PRODUCT SUITABILITY (0.0-1.0): Is it suitable for product mockups?
4. SAFETY (0.0-1.0): Is the content appropriate and safe?

Provide scores in the format:
PROMPT_ADHERENCE: 0.X
AESTHETICS: 0.X
PRODUCT_SUITABILITY: 0.X
SAFETY: 0.X

Then provide brief feedback explaining your scores.
"""

    def _get_evaluation_system_prompt(self) -> str:
        """Get system prompt for evaluation."""
        return """You are an expert image quality evaluator.
Score images objectively based on technical quality, aesthetic appeal, and commercial viability.
Be consistent in your scoring and provide constructive feedback."""

    def _parse_scores(self, evaluation_text: str) -> Dict[str, float]:
        """Parse scores from evaluation text."""
        scores = {
            "prompt_adherence": 0.5,
            "aesthetics": 0.5,
            "product_suitability": 0.5,
            "safety": 0.9,  # Default high for safety
        }

        # Parse scores from text
        lines = evaluation_text.split("\n")
        for line in lines:
            line = line.strip().upper()

            if "PROMPT_ADHERENCE:" in line or "PROMPT ADHERENCE:" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                    scores["prompt_adherence"] = max(0.0, min(1.0, score))
                except:
                    pass

            elif "AESTHETICS:" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                    scores["aesthetics"] = max(0.0, min(1.0, score))
                except:
                    pass

            elif "PRODUCT_SUITABILITY:" in line or "PRODUCT SUITABILITY:" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                    scores["product_suitability"] = max(0.0, min(1.0, score))
                except:
                    pass

            elif "SAFETY:" in line:
                try:
                    score = float(line.split(":")[-1].strip())
                    scores["safety"] = max(0.0, min(1.0, score))
                except:
                    pass

        return scores

    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weights = {
            "prompt_adherence": 0.3,
            "aesthetics": 0.25,
            "product_suitability": 0.25,
            "safety": 0.2,
        }

        overall = sum(scores.get(k, 0.0) * v for k, v in weights.items())
        return round(overall, 2)

    def _select_best_image(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the best image from evaluations."""
        if not evaluations:
            return {}

        # Sort by overall score (or total_score for backward compatibility)
        sorted_evals = sorted(
            evaluations,
            key=lambda e: e.get("overall_score", e.get("total_score", 0.0) / 100.0),
            reverse=True,
        )

        return sorted_evals[0]


# ==============================================================================
# Sub-Agent Stubs - Placeholder for Future Enhancements
# ==============================================================================
# NOTE: These sub-agents are currently NOT used in the evaluation pipeline.
# The EvaluationAgent._evaluate_image() method directly calls the reasoning model
# to perform all evaluations in a single call for efficiency.
#
# Future Enhancement Options:
# 1. Parallel Evaluation: Use Google ADK ParallelAgent to run these sub-agents
#    concurrently for faster evaluation of multiple images
# 2. Specialized Models: Each sub-agent could use a specialized model
#    (e.g., safety model for SafetyAgent, aesthetic model for AestheticsAgent)
# 3. Multi-modal Analysis: Sub-agents could perform specific vision tasks
#    (e.g., OCR for text detection, object detection for product suitability)
# ==============================================================================


class PromptAdherenceAgent(BaseAgent):
    """
    Sub-agent for prompt adherence evaluation.

    TODO: Implement with specialized vision model or reasoning chain.
    """

    def __init__(self):
        super().__init__(name="PromptAdherenceAgent")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder implementation - NOT currently used
        # TODO: Implement actual prompt adherence scoring
        return {"score": 0.8}


class AestheticsAgent(BaseAgent):
    """
    Sub-agent for aesthetics evaluation.

    TODO: Implement with aesthetic quality model.
    """

    def __init__(self):
        super().__init__(name="AestheticsAgent")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder implementation - NOT currently used
        # TODO: Implement aesthetic scoring with models like NIMA
        return {"score": 0.8}


class ProductSuitabilityAgent(BaseAgent):
    """
    Sub-agent for product suitability evaluation.

    TODO: Implement with product-specific vision model.
    """

    def __init__(self):
        super().__init__(name="ProductSuitabilityAgent")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder implementation - NOT currently used
        # TODO: Implement product suitability analysis
        return {"score": 0.8}


class SafetyAgent(BaseAgent):
    """
    Sub-agent for safety evaluation.

    TODO: Implement with content moderation model.
    """

    def __init__(self):
        super().__init__(name="SafetyAgent")

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder implementation - NOT currently used
        # TODO: Implement safety/moderation scoring
        return {"score": 0.9}
