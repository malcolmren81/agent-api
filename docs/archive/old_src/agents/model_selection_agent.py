"""
Model-Selection Agent - Dynamic model routing with UCB1 bandit algorithm.

Uses Google ADK BaseAgent.
Implements intelligent exploration/exploitation using database-tracked performance stats.
"""
from typing import Any, Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import asdict
import math
from src.agents.base_agent import BaseAgent, AgentContext
from src.connectors.engine_interface import ModelMetadata
from src.connectors.gemini_reasoning import GeminiReasoningEngine
from src.connectors.chatgpt_reasoning import ChatGPTReasoningEngine
from src.connectors.flux_image import FluxImageEngine
# DISABLED: Vertex AI SDK causes Cloud Run crashes due to gRPC subprocesses
# from src.connectors.gemini_image import Imagen3Engine
from src.config.policy_loader import policy
from src.utils import get_logger
from src.models.schemas import ReasoningModel, ImageModel

logger = get_logger(__name__)

# Import Prisma client for database access
try:
    import sys
    from pathlib import Path
    admin_frontend_path = Path(__file__).parent.parent.parent.parent / "admin-frontend"
    sys.path.insert(0, str(admin_frontend_path))
    from prisma import Prisma
    PRISMA_AVAILABLE = True
except ImportError:
    logger.warning("Prisma client not available for Model Selection")
    PRISMA_AVAILABLE = False
    Prisma = None


class ModelSelectionAgent(BaseAgent):
    """
    Model-Selection Agent dynamically chooses optimal models.

    Responsibilities:
    - Evaluate cost vs performance tradeoffs
    - Select reasoning model (Gemini 2.0 Flash vs GPT-4o)
    - Select image model (Flux vs Imagen 3)
    - Consider user preferences and requirements
    """

    def __init__(self, name: str = "ModelSelectionAgent") -> None:
        """Initialize Model-Selection Agent with UCB1 bandit support."""
        super().__init__(name=name)

        # Initialize Prisma client if available
        self.prisma = Prisma() if PRISMA_AVAILABLE else None
        self.db_connected = False

        # Initialize all engines for metadata
        self.gemini_reasoning = GeminiReasoningEngine()
        self.chatgpt_reasoning = ChatGPTReasoningEngine()
        self.flux_image = FluxImageEngine()
        # DISABLED: Vertex AI SDK causes Cloud Run crashes due to gRPC subprocesses
        # self.imagen3 = Imagen3Engine()
        self.imagen3 = None  # Disabled until REST API implementation

        logger.info(
            "Model-Selection Agent initialized",
            db_available=PRISMA_AVAILABLE,
            bandit_enabled=policy.get_feature_flag("model_selector_bandit_enabled")
        )

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select optimal models for the task.

        Args:
            input_data: Contains:
                - context: User context with preferences
                - prompts: Assembled prompts from Prompt Manager

        Returns:
            Selected models with rationale
        """
        context = input_data.get("context", {})
        prompts = input_data.get("prompts", {})

        # Get user preferences
        user_reasoning_pref = context.get("reasoning_model")
        user_image_pref = context.get("image_model")

        logger.info(
            "Model-Selection Agent evaluating options",
            user_reasoning_pref=user_reasoning_pref,
            user_image_pref=user_image_pref,
        )

        try:
            # Select reasoning model
            reasoning_model, reasoning_rationale = await self._select_reasoning_model(
                user_preference=user_reasoning_pref,
                context=context,
            )

            # Select image model
            image_model, image_rationale = await self._select_image_model(
                user_preference=user_image_pref,
                prompts=prompts,
                context=context,
            )

            # Calculate total estimated cost
            total_cost = await self._calculate_total_cost(
                reasoning_model, image_model, context
            )

            logger.info(
                "Models selected",
                reasoning=reasoning_model.value,
                image=image_model.value,
                estimated_cost=total_cost,
            )

            return {
                "success": True,
                "selected_models": {
                    "reasoning": {
                        "model": reasoning_model,
                        "rationale": reasoning_rationale,
                        "metadata": self._get_reasoning_metadata(reasoning_model),
                    },
                    "image": {
                        "model": image_model,
                        "rationale": image_rationale,
                        "metadata": self._get_image_metadata(image_model),
                    },
                },
                "estimated_cost": total_cost,
                "context": {
                    **asdict(context),
                    "reasoning_model": reasoning_model,
                    "image_model": image_model,
                },
                "prompts": prompts,
            }

        except Exception as e:
            logger.error("Model selection failed", error=str(e), exc_info=True)
            return {
                "success": False,
                "error": f"Model selection failed: {str(e)}",
                "context": context,
            }

    async def _select_reasoning_model(
        self, user_preference: Any, context: Dict[str, Any]
    ) -> tuple[ReasoningModel, str]:
        """
        Select optimal reasoning model.

        Args:
            user_preference: User's model preference (ignored - agent always decides)
            context: Execution context

        Returns:
            Tuple of (selected_model, rationale)
        """
        # Agent-controlled selection - user preferences are ignored
        # Default strategy: Use Gemini 2.0 Flash (33x cheaper, very fast)
        # Only use GPT-4o for specific requirements

        prompt = context.shared_data.get("prompt", "")

        # Check if task requires specific GPT-4o features
        requires_gpt4o = any(keyword in prompt.lower() for keyword in [
            "code", "programming", "technical", "api", "function"
        ])

        if requires_gpt4o:
            return ReasoningModel.CHATGPT, "Complex technical reasoning requires GPT-4o"

        # Default to Gemini (cheaper, faster, 1M context)
        return ReasoningModel.GEMINI, "Gemini 2.0 Flash: 33x cheaper, 1M context, thinking budget"

    async def _select_image_model(
        self, user_preference: Any, prompts: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[ImageModel, str]:
        """
        Select optimal image generation model using UCB1 bandit algorithm.

        Args:
            user_preference: User's model preference (ignored - agent always decides)
            prompts: Assembled prompts
            context: Execution context

        Returns:
            Tuple of (selected_model, rationale)
        """
        # Check if bandit is enabled in policy
        bandit_enabled = policy.get_feature_flag("model_selector_bandit_enabled")

        if bandit_enabled and self.prisma:
            # Use UCB1 bandit algorithm
            return await self._ucb1_selection(prompts, context)
        else:
            # Fallback to rule-based selection
            return self._rule_based_image_selection(prompts, context)

    def _rule_based_image_selection(
        self, prompts: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[ImageModel, str]:
        """
        Rule-based image model selection (fallback when bandit disabled).

        Args:
            prompts: Assembled prompts
            context: Execution context

        Returns:
            Tuple of (selected_model, rationale)
        """
        primary_prompt = prompts.get("primary", "")
        prompt_lower = primary_prompt.lower()

        # Flux 1 Kontext is better for:
        # - Fast iteration, Style transfer, Character consistency, Local editing

        # Imagen 3 is better for:
        # - Multi-image blending, Photorealism, Complex compositions

        # Check for multi-image requirements
        if "blend" in prompt_lower or "combine" in prompt_lower or "merge" in prompt_lower:
            # DISABLED: Vertex AI SDK causes crashes, defaulting to Flux
            return ImageModel.FLUX, "Flux 1 Kontext: Multi-image support (Imagen 3 disabled)"

        # Check for photorealism requirements
        if any(word in prompt_lower for word in ["photorealistic", "realistic", "photo"]):
            # DISABLED: Vertex AI SDK causes crashes, defaulting to Flux
            return ImageModel.FLUX, "Flux 1 Kontext: Photorealistic generation (Imagen 3 disabled)"

        # Check for style transfer
        if "style" in prompt_lower or "transfer" in prompt_lower:
            return ImageModel.FLUX, "Flux 1 Kontext: Excellent style transfer, 8x faster"

        # Check for character consistency
        if "character" in prompt_lower or "consistent" in prompt_lower:
            return ImageModel.FLUX, "Flux 1 Kontext: Superior character consistency"

        # Default to Flux for speed and general quality
        return ImageModel.FLUX, "Flux 1 Kontext: 8x faster inference, excellent quality"

    async def _ucb1_selection(
        self, prompts: Dict[str, Any], context: Dict[str, Any]
    ) -> tuple[ImageModel, str]:
        """
        UCB1 bandit algorithm for model selection.

        Balances exploration (trying new models) with exploitation (using proven models).
        UCB score = μ_i + sqrt((2 * ln N) / n_i)
        where:
        - μ_i = mean reward for model i
        - N = total impressions across all models
        - n_i = impressions for model i

        Args:
            prompts: Assembled prompts
            context: Execution context

        Returns:
            Tuple of (selected_model, rationale)
        """
        # Connect to database if not already connected
        if not self.db_connected:
            try:
                await self.prisma.connect()
                self.db_connected = True
                logger.debug("Connected to database for UCB1 selection")
            except Exception as e:
                logger.warning(f"Failed to connect to database for UCB1: {e}, using rule-based fallback")
                return self._rule_based_image_selection(prompts, context)

        try:
            # Determine bucket for this request
            bucket = self._determine_bucket(prompts, context)

            # Get all ModelStats for this bucket
            model_stats = await self.prisma.modelstats.find_many(
                where={"bucket": bucket}
            )

            if not model_stats:
                logger.warning(f"No ModelStats found for bucket={bucket}, using rule-based fallback")
                return self._rule_based_image_selection(prompts, context)

            # Filter to only image models (Flux only - Imagen3 disabled)
            # DISABLED: Vertex AI SDK causes crashes, only allow Flux
            image_model_names = [ImageModel.FLUX.value]
            model_stats = [m for m in model_stats if m.modelName in image_model_names]

            if not model_stats:
                logger.warning(f"No image model stats found for bucket={bucket}")
                return self._rule_based_image_selection(prompts, context)

            # Calculate total impressions across all models
            total_impressions = sum(m.impressions for m in model_stats)

            # Get minimum trials threshold from policy
            min_trials = policy.get("model_selection.exploration.min_trials_per_model", 1)

            # Cold start: If any model has < min_trials, force exploration
            underexplored = [m for m in model_stats if m.impressions < min_trials]
            if underexplored:
                selected_stat = underexplored[0]  # Select first underexplored model
                model_enum = ImageModel(selected_stat.modelName)
                logger.info(
                    f"Cold start exploration: {selected_stat.modelName}",
                    impressions=selected_stat.impressions,
                    min_trials=min_trials
                )
                return model_enum, f"Exploring {selected_stat.modelName} (cold start: {selected_stat.impressions}/{min_trials} trials)"

            # Calculate UCB scores for each model
            ucb_scores = []
            for stat in model_stats:
                ucb = self._calculate_ucb_score(
                    mean_reward=stat.rewardMean,
                    impressions=stat.impressions,
                    total_impressions=total_impressions
                )
                ucb_scores.append({
                    "model": stat.modelName,
                    "ucb": ucb,
                    "mean_reward": stat.rewardMean,
                    "impressions": stat.impressions,
                    "stat_id": stat.id
                })

            # Sort by UCB score (descending)
            ucb_scores.sort(key=lambda x: x["ucb"], reverse=True)

            # Select best model
            best = ucb_scores[0]
            selected_model = ImageModel(best["model"])

            # Store selection in context for later reward update
            context.shared_data["selected_model_stat_id"] = best["stat_id"]
            context.shared_data["bucket"] = bucket

            logger.info(
                f"UCB1 selected {best['model']}",
                ucb=round(best["ucb"], 4),
                mean_reward=round(best["mean_reward"], 4),
                impressions=best["impressions"],
                bucket=bucket
            )

            rationale = (
                f"UCB1: {best['model']} (score={best['ucb']:.3f}, "
                f"μ={best['mean_reward']:.3f}, n={best['impressions']})"
            )

            return selected_model, rationale

        except Exception as e:
            logger.error(f"UCB1 selection failed: {e}, using rule-based fallback", exc_info=True)
            return self._rule_based_image_selection(prompts, context)
        finally:
            # Disconnect from database
            if self.db_connected:
                await self.prisma.disconnect()
                self.db_connected = False

    def _determine_bucket(self, prompts: Dict[str, Any], context: AgentContext) -> str:
        """
        Determine use-case bucket for this request.

        Buckets are defined in policy as combinations of:
        - category: product | creative
        - style: realistic | artistic
        - context: white-bg | lifestyle | abstract | scene

        Args:
            prompts: Assembled prompts
            context: Execution context

        Returns:
            Bucket string (e.g., "product:realistic:white-bg")
        """
        primary_prompt = prompts.get("primary", "")
        prompt_lower = primary_prompt.lower()

        # Determine category
        if any(kw in prompt_lower for kw in ["product", "commercial", "merchandise"]):
            category = "product"
        else:
            category = "creative"

        # Determine style
        style_from_prompts = prompts.get("style", "").lower()
        if "artistic" in style_from_prompts or "abstract" in prompt_lower or "creative" in prompt_lower:
            style = "artistic"
        else:
            style = "realistic"

        # Determine context
        if "white background" in prompt_lower or "white bg" in prompt_lower or "isolated" in prompt_lower:
            bg_context = "white-bg"
        elif "lifestyle" in prompt_lower or "in-use" in prompt_lower or "authentic" in prompt_lower:
            bg_context = "lifestyle"
        elif "abstract" in prompt_lower:
            bg_context = "abstract"
        else:
            bg_context = "scene"

        bucket = f"{category}:{style}:{bg_context}"
        logger.debug(f"Determined bucket: {bucket}")

        return bucket

    def _calculate_ucb_score(
        self, mean_reward: float, impressions: int, total_impressions: int
    ) -> float:
        """
        Calculate UCB1 score for a model.

        UCB score = μ_i + sqrt((2 * ln N) / n_i)

        Args:
            mean_reward: Mean reward for this model (0.0-1.0)
            impressions: Number of times this model has been selected
            total_impressions: Total impressions across all models

        Returns:
            UCB score
        """
        if impressions == 0:
            return float('inf')  # Force exploration of untried models

        if total_impressions <= 1:
            return mean_reward  # Not enough data for exploration bonus

        # UCB1 formula
        exploration_bonus = math.sqrt((2 * math.log(total_impressions)) / impressions)
        ucb = mean_reward + exploration_bonus

        return ucb

    async def update_bandit_reward(
        self, context: Dict[str, Any], evaluation_result: Dict[str, Any]
    ) -> None:
        """
        Update bandit statistics after image generation and evaluation.

        Should be called by the orchestrator after evaluation is complete.

        Args:
            context: Execution context (contains stat_id and bucket)
            evaluation_result: Evaluation results with scores
        """
        if not PRISMA_AVAILABLE or not self.prisma:
            logger.debug("Prisma not available, skipping bandit reward update")
            return

        stat_id = context.shared_data.get("selected_model_stat_id")
        if not stat_id:
            logger.warning("No stat_id in context, skipping bandit reward update")
            return

        try:
            # Connect to database
            if not self.db_connected:
                await self.prisma.connect()
                self.db_connected = True

            # Extract reward from evaluation result
            reward = self._calculate_reward(evaluation_result)

            # Get current stats
            current_stat = await self.prisma.modelstats.find_unique(
                where={"id": stat_id}
            )

            if not current_stat:
                logger.warning(f"ModelStat {stat_id} not found")
                return

            # Get decay alpha from policy
            decay_alpha = policy.get("model_selection.exploration.decay_alpha", 0.031)

            # Update using EMA (Exponential Moving Average)
            # new_mean = α * reward + (1 - α) * old_mean
            new_mean = decay_alpha * reward + (1 - decay_alpha) * current_stat.rewardMean

            # Update database
            await self.prisma.modelstats.update(
                where={"id": stat_id},
                data={
                    "impressions": {"increment": 1},
                    "rewardMean": new_mean,
                    "lastUpdated": datetime.now()
                }
            )

            logger.info(
                f"Updated bandit stats",
                model=current_stat.modelName,
                bucket=current_stat.bucket,
                old_mean=round(current_stat.rewardMean, 4),
                new_mean=round(new_mean, 4),
                reward=round(reward, 4),
                impressions=current_stat.impressions + 1
            )

        except Exception as e:
            logger.error(f"Failed to update bandit reward: {e}", exc_info=True)
        finally:
            if self.db_connected:
                await self.prisma.disconnect()
                self.db_connected = False

    def _calculate_reward(self, evaluation_result: Dict[str, Any]) -> float:
        """
        Calculate reward from evaluation result.

        Reward is a weighted combination of:
        - Overall evaluation score (0.60 weight)
        - Accept flag (0.30 weight)
        - Cost efficiency (0.05 weight)
        - Latency efficiency (0.05 weight)

        Args:
            evaluation_result: Evaluation results

        Returns:
            Reward score (0.0-1.0)
        """
        # Get weights from policy
        weights = policy.get("model_selection.reward_weights", {
            "overall_score": 0.60,
            "accept_flag": 0.30,
            "cost_efficiency": 0.05,
            "latency_efficiency": 0.05
        })

        # Extract components
        overall_score = evaluation_result.get("overall_score", 0.5)
        accepted = 1.0 if evaluation_result.get("accepted", False) else 0.0

        # Cost efficiency (inverse of cost, normalized)
        cost = evaluation_result.get("cost", 0.01)
        cost_efficiency = max(0.0, 1.0 - (cost / 0.10))  # Normalize to 0-1 (assuming max $0.10)

        # Latency efficiency (inverse of latency, normalized)
        latency = evaluation_result.get("latency", 5.0)
        latency_efficiency = max(0.0, 1.0 - (latency / 10.0))  # Normalize to 0-1 (assuming max 10s)

        # Weighted combination
        reward = (
            weights["overall_score"] * overall_score +
            weights["accept_flag"] * accepted +
            weights["cost_efficiency"] * cost_efficiency +
            weights["latency_efficiency"] * latency_efficiency
        )

        return min(max(reward, 0.0), 1.0)  # Clamp to [0, 1]

    async def _calculate_total_cost(
        self, reasoning_model: ReasoningModel, image_model: ImageModel, context: Dict[str, Any]
    ) -> float:
        """
        Calculate estimated total cost.

        Args:
            reasoning_model: Selected reasoning model
            image_model: Selected image model
            context: Execution context

        Returns:
            Estimated cost in USD
        """
        from config import settings

        total = 0.0

        # Get actual number of images from context (not hardcoded)
        num_images = context.shared_data.get("num_images", 2)

        # Reasoning cost - more accurate estimation
        # Planning: 1 call, Evaluation: num_images calls (per-image), Refinement: 1 call
        num_reasoning_calls = 2 + num_images  # Dynamic based on images to evaluate

        if reasoning_model == ReasoningModel.GEMINI:
            total += settings.gemini_reasoning_cost * num_reasoning_calls
        else:
            total += settings.chatgpt_reasoning_cost * num_reasoning_calls

        # Image generation cost - use actual num_images from context
        if image_model == ImageModel.FLUX:
            total += settings.flux_pro_cost * num_images  # Dynamic (using Pro tier as default)
        else:
            total += settings.gemini_image_cost * num_images  # Dynamic

        # Evaluation cost (scales with images)
        total += 0.001 * num_images

        # Product generation cost (typically constant)
        total += 0.01

        return round(total, 4)

    def _get_reasoning_metadata(self, model: ReasoningModel) -> Dict[str, Any]:
        """Get metadata for reasoning model."""
        if model == ReasoningModel.GEMINI:
            meta = self.gemini_reasoning.get_metadata()
        else:
            meta = self.chatgpt_reasoning.get_metadata()

        return {
            "model_name": meta.model_name,
            "provider": meta.provider,
            "cost_per_request": meta.cost_per_request,
            "average_latency": meta.average_latency,
            "max_tokens": meta.max_tokens,
            "supports_streaming": meta.supports_streaming,
        }

    def _get_image_metadata(self, model: ImageModel) -> Dict[str, Any]:
        """Get metadata for image model."""
        if model == ImageModel.FLUX:
            meta = self.flux_image.get_metadata()
        else:
            # DISABLED: Vertex AI causes crashes, fallback to Flux
            logger.warning(f"ImageModel.GEMINI requested but disabled, using Flux metadata instead")
            meta = self.flux_image.get_metadata()

        return {
            "model_name": meta.model_name,
            "provider": meta.provider,
            "cost_per_request": meta.cost_per_request,
            "average_latency": meta.average_latency,
            "supports_images": meta.supports_images,
        }
