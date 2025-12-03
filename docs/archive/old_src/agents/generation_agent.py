"""
Generation Agent - Orchestrates image generation with dual providers.

Uses Google ADK BaseAgent.

REFACTORED: Credit deduction moved to CreditService (called by orchestrator).
This agent now only generates images and calculates costs.
"""
from typing import Any, Dict, List, Optional
from src.agents.base_agent import BaseAgent, AgentContext, AgentResult
from src.api.clients import Palet8APIClient, GenerationType
from src.connectors.flux_image import FluxImageEngine
# DISABLED: Vertex AI SDK causes Cloud Run crashes due to gRPC subprocesses
# from src.connectors.gemini_image import Imagen3Engine
from src.utils import get_logger
from src.models.schemas import ImageModel
import asyncio
import base64
from uuid import uuid4

logger = get_logger(__name__)


class GenerationAgent(BaseAgent):
    """
    Generation Agent handles image creation using Flux or Imagen 3.

    Responsibilities:
    - Execute image generation with selected model
    - Handle multiple image requests
    - Manage retries and fallbacks
    - Calculate generation costs (domain logic)
    - Defensive credit validation

    NOTE: Credit deduction is handled by CreditService (called by orchestrator),
    NOT by this agent. This agent only calculates costs and validates.
    """

    def __init__(
        self,
        name: str = "GenerationAgent",
        api_client: Optional[Palet8APIClient] = None
    ) -> None:
        """
        Initialize Generation Agent.

        Args:
            name: Agent name
            api_client: Palet8 API client for fetching generation costs
        """
        super().__init__(name=name)

        # Initialize image engines
        self.flux_engine = FluxImageEngine()
        # DISABLED: Vertex AI SDK causes Cloud Run crashes due to gRPC subprocesses
        # self.imagen3_engine = Imagen3Engine()
        self.imagen3_engine = None  # Disabled until REST API implementation

        # API client for fetching generation costs (not for deduction)
        self.api_client = api_client or Palet8APIClient()

        logger.info("Generation Agent initialized with Flux and Imagen 3")

    def _map_dimensions_to_generation_type(self, dimensions: str) -> GenerationType:
        """
        Map image dimensions to generation type for credit calculation.

        Args:
            dimensions: Image dimensions (e.g., "1024x1024")

        Returns:
            GenerationType enum value
        """
        dim_map = {
            "512x512": GenerationType.S,
            "1024x1024": GenerationType.M,
            "1536x1536": GenerationType.L,
            "2048x2048": GenerationType.XL,
        }
        return dim_map.get(dimensions, GenerationType.M)  # Default to M

    async def run(self, context: AgentContext) -> AgentResult:
        """
        Generate images using the selected model.

        REFACTORED: This method now only:
        1. Determines generation type and calculates credit cost (domain logic)
        2. Defensive credit validation (sanity check)
        3. Generates images
        4. Returns images and cost information

        Credit deduction is handled by CreditService in the orchestrator AFTER
        this agent completes successfully.

        Args:
            context: AgentContext with prompts, model selection, and user credit info

        Returns:
            AgentResult with generated images and cost calculation
        """
        import time

        try:
            # Extract data from shared_data
            prompts = context.shared_data.get("prompts", {})
            prompt = prompts.get("optimized_prompt", prompts.get("primary", ""))
            dimensions = context.shared_data.get("dimensions", "1024x1024")
            num_images = context.shared_data.get("num_images", 1)

            # Select engine based on model
            if context.image_model == ImageModel.FLUX:
                engine = self.flux_engine
            else:
                engine = self.imagen3_engine

            logger.info(
                "Generation Agent starting",
                task_id=context.task_id,
                model=context.image_model.value,
                dimensions=dimensions,
                num_images=num_images,
                current_balance=context.credit_balance,
            )

            # Phase 5: Determine generation type for credit calculation
            generation_type = self._map_dimensions_to_generation_type(dimensions)

            # Phase 5: Get generation costs from external API
            try:
                costs = await self.api_client.get_generation_costs()
                required_credits = costs[generation_type.value]["credit_cost"]
            except Exception as e:
                logger.error("Failed to fetch generation costs", error=str(e))
                # Fallback to default costs
                cost_map = {"S": 10, "M": 25, "L": 50, "XL": 100}
                required_credits = cost_map.get(generation_type.value, 25)

            # Phase 5: Pre-check credits (double validation after Interactive Agent)
            if context.credit_balance < required_credits:
                logger.warning(
                    "Insufficient credits at generation time",
                    current_balance=context.credit_balance,
                    required=required_credits,
                )
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error="Insufficient credits for generation",
                    metadata={
                        "error_code": "INSUFFICIENT_CREDITS",
                        "current_balance": context.credit_balance,
                        "required": required_credits,
                        "generation_type": generation_type.value
                    }
                )

            # Generate images
            start_time = time.time()

            images_bytes = await engine.generate_image(
                prompt=prompt,
                num_images=num_images,
            )

            generation_time = time.time() - start_time

            if not images_bytes or len(images_bytes) == 0:
                return AgentResult(
                    agent_name=self.name,
                    success=False,
                    error="Image generation returned no images",
                )

            # Package images with metadata
            generated_images = []
            for idx, img_bytes in enumerate(images_bytes):
                image_id = str(uuid4())
                base64_data = base64.b64encode(img_bytes).decode('utf-8')

                generated_images.append({
                    "image_id": image_id,
                    "base64_data": base64_data,
                    "model_used": context.image_model.value,
                    "prompt": prompt,
                    "index": idx,
                    "size_bytes": len(img_bytes),
                })

            logger.info(
                "Image generation successful",
                num_images=len(generated_images),
                generation_time=generation_time,
            )

            # Calculate cost for internal tracking
            cost = await engine.estimate_cost(num_images=len(generated_images))

            # Store cost information in shared_data for orchestrator
            context.shared_data["generation_cost"] = {
                "required_credits": required_credits,
                "generation_type": generation_type.value,
                "num_images": len(generated_images),
                "prompt": prompt[:100],
                "model": context.image_model.value,
            }

            return AgentResult(
                agent_name=self.name,
                success=True,
                data={
                    "images": generated_images,
                    "generation_metadata": {
                        "model": context.image_model.value,
                        "num_images": len(generated_images),
                        "generation_time": generation_time,
                        "dimensions": dimensions,
                        "prompt_used": prompt,
                    },
                    "cost_calculation": {
                        "required_credits": required_credits,
                        "generation_type": generation_type.value,
                    }
                },
                cost=cost,
                metadata={
                    "generation_time": generation_time,
                    "num_images": len(generated_images),
                    "required_credits": required_credits,
                    "generation_type": generation_type.value,
                }
            )

        except Exception as e:
            logger.error(
                "Generation Agent failed",
                error=str(e),
                exc_info=True,
            )
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=f"Generation failed: {str(e)}",
            )

    async def _fallback_generation(
        self, prompts: Dict[str, Any], context: Dict[str, Any], original_error: str
    ) -> Dict[str, Any]:
        """
        Fallback to alternative image model on failure.

        Args:
            prompts: Prompt data
            context: Execution context
            original_error: Error from primary model

        Returns:
            Generation result from fallback model
        """
        logger.warning("Attempting fallback image generation")

        # Try the other engine
        current_model = context.get("image_model", ImageModel.FLUX)

        if current_model == ImageModel.FLUX:
            fallback_engine = self.imagen3_engine
            fallback_model = ImageModel.GEMINI
            prompt = prompts.get("optimized", {}).get(ImageModel.GEMINI.value, prompts.get("primary", ""))
        else:
            fallback_engine = self.flux_engine
            fallback_model = ImageModel.FLUX
            prompt = prompts.get("optimized", {}).get(ImageModel.FLUX.value, prompts.get("primary", ""))

        try:
            import time
            start_time = time.time()

            images_bytes = await fallback_engine.generate_image(
                prompt=prompt,
                num_images=1,  # Generate fewer on fallback
            )

            generation_time = time.time() - start_time

            # Package fallback images
            generated_images = []
            for idx, img_bytes in enumerate(images_bytes):
                image_id = str(uuid4())
                base64_data = base64.b64encode(img_bytes).decode('utf-8')

                generated_images.append({
                    "image_id": image_id,
                    "base64_data": base64_data,
                    "model_used": fallback_model.value,
                    "prompt": prompt,
                    "index": idx,
                    "is_fallback": True,
                    "size_bytes": len(img_bytes),
                })

            cost = await fallback_engine.estimate_cost(num_images=len(images_bytes))

            logger.info(
                "Fallback generation successful",
                fallback_model=fallback_model.value,
                num_images=len(generated_images),
            )

            return {
                "success": True,
                "images": generated_images,
                "generation_metadata": {
                    "model": fallback_model.value,
                    "num_images": len(generated_images),
                    "generation_time": generation_time,
                    "cost": cost,
                    "is_fallback": True,
                    "original_error": original_error,
                },
                "context": context,
            }

        except Exception as fallback_error:
            logger.error(
                "Fallback generation also failed",
                error=str(fallback_error),
                exc_info=True,
            )

            return {
                "success": False,
                "error": f"Both models failed. Primary: {original_error}, Fallback: {str(fallback_error)}",
                "context": context,
            }
