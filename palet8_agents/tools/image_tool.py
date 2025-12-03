"""
Image Tool - Image generation.

This tool provides agents with the ability to generate images
using the Image Generation Service.

Documentation Reference: Section 5.3.2
"""

from typing import Any, Dict, List, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult

from palet8_agents.services.image_generation_service import (
    ImageGenerationService,
    ImageGenerationRequest,
    ImageGenerationError,
)

logger = logging.getLogger(__name__)


class ImageTool(BaseTool):
    """
    Image generation tool.

    Provides access to image generation via the Image Generation Service.
    Supports multiple providers (Runware, Flux) with automatic fallback.

    Methods (from Documentation Section 5.3.2):
    - generate(prompt, negative_prompt, model, parameters) -> ImageResult
    - estimate_cost(prompt, model, parameters) -> CostEstimate
    """

    def __init__(
        self,
        image_service: Optional[ImageGenerationService] = None,
    ):
        """
        Initialize the Image Tool.

        Args:
            image_service: Optional ImageGenerationService for dependency injection
        """
        super().__init__(
            name="image",
            description="Generate images using AI models",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform: generate or estimate_cost",
                    required=True,
                    enum=["generate", "estimate_cost"],
                ),
                ToolParameter(
                    name="prompt",
                    type=ParameterType.STRING,
                    description="Positive prompt describing the desired image",
                    required=True,
                ),
                ToolParameter(
                    name="negative_prompt",
                    type=ParameterType.STRING,
                    description="Negative prompt describing what to avoid",
                    required=False,
                    default="",
                ),
                ToolParameter(
                    name="model",
                    type=ParameterType.STRING,
                    description="Model ID to use for generation (e.g., flux-1-kontext-pro)",
                    required=False,
                ),
                ToolParameter(
                    name="width",
                    type=ParameterType.INTEGER,
                    description="Image width in pixels",
                    required=False,
                    default=1024,
                ),
                ToolParameter(
                    name="height",
                    type=ParameterType.INTEGER,
                    description="Image height in pixels",
                    required=False,
                    default=1024,
                ),
                ToolParameter(
                    name="steps",
                    type=ParameterType.INTEGER,
                    description="Number of inference steps",
                    required=False,
                    default=30,
                ),
                ToolParameter(
                    name="guidance_scale",
                    type=ParameterType.NUMBER,
                    description="Guidance scale for generation",
                    required=False,
                    default=7.5,
                ),
                ToolParameter(
                    name="seed",
                    type=ParameterType.INTEGER,
                    description="Random seed for reproducibility",
                    required=False,
                ),
                ToolParameter(
                    name="num_images",
                    type=ParameterType.INTEGER,
                    description="Number of images to generate",
                    required=False,
                    default=1,
                ),
                ToolParameter(
                    name="reference_image_url",
                    type=ParameterType.STRING,
                    description="URL of reference image for image-to-image generation",
                    required=False,
                ),
                ToolParameter(
                    name="reference_strength",
                    type=ParameterType.NUMBER,
                    description="Strength of reference image influence (0.0 to 1.0)",
                    required=False,
                    default=0.75,
                ),
            ],
        )

        self._image_service = image_service
        self._owns_service = image_service is None

    async def _get_image_service(self) -> ImageGenerationService:
        """Get or create image generation service."""
        if self._image_service is None:
            self._image_service = ImageGenerationService()
        return self._image_service

    async def close(self) -> None:
        """Close resources."""
        if self._image_service and self._owns_service:
            await self._image_service.close()
            self._image_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute image generation action.

        Args:
            action: Action to perform (generate or estimate_cost)
            prompt: Positive prompt
            negative_prompt: Negative prompt
            model: Model ID
            width: Image width
            height: Image height
            steps: Inference steps
            guidance_scale: Guidance scale
            seed: Random seed
            num_images: Number of images
            reference_image_url: Reference image URL
            reference_strength: Reference image strength

        Returns:
            ToolResult with generated image info or cost estimate
        """
        action = kwargs.get("action", "generate")
        prompt = kwargs.get("prompt")
        negative_prompt = kwargs.get("negative_prompt", "")
        model = kwargs.get("model")
        width = kwargs.get("width", 1024)
        height = kwargs.get("height", 1024)
        steps = kwargs.get("steps", 30)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        seed = kwargs.get("seed")
        num_images = kwargs.get("num_images", 1)
        reference_image_url = kwargs.get("reference_image_url")
        reference_strength = kwargs.get("reference_strength", 0.75)

        try:
            if action == "generate":
                return await self._generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    num_images=num_images,
                    reference_image_url=reference_image_url,
                    reference_strength=reference_strength,
                )
            elif action == "estimate_cost":
                return await self._estimate_cost(
                    prompt=prompt,
                    model=model,
                    width=width,
                    height=height,
                    steps=steps,
                    num_images=num_images,
                )
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except ImageGenerationError as e:
            logger.error(f"Image generation error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Image generation failed: {e}",
                error_code="IMAGE_GENERATION_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in ImageTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        model: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        num_images: int = 1,
        reference_image_url: Optional[str] = None,
        reference_strength: float = 0.75,
    ) -> ToolResult:
        """Generate images."""
        if not prompt:
            return ToolResult(
                success=False,
                data=None,
                error="prompt is required for image generation",
                error_code="MISSING_PARAMETER",
            )

        image_service = await self._get_image_service()

        # Build generation request
        request = ImageGenerationRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            model_id=model,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
            reference_image_url=reference_image_url,
            reference_strength=reference_strength,
        )

        # Generate images
        result = await image_service.generate_images(request)

        return ToolResult(
            success=result.success,
            data={
                "images": [img.to_dict() for img in result.images],
                "model_used": result.model_used,
                "provider_used": result.provider_used,
                "generation_time_ms": result.generation_time_ms,
                "total_cost": result.total_cost,
                "metadata": result.metadata,
            },
            error=result.error,
            error_code=result.error_code if not result.success else None,
        )

    async def _estimate_cost(
        self,
        prompt: str,
        model: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: int = 30,
        num_images: int = 1,
    ) -> ToolResult:
        """Estimate generation cost."""
        image_service = await self._get_image_service()

        request = ImageGenerationRequest(
            prompt=prompt,
            model_id=model,
            width=width,
            height=height,
            steps=steps,
            num_images=num_images,
        )

        estimated_cost = await image_service.estimate_cost(request)

        return ToolResult(
            success=True,
            data={
                "estimated_cost": estimated_cost,
                "model": model,
                "width": width,
                "height": height,
                "steps": steps,
                "num_images": num_images,
            },
        )

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        model: Optional[str] = None,
        **parameters,
    ) -> Dict[str, Any]:
        """
        Generate an image (convenience method).

        Args:
            prompt: Positive prompt
            negative_prompt: Negative prompt
            model: Model ID
            **parameters: Additional generation parameters

        Returns:
            Generated image info including URL

        Raises:
            Exception: If generation fails
        """
        result = await self.execute(
            action="generate",
            prompt=prompt,
            negative_prompt=negative_prompt,
            model=model,
            **parameters,
        )

        if result.success:
            return result.data
        else:
            raise Exception(result.error)

    async def __aenter__(self) -> "ImageTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
