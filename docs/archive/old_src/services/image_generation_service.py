"""
Image Generation Service

High-level service for AI image generation using various providers
(Runware, Flux, etc.) with automatic retry and failover support.

Documentation Reference: Section 4.4
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import os
import httpx
import asyncio
import base64
import yaml

from palet8_agents.core.config import get_config

logger = logging.getLogger(__name__)


class ImageGenerationError(Exception):
    """Base exception for ImageGenerationService errors."""
    pass


class ProviderError(ImageGenerationError):
    """Raised when image provider fails."""
    pass


class GenerationTimeoutError(ImageGenerationError):
    """Raised when generation times out."""
    pass


class AspectRatio(Enum):
    """Supported aspect ratios."""
    SQUARE = "1:1"
    PORTRAIT_9_16 = "9:16"
    LANDSCAPE_16_9 = "16:9"
    PORTRAIT_3_4 = "3:4"
    LANDSCAPE_4_3 = "4:3"
    PORTRAIT_2_3 = "2:3"
    LANDSCAPE_3_2 = "3:2"


@dataclass
class ImageGenerationRequest:
    """Request for image generation."""
    prompt: str
    negative_prompt: Optional[str] = None
    model: Optional[str] = None
    aspect_ratio: str = "1:1"
    width: Optional[int] = None
    height: Optional[int] = None
    num_images: int = 1
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    reference_image_url: Optional[str] = None
    style: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "model": self.model,
            "aspect_ratio": self.aspect_ratio,
            "width": self.width,
            "height": self.height,
            "num_images": self.num_images,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "reference_image_url": self.reference_image_url,
            "style": self.style,
            "metadata": self.metadata,
        }


@dataclass
class GeneratedImage:
    """A single generated image."""
    url: Optional[str] = None
    base64_data: Optional[str] = None
    seed: Optional[int] = None
    revised_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "has_base64": self.base64_data is not None,
            "seed": self.seed,
            "revised_prompt": self.revised_prompt,
            "metadata": self.metadata,
        }


@dataclass
class ImageGenerationResult:
    """Result from image generation."""
    images: List[GeneratedImage]
    model_used: str
    provider: str
    cost_usd: float = 0.0
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "images": [img.to_dict() for img in self.images],
            "model_used": self.model_used,
            "provider": self.provider,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @property
    def first_image_url(self) -> Optional[str]:
        """Get URL of first image."""
        return self.images[0].url if self.images else None


class ImageGenerationService:
    """
    Service for AI image generation.

    Features:
    - Multiple provider support (Runware, Flux)
    - Automatic retry with exponential backoff
    - Failover between providers
    - Cost tracking
    - Aspect ratio handling
    """

    # Dimension mappings for aspect ratios
    ASPECT_RATIO_DIMENSIONS = {
        "1:1": (1024, 1024),
        "9:16": (576, 1024),
        "16:9": (1024, 576),
        "3:4": (768, 1024),
        "4:3": (1024, 768),
        "2:3": (683, 1024),
        "3:2": (1024, 683),
    }

    # Path to image models config
    IMAGE_MODELS_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "image_models_config.yaml"

    # Default cost if model not found (fallback)
    DEFAULT_COST_PER_IMAGE = 0.03

    def __init__(
        self,
        runware_api_key: Optional[str] = None,
        flux_api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize ImageGenerationService.

        Args:
            runware_api_key: Runware API key
            flux_api_key: Flux API key
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.runware_api_key = runware_api_key or os.environ.get("RUNWARE_API_KEY", "")
        self.flux_api_key = flux_api_key or os.environ.get("FLUX_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._config = get_config()
        self._model_costs: Dict[str, float] = {}
        self._load_model_costs()

    def _load_model_costs(self) -> None:
        """
        Load model costs from image_models_config.yaml.

        Costs per image from model_registry entries.
        """
        try:
            if self.IMAGE_MODELS_CONFIG_PATH.exists():
                with open(self.IMAGE_MODELS_CONFIG_PATH, "r") as f:
                    config = yaml.safe_load(f) or {}

                model_registry = config.get("model_registry", {})
                for model_id, model_data in model_registry.items():
                    cost_data = model_data.get("cost", {})
                    # Primary: per_image cost
                    if "per_image" in cost_data:
                        self._model_costs[model_id] = cost_data["per_image"]
                    # Some models have tiered pricing (e.g., per_image_1k, per_image_4k)
                    elif "per_image_1k" in cost_data:
                        # Use 1K as default
                        self._model_costs[model_id] = cost_data["per_image_1k"]
                    # MP-based pricing (e.g., flux-2-flex at 0.06/MP)
                    elif "per_mp" in cost_data:
                        # Estimate for 1MP (1024x1024)
                        self._model_costs[model_id] = cost_data["per_mp"]
                    # Input/output based (e.g., flux-2-pro)
                    elif "output_first_mp" in cost_data:
                        self._model_costs[model_id] = cost_data["output_first_mp"]

                logger.debug(f"Loaded costs for {len(self._model_costs)} models")
        except Exception as e:
            logger.warning(f"Failed to load model costs from config: {e}")

    def _get_model_cost(self, model: str) -> float:
        """
        Get cost per image for a model.

        Args:
            model: Model ID or name

        Returns:
            Cost per image in USD
        """
        # Direct match
        if model in self._model_costs:
            return self._model_costs[model]

        # Try to match by model name substring
        model_lower = model.lower()
        for model_id, cost in self._model_costs.items():
            if model_id.lower() in model_lower or model_lower in model_id.lower():
                return cost

        return self.DEFAULT_COST_PER_IMAGE

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_dimensions(self, request: ImageGenerationRequest) -> tuple:
        """Get width and height from request or aspect ratio."""
        if request.width and request.height:
            return (request.width, request.height)

        aspect_ratio = request.aspect_ratio or "1:1"
        return self.ASPECT_RATIO_DIMENSIONS.get(aspect_ratio, (1024, 1024))

    async def generate_images(
        self,
        request: ImageGenerationRequest,
    ) -> ImageGenerationResult:
        """
        Generate images from a request.

        Args:
            request: ImageGenerationRequest with prompt and parameters

        Returns:
            ImageGenerationResult with generated images

        Raises:
            ImageGenerationError: If generation fails
        """
        import time
        start_time = time.time()

        # Determine model/provider to use
        model = request.model or self._config.image_models.primary or "flux-1-kontext-pro"
        provider = self._detect_provider(model)

        # Get dimensions
        width, height = self._get_dimensions(request)

        # Try generation with retry
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                if provider == "runware":
                    result = await self._generate_runware(request, model, width, height)
                elif provider == "flux":
                    result = await self._generate_flux(request, model, width, height)
                else:
                    result = await self._generate_flux(request, model, width, height)

                result.duration_ms = int((time.time() - start_time) * 1000)
                return result

            except GenerationTimeoutError:
                raise  # Don't retry timeouts
            except Exception as e:
                last_error = e
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue

        raise ImageGenerationError(f"Generation failed after {self.max_retries + 1} attempts: {last_error}")

    def _detect_provider(self, model: str) -> str:
        """Detect provider from model name."""
        model_lower = model.lower()
        if "runware" in model_lower:
            return "runware"
        elif "flux" in model_lower:
            return "flux"
        else:
            return "flux"  # Default to flux

    async def _generate_runware(
        self,
        request: ImageGenerationRequest,
        model: str,
        width: int,
        height: int,
    ) -> ImageGenerationResult:
        """Generate images using Runware API."""
        client = await self._get_client()

        payload = {
            "positivePrompt": request.prompt,
            "model": model,
            "width": width,
            "height": height,
            "numberResults": request.num_images,
            "steps": request.steps,
            "CFGScale": request.guidance_scale,
        }

        if request.negative_prompt:
            payload["negativePrompt"] = request.negative_prompt

        if request.seed is not None:
            payload["seed"] = request.seed

        try:
            response = await client.post(
                "https://api.runware.ai/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.runware_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            if response.status_code != 200:
                raise ProviderError(f"Runware API error ({response.status_code}): {response.text}")

            data = response.json()
            images = []

            for img_data in data.get("data", []):
                images.append(GeneratedImage(
                    url=img_data.get("imageURL"),
                    seed=img_data.get("seed"),
                    metadata={"runware_id": img_data.get("taskUUID")},
                ))

            return ImageGenerationResult(
                images=images,
                model_used=model,
                provider="runware",
                cost_usd=self._get_model_cost(model) * request.num_images,
                metadata={"raw_response": data},
            )

        except httpx.TimeoutException:
            raise GenerationTimeoutError(f"Runware request timed out after {self.timeout}s")

    async def _generate_flux(
        self,
        request: ImageGenerationRequest,
        model: str,
        width: int,
        height: int,
    ) -> ImageGenerationResult:
        """Generate images using Flux API."""
        client = await self._get_client()

        payload = {
            "prompt": request.prompt,
            "model": model,
            "width": width,
            "height": height,
            "num_outputs": request.num_images,
            "num_inference_steps": request.steps,
            "guidance_scale": request.guidance_scale,
        }

        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt

        if request.seed is not None:
            payload["seed"] = request.seed

        if request.reference_image_url:
            payload["image_url"] = request.reference_image_url

        try:
            response = await client.post(
                "https://api.bfl.ml/v1/flux-pro-1.1",
                headers={
                    "X-Key": self.flux_api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
            )

            if response.status_code != 200:
                raise ProviderError(f"Flux API error ({response.status_code}): {response.text}")

            data = response.json()

            # Handle async generation (polling)
            if "id" in data:
                result = await self._poll_flux_result(data["id"])
                data = result

            images = []
            for output in data.get("output", []):
                if isinstance(output, str):
                    images.append(GeneratedImage(url=output))
                elif isinstance(output, dict):
                    images.append(GeneratedImage(
                        url=output.get("url"),
                        seed=output.get("seed"),
                    ))

            # Handle single image response
            if not images and data.get("sample"):
                images.append(GeneratedImage(url=data.get("sample")))

            return ImageGenerationResult(
                images=images,
                model_used=model,
                provider="flux",
                cost_usd=self._get_model_cost(model) * request.num_images,
                metadata={"raw_response": data},
            )

        except httpx.TimeoutException:
            raise GenerationTimeoutError(f"Flux request timed out after {self.timeout}s")

    async def _poll_flux_result(
        self,
        task_id: str,
        max_polls: int = 60,
        poll_interval: float = 2.0,
    ) -> Dict[str, Any]:
        """Poll Flux API for async result."""
        client = await self._get_client()

        for _ in range(max_polls):
            try:
                response = await client.get(
                    f"https://api.bfl.ml/v1/get_result?id={task_id}",
                    headers={"X-Key": self.flux_api_key},
                )

                if response.status_code != 200:
                    raise ProviderError(f"Flux poll error ({response.status_code})")

                data = response.json()
                status = data.get("status")

                if status == "Ready":
                    return data.get("result", data)
                elif status == "Error":
                    raise ProviderError(f"Flux generation failed: {data.get('error')}")

                await asyncio.sleep(poll_interval)

            except httpx.TimeoutException:
                await asyncio.sleep(poll_interval)

        raise GenerationTimeoutError("Flux polling timed out")

    async def estimate_cost(
        self,
        request: ImageGenerationRequest,
    ) -> float:
        """
        Estimate the cost of image generation.

        Uses model-specific costs from image_models_config.yaml.

        Args:
            request: ImageGenerationRequest

        Returns:
            Estimated cost in USD
        """
        model = request.model or self._config.image_models.primary or "flux-1-kontext-pro"
        cost_per_image = self._get_model_cost(model)
        return cost_per_image * request.num_images

    async def __aenter__(self) -> "ImageGenerationService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
