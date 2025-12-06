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
import uuid

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
    reference_strength: float = 0.75
    style: Optional[str] = None
    # Provider-specific settings (varies by model/provider)
    # e.g., {"lora_weights": [...], "scheduler": "euler", "safety_checker": false}
    provider_settings: Dict[str, Any] = field(default_factory=dict)
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
            "reference_strength": self.reference_strength,
            "style": self.style,
            "provider_settings": self.provider_settings,
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
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize ImageGenerationService.

        All generation uses Runware API.

        Args:
            runware_api_key: Runware API key
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
        """
        self.runware_api_key = runware_api_key or os.environ.get("RUNWARE_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: Optional[httpx.AsyncClient] = None
        self._config = get_config()
        self._model_costs: Dict[str, float] = {}
        self._model_air_ids: Dict[str, str] = {}  # model_name -> air_id
        self._model_supports_cfg: Dict[str, bool] = {}  # model_name -> supports CFGScale
        self._model_supports_steps: Dict[str, bool] = {}  # model_name -> supports steps
        self._load_model_config()

    # Models that support steps parameter (whitelist approach for safety)
    # Only diffusion models (FLUX, SD-based) support steps
    MODELS_WITH_STEPS_SUPPORT = frozenset({
        "flux-2-flex",
        "flux-2-pro",
        # Add other diffusion models here if needed
    })

    # Models that do NOT support steps (provider-hosted models)
    # These models use proprietary inference and don't expose steps
    MODELS_WITHOUT_STEPS = frozenset({
        "midjourney-v7",
        "ideogram-3",
        "imagen-4-preview",
        "imagen-4-ultra",
        "nano-banana",
        "nano-banana-2-pro",
        "seedream-4",
        "flux-1-kontext-pro",
        "qwen-image",
        "qwen-image-edit",
    })

    def _load_model_config(self) -> None:
        """
        Load model costs and AIR IDs from image_models_config.yaml.

        Extracts:
        - Costs per image from model_registry entries
        - AIR IDs for Runware API calls
        - CFGScale support per model
        - Steps support per model (with whitelist fallback)
        """
        try:
            if self.IMAGE_MODELS_CONFIG_PATH.exists():
                with open(self.IMAGE_MODELS_CONFIG_PATH, "r") as f:
                    config = yaml.safe_load(f) or {}

                model_registry = config.get("model_registry", {})
                for model_id, model_data in model_registry.items():
                    # Load AIR ID (required for Runware API)
                    if "air_id" in model_data:
                        self._model_air_ids[model_id] = model_data["air_id"]

                    # Check if model supports CFGScale (only if specs.cfg_scale exists)
                    specs = model_data.get("specs", {})
                    self._model_supports_cfg[model_id] = "cfg_scale" in specs

                    # Check if model supports steps:
                    # 1. Check if specs.steps exists in YAML
                    # 2. Fallback to whitelist if not in YAML
                    if "steps" in specs:
                        self._model_supports_steps[model_id] = True
                    elif model_id in self.MODELS_WITH_STEPS_SUPPORT:
                        self._model_supports_steps[model_id] = True
                    else:
                        # Default to False - safer to not send steps
                        self._model_supports_steps[model_id] = False

                    # Load cost data
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

                # Log which models support steps for debugging
                steps_supported = [m for m, s in self._model_supports_steps.items() if s]
                steps_not_supported = [m for m, s in self._model_supports_steps.items() if not s]
                logger.info(
                    f"image_generation_service.config.loaded: "
                    f"total_models={len(self._model_air_ids)}, "
                    f"steps_supported={steps_supported}, "
                    f"steps_blocked={steps_not_supported[:5]}..."  # Show first 5 for brevity
                )
                logger.debug(f"image_generation_service.config.details: air_ids={list(self._model_air_ids.keys())}")
            else:
                # Config file not found - log error but still use hardcoded lists
                logger.error(
                    f"image_generation_service.config.not_found: "
                    f"path={self.IMAGE_MODELS_CONFIG_PATH}, "
                    f"cwd={Path.cwd()}, "
                    f"falling back to hardcoded MODELS_WITH_STEPS_SUPPORT and MODELS_WITHOUT_STEPS"
                )
        except Exception as e:
            logger.error(
                f"image_generation_service.config.load_error: {e}, "
                f"falling back to hardcoded lists. "
                f"MODELS_WITH_STEPS_SUPPORT={list(self.MODELS_WITH_STEPS_SUPPORT)}, "
                f"MODELS_WITHOUT_STEPS={list(self.MODELS_WITHOUT_STEPS)}"
            )

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

    def _get_air_id(self, model: str) -> str:
        """
        Get AIR ID for a model name.

        Converts model names like 'flux-1-kontext-pro' to AIR IDs like 'bfl:3@1'.
        If the model is already an AIR ID format (contains ':'), returns as-is.

        Args:
            model: Model name or AIR ID

        Returns:
            AIR ID for Runware API
        """
        # If already in AIR format, return as-is
        if ":" in model:
            return model

        # Direct lookup
        if model in self._model_air_ids:
            return self._model_air_ids[model]

        # Try case-insensitive match
        model_lower = model.lower()
        for model_id, air_id in self._model_air_ids.items():
            if model_id.lower() == model_lower:
                return air_id

        # Fallback to the model name itself (may fail at Runware)
        logger.warning(f"No AIR ID found for model '{model}', using as-is")
        return model

    def _supports_cfg_scale(self, model_name: str) -> bool:
        """
        Check if a model supports CFGScale parameter.

        Args:
            model_name: Model name (not AIR ID)

        Returns:
            True if model supports CFGScale, False otherwise
        """
        # Direct lookup
        if model_name in self._model_supports_cfg:
            return self._model_supports_cfg[model_name]

        # Try case-insensitive match
        model_lower = model_name.lower()
        for model_id, supports in self._model_supports_cfg.items():
            if model_id.lower() == model_lower:
                return supports

        # Default to False for unknown models (safer)
        return False

    def _supports_steps(self, model_name: str) -> bool:
        """
        Check if a model supports steps parameter.

        Uses a multi-layer approach:
        1. Check loaded config (exact match)
        2. Check loaded config (case-insensitive match)
        3. Check whitelist MODELS_WITH_STEPS_SUPPORT
        4. Check blocklist MODELS_WITHOUT_STEPS (with prefix matching)
        5. Default to False (safe default)

        Args:
            model_name: Model name (not AIR ID)

        Returns:
            True if model supports steps, False otherwise
        """
        # Handle None or empty model name
        if not model_name:
            logger.warning("_supports_steps: empty model_name, returning False")
            return False

        # Strip AIR format if passed (e.g., "midjourney:3@1" -> use blocklist check)
        if ":" in model_name:
            logger.debug(f"_supports_steps: model_name contains AIR format: {model_name}")
            # AIR format models are provider-hosted, default to no steps
            return False

        # Direct lookup from loaded config
        if model_name in self._model_supports_steps:
            result = self._model_supports_steps[model_name]
            logger.debug(f"_supports_steps: model={model_name}, result={result} (from config)")
            return result

        # Try case-insensitive match in config
        model_lower = model_name.lower()
        for model_id, supports in self._model_supports_steps.items():
            if model_id.lower() == model_lower:
                logger.debug(f"_supports_steps: model={model_name}, result={supports} (case-insensitive match)")
                return supports

        # Fallback to whitelist (exact and prefix match)
        if model_name in self.MODELS_WITH_STEPS_SUPPORT:
            logger.debug(f"_supports_steps: model={model_name}, result=True (whitelist exact)")
            return True

        # Check if model starts with a whitelisted prefix (e.g., "flux-2-flex-v2" matches "flux-2-flex")
        for whitelisted in self.MODELS_WITH_STEPS_SUPPORT:
            if model_name.startswith(whitelisted) or whitelisted.startswith(model_name):
                logger.debug(f"_supports_steps: model={model_name}, result=True (whitelist prefix: {whitelisted})")
                return True

        # Check blocklist explicitly (exact match)
        if model_name in self.MODELS_WITHOUT_STEPS:
            logger.debug(f"_supports_steps: model={model_name}, result=False (blocklist exact)")
            return False

        # Check blocklist with prefix matching (e.g., "midjourney-v7-fast" matches "midjourney-v7")
        for blocked in self.MODELS_WITHOUT_STEPS:
            if model_name.startswith(blocked) or blocked.startswith(model_name):
                logger.debug(f"_supports_steps: model={model_name}, result=False (blocklist prefix: {blocked})")
                return False

        # Check base model name (split on hyphen, take first 2 parts)
        # e.g., "midjourney-v7-experimental" -> "midjourney-v7"
        parts = model_name.split("-")
        if len(parts) >= 2:
            base_name = "-".join(parts[:2])
            if base_name in self.MODELS_WITHOUT_STEPS:
                logger.debug(f"_supports_steps: model={model_name}, result=False (base name match: {base_name})")
                return False

        # Default to False for unknown models (safer - don't send unsupported params)
        logger.warning(f"_supports_steps: model={model_name}, result=False (unknown model, defaulting to safe)")
        return False

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
        Generate images from a request using Runware API.

        Args:
            request: ImageGenerationRequest with prompt and parameters

        Returns:
            ImageGenerationResult with generated images

        Raises:
            ImageGenerationError: If generation fails
        """
        import time
        start_time = time.time()

        # Determine model to use (all via Runware)
        # Model comes from request or config - no hardcoded fallback
        model_name = request.model or self._config.image_models.primary
        if not model_name:
            raise ImageGenerationError("No model specified and no default model configured")

        # Convert model name to AIR ID for Runware API
        model = self._get_air_id(model_name)
        logger.debug(f"Model selection: {model_name} -> AIR ID: {model}")

        # Get dimensions
        width, height = self._get_dimensions(request)

        # Try generation with retry
        last_error: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._generate_runware(request, model, model_name, width, height)
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

    async def _generate_runware(
        self,
        request: ImageGenerationRequest,
        model: str,
        model_name: str,
        width: int,
        height: int,
    ) -> ImageGenerationResult:
        """Generate images using Runware API."""
        client = await self._get_client()

        # Generate unique taskUUID for this request (required by Runware API)
        task_uuid = str(uuid.uuid4())

        payload = {
            "taskType": "imageInference",  # Required by Runware API
            "taskUUID": task_uuid,  # Required: UUID v4 for matching async responses
            "outputType": "URL",  # Return image as URL
            "positivePrompt": request.prompt,
            "model": model,
            "width": width,
            "height": height,
            "numberResults": request.num_images,
        }

        # Only add steps if the model supports it (e.g., flux-2-flex)
        # Provider-hosted models (Midjourney, Ideogram, Google) don't support steps
        supports_steps = self._supports_steps(model_name)
        if supports_steps:
            payload["steps"] = request.steps
            logger.info(f"runware.payload.steps_added: model={model_name}, steps={request.steps}")
        else:
            logger.info(f"runware.payload.steps_skipped: model={model_name} (not supported)")

        # Only add CFGScale if the model supports it (e.g., flux-2-flex)
        # Provider-hosted models don't support CFGScale
        supports_cfg = self._supports_cfg_scale(model_name)
        if supports_cfg:
            payload["CFGScale"] = request.guidance_scale
            logger.info(f"runware.payload.cfg_added: model={model_name}, cfg={request.guidance_scale}")
        else:
            logger.debug(f"runware.payload.cfg_skipped: model={model_name} (not supported)")

        if request.negative_prompt:
            payload["negativePrompt"] = request.negative_prompt

        if request.seed is not None:
            payload["seed"] = request.seed

        if request.reference_image_url:
            # Runware uses seedImage for image-to-image (not inputImage)
            # Can be URL, base64, data URI, or UUID of uploaded image
            payload["seedImage"] = request.reference_image_url
            payload["strength"] = request.reference_strength

        # Apply provider-specific settings (Runware-specific params)
        # e.g., lora, scheduler, clipSkip, etc.
        if request.provider_settings:
            for key, value in request.provider_settings.items():
                # Map common settings to Runware-specific keys
                runware_key_map = {
                    "scheduler": "scheduler",
                    "lora": "lora",
                    "lora_weights": "lora",
                    "clip_skip": "clipSkip",
                    "safety_checker": "checkNSFW",
                }
                mapped_key = runware_key_map.get(key, key)
                payload[mapped_key] = value

        try:
            # Log payload keys for debugging (don't log full prompt for privacy)
            payload_keys = list(payload.keys())
            logger.info(f"runware.api.request: taskUUID={task_uuid}, model_name={model_name}, air_id={model}, size={width}x{height}, payload_keys={payload_keys}")

            # Runware API requires the request payload to be an array of objects
            response = await client.post(
                "https://api.runware.ai/v1/images/generations",
                headers={
                    "Authorization": f"Bearer {self.runware_api_key}",
                    "Content-Type": "application/json",
                },
                json=[payload],  # Wrap in array as required by Runware API
            )

            if response.status_code != 200:
                logger.error(f"Runware API error: status={response.status_code}, taskUUID={task_uuid}")
                raise ProviderError(f"Runware API error ({response.status_code}): {response.text}")

            data = response.json()
            images = []

            for img_data in data.get("data", []):
                images.append(GeneratedImage(
                    url=img_data.get("imageURL"),
                    seed=img_data.get("seed"),
                    metadata={"runware_id": img_data.get("taskUUID"), "imageUUID": img_data.get("imageUUID")},
                ))

            logger.info(f"Runware API success: taskUUID={task_uuid}, images={len(images)}")

            return ImageGenerationResult(
                images=images,
                model_used=model,
                provider="runware",
                cost_usd=self._get_model_cost(model) * request.num_images,
                metadata={"raw_response": data, "task_uuid": task_uuid},
            )

        except httpx.TimeoutException:
            raise GenerationTimeoutError(f"Runware request timed out after {self.timeout}s")

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
        model = request.model or self._config.image_models.primary
        if not model:
            return 0.04  # Default cost estimate if no model specified
        cost_per_image = self._get_model_cost(model)
        return cost_per_image * request.num_images

    async def __aenter__(self) -> "ImageGenerationService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
