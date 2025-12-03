"""
Flux 1 Kontext image generation connector.
Uses BFL API with async polling pattern.
"""
import base64
import asyncio
import httpx
from typing import Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from config import settings
from src.connectors.engine_interface import ImageEngineInterface, ModelMetadata
from src.utils import get_logger

logger = get_logger(__name__)


class FluxImageEngine(ImageEngineInterface):
    """
    Flux 1 Kontext image generation connector.

    Supports:
    - In-context text-to-image generation
    - Local editing
    - Character consistency
    - Style transfer
    - 8x faster inference

    Uses BFL API v1 with async polling pattern:
    1. Submit request → get polling_url
    2. Poll until status == "Ready"
    3. Download image from result.sample URL
    """

    def __init__(self, model_tier: str = "pro") -> None:
        """
        Initialize Flux image engine.

        Args:
            model_tier: "pro" (default, lower cost) or "max" (premium quality)
        """
        self.api_key = settings.flux_api_key
        self.model_tier = model_tier.lower()

        # Select model based on tier
        # Note: BFL API endpoint is "flux-kontext-pro" not "flux-1-kontext-pro"
        if self.model_tier == "max":
            self.model_name = "flux-kontext-max"  # BFL API endpoint name
            self.cost_per_image = settings.flux_max_cost
        else:
            self.model_name = "flux-kontext-pro"  # BFL API endpoint name
            self.cost_per_image = settings.flux_pro_cost

        # BFL API base URL (no /v1 in base)
        self.client = httpx.AsyncClient(
            base_url="https://api.bfl.ai",
            headers={
                "x-key": self.api_key,  # BFL uses x-key, not Authorization
                "Content-Type": "application/json",
                "accept": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),  # 2 min for polling
        )

        logger.info(
            "Flux image engine initialized",
            model=self.model_name,
            tier=self.model_tier,
            cost_per_image=self.cost_per_image
        )

    async def _poll_result(self, polling_url: str, task_id: str, max_wait: int = 120) -> dict[str, Any]:
        """
        Poll BFL API until image generation is complete.

        Args:
            polling_url: URL to poll for results (absolute URL from BFL, may be regional endpoint)
            task_id: Task ID for logging
            max_wait: Maximum wait time in seconds

        Returns:
            Result dict with image URL

        Raises:
            TimeoutError: If polling exceeds max_wait
            ValueError: If generation fails
        """
        start_time = asyncio.get_event_loop().time()
        poll_interval = 0.5  # Start with 500ms

        # Create a separate client for polling since BFL returns absolute URLs to regional endpoints
        # This ensures proper handling of cross-domain requests
        async with httpx.AsyncClient(
            headers={"x-key": self.api_key, "accept": "application/json"},
            timeout=httpx.Timeout(30.0, connect=10.0)
        ) as poll_client:
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time

                if elapsed > max_wait:
                    raise TimeoutError(f"Image generation timed out after {max_wait}s for task {task_id}")

                # Poll the status (polling_url is absolute, e.g. https://api.us1.bfl.ai/...)
                response = await poll_client.get(polling_url)
                response.raise_for_status()
                result = response.json()

                status = result.get("status")
                logger.debug(f"Poll result for {task_id}: status={status}, elapsed={elapsed:.1f}s")

                if status == "Ready":
                    logger.info(f"Image generation complete for {task_id}", elapsed=f"{elapsed:.1f}s")
                    return result

                elif status in ["Error", "Failed"]:
                    error_msg = result.get("error", "Unknown error")
                    logger.error(f"Image generation failed for {task_id}", error=error_msg)
                    raise ValueError(f"BFL generation failed: {error_msg}")

                elif status in ["Pending", "Request Moderated"]:
                    # Still processing, continue polling
                    await asyncio.sleep(poll_interval)
                    # Gradually increase poll interval (exponential backoff)
                    poll_interval = min(poll_interval * 1.2, 2.0)  # Max 2s between polls

                else:
                    logger.warning(f"Unknown status '{status}' for {task_id}")
                    await asyncio.sleep(poll_interval)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        )),
        reraise=True,
    )
    async def generate_image(
        self,
        prompt: str,
        style: Optional[str] = None,
        reference_image: Optional[bytes] = None,
        num_images: int = 1,
        **kwargs: Any,
    ) -> list[bytes]:
        """
        Generate images from a text prompt using BFL API async polling.

        Args:
            prompt: Text description
            style: Style guidance
            reference_image: Reference image for style transfer
            num_images: Number of images to generate
            **kwargs: Additional parameters (aspect_ratio, guidance, etc.)

        Returns:
            List of generated images as bytes

        Raises:
            ValueError: Invalid request parameters or generation failed
            PermissionError: API key invalid (401)
            TimeoutError: Generation took too long
            httpx.HTTPStatusError: HTTP errors from Flux API
        """
        try:
            logger.info(
                "Generating images with Flux BFL API",
                prompt_length=len(prompt),
                num_images=num_images,
                has_reference=reference_image is not None,
                model=self.model_name,
            )

            images = []

            for i in range(num_images):
                # Build request payload (BFL API format)
                payload: dict[str, Any] = {
                    "prompt": prompt,
                }

                # Add style if provided
                if style:
                    payload["prompt"] = f"{style} style: {prompt}"

                # Handle dimensions - BFL uses aspect_ratio instead of width/height
                width = kwargs.get("width", 1024)
                height = kwargs.get("height", 1024)

                # Convert to aspect ratio (BFL supports 21:9 to 9:21)
                if width == height:
                    payload["aspect_ratio"] = "1:1"
                elif width > height:
                    ratio = round(width / height)
                    payload["aspect_ratio"] = f"{ratio}:1"
                else:
                    ratio = round(height / width)
                    payload["aspect_ratio"] = f"1:{ratio}"

                # Add optional parameters (using correct BFL field names)
                if "seed" in kwargs:
                    payload["seed"] = kwargs["seed"]
                if "safety_tolerance" in kwargs:
                    payload["safety_tolerance"] = kwargs["safety_tolerance"]
                if "output_format" in kwargs:
                    payload["output_format"] = kwargs["output_format"]
                if "prompt_upsampling" in kwargs:
                    payload["prompt_upsampling"] = kwargs["prompt_upsampling"]

                # Add reference image for in-context generation if provided
                # BFL uses "input_image" not "image"
                if reference_image:
                    payload["input_image"] = base64.b64encode(reference_image).decode()

                # Step 1: Submit generation request
                endpoint = f"/v1/{self.model_name}"
                logger.debug(f"Submitting to {endpoint}", payload_keys=list(payload.keys()))

                # DIAGNOSTIC: Log exact payload being sent to BFL API (using print)
                print("=" * 80)
                print(f"🔍 DIAGNOSTIC - Flux BFL API Request #{i+1}/{num_images}")
                print(f"  Endpoint: {endpoint}")
                print(f"  Prompt: '{payload.get('prompt')}'")
                print(f"  Aspect Ratio: {payload.get('aspect_ratio')}")
                print(f"  Has Seed: {('seed' in payload)}")
                print(f"  Has Input Image: {('input_image' in payload)}")
                print(f"  Full Payload: {payload}")
                print("=" * 80)

                response = await self.client.post(endpoint, json=payload)

                # Handle HTTP errors
                if response.status_code == 401:
                    logger.error("Flux API key invalid")
                    raise PermissionError("Flux API authentication failed - invalid API key")
                elif response.status_code == 400:
                    error_detail = response.text
                    logger.error("Invalid Flux request", status=400, detail=error_detail)
                    raise ValueError(f"Invalid request to Flux API: {error_detail}")
                elif response.status_code == 429:
                    logger.warning("Flux rate limit hit")
                    raise httpx.HTTPStatusError(
                        "Rate limit exceeded",
                        request=response.request,
                        response=response
                    )

                response.raise_for_status()
                submit_result = response.json()

                # Step 2: Get polling URL
                task_id = submit_result.get("id")
                polling_url = submit_result.get("polling_url")

                if not polling_url:
                    logger.error("No polling_url in response", result=submit_result)
                    raise ValueError("No polling_url returned from BFL API")

                logger.info(f"Submitted image {i+1}/{num_images}", task_id=task_id[:8] if task_id else None)

                # Step 3: Poll until ready
                result = await self._poll_result(polling_url, task_id or f"task-{i}")

                # Step 4: Download image from result URL
                image_url = result.get("result", {}).get("sample")
                if not image_url:
                    logger.error("No image URL in result", result=result)
                    raise ValueError("No image URL in BFL API result")

                # Download the image (images expire in 10 minutes!)
                # Note: image_url is absolute (e.g. https://delivery-us1.bfl.ai/...)
                logger.debug(f"Downloading image from {image_url}")

                # Use separate client for downloading since delivery URLs are on different domain
                async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as download_client:
                    image_response = await download_client.get(image_url)
                    image_response.raise_for_status()

                    image_bytes = image_response.content
                    images.append(image_bytes)
                    logger.info(f"Downloaded Flux image {i+1}/{num_images}", size_kb=len(image_bytes) // 1024)

            return images

        except httpx.TimeoutException as e:
            logger.warning("Flux API timeout, retrying", error=str(e))
            raise  # Will be retried
        except httpx.NetworkError as e:
            logger.warning("Flux network error, retrying", error=str(e))
            raise  # Will be retried
        except (ValueError, PermissionError, TimeoutError):
            raise  # Don't retry client errors
        except Exception as e:
            logger.error("Flux generation error", error=str(e), exc_info=True)
            raise

    async def edit_image(
        self,
        image: bytes,
        prompt: str,
        mask: Optional[bytes] = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Edit an existing image based on a prompt using BFL API.

        Note: Uses the same Kontext model endpoint with image parameter.

        Args:
            image: Input image
            prompt: Edit instructions
            mask: Optional mask for local editing
            **kwargs: Additional parameters

        Returns:
            Edited image as bytes
        """
        try:
            logger.info("Editing image with Flux BFL API", has_mask=mask is not None)

            # Build edit request payload (BFL API format)
            # BFL uses "input_image" not "image"
            payload: dict[str, Any] = {
                "prompt": prompt,
                "input_image": base64.b64encode(image).decode(),
            }

            # Add mask if provided (for inpainting)
            if mask:
                payload["mask"] = base64.b64encode(mask).decode()

            # Add optional parameters
            if "seed" in kwargs:
                payload["seed"] = kwargs["seed"]
            if "safety_tolerance" in kwargs:
                payload["safety_tolerance"] = kwargs["safety_tolerance"]

            # Submit edit request
            endpoint = f"/v1/{self.model_name}"
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()

            submit_result = response.json()
            polling_url = submit_result.get("polling_url")
            task_id = submit_result.get("id", "edit-task")

            if not polling_url:
                raise ValueError("No polling_url returned from BFL API")

            # Poll until ready
            result = await self._poll_result(polling_url, task_id)

            # Download edited image (absolute URL to delivery domain)
            image_url = result.get("result", {}).get("sample")
            if not image_url:
                raise ValueError("No image URL in BFL API result")

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as download_client:
                image_response = await download_client.get(image_url)
                image_response.raise_for_status()
                return image_response.content

        except Exception as e:
            logger.error("Flux editing error", error=str(e), exc_info=True)
            raise

    async def blend_images(
        self,
        images: list[bytes],
        prompt: str,
        **kwargs: Any,
    ) -> bytes:
        """
        Blend multiple images together using BFL API.

        Note: Uses the first image as base with image_prompt_strength for blending.

        Args:
            images: List of images to blend
            prompt: Blending instructions
            **kwargs: Additional parameters

        Returns:
            Blended image as bytes
        """
        try:
            logger.info("Blending images with Flux BFL API", num_images=len(images))

            if len(images) < 1:
                raise ValueError("Need at least 1 image for blending")

            # Use first image as base
            base_image = images[0]

            # Build blend request (using BFL "input_image" parameter)
            payload: dict[str, Any] = {
                "prompt": prompt,
                "input_image": base64.b64encode(base_image).decode(),
            }

            # Add optional parameters
            if "seed" in kwargs:
                payload["seed"] = kwargs["seed"]
            if "safety_tolerance" in kwargs:
                payload["safety_tolerance"] = kwargs["safety_tolerance"]

            # Submit blend request
            endpoint = f"/v1/{self.model_name}"
            response = await self.client.post(endpoint, json=payload)
            response.raise_for_status()

            submit_result = response.json()
            polling_url = submit_result.get("polling_url")
            task_id = submit_result.get("id", "blend-task")

            if not polling_url:
                raise ValueError("No polling_url returned from BFL API")

            # Poll until ready
            result = await self._poll_result(polling_url, task_id)

            # Download blended image (absolute URL to delivery domain)
            image_url = result.get("result", {}).get("sample")
            if not image_url:
                raise ValueError("No image URL in BFL API result")

            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as download_client:
                image_response = await download_client.get(image_url)
                image_response.raise_for_status()
                return image_response.content

        except Exception as e:
            logger.error("Flux blending error", error=str(e), exc_info=True)
            raise

    async def estimate_cost(self, num_images: int = 1) -> float:
        """
        Estimate the cost of image generation based on model tier.

        Args:
            num_images: Number of images

        Returns:
            Estimated cost in USD
        """
        return self.cost_per_image * num_images

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata based on model tier.

        Returns:
            Model metadata
        """
        return ModelMetadata(
            model_name=self.model_name,
            provider="bfl",
            cost_per_request=self.cost_per_image,
            average_latency=1.2 if self.model_tier == "pro" else 1.5,  # Max slightly slower
            supports_streaming=False,
            supports_images=True,
        )

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the model supports a specific feature.

        Args:
            feature: Feature name

        Returns:
            True if feature is supported
        """
        supported_features = {
            "text_to_image",
            "image_to_image",
            "local_editing",
            "style_transfer",
            "character_consistency",
            "fast_inference",
            "in_context_generation",
        }
        return feature in supported_features
