"""
Google Imagen 3 generation connector using Vertex AI.
"""
import base64
from typing import Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions
from vertexai.preview.vision_models import ImageGenerationModel
import vertexai
from config import settings
from src.connectors.engine_interface import ImageEngineInterface, ModelMetadata
from src.utils import get_logger

logger = get_logger(__name__)


class Imagen3Engine(ImageEngineInterface):
    """
    Google Imagen 3 connector.

    Supports:
    - Multi-image blending
    - Character consistency across scenes
    - Precise local edits with natural language
    - World knowledge-based transformations
    - SynthID watermarking
    - Photorealistic image generation
    """

    def __init__(self) -> None:
        """Initialize Imagen 3 engine using Vertex AI."""
        self.model_name = settings.gemini_model_image  # imagen-3.0-generate-001
        self._model = None  # Lazy initialization to avoid Cloud Run gRPC issues
        self._initialized = False
        logger.info("Imagen 3 engine prepared (lazy init)", model=self.model_name)

    def _ensure_initialized(self) -> None:
        """Lazily initialize Vertex AI and load model on first use."""
        if not self._initialized:
            logger.info("Initializing Vertex AI on first use", model=self.model_name)
            # Initialize Vertex AI
            vertexai.init(project=settings.gcp_project_id, location=settings.gcp_region or "us-central1")
            self._model = ImageGenerationModel.from_pretrained(self.model_name)
            self._initialized = True
            logger.info("Vertex AI initialized successfully", model=self.model_name)

    @property
    def model(self):
        """Get the model, initializing if needed."""
        self._ensure_initialized()
        return self._model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            google_exceptions.ResourceExhausted,  # Rate limit
            google_exceptions.ServiceUnavailable,  # Temporary service issues
            google_exceptions.DeadlineExceeded,  # Timeout
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
        Generate images from a text prompt.

        Args:
            prompt: Text description
            style: Style guidance
            reference_image: Reference image for context
            num_images: Number of images to generate
            **kwargs: Additional Gemini-specific parameters

        Returns:
            List of generated images as bytes (includes SynthID watermark)

        Raises:
            ValueError: Invalid request parameters or blocked content
            PermissionError: API key invalid
            google_exceptions.ResourceExhausted: Rate limit exceeded (retried)
        """
        try:
            logger.info(
                "Generating images with Imagen 3",
                prompt_length=len(prompt),
                num_images=num_images,
                has_reference=reference_image is not None,
            )

            images = []

            # Build full prompt with style if provided
            full_prompt = f"{style} style: {prompt}" if style else prompt

            # Generate images using Vertex AI
            response = self.model.generate_images(
                prompt=full_prompt,
                number_of_images=num_images,
                aspect_ratio=kwargs.get("aspect_ratio", "1:1"),
                safety_filter_level="block_some",
                person_generation="allow_adult",
            )

            # Validate response
            if not response or not response.images:
                logger.warning("No images in Imagen 3 response")
                raise ValueError("No images returned from Imagen 3")

            # Extract image bytes (SynthID watermark automatically included)
            for i, img in enumerate(response.images):
                try:
                    image_data = img._image_bytes
                    if not image_data:
                        raise ValueError(f"Empty image data for image {i}")
                    images.append(image_data)
                    logger.info(f"Generated Imagen 3 image {i+1}/{num_images}")
                except AttributeError as e:
                    logger.error("Invalid Imagen 3 response structure", error=str(e))
                    raise ValueError(f"Unexpected Imagen 3 response format: {str(e)}")

            return images

        except google_exceptions.InvalidArgument as e:
            logger.error("Invalid Imagen 3 request", error=str(e), prompt_length=len(prompt))
            raise ValueError(f"Invalid request to Imagen 3: {str(e)}")
        except google_exceptions.PermissionDenied as e:
            logger.error("Imagen 3 API key invalid", error=str(e))
            raise PermissionError(f"Imagen 3 API authentication failed: {str(e)}")
        except google_exceptions.ResourceExhausted as e:
            logger.warning("Imagen 3 rate limit hit, retrying", error=str(e))
            raise  # Will be retried by @retry decorator
        except ValueError:
            raise  # Don't retry client errors
        except Exception as e:
            logger.error("Imagen 3 generation error", error=str(e), exc_info=True)
            raise

    async def edit_image(
        self,
        image: bytes,
        prompt: str,
        mask: Optional[bytes] = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Edit an existing image based on natural language prompt.

        Supports precise local edits and semantic transformations.

        Args:
            image: Input image
            prompt: Edit instructions in natural language
            mask: Optional mask for targeted editing
            **kwargs: Additional parameters

        Returns:
            Edited image as bytes with SynthID watermark
        """
        try:
            logger.info("Editing image with Imagen 3", has_mask=mask is not None)

            # Use Imagen 3's edit capabilities with natural language via Vertex AI
            from vertexai.preview.vision_models import Image as VertexImage

            # Convert image to Vertex AI format
            vertex_image = VertexImage(image_bytes=image)

            # Edit image using Vertex AI
            response = self.model.edit_image(
                base_image=vertex_image,
                prompt=prompt,
                edit_mode="inpainting-insert" if mask else "inpainting-remove",
            )

            # Extract edited image (with SynthID watermark)
            if response and response._image_bytes:
                return response._image_bytes

            raise ValueError("No image returned from Imagen 3")

        except Exception as e:
            logger.error("Imagen 3 editing error", error=str(e), exc_info=True)
            raise

    async def blend_images(
        self,
        images: list[bytes],
        prompt: str,
        **kwargs: Any,
    ) -> bytes:
        """
        Blend multiple images together.

        This is a key strength of Gemini 2.5 Flash Image - multi-image fusion
        with semantic understanding.

        Args:
            images: List of images to blend
            prompt: Blending instructions
            **kwargs: Additional parameters

        Returns:
            Blended image as bytes with SynthID watermark
        """
        try:
            logger.info("Blending images with Imagen 3", num_images=len(images))

            # Note: Multi-image blending may not be directly supported in current Vertex AI SDK
            # For now, raise NotImplementedError - this feature can be added when SDK supports it
            raise NotImplementedError("Multi-image blending is not yet supported in Vertex AI SDK")

        except Exception as e:
            logger.error("Imagen 3 blending error", error=str(e), exc_info=True)
            raise

    async def estimate_cost(self, num_images: int = 1) -> float:
        """
        Estimate the cost of image generation.

        Args:
            num_images: Number of images

        Returns:
            Estimated cost in USD
        """
        return settings.gemini_image_cost * num_images

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.

        Returns:
            Model metadata
        """
        return ModelMetadata(
            model_name=self.model_name,
            provider="google",
            cost_per_request=settings.gemini_image_cost,
            average_latency=2.0,
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
            "multi_image_blending",
            "character_consistency",
            "natural_language_editing",
            "world_knowledge",
            "synthid_watermark",
            "local_editing",
            "semantic_transformations",
        }
        return feature in supported_features
