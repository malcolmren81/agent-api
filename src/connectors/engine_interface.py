"""
Abstract interfaces for reasoning and image generation engines.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional


@dataclass
class ModelMetadata:
    """Model metadata and performance metrics."""
    model_name: str
    provider: str
    cost_per_request: float
    average_latency: float
    supports_streaming: bool
    max_tokens: Optional[int] = None
    supports_images: bool = False


class ReasoningEngineInterface(ABC):
    """Abstract interface for reasoning models (Gemini, ChatGPT)."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        Generate a response using the reasoning model.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated text or async iterator for streaming
        """
        pass

    @abstractmethod
    async def estimate_cost(self, prompt: str, max_tokens: Optional[int] = None) -> float:
        """
        Estimate the cost of a generation request.

        Args:
            prompt: Input prompt
            max_tokens: Expected output tokens

        Returns:
            Estimated cost in USD
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.

        Returns:
            Model metadata
        """
        pass


class ImageEngineInterface(ABC):
    """Abstract interface for image generation models (Flux, Gemini Image)."""

    @abstractmethod
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
            reference_image: Reference image for style transfer
            num_images: Number of images to generate
            **kwargs: Additional model-specific parameters

        Returns:
            List of generated images as bytes
        """
        pass

    @abstractmethod
    async def edit_image(
        self,
        image: bytes,
        prompt: str,
        mask: Optional[bytes] = None,
        **kwargs: Any,
    ) -> bytes:
        """
        Edit an existing image based on a prompt.

        Args:
            image: Input image
            prompt: Edit instructions
            mask: Optional mask for local editing
            **kwargs: Additional model-specific parameters

        Returns:
            Edited image as bytes
        """
        pass

    @abstractmethod
    async def blend_images(
        self,
        images: list[bytes],
        prompt: str,
        **kwargs: Any,
    ) -> bytes:
        """
        Blend multiple images together.

        Args:
            images: List of images to blend
            prompt: Blending instructions
            **kwargs: Additional model-specific parameters

        Returns:
            Blended image as bytes
        """
        pass

    @abstractmethod
    async def estimate_cost(self, num_images: int = 1) -> float:
        """
        Estimate the cost of image generation.

        Args:
            num_images: Number of images

        Returns:
            Estimated cost in USD
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.

        Returns:
            Model metadata
        """
        pass

    @abstractmethod
    def supports_feature(self, feature: str) -> bool:
        """
        Check if the model supports a specific feature.

        Args:
            feature: Feature name (e.g., "local_editing", "style_transfer")

        Returns:
            True if feature is supported
        """
        pass
