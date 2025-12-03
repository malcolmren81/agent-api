"""
Embedding Service

Service for generating text and image embeddings for vector search and RAG.
Uses Google Vertex AI via Gemini API:
- Text: gemini-embedding-001 (768 dimensions)
- Image: multimodalembedding@001 (1408 dimensions)

Documentation Reference: Section 4.2
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import os
import httpx

from palet8_agents.core.config import get_config
from palet8_agents.core.exceptions import LLMClientError

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Base exception for EmbeddingService errors."""
    pass


class EmbeddingModelError(EmbeddingServiceError):
    """Raised when embedding model fails."""
    pass


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""
    embedding: List[float]
    model_used: str
    dimensions: int
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embedding": self.embedding,
            "model_used": self.model_used,
            "dimensions": self.dimensions,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


@dataclass
class BatchEmbeddingResult:
    """Result from batch embedding generation."""
    embeddings: List[List[float]]
    model_used: str
    dimensions: int
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "embeddings": self.embeddings,
            "model_used": self.model_used,
            "dimensions": self.dimensions,
            "total_tokens": self.total_tokens,
            "total_cost_usd": self.total_cost_usd,
            "metadata": self.metadata,
        }


class EmbeddingService:
    """
    Service for generating embeddings.

    Features:
    - Text embedding generation
    - Batch embedding for efficiency
    - Image embedding (optional, model-dependent)
    - Cost tracking
    - Automatic failover
    """

    # OpenRouter embedding endpoint
    BASE_URL = "https://openrouter.ai/api/v1"

    # Default batch size for efficiency
    DEFAULT_BATCH_SIZE = 100

    # Cost per 1M tokens (from agent_routing_policy.yaml embedding_models)
    # text: openai/text-embedding-3-small = $0.02/1M = $0.00002/1K
    DEFAULT_COST_PER_1K = 0.00002

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        """
        Initialize EmbeddingService.

        Args:
            api_key: OpenRouter API key. If None, reads from environment.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._config = get_config()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://palet8.com",
                    "X-Title": "Palet8 Agent System",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_text_model(self) -> str:
        """Get configured text embedding model."""
        return self._config.embedding_models.text_model or "gemini-embedding-001"

    def _get_text_dimensions(self) -> int:
        """Get configured text embedding dimensions."""
        return self._config.embedding_models.text_dimensions or 768

    def _get_image_model(self) -> str:
        """Get configured image embedding model."""
        return self._config.embedding_models.image_model or "multimodalembedding@001"

    def _get_image_dimensions(self) -> int:
        """Get configured image embedding dimensions."""
        return self._config.embedding_models.image_dimensions or 1408

    async def embed_text(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed
            model: Optional model override

        Returns:
            EmbeddingResult with embedding vector

        Raises:
            EmbeddingModelError: If embedding generation fails
        """
        model = model or self._get_text_model()
        client = await self._get_client()

        try:
            response = await client.post(
                f"{self.BASE_URL}/embeddings",
                json={
                    "model": model,
                    "input": text,
                },
            )

            if response.status_code != 200:
                error_msg = response.text
                raise EmbeddingModelError(f"Embedding API error ({response.status_code}): {error_msg}")

            data = response.json()
            embedding_data = data.get("data", [{}])[0]
            embedding = embedding_data.get("embedding", [])
            usage = data.get("usage", {})

            tokens_used = usage.get("total_tokens", 0)
            cost_usd = (tokens_used / 1000) * self.DEFAULT_COST_PER_1K

            return EmbeddingResult(
                embedding=embedding,
                model_used=data.get("model", model),
                dimensions=len(embedding),
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                metadata={"raw_usage": usage},
            )

        except httpx.TimeoutException:
            raise EmbeddingModelError(f"Embedding request timed out after {self.timeout}s")
        except EmbeddingModelError:
            raise
        except Exception as e:
            raise EmbeddingModelError(f"Unexpected embedding error: {str(e)}")

    async def embed_text_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> BatchEmbeddingResult:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Optional model override
            batch_size: Number of texts per API call

        Returns:
            BatchEmbeddingResult with all embeddings

        Raises:
            EmbeddingModelError: If embedding generation fails
        """
        if not texts:
            return BatchEmbeddingResult(
                embeddings=[],
                model_used=model or self._get_text_model(),
                dimensions=self._get_text_dimensions(),
            )

        model = model or self._get_text_model()
        batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        client = await self._get_client()

        all_embeddings: List[List[float]] = []
        total_tokens = 0
        total_cost = 0.0

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            try:
                response = await client.post(
                    f"{self.BASE_URL}/embeddings",
                    json={
                        "model": model,
                        "input": batch,
                    },
                )

                if response.status_code != 200:
                    error_msg = response.text
                    raise EmbeddingModelError(
                        f"Batch embedding API error ({response.status_code}): {error_msg}"
                    )

                data = response.json()
                usage = data.get("usage", {})

                # Extract embeddings in order
                embedding_items = sorted(
                    data.get("data", []),
                    key=lambda x: x.get("index", 0)
                )

                for item in embedding_items:
                    all_embeddings.append(item.get("embedding", []))

                batch_tokens = usage.get("total_tokens", 0)
                total_tokens += batch_tokens
                total_cost += (batch_tokens / 1000) * self.DEFAULT_COST_PER_1K

            except httpx.TimeoutException:
                raise EmbeddingModelError(
                    f"Batch embedding request timed out at batch {i // batch_size + 1}"
                )
            except EmbeddingModelError:
                raise
            except Exception as e:
                raise EmbeddingModelError(f"Unexpected batch embedding error: {str(e)}")

        return BatchEmbeddingResult(
            embeddings=all_embeddings,
            model_used=model,
            dimensions=len(all_embeddings[0]) if all_embeddings else self._get_text_dimensions(),
            total_tokens=total_tokens,
            total_cost_usd=total_cost,
            metadata={"batch_count": (len(texts) + batch_size - 1) // batch_size},
        )

    async def embed_for_search(
        self,
        query: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generate embedding optimized for search queries.

        This is a convenience method that returns just the embedding vector,
        suitable for direct use in similarity search.

        Args:
            query: Search query text
            model: Optional model override

        Returns:
            Embedding vector as list of floats
        """
        result = await self.embed_text(query, model)
        return result.embedding

    async def embed_for_storage(
        self,
        text: str,
        model: Optional[str] = None,
    ) -> List[float]:
        """
        Generate embedding optimized for storage/indexing.

        This is a convenience method for embedding documents to be stored
        in a vector database.

        Args:
            text: Document text
            model: Optional model override

        Returns:
            Embedding vector as list of floats
        """
        result = await self.embed_text(text, model)
        return result.embedding

    async def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between -1 and 1
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimensions")

        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def __aenter__(self) -> "EmbeddingService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
