"""
Context Service

Central RAG entry point for retrieving context from various sources
including user history, art library, and similar prompts.

Documentation Reference: Section 6.3
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

from prisma import Prisma

from src.services.embedding_service import EmbeddingService, EmbeddingServiceError

logger = logging.getLogger(__name__)


class ContextServiceError(Exception):
    """Base exception for ContextService errors."""
    pass


@dataclass
class ArtItem:
    """Item from the art library."""
    id: str
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    image_url: Optional[str] = None
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "image_url": self.image_url,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
        }


@dataclass
class PromptReference:
    """Reference prompt from successful generations."""
    id: str
    prompt: str
    negative_prompt: Optional[str] = None
    style_tags: List[str] = field(default_factory=list)
    product_type: Optional[str] = None
    evaluation_score: float = 0.0
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "style_tags": self.style_tags,
            "product_type": self.product_type,
            "evaluation_score": self.evaluation_score,
            "similarity_score": self.similarity_score,
            "metadata": self.metadata,
        }


@dataclass
class DesignHistory:
    """User's design history item."""
    id: str
    job_id: str
    prompt: str
    image_url: Optional[str] = None
    product_type: Optional[str] = None
    style: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "prompt": self.prompt,
            "image_url": self.image_url,
            "product_type": self.product_type,
            "style": self.style,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class Context:
    """Aggregated context for generation."""
    user_history: List[DesignHistory]
    art_references: List[ArtItem]
    similar_prompts: List[PromptReference]
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Error tracking for drop point detection
    errors: List[str] = field(default_factory=list)
    partial_failure: bool = False  # True if some sources failed but others succeeded

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_history": [h.to_dict() for h in self.user_history],
            "art_references": [a.to_dict() for a in self.art_references],
            "similar_prompts": [p.to_dict() for p in self.similar_prompts],
            "metadata": self.metadata,
            "errors": self.errors,
            "partial_failure": self.partial_failure,
        }

    @property
    def has_context(self) -> bool:
        """Check if any context was found."""
        return bool(self.user_history or self.art_references or self.similar_prompts)

    @property
    def has_errors(self) -> bool:
        """Check if any errors occurred during retrieval."""
        return len(self.errors) > 0

    def to_prompt_context(self) -> str:
        """Format context for inclusion in prompts."""
        parts = []

        if self.user_history:
            history_items = [
                f"- {h.prompt} ({h.product_type or 'unknown type'})"
                for h in self.user_history[:3]
            ]
            parts.append("User's previous designs:\n" + "\n".join(history_items))

        if self.art_references:
            art_items = [
                f"- {a.name}: {a.description or 'No description'}"
                for a in self.art_references[:3]
            ]
            parts.append("Relevant art references:\n" + "\n".join(art_items))

        if self.similar_prompts:
            prompt_items = [
                f"- {p.prompt} (score: {p.evaluation_score:.2f})"
                for p in self.similar_prompts[:3]
            ]
            parts.append("Similar successful prompts:\n" + "\n".join(prompt_items))

        return "\n\n".join(parts) if parts else "No additional context available."


class ContextService:
    """
    Central service for context retrieval (RAG).

    Features:
    - User history retrieval from relational DB
    - Art library search using vector similarity
    - Similar prompt search using embeddings
    - Context aggregation and formatting
    """

    def __init__(
        self,
        prisma: Optional[Prisma] = None,
        embedding_service: Optional[EmbeddingService] = None,
    ):
        """
        Initialize ContextService.

        Args:
            prisma: Prisma client for relational DB access
            embedding_service: EmbeddingService for vector operations
        """
        self._prisma = prisma
        self._embedding_service = embedding_service
        self._owns_embedding_service = embedding_service is None

    async def _get_embedding_service(self) -> EmbeddingService:
        """Get or create embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    async def close(self) -> None:
        """Close the service and release resources."""
        if self._embedding_service and self._owns_embedding_service:
            await self._embedding_service.close()
            self._embedding_service = None

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10,
    ) -> List[DesignHistory]:
        """
        Get user's design history from relational DB.

        Args:
            user_id: User identifier
            limit: Maximum number of items to return

        Returns:
            List of DesignHistory items
        """
        if not self._prisma:
            logger.warning("Prisma client not available, returning empty history")
            return []

        try:
            # Query designs for user
            designs = await self._prisma.design.find_many(
                where={
                    "job": {
                        "userId": user_id,
                    },
                    "status": "approved",
                },
                include={"job": True},
                order_by={"createdAt": "desc"},
                take=limit,
            )

            return [
                DesignHistory(
                    id=d.id,
                    job_id=d.jobId,
                    prompt=d.prompt,
                    image_url=d.asset.imageUrl if hasattr(d, 'asset') and d.asset else None,
                    product_type=d.job.requirements.get("product_type") if d.job and d.job.requirements else None,
                    style=d.job.requirements.get("style") if d.job and d.job.requirements else None,
                    created_at=d.createdAt.isoformat() if d.createdAt else None,
                    metadata={"model_used": d.modelUsed},
                )
                for d in designs
            ]

        except Exception as e:
            logger.error(f"Failed to get user history: {e}")
            return []

    async def search_art_library(
        self,
        query: str,
        limit: int = 5,
        category: Optional[str] = None,
    ) -> List[ArtItem]:
        """
        Search art library using vector similarity.

        Args:
            query: Search query text
            limit: Maximum number of items to return
            category: Optional category filter

        Returns:
            List of ArtItem matches
        """
        if not self._prisma:
            logger.warning("Prisma client not available, returning empty results")
            return []

        try:
            embedding_service = await self._get_embedding_service()
            query_embedding = await embedding_service.embed_for_search(query)

            # Vector similarity search using raw SQL
            # Note: This requires pgvector extension
            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            category_filter = f"AND category = '{category}'" if category else ""

            results = await self._prisma.query_raw(
                f"""
                SELECT id, name, description, category, tags, image_url,
                       1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM art_library
                WHERE embedding IS NOT NULL
                {category_filter}
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT {limit}
                """
            )

            return [
                ArtItem(
                    id=str(r["id"]),
                    name=r["name"],
                    description=r.get("description"),
                    category=r.get("category"),
                    tags=r.get("tags", []),
                    image_url=r.get("image_url"),
                    similarity_score=float(r.get("similarity", 0)),
                )
                for r in results
            ]

        except EmbeddingServiceError as e:
            logger.error(f"Embedding service error: {e}")
            return []
        except Exception as e:
            logger.error(f"Art library search failed: {e}")
            return []

    async def search_similar_prompts(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> List[PromptReference]:
        """
        Search for similar successful prompts.

        Args:
            query: Search query (usually current prompt)
            limit: Maximum number of items to return
            min_score: Minimum evaluation score filter

        Returns:
            List of PromptReference matches
        """
        if not self._prisma:
            logger.warning("Prisma client not available, returning empty results")
            return []

        try:
            embedding_service = await self._get_embedding_service()
            query_embedding = await embedding_service.embed_for_search(query)

            embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

            score_filter = f"AND evaluation_score >= {min_score}" if min_score > 0 else ""

            results = await self._prisma.query_raw(
                f"""
                SELECT id, prompt, negative_prompt, style_tags, product_type,
                       evaluation_score,
                       1 - (embedding <=> '{embedding_str}'::vector) as similarity
                FROM prompt_embeddings
                WHERE embedding IS NOT NULL
                {score_filter}
                ORDER BY embedding <=> '{embedding_str}'::vector
                LIMIT {limit}
                """
            )

            return [
                PromptReference(
                    id=str(r["id"]),
                    prompt=r["prompt"],
                    negative_prompt=r.get("negative_prompt"),
                    style_tags=r.get("style_tags", []),
                    product_type=r.get("product_type"),
                    evaluation_score=float(r.get("evaluation_score", 0)),
                    similarity_score=float(r.get("similarity", 0)),
                )
                for r in results
            ]

        except EmbeddingServiceError as e:
            logger.error(f"Embedding service error: {e}")
            return []
        except Exception as e:
            logger.error(f"Similar prompt search failed: {e}")
            return []

    async def build_context(
        self,
        user_id: str,
        requirements: Dict[str, Any],
        query: Optional[str] = None,
    ) -> Context:
        """
        Build aggregated context for generation.

        This is the main entry point for RAG, combining:
        - User history
        - Art library references
        - Similar prompts

        Args:
            user_id: User identifier
            requirements: Generation requirements
            query: Search query (defaults to prompt from requirements)

        Returns:
            Context object with all retrieved information
        """
        query = query or requirements.get("prompt", "")
        product_type = requirements.get("product_type")
        errors: List[str] = []

        # Check Prisma availability upfront
        if not self._prisma:
            errors.append("Database unavailable: Prisma client not initialized")
            logger.warning("Prisma client not available for context retrieval")

        # Fetch all context sources concurrently
        import asyncio

        user_history_task = self.get_user_history(user_id, limit=5)
        art_library_task = self.search_art_library(
            query,
            limit=5,
            category=product_type,
        )
        similar_prompts_task = self.search_similar_prompts(
            query,
            limit=5,
            min_score=0.45,  # Only high-quality prompts
        )

        # Use return_exceptions to capture individual failures
        results = await asyncio.gather(
            user_history_task,
            art_library_task,
            similar_prompts_task,
            return_exceptions=True,
        )

        # Process results and track errors
        user_history = []
        art_references = []
        similar_prompts = []

        if isinstance(results[0], Exception):
            errors.append(f"User history retrieval failed: {results[0]}")
            logger.error(f"User history retrieval failed: {results[0]}")
        else:
            user_history = results[0]

        if isinstance(results[1], Exception):
            errors.append(f"Art library search failed: {results[1]}")
            logger.error(f"Art library search failed: {results[1]}")
        else:
            art_references = results[1]

        if isinstance(results[2], Exception):
            errors.append(f"Similar prompts search failed: {results[2]}")
            logger.error(f"Similar prompts search failed: {results[2]}")
        else:
            similar_prompts = results[2]

        # Determine if partial failure occurred
        partial_failure = len(errors) > 0 and (user_history or art_references or similar_prompts)

        return Context(
            user_history=user_history,
            art_references=art_references,
            similar_prompts=similar_prompts,
            metadata={
                "user_id": user_id,
                "query": query,
                "product_type": product_type,
            },
            errors=errors,
            partial_failure=partial_failure,
        )

    async def save_design_summary(
        self,
        job_id: str,
        user_id: str,
        summary: str,
        product_type: Optional[str] = None,
        style: Optional[str] = None,
    ) -> bool:
        """
        Save design summary with embedding for future RAG.

        This is called after successful generation to populate
        the design_summaries table.

        Args:
            job_id: Job identifier
            user_id: User identifier
            summary: Text summary of the design
            product_type: Product type
            style: Style used

        Returns:
            True if saved successfully
        """
        if not self._prisma:
            logger.warning("Prisma client not available, cannot save summary")
            return False

        try:
            embedding_service = await self._get_embedding_service()
            embedding = await embedding_service.embed_for_storage(summary)

            embedding_str = f"[{','.join(str(x) for x in embedding)}]"

            await self._prisma.execute_raw(
                f"""
                INSERT INTO design_summaries (job_id, user_id, summary, product_type, style, embedding)
                VALUES ('{job_id}', '{user_id}', '{summary}', '{product_type or ''}', '{style or ''}', '{embedding_str}'::vector)
                """
            )

            logger.info(f"Saved design summary for job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to save design summary: {e}")
            return False

    async def __aenter__(self) -> "ContextService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
