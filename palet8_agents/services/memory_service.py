"""
Memory Service

Long-term memory storage and retrieval using PostgreSQL with pgvector.
Provides RAG functionality for design summaries, prompt embeddings, and art library.

Documentation Reference: Section 4.2 (Embedding Service), vector_db_setup.sql
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
import logging
import os

from palet8_agents.services.embedding_service import EmbeddingService, EmbeddingServiceError

logger = logging.getLogger(__name__)


class MemoryServiceError(Exception):
    """Base exception for MemoryService errors."""
    pass


class DatabaseConnectionError(MemoryServiceError):
    """Raised when database connection fails."""
    pass


class StorageError(MemoryServiceError):
    """Raised when storage operation fails."""
    pass


class RetrievalError(MemoryServiceError):
    """Raised when retrieval operation fails."""
    pass


@dataclass
class MemorySearchResult:
    """Result from a memory search operation."""
    id: str
    content: str
    similarity: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "similarity": self.similarity,
            "metadata": self.metadata,
        }


class MemoryService:
    """
    Long-term memory storage using PostgreSQL with pgvector.

    This service provides:
    - Storage and retrieval of design summaries (user history)
    - Storage and retrieval of successful prompts (RAG for prompt generation)
    - Search of art library references
    - Vector similarity search across all memory types

    Tables used (from vector_db_setup.sql):
    - design_summaries: User design history with embeddings
    - prompt_embeddings: Successful prompts for RAG
    - art_library: Art references with image embeddings
    """

    # Default embedding dimensions (from config)
    TEXT_EMBEDDING_DIM = 768  # gemini-embedding-001
    IMAGE_EMBEDDING_DIM = 1408  # multimodalembedding@001

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        db_url: Optional[str] = None,
    ):
        """
        Initialize MemoryService.

        Args:
            embedding_service: EmbeddingService for generating embeddings.
                             Creates one if not provided.
            db_url: PostgreSQL connection URL. Reads from DATABASE_URL env if not provided.
        """
        self._embedding_service = embedding_service
        self._owns_embedding_service = embedding_service is None
        self._db_url = db_url or os.environ.get("DATABASE_URL")
        self._pool = None  # asyncpg pool, initialized on first use

    async def _get_pool(self):
        """Get or create connection pool (lazy initialization)."""
        if self._pool is None:
            try:
                import asyncpg
                self._pool = await asyncpg.create_pool(self._db_url)
            except ImportError:
                logger.error("asyncpg not installed. Install with: pip install asyncpg")
                raise DatabaseConnectionError("asyncpg package not installed")
            except Exception as e:
                logger.error(f"Failed to create database pool: {e}")
                raise DatabaseConnectionError(f"Database connection failed: {e}")
        return self._pool

    async def _get_embedding_service(self) -> EmbeddingService:
        """Get or create embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    async def close(self) -> None:
        """Close the service and release resources."""
        if self._pool:
            await self._pool.close()
            self._pool = None

        if self._embedding_service and self._owns_embedding_service:
            await self._embedding_service.close()
            self._embedding_service = None

    # =========================================================================
    # Design Summaries (User History)
    # =========================================================================

    async def store_design_summary(
        self,
        job_id: str,
        user_id: str,
        summary: str,
        prompt: str = "",
        title: Optional[str] = None,
        product_type: Optional[str] = None,
        style: Optional[str] = None,
        image_url: Optional[str] = None,
        thumbnail_url: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a design summary with embedding for later retrieval.

        Args:
            job_id: Job identifier
            user_id: User identifier
            summary: Design summary text
            prompt: Final prompt used
            title: Optional title
            product_type: Product type (e.g., 'tshirt', 'poster')
            style: Design style
            image_url: URL to generated image
            thumbnail_url: URL to thumbnail
            tags: List of tags
            metadata: Additional metadata

        Returns:
            ID of stored summary

        Raises:
            StorageError: If storage fails
        """
        try:
            # Generate embedding
            embedding_service = await self._get_embedding_service()
            embedding = await embedding_service.embed_for_storage(summary)

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchrow(
                    """
                    INSERT INTO design_summaries
                    (job_id, user_id, summary, title, product_type, style,
                     final_prompt, image_url, thumbnail_url, tags, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                    RETURNING id
                    """,
                    job_id, user_id, summary, title, product_type, style,
                    prompt, image_url, thumbnail_url, tags or [],
                    embedding, json.dumps(metadata or {}),
                )
                return str(result["id"])

        except EmbeddingServiceError as e:
            raise StorageError(f"Failed to generate embedding: {e}")
        except Exception as e:
            logger.error(f"Failed to store design summary: {e}")
            raise StorageError(f"Storage failed: {e}")

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Get recent design history for a user.

        Args:
            user_id: User identifier
            limit: Maximum results
            offset: Pagination offset

        Returns:
            List of design summary dicts
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, job_id, summary, title, product_type, style,
                           image_url, thumbnail_url, tags, created_at
                    FROM design_summaries
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    user_id, limit, offset,
                )
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get user history: {e}")
            raise RetrievalError(f"Retrieval failed: {e}")

    async def search_user_history(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> List[MemorySearchResult]:
        """
        Search user's design history by similarity.

        Args:
            query: Search query text
            user_id: User identifier
            limit: Maximum results

        Returns:
            List of MemorySearchResult
        """
        try:
            embedding_service = await self._get_embedding_service()
            query_embedding = await embedding_service.embed_for_search(query)

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, summary, title, product_type, style,
                           image_url, created_at,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM design_summaries
                    WHERE user_id = $2
                      AND embedding IS NOT NULL
                    ORDER BY embedding <=> $1::vector
                    LIMIT $3
                    """,
                    query_embedding, user_id, limit,
                )

                return [
                    MemorySearchResult(
                        id=str(row["id"]),
                        content=row["summary"],
                        similarity=float(row["similarity"]),
                        metadata={
                            "title": row["title"],
                            "product_type": row["product_type"],
                            "style": row["style"],
                            "image_url": row["image_url"],
                            "created_at": str(row["created_at"]) if row["created_at"] else None,
                        },
                    )
                    for row in rows
                ]

        except EmbeddingServiceError as e:
            raise RetrievalError(f"Failed to generate query embedding: {e}")
        except Exception as e:
            logger.error(f"Failed to search user history: {e}")
            raise RetrievalError(f"Search failed: {e}")

    # =========================================================================
    # Prompt Embeddings (RAG for Prompt Generation)
    # =========================================================================

    async def store_prompt(
        self,
        job_id: str,
        user_id: str,
        prompt: str,
        negative_prompt: str = "",
        product_type: Optional[str] = None,
        style: Optional[str] = None,
        style_tags: Optional[List[str]] = None,
        evaluation_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Store a successful prompt with embedding for RAG.

        Args:
            job_id: Job identifier
            user_id: User identifier
            prompt: Prompt text
            negative_prompt: Negative prompt
            product_type: Product type
            style: Design style
            style_tags: Style tags
            evaluation_score: Quality score from evaluation
            metadata: Additional metadata

        Returns:
            ID of stored prompt
        """
        try:
            embedding_service = await self._get_embedding_service()
            embedding = await embedding_service.embed_for_storage(prompt)

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                result = await conn.fetchrow(
                    """
                    INSERT INTO prompt_embeddings
                    (job_id, user_id, prompt, negative_prompt, product_type,
                     style, style_tags, evaluation_score, embedding, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    RETURNING id
                    """,
                    job_id, user_id, prompt, negative_prompt, product_type,
                    style, style_tags or [], evaluation_score,
                    embedding, json.dumps(metadata or {}),
                )
                return str(result["id"])

        except EmbeddingServiceError as e:
            raise StorageError(f"Failed to generate embedding: {e}")
        except Exception as e:
            logger.error(f"Failed to store prompt: {e}")
            raise StorageError(f"Storage failed: {e}")

    async def search_similar_prompts(
        self,
        query: str,
        user_id: Optional[str] = None,
        product_type: Optional[str] = None,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> List[MemorySearchResult]:
        """
        Search for similar prompts using vector similarity.

        Args:
            query: Search query text
            user_id: Optional filter by user
            product_type: Optional filter by product type
            min_score: Minimum evaluation score
            limit: Maximum results

        Returns:
            List of MemorySearchResult
        """
        try:
            embedding_service = await self._get_embedding_service()
            query_embedding = await embedding_service.embed_for_search(query)

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                # Build query with optional filters
                sql = """
                    SELECT id, prompt, negative_prompt, style, product_type,
                           evaluation_score,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM prompt_embeddings
                    WHERE embedding IS NOT NULL
                      AND evaluation_score >= $2
                """
                params = [query_embedding, min_score]
                param_idx = 3

                if user_id:
                    sql += f" AND user_id = ${param_idx}"
                    params.append(user_id)
                    param_idx += 1

                if product_type:
                    sql += f" AND product_type = ${param_idx}"
                    params.append(product_type)
                    param_idx += 1

                sql += f" ORDER BY embedding <=> $1::vector LIMIT ${param_idx}"
                params.append(limit)

                rows = await conn.fetch(sql, *params)

                return [
                    MemorySearchResult(
                        id=str(row["id"]),
                        content=row["prompt"],
                        similarity=float(row["similarity"]),
                        metadata={
                            "negative_prompt": row["negative_prompt"],
                            "style": row["style"],
                            "product_type": row["product_type"],
                            "evaluation_score": row["evaluation_score"],
                        },
                    )
                    for row in rows
                ]

        except EmbeddingServiceError as e:
            raise RetrievalError(f"Failed to generate query embedding: {e}")
        except Exception as e:
            logger.error(f"Failed to search prompts: {e}")
            raise RetrievalError(f"Search failed: {e}")

    # =========================================================================
    # Art Library (Reference Images)
    # =========================================================================

    async def get_art_references(
        self,
        query: str,
        category: Optional[str] = None,
        limit: int = 10,
    ) -> List[MemorySearchResult]:
        """
        Search art library for reference images.

        Args:
            query: Search query text
            category: Optional category filter
            limit: Maximum results

        Returns:
            List of MemorySearchResult
        """
        try:
            embedding_service = await self._get_embedding_service()
            query_embedding = await embedding_service.embed_for_search(query)

            pool = await self._get_pool()
            async with pool.acquire() as conn:
                sql = """
                    SELECT id, name, description, category, tags,
                           image_url, thumbnail_url,
                           1 - (embedding <=> $1::vector) as similarity
                    FROM art_library
                    WHERE embedding IS NOT NULL
                """
                params = [query_embedding]
                param_idx = 2

                if category:
                    sql += f" AND category = ${param_idx}"
                    params.append(category)
                    param_idx += 1

                sql += f" ORDER BY embedding <=> $1::vector LIMIT ${param_idx}"
                params.append(limit)

                rows = await conn.fetch(sql, *params)

                return [
                    MemorySearchResult(
                        id=str(row["id"]),
                        content=row["description"] or row["name"],
                        similarity=float(row["similarity"]),
                        metadata={
                            "name": row["name"],
                            "category": row["category"],
                            "tags": row["tags"],
                            "image_url": row["image_url"],
                            "thumbnail_url": row["thumbnail_url"],
                        },
                    )
                    for row in rows
                ]

        except EmbeddingServiceError as e:
            raise RetrievalError(f"Failed to generate query embedding: {e}")
        except Exception as e:
            logger.error(f"Failed to search art library: {e}")
            raise RetrievalError(f"Search failed: {e}")

    async def get_art_by_category(
        self,
        category: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Get art references by category (no similarity search).

        Args:
            category: Category to filter by
            limit: Maximum results

        Returns:
            List of art reference dicts
        """
        try:
            pool = await self._get_pool()
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT id, name, description, category, tags,
                           image_url, thumbnail_url, created_at
                    FROM art_library
                    WHERE category = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                    """,
                    category, limit,
                )
                return [dict(row) for row in rows]

        except Exception as e:
            logger.error(f"Failed to get art by category: {e}")
            raise RetrievalError(f"Retrieval failed: {e}")

    # =========================================================================
    # Combined RAG Search
    # =========================================================================

    async def search_all(
        self,
        query: str,
        user_id: Optional[str] = None,
        limit_per_type: int = 3,
    ) -> Dict[str, List[MemorySearchResult]]:
        """
        Search across all memory types for RAG context.

        Args:
            query: Search query
            user_id: Optional user ID for history search
            limit_per_type: Max results per memory type

        Returns:
            Dict with 'prompts', 'history', 'art' keys
        """
        results = {
            "prompts": [],
            "history": [],
            "art": [],
        }

        try:
            # Search prompts
            results["prompts"] = await self.search_similar_prompts(
                query=query,
                limit=limit_per_type,
            )
        except RetrievalError as e:
            logger.warning(f"Prompt search failed: {e}")

        if user_id:
            try:
                # Search user history
                results["history"] = await self.search_user_history(
                    query=query,
                    user_id=user_id,
                    limit=limit_per_type,
                )
            except RetrievalError as e:
                logger.warning(f"History search failed: {e}")

        try:
            # Search art library
            results["art"] = await self.get_art_references(
                query=query,
                limit=limit_per_type,
            )
        except RetrievalError as e:
            logger.warning(f"Art search failed: {e}")

        return results

    async def __aenter__(self) -> "MemoryService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
