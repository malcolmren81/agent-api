-- ============================================================================
-- Palet8 Agent System - Vector Database Schema
-- Phase 1: Foundation setup for RAG functionality
-- ============================================================================
--
-- Purpose: Art library embeddings, prompt embeddings, design summaries for RAG
-- Database: PostgreSQL with pgvector extension
--
-- Configuration (aligned with agent_routing_policy.yaml):
-- - Text embedding: Google gemini-embedding-001 (768 dimensions)
-- - Image embedding: Google multimodalembedding@001 (1408 dimensions)
-- - Index type: ivfflat (switch to hnsw when >100k vectors)
-- ============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================================
-- Art Library Table
-- Purpose: Store art references with embeddings for similarity search
-- ============================================================================

CREATE TABLE IF NOT EXISTS art_library (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Basic info
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    tags TEXT[],

    -- Image reference
    image_url TEXT,
    thumbnail_url TEXT,

    -- Vector embedding (1408 dimensions for Google multimodalembedding@001)
    embedding vector(1408),

    -- Metadata
    metadata JSONB,
    source VARCHAR(100),  -- Where this art came from (internal, stock, etc.)
    license VARCHAR(100),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for similarity search (ivfflat for <100k vectors)
CREATE INDEX IF NOT EXISTS art_library_embedding_idx
ON art_library USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Additional indexes for filtering
CREATE INDEX IF NOT EXISTS art_library_category_idx ON art_library(category);
CREATE INDEX IF NOT EXISTS art_library_tags_idx ON art_library USING gin(tags);
CREATE INDEX IF NOT EXISTS art_library_created_idx ON art_library(created_at);

-- ============================================================================
-- Prompt Embeddings Table
-- Purpose: Store successful prompts with embeddings for RAG retrieval
-- ============================================================================

CREATE TABLE IF NOT EXISTS prompt_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Reference to job/task
    job_id VARCHAR(255),
    user_id VARCHAR(255),

    -- Prompt content
    prompt TEXT NOT NULL,
    negative_prompt TEXT,

    -- Categorization
    style_tags TEXT[],
    product_type VARCHAR(100),
    style VARCHAR(100),

    -- Quality metrics (from evaluation)
    evaluation_score FLOAT,

    -- Vector embedding (768 dimensions for Google gemini-embedding-001)
    embedding vector(768),

    -- Full metadata
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for similarity search
CREATE INDEX IF NOT EXISTS prompt_embedding_idx
ON prompt_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Additional indexes
CREATE INDEX IF NOT EXISTS prompt_user_idx ON prompt_embeddings(user_id);
CREATE INDEX IF NOT EXISTS prompt_product_type_idx ON prompt_embeddings(product_type);
CREATE INDEX IF NOT EXISTS prompt_style_idx ON prompt_embeddings(style);
CREATE INDEX IF NOT EXISTS prompt_score_idx ON prompt_embeddings(evaluation_score);
CREATE INDEX IF NOT EXISTS prompt_created_idx ON prompt_embeddings(created_at);

-- ============================================================================
-- Design Summaries Table
-- Purpose: Store final task summaries with embeddings for user history RAG
-- ============================================================================

CREATE TABLE IF NOT EXISTS design_summaries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- References
    job_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,

    -- Summary content
    summary TEXT NOT NULL,
    title VARCHAR(255),

    -- Categorization
    product_type VARCHAR(100),
    style VARCHAR(100),
    tags TEXT[],

    -- Generated content references
    final_prompt TEXT,
    image_url TEXT,
    thumbnail_url TEXT,

    -- Vector embedding (768 dimensions for Google gemini-embedding-001)
    embedding vector(768),

    -- Full metadata
    metadata JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for similarity search
CREATE INDEX IF NOT EXISTS design_summary_embedding_idx
ON design_summaries USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Additional indexes
CREATE INDEX IF NOT EXISTS design_summary_user_idx ON design_summaries(user_id);
CREATE INDEX IF NOT EXISTS design_summary_job_idx ON design_summaries(job_id);
CREATE INDEX IF NOT EXISTS design_summary_product_type_idx ON design_summaries(product_type);
CREATE INDEX IF NOT EXISTS design_summary_style_idx ON design_summaries(style);
CREATE INDEX IF NOT EXISTS design_summary_created_idx ON design_summaries(created_at);

-- ============================================================================
-- Helper Functions
-- ============================================================================

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for art_library updated_at
DROP TRIGGER IF EXISTS update_art_library_updated_at ON art_library;
CREATE TRIGGER update_art_library_updated_at
    BEFORE UPDATE ON art_library
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Sample Similarity Search Queries (for reference)
-- ============================================================================

-- Example: Search art library for similar items
-- SELECT id, name, description, 1 - (embedding <=> $1::vector) as similarity
-- FROM art_library
-- WHERE embedding IS NOT NULL
-- ORDER BY embedding <=> $1::vector
-- LIMIT 10;

-- Example: Search prompts by similarity with filters
-- SELECT id, prompt, evaluation_score, 1 - (embedding <=> $1::vector) as similarity
-- FROM prompt_embeddings
-- WHERE product_type = 'tshirt'
--   AND evaluation_score > 0.7
--   AND embedding IS NOT NULL
-- ORDER BY embedding <=> $1::vector
-- LIMIT 5;

-- Example: Get user's recent similar designs
-- SELECT id, summary, title, 1 - (embedding <=> $1::vector) as similarity
-- FROM design_summaries
-- WHERE user_id = $2
--   AND embedding IS NOT NULL
-- ORDER BY embedding <=> $1::vector
-- LIMIT 10;

-- ============================================================================
-- Maintenance Notes
-- ============================================================================
--
-- Index Maintenance:
-- - ivfflat indexes need reindexing as data grows significantly
-- - Consider switching to hnsw for datasets > 100k vectors
-- - REINDEX INDEX CONCURRENTLY art_library_embedding_idx;
--
-- Vacuuming:
-- - Regular VACUUM ANALYZE recommended for vector tables
-- - VACUUM ANALYZE art_library;
-- - VACUUM ANALYZE prompt_embeddings;
-- - VACUUM ANALYZE design_summaries;
