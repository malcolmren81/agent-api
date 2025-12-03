# LLM Model Selection Guide

## Overview

This document defines the LLM model configuration for each agent in the Palet8 Agent System.
Each agent has specific requirements based on its role in the pipeline.

---

## 1. Pali Agent (User-Facing Conversational)

**Role:** Handles direct user interaction, gathers requirements through conversation.

**Primary Model:** Grok 4 Fast (xAI)
**Fallback Model:** Gemini 2.5 Flash Preview 09-2025 (Google)
**Temperature:** 0.7
**Max Tokens:** 1000

| LLM model | Context window | Disable reasoning | Input price (per 1M tokens) | Output price (per 1M tokens) | Instruction following | Multi-turn memory | Long-query handling |
|-----------|----------------|-------------------|-----------------------------|-----------------------------|----------------------|-------------------|---------------------|
| **Grok 4 Fast (xAI)** | 2 M tokens | Yes – can toggle with the `reasoning.enabled` parameter | US$0.20 | US$0.50 | Good – ranks #8 in LMArena's text arena | Good – long context allows sustained conversation | Excellent – 2 M tokens support very long inputs |
| **Gemini 2.5 Flash Preview 09-2025 (Google)** | 1,048,576 tokens | Partially – uses a thinking budget; you can adjust reasoning tokens but not fully disable | US$0.30 | US$2.50 | Good – preview checkpoint is positioned for advanced reasoning and structured outputs | Good – can handle standard conversations with its 1 M token context | High – 1 M tokens allow long inputs but less than Grok 4 Fast |

---

## 2. Planner Agent (Reasoning & Prompt Construction)

**Role:** Transforms creative briefs into generation plans, constructs prompts, handles refinements.

**Primary Model:** Claude Haiku 4.5 (Anthropic)
**Fallback Model:** Kimi K2 Thinking (Moonshot)
**Temperature:** 0.3
**Max Tokens:** 2000

| LLM model | Context window | Disable reasoning | Input price (per 1M tokens) | Output price (per 1M tokens) | Instruction following | Multi-turn memory | Long-query handling |
|-----------|----------------|-------------------|-----------------------------|-----------------------------|----------------------|-------------------|---------------------|
| **Claude Haiku 4.5** (Oct 2025) | 200,000 tokens | Partial – introduces controllable reasoning depth and summarised thought output | US$1.00 | US$5.00 | High – frontier-level capability with tool use and coding | Good – 200k context supports multi-turn | Good – efficient for real-time tasks; output still expensive |
| **Kimi K2 Thinking** (Nov 2025) | 262,144 tokens | Yes – persistent step-by-step reasoning is exposed; users can request thought traces or disable them via API settings | US$0.45 | US$2.35 | High – designed for long-horizon reasoning and complex tool use | Good – 256k context supports many turns | High – excellent for autonomous agents; output cost slightly above $2 |

---

## 3. Evaluator Agent (Vision + Quality Assessment)

**Role:** Quality gate for generation - evaluates prompts before generation and images after.

**Primary Model:** Qwen3 VL 30B A3B Instruct (Alibaba)
**Fallback Model:** GPT-4o Mini (OpenAI)
**Temperature:** 0.2
**Max Tokens:** 800

| LLM model (provider) | Context window | Vision & reasoning features | Reasoning disable/control | Input price (per 1M tokens) | Output price (per 1M tokens) | Suitability for evaluation agent |
|----------------------|----------------|----------------------------|---------------------------|-----------------------------|-----------------------------|----------------------------------|
| **Qwen3 VL 30B A3B Instruct** (Alibaba) | **262 K tokens** | Multimodal model that unifies strong text generation with visual understanding; the **Instruct variant** excels at instruction-following, perception of real/synthetic categories, 2D/3D spatial grounding and long-form visual comprehension. Handles multi-image multi-turn instructions, video timeline alignments, GUI automation and visual coding. | Reasoning toggle available | **$0.15** | **$0.60** | Well-rounded VL model for evaluation tasks; cost meets requirement. |
| **GPT-4o Mini** (OpenAI) | **128 K tokens** | Multimodal model with strong visual and reasoning capabilities | Reasoning can be enabled or disabled via `reasoning` parameter | **$0.15** | **$0.60** | Balanced option for image evaluation and scoring. |

---

## 4. Safety Agent (Vision + Content Monitoring)

**Role:** Monitors for unsafe content and IP risks across text and images.

**Primary Model:** Qwen3 VL 30B A3B Instruct (Alibaba)
**Fallback Model:** GPT-4o Mini (OpenAI)
**Temperature:** 0.0
**Max Tokens:** 500

| LLM model (provider) | Context window | Vision & reasoning features | Reasoning disable/control | Input price (per 1M tokens) | Output price (per 1M tokens) | Suitability for safety agent |
|----------------------|----------------|----------------------------|---------------------------|-----------------------------|-----------------------------|------------------------------|
| **Qwen3 VL 30B A3B Instruct** (Alibaba) | **262 K tokens** | Multimodal model that unifies strong text generation with visual understanding; the **Instruct variant** excels at instruction-following, perception of real/synthetic categories, 2D/3D spatial grounding and long-form visual comprehension. Handles multi-image multi-turn instructions, video timeline alignments, GUI automation and visual coding. | Reasoning toggle available | **$0.15** | **$0.60** | Well-rounded VL model for content safety scanning; cost-effective. |
| **GPT-4o Mini** (OpenAI) | **128 K tokens** | Multimodal model with strong visual and reasoning capabilities | Reasoning can be enabled or disabled via `reasoning` parameter | **$0.15** | **$0.60** | Reliable fallback for safety classification. |

---

## 5. Embedding Models (Vector DB Indexing & Search)

**Role:** Convert text and images to vector embeddings for semantic search and RAG retrieval.

### Text Embedding

**Model:** OpenAI text-embedding-3-small
**Use Case:** Text content indexing, prompt similarity search, user history embedding

| Model | Dimensions | Max Tokens | Price (per 1M tokens) | Performance |
|-------|------------|------------|----------------------|-------------|
| **text-embedding-3-small** (OpenAI) | 1536 | 8191 | $0.02 | High quality, cost-effective for most use cases |

### Image Embedding

**Model:** Qwen3 Embedding 8B
**Use Case:** Image content indexing, visual similarity search, art reference matching

| Model | Dimensions | Modality | Price (per 1M tokens) | Performance |
|-------|------------|----------|----------------------|-------------|
| **qwen3-embedding-8b** (Qwen) | 4096 | Text + Image | TBD | Multimodal embedding for visual content |

---

## Summary Table

| Agent/Service | Primary | Fallback | Temp | Tokens | Key Requirement |
|---------------|---------|----------|------|--------|-----------------|
| Pali | Grok 4 Fast (xAI) | Gemini 2.5 Flash Preview 09-2025 | 0.7 | 1000 | Fast, conversational |
| Planner | Claude Haiku 4.5 | Kimi K2 Thinking | 0.3 | 2000 | Reasoning/thinking |
| Evaluator | Qwen3 VL 30B A3B Instruct | GPT-4o Mini | 0.2 | 800 | Vision, evaluation |
| Safety | Qwen3 VL 30B A3B Instruct | GPT-4o Mini | 0.0 | 500 | Vision, safety scanning |
| **Composer** | Claude Sonnet 4.5 | Claude Haiku 4.5 | 0.4 | 1000 | Creative prompt writing |
| **Search** | GPT-4o-mini Search Preview | Tavily API | 0.0 | 2000 | Web search |
| **Text Embedding** | text-embedding-3-small | - | - | - | Vector DB text indexing |
| **Image Embedding** | qwen3-embedding-8b | - | - | - | Vector DB image indexing |

---

## OpenRouter Model IDs

For configuration in `agent_routing_policy.yaml`:

```yaml
# Pali
primary: "x-ai/grok-4-fast"
fallback: "google/gemini-2.5-flash-preview-09-25"

# Planner
primary: "anthropic/claude-haiku-4.5"
fallback: "moonshotai/kimi-k2-thinking"

# Evaluator
primary: "alibaba/qwen3-vl-30b-a3b-instruct"
fallback: "openai/gpt-4o-mini"

# Safety
primary: "alibaba/qwen3-vl-30b-a3b-instruct"
fallback: "openai/gpt-4o-mini"

# Composer (Prompt Composer Service)
primary: "anthropic/claude-sonnet-4.5"
fallback: "anthropic/claude-haiku-4.5"

# Search (Web Search Service)
primary: "openai/gpt-4o-mini-search-preview"
fallback: "tavily"  # Tavily API fallback

# Embedding (Vector DB)
text_embedding: "openai/text-embedding-3-small"
image_embedding: "qwen/qwen3-embedding-8b"
```

---

*Last Updated: December 2025*
