# Phase 2 Complete - Agent System Restructure

**Date:** December 2, 2025
**Status:** Complete

---

## Overview

Phase 2 restructured the Palet8 Agent System with clear separation of concerns, fixed workflow issues, and consolidated the codebase into a clean architecture.

---

## Architecture

```
agents-api/
├── palet8_agents/              # Agent Business Logic
│   ├── agents/                 # Agent implementations
│   │   ├── pali_agent.py       # User-facing conversational agent
│   │   ├── planner_agent.py    # Planning, model selection, pipeline decisions
│   │   ├── evaluator_agent.py  # Quality evaluation (prompt + result)
│   │   └── safety_agent.py     # Content safety monitoring
│   │
│   ├── core/                   # Core framework
│   │   ├── agent.py            # BaseAgent class
│   │   ├── config.py           # Configuration loading
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── llm_client.py       # LLM client wrapper
│   │   └── message.py          # Message types
│   │
│   ├── services/               # Business services
│   │   ├── text_llm_service.py
│   │   ├── reasoning_service.py
│   │   ├── context_service.py      # RAG context retrieval
│   │   ├── embedding_service.py    # Google Vertex AI embeddings
│   │   ├── web_search_service.py
│   │   ├── model_info_service.py
│   │   ├── prompt_template_service.py
│   │   ├── prompt_composer_service.py
│   │   └── image_generation_service.py
│   │
│   └── tools/                  # Agent tools
│       ├── context_tool.py
│       ├── image_tool.py
│       ├── job_tool.py
│       └── search_tool.py
│
├── src/                        # API Layer
│   ├── api/                    # FastAPI routes, orchestrator
│   ├── connectors/             # External API connectors
│   ├── database/               # Prisma client
│   ├── models/                 # Schemas
│   └── utils/                  # Logger, metrics
│
├── config/                     # Configuration files
│   ├── agent_routing_policy.yaml
│   ├── image_models_config.yaml
│   ├── safety_config.yaml
│   └── evaluation_config.yaml
│
└── docs/                       # Documentation
    ├── Palet8 Agent Swimlane.drawio.html
    ├── Palet8-Agent-System-Restructure-Plan.md
    ├── llm_model_selection.md
    └── archive/                # Archived old files
```

---

## Key Changes

### 1. Agent Workflow (Matches Swimlane)

```
User → Pali Agent → Planner Agent → [Enough Context?]
                         ↓ NO
                    Pali Agent (clarify) → back to Planner
                         ↓ YES
                    Knowledge Acquire (RAG + Search)
                         ↓
                    Model Selection + Pipeline Decision
                         ↓
                    Prompt Compose → Evaluate Quality
                         ↓
                    Evaluator Agent → [Pass?]
                         ↓ NO
                    Planner Fix Plan → retry
                         ↓ YES
                    Pali Agent → User

Safety Agent runs continuously as "Stay Live Service"
```

### 2. Completeness Check (Fixed Reverse Logic)

**Before:** Both Pali and Planner had independent completeness checks causing ping-pong.

**After:**
- **Pali:** Minimal check - only `subject is not None`
- **Planner:** Thorough "Enough Context?" evaluation with weighted fields

### 3. RAG Error Handling (Fixed Drop Point)

**Before:** RAG silently returned empty results on failure.

**After:** Context dataclass tracks errors:
```python
@dataclass
class Context:
    user_history: List[DesignHistory]
    art_references: List[ArtItem]
    similar_prompts: List[PromptReference]
    errors: List[str] = field(default_factory=list)
    partial_failure: bool = False
```

### 4. Pipeline Decision (Single vs Dual Model)

Planner decides based on **task scope**, not mode:

| Trigger | Pipeline | Stage 1 | Stage 2 |
|---------|----------|---------|---------|
| Text in image | Dual | Midjourney/FLUX | Nano Banana 2 Pro |
| Character refinement | Dual | Generator | Editor |
| Multi-element | Dual | Based on style | Editor |
| Simple request | Single | Selected model | - |

**Three Dual Pipelines:**
1. `creative_art`: Midjourney V7 → Nano Banana 2 Pro
2. `photorealistic`: Imagen 4 Ultra → Nano Banana 2 Pro
3. `layout_poster`: FLUX.2 Flex → Qwen Image Edit

### 5. Safety Removed from Evaluator

- Evaluator focuses on **quality only** (prompt fidelity, technical, aesthetic)
- Safety Agent handles all safety checks as standalone "Stay Live Service"

### 6. Embedding Models (Google Vertex AI)

| Type | Model | Dimensions | Use Case |
|------|-------|------------|----------|
| Text | gemini-embedding-001 | 768 | Prompts, summaries, user history |
| Image | multimodalembedding@001 | 1408 | Art library, generated images |

### 7. Prompt Composer Rules

```
- Use structural language
- One point per sentence
- No complex sentences
- All info provided by Planner Agent
- Scenario-specific rules (layout, text, character, photorealistic)
```

---

## Files Modified

### Agents
- `palet8_agents/agents/pali_agent.py` - Minimal completeness check
- `palet8_agents/agents/planner_agent.py` - Pipeline decision, thorough context check
- `palet8_agents/agents/evaluator_agent.py` - Safety removed, quality focus
- `palet8_agents/agents/safety_agent.py` - Standalone safety monitoring

### Services
- `palet8_agents/services/context_service.py` - Error tracking added
- `palet8_agents/services/prompt_composer_service.py` - Structural rules
- `palet8_agents/services/embedding_service.py` - Google Vertex AI config

### Config
- `config/agent_routing_policy.yaml` - Google embeddings, model profiles
- `scripts/vector_db_setup.sql` - Updated dimensions (768/1408)

---

## Archived Files

Moved to `docs/archive/`:

**Root cleanup:**
- Old Dockerfiles, test scripts, migration scripts
- ADK installation docs, phase completion docs

**Old src:**
- `src/agents/` - Old agent implementations
- `src/services/` - Old service copies
- `src/config/` - Old config

---

## Next Steps (Phase 3+)

1. **User Credit Integration** - Pipeline cost affects model selection
2. **Orchestrator Update** - Handle dual pipeline execution
3. **Testing** - End-to-end workflow tests
4. **Monitoring** - Pipeline metrics and cost tracking

---

## Reference Documents

- `docs/Palet8 Agent Swimlane.drawio.html` - Visual workflow
- `docs/Palet8-Agent-System-Restructure-Plan.md` - Original plan
- `docs/llm_model_selection.md` - Model selection logic
- `config/image_models_config.yaml` - Full model registry
