# Palet8 Agent System Restructure - Development Plan

> **Version:** 3.1
> **Scope:** Backend service restructure aligned with agent-api Development Documentation v0.4
> **Strategy:** Rewrite from scratch following HelloAgents patterns
> **Deploy Target:** GCP Cloud Run (palet8-agents)
> **Transition:** Delete old `src/agents/` after Phase 3 validation
> **Last Updated:** 2025-12-03

---

## Current Status: PHASE 2-3 COMPLETE

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Structure & Foundation | ✅ COMPLETE | All core components, models, configs created |
| Phase 2: Agent & Service Implementation | ✅ COMPLETE | All agents, services, tools implemented |
| Phase 3: Deploy & Test | ✅ COMPLETE | Deployed to Cloud Run (revision 175) |

### Deployment Details
- **Service URL:** `https://palet8-agents-kshhjydolq-uc.a.run.app/`
- **Current Revision:** `palet8-agents-00175-2zh` (100% traffic)
- **Status:** Running and healthy

### Implementation Summary (PRs Completed)
| PR | Description | Status |
|----|-------------|--------|
| PR 1 | Models Package + Configs | ✅ Complete |
| PR 2 | New Services (Part 1 - Core) | ✅ Complete |
| PR 3 | New Services (Part 2 - Evaluation & Selection) | ✅ Complete |
| PR 4 | Tools + Registry | ✅ Complete |
| PR 5-9 | Agent Refactoring (Planner, Evaluator, Pali, Safety) | ✅ Complete |
| PR 10 | AssemblyService + Integration | ✅ Complete |

### Key Changes
- **AssemblyService:** Single/dual pipeline execution with Runware API
- **Removed:** Flux API (all generation now uses Runware only)
- **Fixed:** `src/api/routes/tasks.py` import error (task_aggregator commented out)

---

## Executive Summary

This plan restructures the existing agent-api from `src/agents/` structure to the documented `palet8_agents/` architecture, implementing a clean multi-agent framework with dedicated model-facing services. The rewrite follows HelloAgents reference patterns while maintaining compatibility with existing integrations.

---

## Swimlane Flow Reference (from Palet8 Agent Swimlane.drawio)

```
User → palet8-customer-app → palet8-agents → AWS Profilebackend
                ↓                   ↓
        palet8-admin-api      Databases (Relational + Vector)

Agent Flow (with Concurrent Execution):
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ║ SAFETY AGENT - RUNS ALONGSIDE ENTIRE TASK (Never a bottleneck) ║    │
│  ║ Continuous monitoring, non-blocking, parallel to all steps     ║    │
│                                                                         │
│  Pali Agent ←──────────────────────────────────────────────────────┐    │
│      ↓                                                              │    │
│  ┌─────────────────────────────────────────┐                        │    │
│  │ CONCURRENT: When task received          │                        │    │
│  │  • Planner Agent: Evaluate Task         │                        │    │
│  │  • Model Info Service: Model Selection  │                        │    │
│  └─────────────────────────────────────────┘                        │    │
│      ↓                                                              │    │
│  Enough Context? ──NO──→ Back to Pali Agent ───────────────────────┘    │
│      ↓ YES                                                               │
│  Knowledge Acquire (RAG Tool)                                            │
│    • User History (Relational DB)                                        │
│    • Art Library RAG (Vector DB)                                         │
│    • Online Search                                                       │
│      ↓                                                                   │
│  Planner Agent: Assembly Request                                         │
│    • Enhanced Prompt                                                     │
│    • Reference Images                                                    │
│    • Model Parameters (from concurrent Model Selection)                  │
│      ↓                                                                   │
│  ┌─────────────────────────────────────────┐                             │
│  │ BEFORE Request Submit:                  │                             │
│  │  • Planner Agent: Evaluate Prompt Quality│                            │
│  │  • Evaluation Agent: Create Eval Plan   │                             │
│  └─────────────────────────────────────────┘                             │
│      ↓                                                                   │
│  Generation Service → Generated Image                                    │
│      ↓                                                                   │
│  Evaluation Agent: Execute Eval Plan (Quality Check)                     │
│      ↓                                                                   │
│  Result Not Pass? ──YES──→ Planner Agent: Fix Plan (loop)                │
│      ↓ PASS                                                              │
│  Pali Agent: Pass Result                                                 │
│      ↓                                                                   │
│  Final Task Summary → Save to DB → Embedding to Vector DB                │
└──────────────────────────────────────────────────────────────────────────┘

Concurrent Execution Points:
1. Task Reception: Planner Agent + Model Selection run in parallel
2. Before Submit: Prompt Quality Check + Evaluation Plan Creation
3. Safety Agent: Runs alongside entire task (non-blocking, continuous monitoring)

Stay Live Services: Safety Agent, Job Service, Cost Service, Middleware/Ratelimit
External: AWS Profilebackend (Check Credit, Deduct Credit)
```

---

## Document Structure Verification Matrix

| Documentation Section | Target Implementation | Phase |
|-----------------------|----------------------|-------|
| 1.2 Responsibilities | All 4 agents + services | 2 |
| 2.1 Logical Layers | HTTP → Middleware → Services → Agents | 1-3 |
| 4.1 Text LLM Service | `text_llm_service.py` | 2 |
| 4.2 Embedding Service | `embedding_service.py` | 2 |
| 4.3 Reasoning Service | `reasoning_service.py` | 2 |
| 4.4 Image Generation Service | `image_generation_service.py` | 2 |
| 5.1 Core (agent/message/config) | `palet8_agents/core/` | 1 |
| 5.2 Agents (Pali/Planner/Evaluator/Safety) | `palet8_agents/agents/` | 2 |
| 5.3 Tools (Context/Image/Job) | `palet8_agents/tools/` | 2 |
| 6.x Core Domain Services | `src/services/` | 2 |
| Appendix A Repository Structure | Full directory alignment | 1 |

---

## TBD Items - Require Alignment During Development

The following items are intentionally marked TBD and **require alignment with stakeholder** during development of each component:

### 1. System Prompts (Per Agent)
| Agent | TBD Item | Alignment Required |
|-------|----------|-------------------|
| Pali Agent | System prompt defining role, responsibilities, constraints, output format | Yes - during Phase 2 |
| Planner Agent | System prompt for planning, RAG usage, prompt building | Yes - during Phase 2 |
| Evaluator Agent | System prompt for quality assessment criteria | Yes - during Phase 2 |
| Safety Agent | System prompt for content policy enforcement | Yes - during Phase 2 |

### 2. LLM Selection (Per Agent/Service)
| Component | TBD Item | Alignment Required |
|-----------|----------|-------------------|
| Pali Agent | Primary model, fallback model, temperature | Yes - during Phase 2 |
| Planner Agent | Primary model, fallback model, temperature | Yes - during Phase 2 |
| Evaluator Agent | Vision model selection for image analysis | Yes - during Phase 2 |
| Safety Agent | Classification model, deterministic settings | Yes - during Phase 2 |
| Text LLM Service | Default models, failover triggers | Yes - during Phase 2 |
| Embedding Service | Embedding model selection, dimensions | Yes - during Phase 2 |
| Reasoning Service | Model per reasoning pattern | Yes - during Phase 2 |

### 3. Evaluation Standards
| Item | TBD Item | Alignment Required |
|------|----------|-------------------|
| Quality Threshold | Minimum score for approval (e.g., 0.45) | Yes - during Phase 2 |
| Scoring Weights | Aesthetics, prompt adherence, suitability weights | Yes - during Phase 2 |
| Objective Checks | Resolution, coverage, background thresholds | Yes - during Phase 2 |
| Re-generation Policy | Max retries, prompt tweak strategies | Yes - during Phase 2 |

### 4. Content Safety Boundaries
| Item | TBD Item | Alignment Required |
|------|----------|-------------------|
| Safety Categories | NSFW, violence, hate, IP/trademark, illegal definitions | Yes - during Phase 2 |
| Blocking Thresholds | Confidence levels per category | Yes - during Phase 2 |
| IP/Trademark List | Specific brands/characters to block | Yes - during Phase 2 |
| User Messaging | Rejection messages and alternatives | Yes - during Phase 2 |

---

# PHASE 1: Structure & Foundation

## Phase 1 Goal
**Establish the `palet8_agents/` directory structure exactly as specified in Documentation Appendix A, with working core components and database schemas.**

### Verification Checklist (Against Documentation)
- [ ] Directory matches Appendix A structure exactly
- [ ] `palet8_agents/core/` has all 5 files
- [ ] `palet8_agents/agents/` has all 4 agent files (placeholder)
- [ ] `palet8_agents/tools/` has base.py, registry.py, and 3 tool files (placeholder)
- [ ] Relational DB schema updated (Prisma)
- [ ] Vector DB schema established

---

## 1.1 Create Directory Structure

```
agent-api/
├── palet8_agents/                    # NEW: Agent framework
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py                  # Base agent class
│   │   ├── llm_client.py             # Low-level LLM client
│   │   ├── message.py                # Message/schema definitions
│   │   ├── config.py                 # Agent configuration
│   │   └── exceptions.py             # Agent exceptions
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── pali_agent.py             # User-facing orchestrator
│   │   ├── planner_agent.py          # Planning & RAG
│   │   ├── evaluator_agent.py        # Quality control
│   │   └── safety_agent.py           # Safety & IP checks
│   └── tools/
│       ├── __init__.py
│       ├── base.py                   # Base tool class
│       ├── registry.py               # Tool registry
│       ├── context_tool.py           # RAG/context access
│       ├── image_tool.py             # Image generation
│       └── job_tool.py               # Job state management
├── src/
│   ├── services/                     # Model-facing services (Phase 2)
│   ├── adapters/                     # Provider clients (Phase 2)
│   └── ... (keep existing)
└── tests/
    └── test_palet8_agents/           # NEW: Agent tests
```

---

## 1.2 Database Setup

### 1.2.1 Relational Database (PostgreSQL via Prisma)

**Purpose:** User history, jobs, conversations, sessions, assets, agent logs

**Schema Updates to `prisma/schema.prisma`:**

```prisma
// NEW: Conversation model for multi-turn chat
model Conversation {
  id          String    @id @default(cuid())
  userId      String
  jobId       String?
  messages    Message[]
  status      String    @default("active")  // active, completed, abandoned
  createdAt   DateTime  @default(now())
  updatedAt   DateTime  @updatedAt

  job         Job?      @relation(fields: [jobId], references: [id])

  @@index([userId])
  @@index([jobId])
}

// NEW: Message model for conversation history
model Message {
  id              String       @id @default(cuid())
  conversationId  String
  role            String       // system, user, assistant, tool
  content         String
  metadata        Json?
  createdAt       DateTime     @default(now())

  conversation    Conversation @relation(fields: [conversationId], references: [id])

  @@index([conversationId])
}

// NEW: Job model aligned with documentation
model Job {
  id              String        @id @default(cuid())
  userId          String
  status          String        @default("INIT")  // INIT, COLLECTING_REQUIREMENTS, PLANNING, GENERATING, EVALUATING, COMPLETED, REJECTED, FAILED, ABANDONED
  requirements    Json?         // User requirements from Pali Agent
  plan            Json?         // Execution plan from Planner Agent
  prompt          Json?         // Final prompt (positive, negative, parameters)
  evaluation      Json?         // Evaluation results
  safetyCheck     Json?         // Safety check results
  creditCost      Float         @default(0)
  createdAt       DateTime      @default(now())
  updatedAt       DateTime      @updatedAt

  conversations   Conversation[]
  designs         Design[]

  @@index([userId])
  @@index([status])
}

// NEW: Design model for generated outputs
model Design {
  id              String    @id @default(cuid())
  jobId           String
  assetId         String?
  prompt          String
  negativePrompt  String?
  modelUsed       String
  parameters      Json?
  evaluationScore Float?
  status          String    @default("pending")  // pending, approved, rejected
  createdAt       DateTime  @default(now())

  job             Job       @relation(fields: [jobId], references: [id])
  asset           Asset?    @relation(fields: [assetId], references: [id])

  @@index([jobId])
}

// UPDATE: Existing models may need relations added
```

**Migration Steps:**
1. Update `prisma/schema.prisma` with new models
2. Run `npx prisma migrate dev --name add_agent_models`
3. Generate client `npx prisma generate`

---

### 1.2.2 Vector Database (pgvector via Supabase or standalone)

**Purpose:** Art library embeddings, prompt embeddings, design summaries for RAG

**Schema:**

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Art Library embeddings
CREATE TABLE art_library (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name VARCHAR(255) NOT NULL,
  description TEXT,
  category VARCHAR(100),
  tags TEXT[],
  image_url TEXT,
  embedding vector(1536),  -- OpenAI text-embedding-3-small dimension
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Create index for similarity search
CREATE INDEX art_library_embedding_idx ON art_library
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Prompt embeddings (successful prompts for RAG)
CREATE TABLE prompt_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id VARCHAR(255),
  prompt TEXT NOT NULL,
  negative_prompt TEXT,
  style_tags TEXT[],
  product_type VARCHAR(100),
  evaluation_score FLOAT,
  embedding vector(1536),
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX prompt_embedding_idx ON prompt_embeddings
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Design summaries (final task summaries for RAG)
CREATE TABLE design_summaries (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id VARCHAR(255) NOT NULL,
  user_id VARCHAR(255) NOT NULL,
  summary TEXT NOT NULL,
  product_type VARCHAR(100),
  style VARCHAR(100),
  embedding vector(1536),
  metadata JSONB,
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX design_summary_embedding_idx ON design_summaries
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

CREATE INDEX design_summary_user_idx ON design_summaries(user_id);
```

**Decision: Add pgvector to existing PostgreSQL instance**
- Use the same PostgreSQL database as Prisma
- Enable pgvector extension
- Vector tables coexist with relational tables

**TBD - Align During Development:**
- [ ] Embedding dimension (1536 for text-embedding-3-small, or other?)
- [ ] Index type (ivfflat vs hnsw)?

---

## 1.3 Core Components Implementation

### 1.3.1 `palet8_agents/core/agent.py`

**Goal:** Base agent class aligned with Documentation Section 5.1

**Implementation:**
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

class AgentState(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"

@dataclass
class AgentContext:
    """Shared execution context across agents"""
    user_id: str
    job_id: str
    conversation_id: Optional[str] = None
    credit_balance: float = 0.0
    reasoning_model: str = ""  # TBD during development
    image_model: str = ""      # TBD during development
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentResult:
    """Standard result from agent execution"""
    success: bool
    data: Any
    error: Optional[str] = None
    cost_usd: float = 0.0
    duration_ms: int = 0
    tokens_used: int = 0

class BaseAgent(ABC):
    """Abstract base class for all agents"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.state = AgentState.IDLE
        self.system_prompt = ""  # TBD - set during development with alignment

    @abstractmethod
    async def run(self, context: AgentContext) -> AgentResult:
        """Execute the agent's task"""
        pass
```

### 1.3.2 `palet8_agents/core/llm_client.py`

**Goal:** Low-level OpenRouter client

**Key Features:**
- API key from GCP Secret Manager: `openrouter-api-key`
- Retry policy: 3 retries, exponential backoff
- Rate limiting: configurable per model
- Cost tracking integration

### 1.3.3 `palet8_agents/core/message.py`

**Goal:** Message definitions for conversations

### 1.3.4 `palet8_agents/core/config.py`

**Goal:** Agent configuration loading from YAML

**Model Configuration Template (values TBD during development):**
```yaml
# config/agent_routing_policy.yaml - model_profiles section
model_profiles:
  pali:
    primary_model: ""      # TBD - align during development
    fallback_model: ""     # TBD
    temperature: 0.7       # TBD
    max_tokens: 1000       # TBD

  planner:
    primary_model: ""      # TBD
    fallback_model: ""     # TBD
    temperature: 0.3       # TBD
    max_tokens: 2000       # TBD

  evaluator:
    primary_model: ""      # TBD - needs vision capability
    fallback_model: ""     # TBD
    temperature: 0.2       # TBD
    max_tokens: 500        # TBD

  safety:
    primary_model: ""      # TBD
    fallback_model: ""     # TBD
    temperature: 0.0       # Deterministic for safety
    max_tokens: 300        # TBD

image_models:
  primary: ""              # TBD
  fallback: ""             # TBD

embedding_models:
  primary: ""              # TBD
  fallback: ""             # TBD
  dimensions: 1536         # TBD - depends on model choice
```

### 1.3.5 `palet8_agents/core/exceptions.py`

**Goal:** Structured exceptions with error codes

---

## 1.4 Tools Base Implementation

### 1.4.1 `palet8_agents/tools/base.py`
- Abstract `BaseTool` class
- Standard execute/validate interface

### 1.4.2 `palet8_agents/tools/registry.py`
- Tool registry singleton
- Dynamic tool registration

---

## Phase 1 Deliverables
- [x] Directory structure created and matches Appendix A
- [x] All 5 core modules implemented
- [x] Tools base (base.py, registry.py) implemented
- [x] Placeholder agent files created
- [x] Placeholder tool files created
- [x] Relational DB schema updated (Prisma migration)
- [x] Vector DB schema created
- [x] Unit tests for core components
- [x] CI passes

---

# PHASE 2: Agent & Service Implementation

## Phase 2 Goal
**Implement all 4 agents, all model-facing services, and all tools. Align with stakeholder on TBD items (system prompts, LLM selection, evaluation standards, safety boundaries) for each component.**

### Verification Checklist (Against Documentation)
- [ ] Text LLM Service matches Section 4.1
- [ ] Embedding Service matches Section 4.2
- [ ] Reasoning Service matches Section 4.3
- [ ] Image Generation Service matches Section 4.4
- [ ] Pali Agent matches Section 5.2.1
- [ ] Planner Agent matches Section 5.2.2
- [ ] Evaluator Agent matches Section 5.2.3
- [ ] Safety Agent matches Section 5.2.4
- [ ] Context Tool matches Section 5.3.1
- [ ] Image Tool matches Section 5.3.2
- [ ] Job Tool matches Section 5.3.3

---

## 2.1 Model-Facing Services

### 2.1.1 Text LLM Service (`src/services/text_llm_service.py`)

**Goal:** Documentation Section 4.1

**Methods:**
- `generate_text(prompt, model_profile, **kwargs) -> str`
- `rewrite_prompt(original, constraints) -> str`
- `generate_clarifying_questions(context) -> List[str]`
- `summarize_conversation(messages) -> str`

**TBD - Align During Development:**
- [ ] Primary/fallback model IDs
- [ ] Failover triggers (HTTP 5xx, rate limit, timeout threshold)
- [ ] Default parameters per use case

---

### 2.1.2 Embedding Service (`src/services/embedding_service.py`)

**Goal:** Documentation Section 4.2

**Methods:**
- `embed_text(text: str) -> List[float]`
- `embed_text_batch(texts: List[str]) -> List[List[float]]`
- `embed_image(image_bytes) -> List[float]` (optional)

**TBD - Align During Development:**
- [ ] Embedding model selection
- [ ] Embedding dimensions
- [ ] Batch size limits

---

### 2.1.3 Reasoning Service (`src/services/reasoning_service.py`)

**Goal:** Documentation Section 4.3

**Methods:**
- `assess_prompt_quality(prompt, constraints) -> QualityScore`
- `propose_prompt_revision(prompt, feedback) -> str`
- `assess_design_alignment(prompt, description) -> AlignmentScore`
- `classify_intent(text, categories) -> str`

**TBD - Align During Development:**
- [ ] Model selection per reasoning pattern
- [ ] Scoring rubrics

---

### 2.1.4 Image Generation Service (`src/services/image_generation_service.py`)

**Goal:** Documentation Section 4.4

**Methods:**
- `generate_images(request: ImageGenerationRequest) -> ImageGenerationResult`

**TBD - Align During Development:**
- [ ] Provider selection (Runware, etc.)
- [ ] Retry policy parameters

---

### 2.1.6 Model Info Service (`src/services/model_info_service.py`)

**Goal:** Model routing for image generation (from Swimlane: Planner Agent → Model Selection → Model Info Service)

**Purpose:** Select optimal image model based on task prompt, job requirements, and model capabilities

**Methods:**
- `select_image_model(prompt, requirements, job_context) -> ModelSelection`
- `get_model_capabilities(model_id) -> ModelCapabilities`
- `get_available_models() -> List[ModelInfo]`

**Model Selection Criteria (from Swimlane):**
- Task/prompt characteristics (complexity, style, subject matter)
- Job requirements (product type, quality level, dimensions)
- Model capabilities (strengths per style/subject)
- Cost considerations
- Availability/rate limits

**Model Selection Flow:**
```
Planner Agent
    ↓
Model Info Service (select_image_model)
    ↓
Analyze: prompt + requirements + job context
    ↓
Match against model capabilities
    ↓
Return: Selected Image Model + rationale
    ↓
Image Generation Service (generate with selected model)
```

**TBD - Align During Development:**
- [ ] Model capability matrix (which models excel at what tasks)
- [ ] Selection algorithm (rule-based, scoring, or hybrid)
- [ ] Model metadata schema
- [ ] Fallback strategy when primary model unavailable

---

### 2.1.5 Context Service (`src/services/context_service.py`)

**Goal:** Documentation Section 6.3 - Central RAG entry point

**Methods:**
- `get_user_history(user_id, limit) -> List[Design]`
- `search_art_library(query, limit) -> List[ArtItem]`
- `search_similar_prompts(query, limit) -> List[Prompt]`
- `build_context(user_id, requirements) -> Context`

**Data Sources:**
- Relational DB: User history, previous jobs
- Vector DB: Art library, prompt embeddings, design summaries

**TBD - Align During Development:**
- [ ] Retrieval limits per source
- [ ] Online search integration (provider, enable/disable)

---

## 2.2 Agent Implementation

### 2.2.1 Pali Agent (`palet8_agents/agents/pali_agent.py`)

**Goal:** Documentation Section 5.2.1 - User-facing orchestrator

**Responsibilities:**
- Receive user input and validate (English only, format check)
- Gather requirements through Q&A
- Determine when requirements are complete
- Delegate to Planner Agent

**TBD - Align During Development:**
- [ ] System prompt
- [ ] LLM selection (primary/fallback)
- [ ] Max Q&A rounds before forcing generation
- [ ] Requirements completeness criteria

---

### 2.2.2 Planner Agent (`palet8_agents/agents/planner_agent.py`)

**Goal:** Documentation Section 5.2.2 - Planning, RAG, prompt creation

**Responsibilities:**
- Evaluate if enough context (loop back to Pali if not)
- Query Context Service for RAG
- Build optimized prompt (positive, negative, parameters)
- **Select image model via Model Info Service** (based on prompt + requirements)
- Pre-generation safety check
- Create execution plan with cost estimate

**Model Selection (CONCURRENT with task evaluation):**
```
When task received:
┌─────────────────────────────────────────────────┐
│ PARALLEL EXECUTION:                             │
│  • Planner Agent: Evaluate Task                 │
│  • Model Info Service: select_image_model()    │
│    - Analyzes: prompt, requirements, job_context│
│    - Returns: model_id, rationale, parameters   │
│      (Temperature, Top K, Top P)                │
└─────────────────────────────────────────────────┘
Results merge at Assembly Request step
```

**TBD - Align During Development:**
- [ ] System prompt
- [ ] LLM selection (primary/fallback)
- [ ] RAG retrieval limits
- [ ] Prompt template structure
- [ ] Model selection integration with Model Info Service

---

### 2.2.3 Evaluator Agent (`palet8_agents/agents/evaluator_agent.py`)

**Goal:** Documentation Section 5.2.3 - Quality review

**Two-Phase Execution:**
1. **BEFORE Request Submit** - Create Evaluation Plan
   - Analyze prompt and requirements
   - Define evaluation criteria for this specific task
   - Set thresholds based on product type
   - Prepare objective check parameters

2. **AFTER Generation** - Execute Evaluation Plan
   - Run objective checks (resolution, coverage, background)
   - Run subjective scoring (aesthetics, adherence, suitability)
   - Calculate combined score
   - Make approve/reject decision
   - Provide feedback for rejected images

**Responsibilities:**
- Create task-specific evaluation plan before generation
- Execute evaluation after image is generated
- Provide actionable feedback for re-generation loop

**TBD - Align During Development:**
- [ ] System prompt
- [ ] Vision LLM selection
- [ ] Objective check thresholds (min resolution, coverage %, etc.)
- [ ] Scoring weights (aesthetics, adherence, suitability)
- [ ] Approval threshold
- [ ] Evaluation plan schema

---

### 2.2.4 Safety Agent (`palet8_agents/agents/safety_agent.py`)

**Goal:** Documentation Section 5.2.4 - Safety and IP checks

**Execution Model: Continuous Non-Blocking Monitoring**
```
║ SAFETY AGENT - RUNS ALONGSIDE ENTIRE TASK ║
║ Event-driven, never a bottleneck          ║

┌─────────────────────────────────────────────────────────────┐
│ Safety Agent spawns at task start                           │
│                                                             │
│  Main Flow                    Safety Agent (parallel)       │
│  ─────────                    ──────────────────────        │
│  Pali Agent ──────────────→   Monitor: user input          │
│       ↓                                                     │
│  Planner Agent ───────────→   Monitor: requirements         │
│       ↓                                                     │
│  RAG + Context ───────────→   Monitor: context/references   │
│       ↓                                                     │
│  Assembly Request ────────→   Monitor: final prompt         │
│       ↓                                                     │
│  Generation ──────────────→   (waits for image)             │
│       ↓                                                     │
│  Generated Image ─────────→   Monitor: generated image      │
│       ↓                                                     │
│  Evaluation ──────────────→   (parallel continues)          │
│       ↓                                                     │
│  Result ──────────────────→   Final safety verdict          │
└─────────────────────────────────────────────────────────────┘

Key Principle: Safety Agent NEVER blocks main flow
- Monitors events as they occur
- Flags issues asynchronously
- Only halts if critical violation detected
- Accumulates safety score throughout task
```

**Responsibilities:**
- **Continuous text monitoring** - User input, requirements, prompts
- **Continuous context monitoring** - RAG results, reference images
- **Post-generation image analysis** - Generated output
- Classify safety violations (accumulative)
- Block only on critical violations with helpful alternatives
- Non-blocking design - never a performance bottleneck

**Implementation Pattern:**
```python
# Event-driven safety monitoring
class SafetyAgent(BaseAgent):
    async def start_monitoring(self, job_id: str):
        """Start background monitoring for a job"""
        # Subscribe to job events
        # Non-blocking event processing

    async def on_event(self, event_type: str, data: Any):
        """Handle incoming events without blocking main flow"""
        # NSFW check, IP check, policy check
        # Update accumulated safety score
        # Only raise if critical threshold exceeded

    async def get_safety_verdict(self, job_id: str) -> SafetyResult:
        """Get final accumulated safety result"""
```

**TBD - Align During Development:**
- [ ] System prompt
- [ ] LLM selection (deterministic, temp=0)
- [ ] Safety categories definitions
- [ ] Blocking thresholds per category (critical vs warning)
- [ ] IP/trademark blocklist
- [ ] User-facing rejection messages
- [ ] Event subscription mechanism
- [ ] Accumulated safety score thresholds

---

## 2.3 Tools Implementation

### 2.3.1 Context Tool (`palet8_agents/tools/context_tool.py`)

**Goal:** Documentation Section 5.3.1

**Methods:**
- `fetch_user_history(user_id, limit) -> List[Dict]`
- `search_art_library(query, limit) -> List[Dict]`
- `search_similar_designs(query, limit) -> List[Dict]`

---

### 2.3.2 Image Tool (`palet8_agents/tools/image_tool.py`)

**Goal:** Documentation Section 5.3.2

**Methods:**
- `generate(prompt, negative_prompt, model, parameters) -> ImageResult`

---

### 2.3.3 Job Tool (`palet8_agents/tools/job_tool.py`)

**Goal:** Documentation Section 5.3.3

**Job State Machine:**
```
INIT → COLLECTING_REQUIREMENTS → PLANNING → GENERATING → EVALUATING → COMPLETED
                ↑                    ↓           ↓            ↓
                └────────────────────┴───────────┴────────────┘
                                  (loops/retries)
                                     ↓
                              REJECTED / FAILED / ABANDONED
```

**Methods:**
- `get_job(job_id) -> Job`
- `update_job(job_id, updates) -> Job`
- `transition_state(job_id, new_state) -> Job`

---

## Phase 2 Deliverables
- [x] All 5 model-facing services implemented
- [x] All 4 agents implemented with TBD items resolved
- [x] All 3 tools implemented
- [x] Integration tests for each agent
- [x] Model profiles configured after alignment
- [x] System prompts finalized after alignment
- [x] Evaluation standards documented
- [x] Safety boundaries documented

---

# PHASE 3: Deploy & Test

## Phase 3 Goal
**Deploy the new agent framework to GCP Cloud Run, validate end-to-end functionality, and remove old `src/agents/` code after successful validation.**

### Verification Checklist
- [x] All secrets loaded from GCP Secret Manager
- [x] Cloud Run deployment successful
- [x] Health checks passing
- [x] End-to-end generation flow working
- [x] Safety blocking scenarios tested
- [x] Evaluation re-generation loop tested
- [x] RAG retrieval working (both DBs)
- [ ] Old `src/agents/` code removed (archived to docs/archive/)

---

## 3.1 API Integration

### 3.1.1 Update Routes
- `POST /chat` → Pali Agent (new)
- `POST /jobs`, `GET /jobs/{id}`, `POST /jobs/{id}/edit`
- Deprecate `/agents/v1/generate` (remove after validation)

### 3.1.2 Update Orchestrator
**Agent Flow with Concurrent Execution:**
```
║ SAFETY AGENT - RUNS ALONGSIDE ENTIRE TASK (non-blocking) ║

Pali Agent
    ↓
┌─────────────────────────────────────┐
│ CONCURRENT (Task Reception):        │
│  • Planner Agent: Evaluate Task     │
│  • Model Info Service: Select Model │
└─────────────────────────────────────┘
    ↓
Planner Agent (context check loop)
    → RAG (User History + Art Library + Online Search)
    → Assembly Request (merge model selection result)
    ↓
┌─────────────────────────────────────┐
│ BEFORE Request Submit:              │
│  • Planner: Evaluate Prompt Quality │
│  • Evaluation Agent: Create Eval Plan│
└─────────────────────────────────────┘
    ↓
Generation Service → Generated Image
    ↓
Evaluation Agent: Execute Eval Plan
    ↓
Quality loop with Planner (if not pass)
    ↓
Pali Agent (present result)
    ↓
Save to DB + Embedding to Vector DB
```

**Concurrency Implementation:**
- Use `asyncio.gather()` or similar for parallel agent execution
- Model Selection result feeds into Assembly Request
- Evaluation Agent creates plan BEFORE generation, executes AFTER
- Safety Agent runs as continuous background task (event-driven, non-blocking)

---

## 3.2 GCP Deployment

### 3.2.1 Secrets Required
| Secret Name | Purpose |
|-------------|---------|
| `openrouter-api-key` | Text LLM, Embedding, Reasoning |
| `runware-api-key` | Image Generation |
| Database connection strings | Relational + Vector DB |

### 3.2.2 Cloud Run Configuration
| Setting | Value |
|---------|-------|
| Memory | 2Gi |
| CPU | 2 vCPU |
| Timeout | 300s |
| Concurrency | 80 |

### 3.2.3 Database Migrations
- [ ] Run Prisma migration for relational DB
- [ ] Run Vector DB schema setup
- [ ] Verify indexes

---

## 3.3 Testing Strategy

### 3.3.1 Test Scenarios
| Scenario | Expected Outcome |
|----------|------------------|
| Simple generation | Success, approved image |
| Insufficient context | Loop back to Pali for clarification |
| Safety block (prompt) | Blocked with helpful message |
| Safety block (image) | Blocked after generation |
| Low quality image | Re-generation triggered |
| Credit insufficient | Error before generation |
| RAG retrieval | Returns relevant context |

---

## 3.4 Cleanup

### 3.4.1 Delete Old Code (After Validation)
- [ ] Delete `src/agents/` directory
- [ ] Remove old agent imports
- [ ] Remove deprecated routes
- [ ] Clean up unused dependencies

---

## Phase 3 Deliverables
- [x] API routes updated
- [x] Orchestrator wired to new agents
- [x] Deployed to GCP Cloud Run
- [x] All test scenarios passing (523 tests)
- [ ] Old code removed (archived to docs/archive/)
- [x] Documentation finalized

---

# Critical Files Reference

## Files to Create (New)
| File | Phase |
|------|-------|
| `palet8_agents/core/agent.py` | 1 |
| `palet8_agents/core/llm_client.py` | 1 |
| `palet8_agents/core/message.py` | 1 |
| `palet8_agents/core/config.py` | 1 |
| `palet8_agents/core/exceptions.py` | 1 |
| `palet8_agents/tools/base.py` | 1 |
| `palet8_agents/tools/registry.py` | 1 |
| `palet8_agents/agents/pali_agent.py` | 2 |
| `palet8_agents/agents/planner_agent.py` | 2 |
| `palet8_agents/agents/evaluator_agent.py` | 2 |
| `palet8_agents/agents/safety_agent.py` | 2 |
| `palet8_agents/tools/context_tool.py` | 2 |
| `palet8_agents/tools/image_tool.py` | 2 |
| `palet8_agents/tools/job_tool.py` | 2 |
| `src/services/text_llm_service.py` | 2 |
| `src/services/embedding_service.py` | 2 |
| `src/services/reasoning_service.py` | 2 |
| `src/services/image_generation_service.py` | 2 |
| `src/services/model_info_service.py` | 2 |
| `src/services/context_service.py` | 2 |

## Database Files
| File | Phase | Type |
|------|-------|------|
| `prisma/schema.prisma` | 1 | Relational DB schema |
| `scripts/vector_db_setup.sql` | 1 | Vector DB schema |

## Files to Delete (After Phase 3)
| File | Reason |
|------|--------|
| `src/agents/interactive_agent.py` | Replaced by Pali Agent |
| `src/agents/planner_agent.py` | Replaced by new Planner Agent |
| `src/agents/prompt_manager_agent.py` | Merged into Planner Agent |
| `src/agents/model_selection_agent.py` | Merged into Planner Agent |
| `src/agents/generation_agent.py` | Replaced by Image Tool |
| `src/agents/evaluation_agent.py` | Replaced by Evaluator Agent |

---

# Success Criteria

## Phase 1 Complete When: ✅ ACHIEVED
- [x] Directory structure matches Documentation Appendix A
- [x] All core components instantiable
- [x] Relational DB migration successful
- [x] Vector DB schema created
- [x] Unit tests passing

## Phase 2 Complete When: ✅ ACHIEVED
- [x] All 4 agents functional
- [x] All TBD items aligned and documented
- [x] All services calling real providers
- [x] Integration tests passing locally

## Phase 3 Complete When: ✅ ACHIEVED (Partial - old code archived, not deleted)
- [x] Deployed to GCP Cloud Run
- [x] All test scenarios passing (523 tests)
- [x] RAG working with both databases
- [ ] Old code removed (archived to docs/archive/ - pending full cleanup)
- [x] No regression in functionality

---

# Next Steps (Post-Restructure)

## Remaining Tasks
1. **Clean up archived code** - Delete `docs/archive/old_src/` once confirmed not needed
2. **Remove `src/services/task_aggregator`** - Currently commented out in tasks.py, create new implementation if needed
3. **Monitor Cloud Run performance** - Verify service stability and response times

## Future Enhancements
- Implement ReactPromptAgent for advanced prompt building (planned in detailed plan)
- Add comprehensive end-to-end integration tests
- Set up monitoring and alerting for production

---

# Appendix: Reference Resources

- **Documentation:** `/Users/malcolmyam/Desktop/alpha project/agent system/services/agents-api/docs/Research/agent-api Development Documentation.md`
- **Swimlane:** `/Users/malcolmyam/Desktop/alpha project/agent system/services/agents-api/docs/Research/Palet8 Agent Swimlane.drawio.html`
- **HelloAgents Reference:** https://github.com/jjyaoao/HelloAgents/tree/main/hello_agents
- **Current Codebase:** `/Users/malcolmyam/Desktop/alpha project/agent system/services/agents-api/`
- **Detailed Implementation Plan:** See `.claude/plans/temporal-brewing-fog.md` for granular PR details
