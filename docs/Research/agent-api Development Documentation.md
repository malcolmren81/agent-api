# agent-api Development Documentation

> Version: v0.4 (aligned with Palet8 Agent Swimlane, model services split, secrets + model routing)
>  Scope: Backend service and internal agent framework for design generation (no production customer-support workflows)

------

## 1. System Overview

### 1.1 Purpose

`agent-api` is a backend service that powers AI-assisted design generation for print-on-demand (POD) products (e.g., t-shirt prints, phone cases, embroidery-friendly graphics).

It provides:

- An HTTP API used by the **palet8-customer-app** “Generator Page”.
- An internal **multi-agent framework** (`palet8_agents`) that:
  - collects and clarifies user requirements,
  - plans and refines prompts,
  - performs safety and quality checks,
  - calls third-party image-generation APIs (no in-house model training),
  - persists user and job history for reuse (RAG, recommendations).

The service itself does **not** handle:

- User authentication UX (login forms, password resets).
- Shopify product creation or order lifecycle.
- Payment or subscription billing logic.

These are handled by other services (see System Context).

------

### 1.2 Responsibilities

`agent-api` is responsible for:

- Exposing **chat** and **job** APIs for design generation.
- Running a **multi-agent orchestration** with **inline orchestration pattern**:
  - **Pali Agent** (always-on communication layer, user ↔ system),
  - **Planner Agent V2** (inline orchestrator - stays active until user confirms),
  - **React Prompt Agent** (prompt building - invoked at checkpoints),
  - **Evaluator Agent V2** (quality control - invoked at checkpoints),
  - **Safety Agent** (policy / IP / NSFW gating).
- Integrating with clearly separated **model-facing services**:
  - **Text LLM Service** – generic text generation,
  - **Embedding Service** – embeddings for text (and optionally images),
  - **Reasoning Service** – reusable higher-level LLM reasoning patterns,
  - **Image Generation Service** – third-party image generation APIs.
- Managing **jobs, conversations, designs, and assets**.
- Enforcing **credit-based usage** via Profilebackend:
  - checking and deducting credits around generation.
- Managing **model routing and fallbacks**:
  - for each model service and each agent/task, define a **primary** and **secondary** model,
  - routing and fallback logic will be implemented during development (TBD in detail).
- Using **secrets from GCP Secret Manager** for all model API keys:
  - `runware-api-key` – for image generation (Runware / image router),
  - `openrouter-api-key` – for text LLM, reasoning, and embedding (OpenRouter / LLM router).
- Providing operational capabilities:
  - logging, tracing, health checks,
  - error handling, retry policies,
  - cost tracking and (TBD) rate limits.

------

### 1.3 System Context and External Dependencies

`agent-api` is deployed as **“palet8-agents”** on GCP and sits in the middle of the system.

**Upstream / clients**

- **palet8-customer-app (GCP)**
  - Hosts the **Generator Page**:
    - **Selectors + Preview** (product template, aesthetics, characters),
    - **Chat UI + Preview** (free-form prompts, Q&A, “Edit Request”).
  - Calls `agent-api` via:
    - `POST /chat`,
    - `POST /jobs`, `POST /jobs/{id}/edit`, etc.

**Side services**

- **palet8-admin-api (GCP)**
  - Owns canonical data for:
    - product templates,
    - aesthetics,
    - characters / motifs and other selectors.
  - Generator Page uses this; selector values are passed into `agent-api`.
- **AWS Profilebackend (credits / user profile)**
  - Source of truth for:
    - user identity (token → user id),
    - credit balance.
  - `agent-api` must:
    - check credits before generation,
    - deduct credits after a job is accepted or completed (exact policy TBD).
- **Product Generation Service (Shopify integration)**
  - Consumes design outputs from `agent-api` to create/update **Shopify Product Pages**.
  - `agent-api` does **not** call Shopify directly.

**Data stores**

- **Relational database (DB)** – users, jobs, conversations, designs, assets.
- **Vector database (Vector DB)** – art library embeddings, prompt/design embeddings, final summaries.
- **Object Storage (GCS, S3, etc.)** – raw generated images and other files.

**External providers via router services**

- **OpenRouter (LLM router)**
  - Used for text LLM, reasoning, and embedding through `openrouter-api-key`.
- **Runware (image router)**
  - Used for image generation through `runware-api-key`.

API keys are never hardcoded; they are loaded from **GCP Secret Manager** at runtime.

------

## 2. High-Level Architecture

### 2.1 Logical Layers

1. **HTTP / API Layer**
   - FastAPI (or similar).
   - Routes: chat, jobs, assets, health.
   - Request validation & authentication token extraction.
2. **Middleware Layer**
   - Trace id injection.
   - Error handling / exception translation.
   - (TBD) Rate limiting and quota hooks.
3. **Core Services**
   - **Domain services**:
     - `conversation_service`,
     - `job_service`,
     - `context_service` (RAG & context),
     - `design_summary_service`,
     - `asset_service`,
     - `url_service`.
   - **Model-facing services**:
     - `text_llm_service`,
     - `embedding_service`,
     - `reasoning_service`,
     - `image_generation_service`.
   - **Support services**:
     - `cost_tracker`,
     - `retry_policy`.
4. **Agent Framework (`palet8_agents`)**
   - Pali / Planner / Evaluator / Safety agents.
   - Tools exposing services to agents.
5. **Adapters**
   - Provider clients (OpenRouter, Runware).
   - External services (Profilebackend, admin-api).
   - Storage (GCS, vector DB, doc store).
6. **Persistence Layer**
   - Repositories for DB entities.
   - Vector DB & doc store integration.

------

## 3. Core Domain Concepts

- **User** – end user of the website; may be authenticated or anonymous.
   Authentication model and enforcement rules are **TBD**.
- **Job** – unit of work for a design creation/edit:
  - Typical states (TBD final list):
     `INIT`, `COLLECTING_REQUIREMENTS`, `PLANNING`,
     `GENERATING`, `EVALUATING`, `COMPLETED`,
     `ABANDONED`, `REJECTED`, `FAILED`.
  - Allowed transitions and irreversibility are **TBD**.
- **Conversation** – ordered messages between user and agents; used for understanding, Q&A, and RAG.
- **Design** – specific design output with:
  - final prompt,
  - generated asset references,
  - template/style metadata,
  - evaluation & safety results.
- **Asset** – stored file (image, thumbnail, reference).
- **Product (Shopify)** – downstream concept; created by Product Generation Service, not `agent-api`.
- **Credits** – consumption units managed by Profilebackend; cost mapping is **TBD**.
- **Safety & IP rules** – definitions of disallowed content are **TBD** (must be defined with policy/legal).

------

## 4. Model-Facing Services (LLM, Embedding, Reasoning, Image)

This section is intentionally separated so developers can quickly find model-related logic.

### 4.1 Text LLM Service (`text_llm_service.py`)

**Purpose**

High-level interface for all **text-only** LLM operations.

**Responsibilities**

- Encapsulate calls to text LLM providers (OpenRouter via `openrouter-api-key`).
- Provide methods such as:
  - `generate_text(prompt, ...)`
  - `rewrite_prompt(context, constraints, ...)`
  - `generate_clarifying_questions(...)`
  - `summarize_conversation(...)`
- Handle:
  - model selection (text models only),
  - default parameters per use case,
  - basic error mapping.

**Secrets & routing**

- API key:
  - Loaded from **GCP Secret Manager** as secret: `openrouter-api-key`.
- Model routing:
  - For each **agent** and **use case**, define:
    - `primary_model` (1st choice),
    - `fallback_model` (2nd choice when primary unavailable).
  - Examples (names are illustrative; actual IDs will be agreed during development):
    - `pali.clarification: primary = model_A, fallback = model_B`
    - `planner.prompting: primary = model_C, fallback = model_D`
  - The concrete routing and failover logic is **TBD** and will be implemented by developers (e.g., handling HTTP 5xx, rate limits, or provider errors by switching to fallback).

**Used by**

- Pali Agent (clarifying questions, user-facing responses),
- Planner Agent (prompt drafting/improvement),
- Evaluator Agent (evaluation summaries),
- Design Summary Service (final summaries),
- Reasoning Service (internally).

------

### 4.2 Embedding Service (`embedding_service.py`)

**Purpose**

Central place to compute and manage **embeddings** (text and optionally image).

**Responsibilities**

- Provide methods:
  - `embed_text(text: str) -> vector`
  - `embed_text_batch(texts: List[str]) -> List[vector]`
  - (Optional) `embed_image(image_bytes) -> vector`
- Hide provider differences (embedding via OpenRouter or other router).
- Handle:
  - default embedding model per use case (RAG vs similarity search),
  - batching if required.

**Secrets & routing**

- API key:
  - Loaded from **GCP Secret Manager** as `openrouter-api-key`.
- Model routing:
  - For each **embedding use case** (e.g., RAG query, summary embedding, art library ingestion), define:
    - `primary_embedding_model`,
    - `fallback_embedding_model`.
  - Exact model IDs and switching conditions are **TBD**.

**Used by**

- Context Service (RAG),
- Design Summary Service (embedding summaries),
- Any other service needing vector representations.

------

### 4.3 Reasoning Service (`reasoning_service.py`)

**Purpose**

Reusable library of **LLM-based reasoning patterns** built on top of Text LLM Service.

**Responsibilities**

- Implement common reasoning patterns:
  - classification (e.g., detect “missing info”),
  - scoring / ranking (e.g., choose best prompt),
  - evaluations (e.g., quality, alignment),
  - multi-step transformations (e.g., extract constraints → propose prompt → self-review).
- Provide high-level functions:
  - `assess_prompt_quality(prompt, constraints)`,
  - `propose_prompt_revision(prompt, feedback)`,
  - `assess_design_alignment(prompt, description, eval_inputs)`.

**Secrets & routing**

- Uses Text LLM Service internally; does not manage API keys directly.
- For each **reasoning pattern**, Reasoning Service will select:
  - a `primary_reasoning_model`,
  - a `fallback_reasoning_model`,
     based on configuration shared with Text LLM Service.
- Model ids and routing rules per reasoning task are **TBD**.

**Used by**

- Planner Agent (plan and prompt refinement),
- Evaluator Agent (structured evaluation),
- Safety Agent (optional classification support).

------

### 4.4 Image Generation Service (`image_generation_service.py`)

**Purpose**

Dedicated service for all **image model** calls.

**Responsibilities**

- Provide a single high-level method:
  - `generate_images(request: ImageGenerationRequest) -> ImageGenerationResult`
- Implement:
  - provider selection and routing (Runware and any additional providers),
  - calling provider clients,
  - retries and backoff (via `retry_policy`),
  - integration with `cost_tracker`,
  - mapping provider responses to internal asset representation.

**Secrets & routing**

- API key:
  - Loaded from **GCP Secret Manager** as secret: `runware-api-key`.
- Model routing:
  - For each **image use case** / agent scenario (e.g., base generation, high-quality rerender, embroidery-friendly variant), define:
    - `primary_image_model`,
    - `fallback_image_model`.
  - Routing rules (e.g., fallback on HTTP 5xx, provider unavailability, quota) are **TBD** and will be implemented by developers.

**Used by**

- Image Tool (agent tool),
- Any non-agent flows that may be added later.

------

### 4.5 Model Routing & Configuration (Cross-Cutting)

- **Config location**:
  - Model routing configuration (primary/fallback per service/agent/use case) will live in:
    - configuration files (YAML/JSON), or
    - environment variables, or
    - a dedicated config module.
  - Exact mechanism is **TBD** and chosen by developers.
- **Per-agent configuration**:
  - Each agent (Pali, Planner, Evaluator, Safety) should have a clear **model profile**:
    - `pali_model_primary`, `pali_model_fallback`,
    - `planner_model_primary`, `planner_model_fallback`,
    - etc.
  - These profiles are used by Text LLM / Reasoning Service to choose models.
- **Fallback strategy**:
  - When primary fails or becomes unavailable (e.g., provider error, rate limit, routing error), services **may**:
    - immediately switch to fallback for that call,
    - or retry primary then fallback,
    - or mark call as failed without fallback.
  - Concrete logic is **TBD** and will be defined with developers during implementation.

------

## 5. Agent Framework (`palet8_agents`)

### 5.1 Core (`palet8_agents/core/`)

#### `agent.py`

- Base class for all agents.
- Defines a standard interface and shared utilities.

#### `llm_client.py`

- Low-level client for calling text LLM router (OpenRouter) with `openrouter-api-key`.
- Used by `text_llm_service`; agents do **not** use it directly.

#### `message.py`

- Message and schema definitions.

#### `config.py`

- Agent-related config, including:
  - default model profile names for each agent,
  - safety settings.
- Actual mapping from profile → model ids is configured in model services and is **TBD**.

#### `exceptions.py`

- Shared agent-level exceptions.

------

### 5.2 Agents (`palet8_agents/agents/`)

The system uses an **inline orchestration pattern** where:
- **Pali** is always on as the **communication layer** (user ↔ system)
- **Planner** stays inline as the **central orchestrator** (doesn't exit until complete)
- **Specialized agents** (ReactPrompt, Evaluator) are invoked only at checkpoints

```
User ←→ /chat/generate ←→ PALI (always on)
                              │
                              ▼
                         PLANNER (inline orchestrator)
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ReactPrompt    AssemblyService   Evaluator
        (checkpoint)    (checkpoint)    (checkpoint)
```

#### Pali Agent (`pali_agent.py`)

- Role: **always-on communication layer** between user and system.
- Key methods:
  - `handle_generate_request()` - Async generator that yields events to frontend while delegating to Planner
  - `confirm_and_complete()` - Handle user confirmation/completion
- Uses:
  - **Text LLM Service** for conversational responses and clarifying questions,
  - **Reasoning Service** (optional) for meta decisions.
- Has its own model profile:
  - e.g., `pali_primary_model`, `pali_fallback_model` (names / IDs TBD).

#### Planner Agent V2 (`planner_agent_v2.py`)

- Role: **inline orchestrator** - stays active throughout generation until user confirms.
- Key methods:
  - `orchestrate_generation()` - Main inline orchestration loop with retry logic
  - `_delegate_to_react_prompt()` - Checkpoint delegation to prompt building
  - `_delegate_to_evaluator()` - Checkpoint delegation for pre/post-gen evaluation
  - `_execute_generation()` - Execute generation via Assembly Service
  - `_emit_progress()` - Send progress updates via Pali callback
- Orchestration flow:
  1. Context check (sufficient? → ask Pali for clarification)
  2. Safety check (safe? → block if not)
  3. Complexity classification (RELAX/STANDARD/COMPLEX)
  4. Delegate to ReactPromptAgent (checkpoint)
  5. Build AssemblyRequest
  6. Delegate to Evaluator for pre-gen check (checkpoint)
  7. Execute generation via AssemblyService (checkpoint)
  8. Delegate to Evaluator for post-gen check (checkpoint)
  9. Handle retry loops on rejection (max 3 retries)
  10. Return result to Pali for user presentation
- Uses:
  - Context Tool → Context Service,
  - Text LLM Service + Reasoning Service,
  - Model info configuration,
  - Safety tools for pre-generation checks.
- Has its own model profile for planning:
  - `planner_primary_model`, `planner_fallback_model` (TBD).

#### React Prompt Agent (`react_prompt_agent.py`)

- Role: **prompt building** - invoked at checkpoints by Planner.
- Key responsibilities:
  - Context building (art refs, history, web search if needed)
  - Dimension selection (subject, aesthetic, style)
  - Prompt composition (final prompt and negative prompt)
  - Quality scoring
- Uses:
  - Context Tool for RAG,
  - Text LLM Service for prompt generation.

#### Evaluator Agent V2 (`evaluator_agent_v2.py`)

- Role: **quality gates** - invoked at checkpoints by Planner.
- Two phases:
  - **Pre-generation** (phase="create_plan"): Evaluate prompt quality before generation
  - **Post-generation** (phase="execute"): Evaluate result quality after generation
- Decision types:
  - `PASS` / `APPROVE` - Continue to next step
  - `FIX_REQUIRED` / `REJECT` - Retry with feedback (if retries remaining)
  - `POLICY_FAIL` - Block for policy violation
- Uses:
  - Reasoning Service for structured evaluations,
  - Text LLM Service for natural language summaries.
- Has model profile:
  - `evaluator_primary_model`, `evaluator_fallback_model` (TBD).

#### Safety Agent (`safety_agent.py`)

- Role: safety and IP checks.
- Uses:
  - Reasoning Service,
  - optional external safety tools / classifiers.
- Has model profile for text-based classification tooling:
  - `safety_primary_model`, `safety_fallback_model` (TBD).

Safety checks are **mandatory** in production flows.

------

### 5.3 Tools (`palet8_agents/tools/`)

Tool names are simplified and aligned with business meaning.

#### Context Tool (`context_tool.py`)

- Bridges agents and Context Service (`context_service`).
- Fetches:
  - user history,
  - designs & prompts,
  - art library entries,
  - optional online search results.

#### Image Tool (`image_tool.py`)

- Bridges agents and Image Generation Service (`image_generation_service`).
- Requests images using the planned prompt and model parameters.

#### Job Tool (`job_tool.py`)

- Bridges agents and Job Service (`job_service`).
- Reads and updates job state and metadata.

------

## 6. Core Services (Domain & Context)

### 6.1 Conversation Service (`conversation_service.py`)

- Manages conversations and messages.
- Bridges HTTP `/chat` requests with Pali Agent.
- Persists messages and conversation state.

### 6.2 Job Service (`job_service.py`)

- Owns job lifecycle and states.
- Enforces valid transitions (state machine details **TBD**).
- Integrates with Profilebackend for credit checks and deductions.

### 6.3 Context Service (`context_service.py`)

- Central RAG entry point.
- Uses:
  - Embedding Service,
  - Vector DB Adapter,
  - Doc Store Adapter,
  - optional online search.

Responsibilities:

- Embed queries via Embedding Service (OpenRouter via `openrouter-api-key`).
- Retrieve related content from:
  - user history,
  - designs and prompts (Vector DB),
  - curated docs (Doc store),
  - online search (provider **TBD**).
- Return structured context for Planner/Evaluator.

### 6.4 Design Summary Service (`design_summary_service.py`)

- Generates and stores final task summaries.
- Uses:
  - Text LLM Service,
  - Reasoning Service,
  - Embedding Service.

Responsibilities:

- Build summaries for completed jobs:
  - user request and selectors,
  - final prompt,
  - models and parameters used,
  - evaluation and safety outcome,
  - interaction statistics.
- Store summaries in DB.
- Embed summaries and store in Vector DB.

### 6.5 Asset Service & URL Service

- **Asset Service (`asset_service.py`)**
  - Manages asset metadata.
  - Uses `gcs_adapter` for file storage.
- **URL Service (`url_service.py`)**
  - Builds public or signed URLs for assets.
  - Access policies **TBD**.

### 6.6 Cost Tracker & Retry Policy

- **Cost Tracker (`cost_tracker.py`)**
  - Records usage and cost for:
    - text LLM,
    - embeddings,
    - image generation,
    - possibly safety tools.
  - Feeds into Job Service / Profilebackend credit logic (mapping cost→credits is **TBD**).
- **Retry Policy (`retry_policy.py`)**
  - Shared retry settings across external calls:
    - OpenRouter, Runware, Profilebackend, online search, etc.
  - Per-provider parameters are **TBD**.

------

## 7. HTTP/API Layer and Adapters

### 7.1 Routes

- `chat_routes.py`
  - `POST /chat` — conversational entry point.
- `job_routes.py`
  - `POST /jobs` — create a new design job.
  - `GET /jobs/{id}` — job details.
  - `POST /jobs/{id}/edit` — edit existing job/design.
  - `GET /jobs/{id}/results` — designs and assets.
- `health_routes.py`
  - Liveness and readiness checks.

------

### 7.2 Middleware and Error Handling

- **Trace Middleware (`trace_middleware.py`)**
  - Injects correlation id into request context.
- **Error Handler (`error_handler.py`)**
  - Maps exceptions to structured HTTP responses.

------

### 7.3 Adapters

#### Model Router Clients

- `text_llm_client.py`
  - Uses `openrouter-api-key` from GCP Secret Manager.
  - Wraps HTTP calls to the OpenRouter text endpoints.
- `embedding_client.py`
  - Uses `openrouter-api-key`.
  - Wraps embedding endpoints.
- `image_provider_client_runware.py` (or similar)
  - Uses `runware-api-key`.
  - Wraps Runware image generation endpoints.

Each client:

- Applies retry policy,
- Reports usage to `cost_tracker`,
- Never logs raw API keys.

#### External Service Adapters

- `profilebackend_client.py`
  - `check_credit` and `deduct_credit` integration.
- `admin_api_client.py`
  - Read-only access to product templates, aesthetics, characters, etc.

#### Storage Adapters

- `gcs_adapter.py` — object storage.
- `vector_db_adapter.py` — vector DB.
- `doc_store_adapter.py` — document store.

------

### 7.4 Database Layer

- `database.py` — DB connection/session.
- Repositories:
  - `user_repository.py`,
  - `job_repository.py`,
  - `conversation_repository.py`,
  - `asset_repository.py`.

------

## 8. Business Flows (Aligned with Swimlane)

> The flows below remain the same in structure as v0.3, but implicitly use:
>
> - Text LLM / Embedding / Reasoning services via OpenRouter (`openrouter-api-key`),
> - Image Generation Service via Runware (`runware-api-key`),
> - primary/fallback model selection configured per agent/use case.

### 8.1 New Design Creation Flow

1. **User opens Generator Page**; provides free-form text, selectors, and optional reference images.
2. **Frontend calls** `POST /chat` or `POST /jobs`.
3. **Authentication & job setup** via Conversation Service + Job Service.
4. **Credit check** via Profilebackend before committing to generation (cost estimation + credit logic **TBD**).
5. **Requirement collection & clarification**:
   - Pali Agent + Text LLM + Reasoning Service,
   - multiple Q&A rounds if needed.
6. **Planning & RAG**:
   - Planner Agent + Context Tool/Service,
   - uses Embedding Service and Vector DB,
   - optional online search.
7. **Prompt creation & model selection**:
   - Planner uses Text LLM + Reasoning Service,
   - uses configured `planner_primary_model`/`planner_fallback_model`,
   - chooses image models/parameters (primary/fallback for image).
8. **Pre-generation safety** via Safety Agent (text-based checks).
9. **Final task summary (pre-generation)** built by Planner.
10. **Image generation** via Image Tool → Image Generation Service:
    - uses `runware-api-key` through Runware client,
    - respects primary/fallback image models and retry policy.
11. **Asset storage** & job update via Asset Service.
12. **Evaluation & post-generation safety**:
    - Evaluator Agent + Reasoning Service,
    - Safety Agent for image checks,
    - may loop with Planner for re-generation.
13. **Present results** to frontend (URLs, metadata, job/design ids).
14. **Final summary + embeddings** via Design Summary Service + Embedding Service.
15. **Product creation** via external Product Generation Service (Shopify), outside this service.

------

### 8.2 Edit Existing Design Flow

Same as v0.3, with the addition that:

- All LLM and reasoning calls use Text LLM / Reasoning Service with primary/fallback models via OpenRouter.
- All image re-generations use Image Generation Service with primary/fallback image models via Runware.
- Cost and credit checks are re-applied for edits.

------

### 8.3 Error Handling and Timeouts

- Provider timeouts and failover use Retry Policy and model routing (primary→fallback) where applicable.
- Precise behavior is **TBD** and will be implemented together with developers.

------

## 9. Rules and TBD Items (Preserved + Extended)

All previous TBDs remain, plus:

- **Model routing & configuration**
  - Concrete model IDs for primary/fallback per service/agent/use case.
  - Decision logic for when to switch to fallback and how often.
  - Config storage format (YAML/env/config service).
- **Authentication model** (JWT vs sessions vs anonymous).
- **Job state machine** (states, transitions, terminal states).
- **Safety policies** (NSFW, hate, violence, IP).
- **Costs and rate limits** (mapping usage→credits, per-user/IP quotas).
- **Data retention and privacy** (retention windows, deletion, training usage).
- **Online search integration** (provider, result filtering, caching).

All of these must be collaboratively defined by product, infra, and legal/policy before full external production rollout.

------

## Appendix A: Suggested Repository Structure

```text
agent-api/
├── palet8_agents/
│   ├── core/
│   │   ├── agent.py
│   │   ├── llm_client.py
│   │   ├── message.py
│   │   ├── config.py
│   │   └── exceptions.py
│   ├── agents/
│   │   ├── pali_agent.py
│   │   ├── planner_agent.py
│   │   ├── evaluator_agent.py
│   │   └── safety_agent.py
│   ├── tools/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── context_tool.py
│   │   ├── image_tool.py
│   │   └── job_tool.py
│   └── __init__.py
├── src/
│   ├── config/
│   │   ├── env.py                # loads secrets from GCP Secret Manager
│   │   └── logger.py
│   ├── api/
│   │   ├── chat_routes.py
│   │   ├── job_routes.py
│   │   └── health_routes.py
│   ├── middlewares/
│   │   ├── trace_middleware.py
│   │   └── error_handler.py
│   ├── services/
│   │   ├── conversation_service.py
│   │   ├── job_service.py
│   │   ├── context_service.py
│   │   ├── text_llm_service.py
│   │   ├── embedding_service.py
│   │   ├── reasoning_service.py
│   │   ├── image_generation_service.py
│   │   ├── design_summary_service.py
│   │   ├── asset_service.py
│   │   ├── url_service.py
│   │   ├── cost_tracker.py
│   │   └── retry_policy.py
│   ├── adapters/
│   │   ├── text_llm_client.py           # uses openrouter-api-key
│   │   ├── embedding_client.py          # uses openrouter-api-key
│   │   ├── image_provider_client_runware.py # uses runware-api-key
│   │   ├── profilebackend_client.py
│   │   ├── admin_api_client.py
│   │   ├── gcs_adapter.py
│   │   ├── vector_db_adapter.py
│   │   └── doc_store_adapter.py
│   └── db/
│       ├── database.py
│       ├── user_repository.py
│       ├── job_repository.py
│       ├── conversation_repository.py
│       └── asset_repository.py
└── tests/
    ├── test_chat_routes.py
    ├── test_job_routes.py
    ├── test_agents.py
    ├── test_image_generation_service.py
    ├── test_context_service.py
    └── ...
```

------

## Appendix B: Developer Review & Next Steps

This document should now be directly usable by the development team:

- Model-facing services are clearly separated and named.
- API key sources (`runware-api-key`, `openrouter-api-key` from GCP Secret Manager) are explicitly defined.
- Model routing (primary/fallback) is clearly described for:
  - Text LLM,
  - Embedding,
  - Reasoning,
  - Image generation,
  - and per-agent configuration.
- All previous TBD content is preserved; new TBDs are clearly marked.

**Recommended first tasks for developers:**

1. Implement `env.py` integration with **GCP Secret Manager** to load:
   - `runware-api-key`,
   - `openrouter-api-key`,
      and expose them to adapters without logging them.
2. Scaffold services and adapters using the structure in Appendix A.
3. Define an initial **minimal model routing config** (even if only one provider/model is used at first) with placeholders for fallback.
4. Implement basic versions of:
   - `text_llm_service`, `embedding_service`, `image_generation_service`,
   - `conversation_service`, `job_service`, `context_service`, `design_summary_service`,
   - Pali + Planner agents with simple logic.
5. Add test scaffolding for:
   - configuration loading,
   - secrets loading,
   - basic generation paths (text + image).

You can copy this file directly into your repo (for example as `docs/agent-api-architecture.md`) and use it as the main reference for implementation planning.



Reference Project

​		Use system architecture and code from project below, do not mention in code base;

​		https://github.com/jjyaoao/HelloAgents/tree/main/hello_agents

​		