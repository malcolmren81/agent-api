# Palet8 Agents API

Multi-agent orchestration service for AI-powered design generation. Transforms user design requests into production-ready product images through intelligent agent coordination.

## Overview

| | |
|---|---|
| **Service** | palet8-agents |
| **URL** | https://palet8-agents-kshhjydolq-uc.a.run.app |
| **Routes** | `/agents/*` |
| **Stack** | Python 3.11, FastAPI, Prisma, PostgreSQL |
| **Port** | 8000 |
| **Platform** | Google Cloud Run |
| **Region** | us-central1 |

## Current Status

| Component | Status |
|-----------|--------|
| Core API | Running |
| Database | Connected (Cloud SQL) |
| Migrations | Auto-run on startup |
| Chat Workflow | Active |
| Image Generation | Runware API (Multi-provider) |

## Architecture

### Agent Flow

The system uses a **modular orchestration** pattern with clear agent boundaries:

```
User ←→ /chat/generate ←→ PALI (always on, communication layer)
                              │
                              ▼
                         PLANNER (pure orchestrator)
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
 GenPlan              ReactPrompt               Evaluator
 (generation planning)  (prompt building)       (quality gates)
    │
    ├── Complexity analysis
    ├── Genflow selection (single/dual pipeline)
    ├── Model selection
    └── Parameter extraction
```

### Agent Responsibilities

```
╔═══════════════════════════════════════════════════════════════════╗
║  PALI AGENT - Always-On Communication Layer                       ║
║  - Multi-turn chat with users                                     ║
║  - UI selector integration (template, aesthetic, character)       ║
║  - Requirements validation & completeness tracking                ║
║  - Delegates generation to Planner                                ║
║  - Presents results and waits for user confirmation               ║
╚═══════════════════════════════════════════════════════════════════╝
                              │
╔═══════════════════════════════════════════════════════════════════╗
║  PLANNER AGENT - Pure Orchestrator                                ║
║  (Follows pipeline_methods.yaml checkpoints)                      ║
║  - Context evaluation & sufficiency check                         ║
║  - Safety classification                                          ║
║  - Delegates generation planning to GenPlan                       ║
║  - Delegates prompt building to ReactPrompt                       ║
║  - Delegates evaluation to Evaluator                              ║
║  - Executes generation via Assembly Service                       ║
║  - Handles retry loops on rejection (max 3 retries)               ║
╚═══════════════════════════════════════════════════════════════════╝
                              │
    ┌─────────────────────────┼─────────────────────────┐
    ▼                         ▼                         ▼
╔═══════════════════╗ ╔═══════════════════╗ ╔═══════════════════╗
║ GENPLAN AGENT     ║ ║ REACT PROMPT      ║ ║ EVALUATOR AGENT   ║
║ (NEW)             ║ ║                   ║ ║                   ║
║ - Complexity      ║ ║ - Context build   ║ ║ - Prompt quality  ║
║   analysis        ║ ║ - Dimension       ║ ║   (pre-gen)       ║
║ - User info       ║ ║   selection       ║ ║ - Result quality  ║
║   parsing         ║ ║ - Prompt compose  ║ ║   (post-gen)      ║
║ - Genflow         ║ ║ - Quality scoring ║ ║ - Pass/Fix/Reject ║
║   selection       ║ ║ - Question gen    ║ ║                   ║
║ - Model selection ║ ╚═══════════════════╝ ╚═══════════════════╝
║ - Parameter       ║
║   extraction      ║
╚═══════════════════╝
```

### Pipeline Types

| Pipeline | Description | Use Case |
|----------|-------------|----------|
| **Single** | One model generation | Simple requests, fast iteration |
| **Dual** | Stage 1 (generate) + Stage 2 (refine) | Text overlays, character refinement |

### Dual Pipeline Variants

| Pipeline | Stage 1 | Stage 2 | Use Case |
|----------|---------|---------|----------|
| `creative_art` | midjourney-v7 | nano-banana-2-pro | Concept art, illustrations |
| `photorealistic` | imagen-4-ultra | nano-banana-2-pro | Product photography |
| `layout_poster` | flux-2-flex | qwen-image-edit | Posters, typography |

## Supported Image Models

### Via Runware API

| Model | Provider | Best For |
|-------|----------|----------|
| `midjourney-v7` | Midjourney | Creative art, cinematic imagery |
| `ideogram-3` | Ideogram | Text rendering, logos |
| `flux-2-flex` | BFL | Typography, posters |
| `flux-2-pro` | BFL | High-quality generations |
| `imagen-4-ultra` | Google | Photorealistic images |
| `nano-banana-2-pro` | Google | Image editing, refinement |
| `seedream-4` | ByteDance | Artistic styles |
| `qwen-image-edit` | Alibaba | Image editing, text correction |

## API Endpoints

### Chat API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/start` | POST | Start conversation with Pali agent |
| `/chat/message` | POST | Send message, receive SSE stream |
| `/chat/history/{id}` | GET | Get conversation history |
| `/chat/generate` | POST | Trigger full generation pipeline |

### Data APIs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tasks` | GET | Query task execution logs |
| `/api/tasks/{id}` | GET | Get specific task details |
| `/api/agent-logs` | GET | Query agent execution logs |
| `/api/templates` | GET | Get prompt templates |

### Planner API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/planner/models` | GET | List available image models |
| `/planner/plan` | POST | Generate execution plan |

### Health & Monitoring
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Service health with component status |
| `/docs` | GET | OpenAPI documentation |

## Project Structure

```
agents-api/
├── palet8_agents/              # Core agent system
│   ├── agents/                 # Agent implementations
│   │   ├── pali_agent.py       # Communication layer
│   │   ├── planner_agent_v2.py # Pure orchestrator
│   │   ├── genplan_agent.py    # Generation planning (NEW)
│   │   ├── react_prompt_agent.py # Prompt building
│   │   ├── evaluator_agent_v2.py # Quality gates
│   │   ├── safety_agent.py     # Safety monitoring
│   │   └── archive/            # Legacy agents
│   ├── core/                   # Framework (BaseAgent, Context)
│   ├── models/                 # Data models
│   │   ├── genplan.py          # GenerationPlan, UserParseResult (NEW)
│   │   ├── requirements.py     # RequirementsStatus
│   │   ├── prompt.py           # PromptDimensions, PromptPlan
│   │   ├── generation.py       # GenerationParameters, PipelineConfig
│   │   └── evaluation.py       # EvaluationPlan, ResultQualityResult
│   ├── services/               # Business logic services
│   │   ├── genflow_service.py  # Pipeline selection (NEW)
│   │   ├── model_selection_service.py # Model selection
│   │   ├── assembly_service.py # Pipeline orchestration
│   │   ├── image_generation_service.py # Runware API
│   │   ├── text_llm_service.py # LLM text generation
│   │   └── reasoning_service.py # Complex reasoning
│   └── tools/                  # Agent tools
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   └── routes/             # API endpoints
│   ├── database/               # Prisma client
│   └── utils/                  # Logger, metrics
├── config/                     # YAML configs
│   ├── image_models_config.yaml # Model registry & specs
│   ├── pipeline_methods.yaml   # Orchestration checkpoints (NEW)
│   └── safety_config.yaml      # Safety rules
├── prompts/
│   └── genplan_system.txt      # GenPlan system prompt (NEW)
├── docs/
│   └── test/
│       └── monitoring_activities.md # Log checkpoint docs
├── tests/
│   └── test_palet8_agents/     # Unit tests
├── Dockerfile
└── requirements.txt
```

## Configuration Files

### image_models_config.yaml
Defines all supported models with:
- AIR IDs for Runware API
- Supported workflows (text-to-image, image-to-image)
- Specs (dimensions, steps, parameters)
- Provider-specific settings
- Cost estimates

### pipeline_methods.yaml
Defines orchestration checkpoints:
- context_check
- safety_check
- generation_plan (GenPlan)
- prompt_build (ReactPrompt)
- pre_evaluation (Evaluator)
- execute_generation (Assembly)
- post_evaluation (Evaluator)

## Local Development

### Prerequisites
- Python 3.11+
- Node.js 20+ (for Prisma)
- PostgreSQL database
- API keys: OPENROUTER_API_KEY, RUNWARE_API_KEY

### Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup database
prisma generate
prisma db push

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run server
uvicorn src.api.main:app --reload --port 8000
```

### Running Tests
```bash
# Run all tests
pytest

# Run specific agent tests
pytest tests/test_palet8_agents/agents/test_genplan_agent.py
pytest tests/test_palet8_agents/services/test_genflow_service.py
```

## Deployment

The service uses automatic database migrations on startup via `migrations/init.sql`.

```bash
# Deploy to Cloud Run
gcloud run deploy palet8-agents \
  --source . \
  --region us-central1 \
  --project palet8-system \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300

# Route traffic to latest
gcloud run services update-traffic palet8-agents \
  --region us-central1 \
  --to-latest
```

### Environment Variables (via Secret Manager)

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (Cloud SQL) |
| `GEMINI_API_KEY` | Google Gemini API key |
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENROUTER_API_KEY` | OpenRouter API key |
| `RUNWARE_API_KEY` | Runware image generation API |
| `FLUX_API_KEY` | Black Forest Labs Flux API |

## AI Models

### Reasoning (via OpenRouter)
- Gemini 2.0 Flash (primary)
- GPT-4o (fallback)
- Automatic routing and fallback handling

### Image Generation (via Runware)
- Midjourney V7
- Ideogram 3
- FLUX Pro/Flex
- Imagen 4 Ultra
- Nano Banana 2 Pro
- Qwen Image Edit

## Key Features

- **Modular Agent Design**: Clear boundaries between GenPlan, ReactPrompt, Evaluator
- **Smart Model Selection**: Scenario-based selection (art vs photo, with/without reference)
- **Dual Pipeline Support**: Two-stage generation for complex requirements
- **Quality Gates**: Multi-dimensional evaluation before and after generation
- **Safety First**: Continuous content monitoring without blocking workflow
- **Observability**: Full execution logging with performance metrics
- **Auto Migrations**: Database schema applied on container startup

## Monitoring

### Log Checkpoints
All agents emit structured log events:
- `genplan.run.start/complete`
- `genplan.complexity.determined`
- `genplan.model.selected`
- `react_prompt.run.start/complete`
- `assembly.execution.start/complete`
- `evaluator.run.start/complete`

See `docs/test/monitoring_activities.md` for full checkpoint documentation.

## Testing

```bash
# Health check
curl https://palet8-agents-kshhjydolq-uc.a.run.app/health

# List available models
curl https://palet8-agents-kshhjydolq-uc.a.run.app/planner/models

# Start conversation
curl -X POST https://palet8-agents-kshhjydolq-uc.a.run.app/chat/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user"}'
```
