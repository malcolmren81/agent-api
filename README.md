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
| Image Generation | Runware API |

## Architecture

```
User Request
    |
+---------------------------------------------------------------+
|  PALI AGENT - Conversational requirement gathering             |
|  - Multi-turn chat with users                                  |
|  - UI selector integration (template, aesthetic, character)    |
|  - Requirements validation & completeness tracking             |
+---------------------------------------------------------------+
    |
+---------------------------------------------------------------+
|  PLANNER AGENT - Central decision making                       |
|  - Context evaluation & RAG (user history, art library)        |
|  - Prompt mode selection (RELAX/STANDARD/COMPLEX)              |
|  - Model selection based on capability/cost analysis           |
|  - Safety classification                                       |
+---------------------------------------------------------------+
    |
+---------------------------------------------------------------+
|  ASSEMBLY SERVICE - Image Generation                           |
|  - Single/Dual pipeline execution                              |
|  - Primary: Runware API (Flux, SDXL models)                    |
|  - Cost tracking per generation                                |
+---------------------------------------------------------------+
    |
+---------------------------------------------------------------+
|  EVALUATOR AGENT - Quality gates                               |
|  - Prompt quality assessment (before generation)               |
|  - Result quality scoring (after generation)                   |
|  - Approval/rejection decisions                                |
+---------------------------------------------------------------+
    |
+---------------------------------------------------------------+
|  SAFETY AGENT - Continuous monitoring (non-blocking)           |
|  - NSFW, violence, hate speech detection                       |
|  - IP/trademark violation checks                               |
|  - Risk categorization & recommendations                       |
+---------------------------------------------------------------+
    |
Approved Image + Metadata
```

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

### Health & Monitoring
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Service health with component status |
| `/docs` | GET | OpenAPI documentation |

## Project Structure

```
agents-api/
+-- palet8_agents/              # Core agent system
|   +-- agents/                 # Agent implementations
|   |   +-- pali_agent.py       # Requirement gathering
|   |   +-- planner_agent.py    # Decision making
|   |   +-- evaluator_agent.py  # Quality gates
|   |   +-- safety_agent.py     # Safety monitoring
|   +-- core/                   # Framework (BaseAgent, Context, Message)
|   +-- models/                 # Data models (shared across agents)
|   |   +-- requirements.py     # RequirementsStatus
|   |   +-- context.py          # ContextCompleteness
|   |   +-- safety.py           # SafetyClassification, SafetyFlag
|   |   +-- prompt.py           # PromptDimensions, PromptQualityResult
|   |   +-- generation.py       # GenerationParameters, PipelineConfig
|   |   +-- evaluation.py       # EvaluationPlan, ResultQualityResult
|   +-- services/               # Business logic services
|   |   +-- text_llm_service.py           # LLM text generation
|   |   +-- reasoning_service.py          # Complex reasoning tasks
|   |   +-- image_generation_service.py   # Image gen via Runware
|   |   +-- assembly_service.py           # Pipeline orchestration
|   |   +-- prompt_composer_service.py    # Prompt construction
|   |   +-- embedding_service.py          # Vector embeddings
|   +-- tools/                  # Agent tools
|       +-- context_tool.py     # RAG, memory, references
|       +-- safety_tool.py      # Content safety checks
|       +-- registry.py         # Tool registry
+-- src/
|   +-- api/
|   |   +-- main.py             # FastAPI app
|   |   +-- routes/             # API endpoints
|   |       +-- chat.py         # Chat API (Pali integration)
|   |       +-- tasks.py        # Task logs API
|   |       +-- agent_logs.py   # Agent execution logs
|   |       +-- templates.py    # Prompt templates
|   +-- database/               # Prisma client
|   +-- models/                 # Pydantic schemas
|   +-- utils/                  # Logger, metrics, health check
+-- prisma/
|   +-- schema.prisma           # Database schema
+-- migrations/
|   +-- init.sql                # Database migrations (auto-run)
+-- config/                     # YAML configs
|   +-- safety_config.yaml      # Safety rules
|   +-- image_models_config.yaml # Model registry
+-- Dockerfile
+-- requirements.txt
```

## Database Models

| Model | Purpose |
|-------|---------|
| **Job** | Agent task execution (status, requirements, plan, evaluation) |
| **Conversation** | Multi-turn chat sessions |
| **ChatMessage** | Conversation history |
| **Design** | Generated design outputs |
| **Task** | Full pipeline logs with performance metrics |
| **AgentLog** | Per-agent execution tracking |
| **Template** | Reusable prompt templates |

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

## Deployment

The service uses automatic database migrations on startup via `migrations/init.sql`.

```bash
# Deploy to Cloud Run
gcloud run deploy palet8-agents \
  --source . \
  --region us-central1 \
  --project palet8-system \
  --allow-unauthenticated

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
| `PALET8_API_URL` | Palet8 backend API URL |
| `PALET8_API_KEY` | Palet8 backend API key |

## AI Models

### Reasoning (via OpenRouter)
- Gemini 2.0 Flash (primary)
- GPT-4o (fallback)
- Automatic routing and fallback handling

### Image Generation (via Runware)
- Flux Pro models
- SDXL variants
- Unified cost tracking

## Key Features

- **Cost Optimization**: Automatic model selection balances quality, speed, and cost
- **Quality Gates**: Multi-dimensional evaluation before and after generation
- **Safety First**: Continuous content monitoring without blocking workflow
- **Observability**: Full execution logging with performance metrics
- **Auto Migrations**: Database schema applied on container startup
- **Fallback Handling**: Automatic model switching on failures

## Testing

```bash
# Health check
curl https://palet8-agents-kshhjydolq-uc.a.run.app/health

# Start conversation
curl -X POST https://palet8-agents-kshhjydolq-uc.a.run.app/chat/start \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user"}'

# Query tasks
curl https://palet8-agents-kshhjydolq-uc.a.run.app/api/tasks
```
