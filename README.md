# Palet8 Agents API

Multi-agent orchestration service for AI-powered design generation. Transforms user design requests into production-ready product images through intelligent agent coordination.

## Overview

| | |
|---|---|
| **Service** | palet8-agents |
| **Routes** | `/agents/*` |
| **Stack** | Python, FastAPI, Prisma |
| **Port** | 8000 |
| **Platform** | Google Cloud Run |

## Architecture

```
User Request
    ↓
┌─────────────────────────────────────────────────────────────┐
│  PALI AGENT - Conversational requirement gathering          │
│  • Multi-turn chat with users                               │
│  • UI selector integration (template, aesthetic, character) │
│  • Requirements validation & completeness tracking          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  PLANNER AGENT - Central decision making                    │
│  • Context evaluation & RAG (user history, art library)     │
│  • Prompt mode selection (RELAX/STANDARD/COMPLEX)           │
│  • Model selection based on capability/cost analysis        │
│  • Safety classification                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  IMAGE GENERATION                                           │
│  • Primary: Flux Pro 1.1 (fast, cost-effective)             │
│  • Fallback: Imagen 3 (photorealistic)                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  EVALUATOR AGENT - Quality gates                            │
│  • Prompt quality assessment (before generation)            │
│  • Result quality scoring (after generation)                │
│  • Approval/rejection decisions                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│  SAFETY AGENT - Continuous monitoring (non-blocking)        │
│  • NSFW, violence, hate speech detection                    │
│  • IP/trademark violation checks                            │
│  • Risk categorization & recommendations                    │
└─────────────────────────────────────────────────────────────┘
    ↓
Approved Image + Metadata
```

## Agents

### Pali Agent
User-facing design assistant that gathers requirements through conversation.
- Collects: subject, style, colors, mood, composition, elements
- Integrates with UI selectors (product category, template, aesthetic)
- Tracks requirement completeness with scoring

### Planner Agent
Central decision-maker that orchestrates the generation pipeline.
- Evaluates context and performs RAG lookups
- Selects prompt mode and image dimensions
- Chooses optimal model based on complexity and cost
- Creates execution plan for downstream services

### Evaluator Agent
Quality gate that ensures output meets standards.
- **Pre-generation**: Validates prompt clarity, coverage, product constraints
- **Post-generation**: Scores prompt fidelity, technical quality, aesthetics
- Approves, requests fixes, or rejects outputs

### Safety Agent
Continuous content safety monitor.
- Risk categories: NSFW, violence, hate, IP violations, illegal content
- Severity levels: none → low → medium → high → critical
- Context-aware evaluation (understands negative prompts)
- Non-blocking by default, escalates when necessary

## API Endpoints

### Chat API
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat/start` | POST | Start conversation with Pali agent |
| `/chat/message` | POST | Send message, receive SSE stream |
| `/chat/history/{id}` | GET | Get conversation history |
| `/chat/generate` | POST | Trigger full generation pipeline |

### Health & Monitoring
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health with component status |
| `/metrics` | GET | Prometheus metrics |

### Workflow & Logs
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/workflow/{task_id}` | GET | Get complete workflow data |
| `/api/agent-logs` | GET | Query agent execution logs |
| `/api/tasks` | GET | Task query and status |

## AI Models

All model calls go through unified API providers (no direct model integrations):

### Reasoning (via OpenRouter)
- Gemini 2.0 Flash, GPT-4o, Claude, and other models
- Automatic routing and fallback handling

### Image Generation (via Runware)
- Flux Pro 1.1, SDXL, and other models
- Unified image generation API

### Embeddings (direct)
- Only embedding calls use direct API integration

## Local Development

### Prerequisites
- Python 3.11+
- Node.js 20+ (for Prisma)
- API keys: OPENROUTER_API_KEY, FLUX_API_KEY

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

```bash
gcloud run deploy palet8-agents \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-secrets="OPENROUTER_API_KEY=openrouter-api-key:latest,FLUX_API_KEY=flux-api-key:latest"
```

## Project Structure

```
agents-api/
├── palet8_agents/              # Core agent system
│   ├── agents/                 # Agent implementations
│   │   ├── pali_agent.py       # Requirement gathering
│   │   ├── planner_agent.py    # Decision making
│   │   ├── evaluator_agent.py  # Quality gates
│   │   └── safety_agent.py     # Safety monitoring
│   ├── core/                   # Framework (BaseAgent, Context, Message)
│   ├── services/               # LLM, embedding, prompt services
│   └── tools/                  # Agent tools (search, image, context)
├── src/
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── websocket.py        # Real-time updates
│   │   └── routes/             # API endpoints
│   │       ├── chat.py         # Chat API (Pali integration)
│   │       ├── workflow.py     # Workflow tracking
│   │       └── agent_logs.py   # Execution logs
│   ├── database/               # Prisma client
│   ├── models/                 # Pydantic schemas
│   └── utils/                  # Logger, metrics, health check
├── prisma/
│   └── schema.prisma           # Database schema
├── config/                     # YAML configs (safety, evaluation, models)
├── docs/archive/               # Archived old code
├── Dockerfile
└── requirements.txt
```

## Database Models

- **Conversation** - Multi-turn chat sessions
- **ChatMessage** - Conversation history
- **Job** - Agent task execution (status, requirements, plan, evaluation)
- **AgentLog** - Per-agent execution tracking
- **Task** - Full pipeline logs with performance metrics

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key (Gemini, GPT-4o) | Yes |
| `FLUX_API_KEY` | Black Forest Labs Flux API | Yes |
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `RUNWARE_API_KEY` | Runware image API | Optional |

## Key Features

- **Cost Optimization**: Automatic model selection balances quality, speed, and cost
- **Quality Gates**: Multi-dimensional evaluation before and after generation
- **Safety First**: Continuous content monitoring without blocking workflow
- **Observability**: Full execution logging with performance metrics
- **Fallback Handling**: Automatic model switching on failures
