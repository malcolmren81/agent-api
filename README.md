# Agents API - AI Orchestration Service

The Agents API orchestrates multi-agent workflows for AI-powered image generation using Gemini 2.0 Flash, GPT-4o, Flux 1 Kontext, and Imagen 3.

## Overview

- **Service Name**: palet8-agents
- **Routes**: `/agents/*`
- **Tech Stack**: Python, FastAPI
- **Port**: 8000
- **Platform**: Google Cloud Run

## Features

- Multi-agent workflow orchestration
- Intelligent model selection (Gemini vs GPT-4o)
- Dual image generation (Flux 1 Kontext + Imagen 3)
- Cost-performance optimization
- Automatic fallback handling
- Prompt management and refinement

## Architecture

The service uses a multi-agent architecture:

1. **Planner Agent** - Analyzes requests and creates execution plans
2. **Prompt Manager Agent** - Refines and optimizes prompts
3. **Model Selection Agent** - Chooses optimal AI models
4. **Generation Agent** - Executes image generation
5. **Evaluation Agent** - Scores and ranks generated images
6. **Refinement Agent** - Improves results based on feedback

## Local Development

### Prerequisites

- Python 3.10+
- Google Cloud credentials (for Gemini/Imagen 3)
- OpenAI API key (for GPT-4o)
- Flux API key

### Setup

1. **Install dependencies**
   ```bash
   cd services/agents-api
   pip install -r requirements.txt
   ```

2. **Configure environment variables**
   ```bash
   cp .env.example .env
   ```

   Edit `.env`:
   ```env
   GEMINI_API_KEY=your_gemini_key
   OPENAI_API_KEY=your_openai_key
   FLUX_API_KEY=your_flux_key
   PYTHONUNBUFFERED=1
   ```

3. **Run development server**
   ```bash
   uvicorn src.main:app --reload --port 8000
   ```

   Server will start at `http://localhost:8000`

## Deployment

Deploy to Google Cloud Run:

```bash
cd services/agents-api

gcloud run deploy palet8-agents \
  --source . \
  --region=us-central1 \
  --platform=managed \
  --allow-unauthenticated \
  --port=8000 \
  --memory=2Gi \
  --cpu=2 \
  --max-instances=10 \
  --timeout=300s \
  --set-env-vars="PYTHONUNBUFFERED=1" \
  --set-secrets="GEMINI_API_KEY=gemini-api-key:latest,FLUX_API_KEY=flux-api-key:latest,OPENAI_API_KEY=openai-api-key:latest"
```

## API Endpoints

### Health Check
```
GET /agents/health
Response: { "status": "healthy", "agents": [...] }
```

### Image Generation
```
POST /agents/v1/generate
Request: {
  "prompt": "Design description",
  "style": "modern",
  "num_images": 2
}
Response: {
  "success": true,
  "images": [...],
  "metadata": {...}
}
```

### Workflow Execution
```
POST /agents/v1/workflow
Executes full multi-agent workflow
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes |
| `FLUX_API_KEY` | Flux API key | Yes |
| `PYTHONUNBUFFERED` | Python output buffering | No |

## Project Structure

```
services/agents-api/
├── src/
│   ├── agents/              # Agent implementations
│   │   ├── planner_agent.py
│   │   ├── prompt_manager_agent.py
│   │   ├── model_selection_agent.py
│   │   ├── generation_agent.py
│   │   ├── evaluation_agent.py
│   │   └── refinement_agent.py
│   ├── connectors/          # AI model connectors
│   │   ├── gemini_reasoning.py
│   │   ├── chatgpt_reasoning.py
│   │   ├── flux_image.py
│   │   └── gemini_image.py
│   ├── models/              # Data models
│   │   └── schemas.py
│   ├── orchestrator.py      # Workflow orchestration
│   ├── main.py             # FastAPI app
│   └── utils.py            # Utilities
├── requirements.txt
└── README.md              # This file
```

## AI Models

### Reasoning Models
- **Gemini 2.0 Flash**: 33x cheaper, 1M context window, thinking budget
- **GPT-4o**: Advanced reasoning, code generation

### Image Generation Models
- **Flux 1 Kontext**: 8x faster inference, excellent style transfer
- **Imagen 3**: Superior photorealism, multi-image blending

## Cost Optimization

The Model Selection Agent automatically chooses the most cost-effective model based on:
- Task complexity
- Required features
- Performance requirements
- Cost constraints

Default strategy: Gemini 2.0 Flash for reasoning, Flux for images.

## Error Handling

Automatic fallback to alternative models on failure:
- If Flux fails → Falls back to Imagen 3
- If Gemini fails → Falls back to GPT-4o

## Monitoring

Key metrics to monitor:
- Agent execution time
- Model selection rationale
- Generation success rate
- Cost per request
- API latency

## Related Documentation

- [Main README](../../README.md)
- [Routing Architecture](../../docs/architecture/routing.md)
- [Agent Implementation Details](src/agents/)

## Support

For issues specific to the Agents API, create an issue on GitHub.
