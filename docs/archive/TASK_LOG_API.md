# Task & Log API Documentation

Complete reference for task and agent log endpoints.

## Quick Reference

**Base URL:** `https://palet8-agents-702210710671.us-central1.run.app/agents`

### Task Endpoints
- `GET /api/tasks` - List tasks (paginated, filtered)
- `GET /api/tasks/{id}` - Get complete task details
- `POST /api/tasks/aggregate/{id}` - Trigger aggregation from logs
- `DELETE /api/tasks/{id}` - Soft delete task

### AgentLog Endpoints  
- `GET /api/agent-logs` - List agent logs (paginated, filtered)
- `GET /api/agent-logs/task/{id}` - Get all logs for a task
- `GET /api/agent-logs/{id}` - Get single log detail
- `GET /api/agent-logs/stats/routing` - Get routing analytics

## Data Models

### Task (Aggregated Report)
```typescript
{
  taskId: string              // Business ID
  originalPrompt: string      // User's prompt
  stages: StageResult[]       // All 10 pipeline stages
  promptJourney: {...}[]      // Prompt transformations
  totalDuration: number       // Total ms
  creditsCost: number         // User credit expense
  performanceBreakdown: {...} // Per-stage metrics
  evaluationResults?: {...}   // Quality scores
  generatedImageUrl?: string  // Final image
  status: string              // completed/failed
}
```

### StageResult
```typescript
{
  stage: string           // Agent name (planner, generation, etc)
  keyInput: object        // Relevant input fields
  keyOutput: object       // Relevant output fields
  duration: number        // Execution time (ms)
  creditsUsed: number     // Credits consumed
  llmTokens?: number      // LLM API tokens (NEW)
  modelName?: string      // Model used (e.g., "gemini-2.5-flash") (NEW)
  status: string          // success/failed/skipped
  reasoning?: string      // Agent reasoning/thinking
}
```

### AgentLog (Raw Execution Record)
```typescript
{
  id: string
  taskId: string
  agentName: string
  input: object           // Full input data
  output: object          // Full output data
  reasoning?: string
  executionTime: number   // ms
  status: string
  routingMode?: string    // rule/llm/hybrid
  usedLlm: boolean
  confidence?: number     // 0.0-1.0
  creditsUsed: number
  llmTokens?: number      // NEW
  modelName?: string      // NEW
  createdAt: datetime
}
```

## Key Features

### LLM Token Tracking (NEW - Nov 2025)
- **llmTokens**: Actual API token consumption
- **modelName**: Which model was used
- **Distinction**: `creditsUsed` (user billing) vs `llmTokens` (internal API cost)

### Filtering & Pagination
- All list endpoints support pagination (`page`, `page_size`)
- Filters: status, date range, agent_name, task_id, shop
- Max page_size: 100

### Task Aggregation Flow
1. Pipeline executes → creates AgentLog records
2. Orchestrator calls `POST /api/tasks/aggregate/{id}`
3. Background worker processes logs → creates Task record
4. Poll `GET /api/tasks/{id}` to check completion

## File Locations

**Backend:**
- API Routes: `src/api/routes/tasks.py`, `src/api/routes/agent_logs.py`
- Aggregator: `src/services/task_aggregator.py`
- Schema: `prisma/schema.prisma`
- Models: `src/models/schemas.py`

**Database Tables:**
- `AgentLog`: Granular per-agent logs (real-time)
- `Task`: Aggregated reports (post-processing)

## Recent Changes

**November 2025:**
- Added `llmTokens` and `modelName` fields
- Enhanced prompt journey extraction
- Improved data preservation in orchestrator
- Added tokens-by-model analytics

See full research output from Plan agent above for complete details on all functions, components, and data flows.
