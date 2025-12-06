# Log Checkpoints Documentation

## Overview

This document describes all structured log checkpoints implemented across the Palet8 Agents API multi-agent pipeline. These checkpoints enable monitoring of forward/reverse flows, detection of interruptions, errors, and timeouts.

## Log Event Naming Convention

Format: `{component}.{operation}.{outcome}`

Examples:
- `pali.session.start`
- `planner_v2.context.evaluated`
- `assembly.execution.complete`
- `evaluator_v2.result_eval.approved`
- `safety.flag.detected`

## Log Levels

| Level | Usage |
|-------|-------|
| INFO | Normal flow checkpoints (start, complete, decisions) |
| WARNING | Anomalies, retries, fix_required, safety flags |
| ERROR | Failures, policy violations, critical safety issues |
| DEBUG | High-volume events (event receipts) |

## Correlation Context

All logs automatically include correlation IDs when set:
- `job_id` - Unique job identifier
- `task_id` - Task within job
- `user_id` - User identifier
- `conversation_id` - Chat conversation ID
- `request_id` - HTTP request ID

---

## API Middleware Logs

**File**: `/src/api/middleware/logging_middleware.py`

| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `api.request.start` | INFO | `method`, `path`, `query_params`, `client_ip` | HTTP request received |
| `api.request.complete` | INFO | `method`, `path`, `status_code`, `duration_ms` | HTTP request completed |
| `api.request.error` | ERROR | `method`, `path`, `duration_ms`, `error`, `error_type` | HTTP request failed |

---

## Pali Agent Logs

**File**: `/palet8_agents/agents/pali_agent.py`

**Pali is always on as the communication layer between user and system.**

### Session & Chat Logs
| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `pali.session.start` | INFO | `job_id`, `has_input`, `input_length`, `has_conversation` | New session started |
| `pali.input.validated` | INFO | `job_id`, `is_valid`, `issues_count` | Input validation completed |
| `pali.input.validation_failed` | WARNING | `job_id`, `issues` | Input validation failed |
| `pali.requirements.analyzed` | INFO | `job_id`, `is_complete`, `missing_count`, `subject` | Requirements analysis done |
| `pali.delegation.triggered` | INFO | `job_id`, `next_agent`, `subject`, `style` | Delegating to another agent |
| `pali.clarification.requested` | INFO | `job_id`, `missing_fields`, `response_length` | Requesting user clarification |
| `pali.chat_turn.start` | INFO | `job_id`, `message_length`, `has_conversation` | Chat turn started |
| `pali.chat_turn.complete` | INFO | `job_id`, `action`, `subject` | Chat turn completed |
| `pali.chat_turn.error` | ERROR | `job_id`, `error`, `error_type` | Chat turn failed |
| `pali.run.llm_error` | ERROR | `error`, `error_type` | LLM service error |
| `pali.run.error` | ERROR | `error`, `error_type` | Unexpected error |
| `pali.requirements.analysis_error` | WARNING | `error`, `error_type` | Requirements analysis failed |
| `pali.response.generation_error` | ERROR | `error`, `error_type` | Response generation failed |

### Generation Orchestration Logs (NEW)
| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `pali.generate.start` | INFO | `job_id`, `has_requirements` | Pali receives generate request |
| `pali.generate.delegating` | INFO | `job_id`, `next_agent` | Pali delegating to Planner |
| `pali.clarification.from_planner` | INFO | `missing_fields` | Planner needs clarification |
| `pali.result.presenting` | INFO | `job_id`, `images_count` | Presenting result to user |
| `pali.user.confirmation_pending` | INFO | `job_id` | Waiting for user confirmation |
| `pali.user.confirmed` | INFO | `job_id` | User confirmed result |
| `pali.user.cancelled` | INFO | `job_id` | User cancelled |
| `pali.session.complete` | INFO | `job_id` | Session ended after confirmation |
| `pali.generate.error` | ERROR | `error`, `error_type` | Generation failed |

---

## Planner Agent V2 Logs

**File**: `/palet8_agents/agents/planner_agent_v2.py`

**Planner stays inline as the central orchestrator throughout the generation flow.**

### Phase-Based Logs (Legacy)
| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `planner_v2.run.start` | INFO | `phase`, `has_feedback`, `requirements_count` | Planner execution started |
| `planner_v2.run.error` | ERROR | `phase`, `error`, `error_type` | Planner execution failed |
| `planner_v2.context.evaluated` | INFO | `score`, `is_sufficient`, `missing_count`, `missing_fields` | Context completeness checked |
| `planner_v2.context.insufficient` | INFO | `score`, `missing_fields`, `questions_count` | Context needs clarification |
| `planner_v2.safety.classified` | INFO | `is_safe`, `risk_level`, `requires_review`, `categories` | Safety classification done |
| `planner_v2.safety.blocked` | WARNING | `reason`, `categories` | Content blocked for safety |
| `planner_v2.complexity.classified` | INFO | `complexity`, `product_type` | Complexity level determined |
| `planner_v2.post_prompt.start` | INFO | `quality_score`, `mode`, `prompt_length` | Post-prompt phase started |
| `planner_v2.model.selected` | INFO | `model_id`, `rationale`, `alternatives_count` | Model selection complete |
| `planner_v2.model.selection_failed` | WARNING | `error`, `fallback_model` | Model selection failed |
| `planner_v2.pipeline.selected` | INFO | `pipeline_type`, `stage_1_model`, `stage_2_model` | Pipeline selection complete |
| `planner_v2.pipeline.selection_failed` | WARNING | `error`, `fallback_pipeline` | Pipeline selection failed |
| `planner_v2.assembly_request.created` | INFO | `model_id`, `pipeline_type`, `prompt_length` | Assembly request built |
| `planner_v2.fix_plan.start` | INFO | `has_feedback`, `issues_count` | Fix plan phase started |

### Inline Orchestration Logs (UPDATED - Now Pure Orchestrator)
| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `planner_v2.orchestration.start` | INFO | `job_id`, `has_pali_callback` | Planner begins inline orchestration |
| `planner_v2.pipeline.selected` | INFO | `pipeline_name`, `checkpoint_count` | Pipeline method selected from config |
| `planner_v2.checkpoint.start` | INFO | `checkpoint_id`, `checkpoint_idx` | Checkpoint execution started |
| `planner_v2.checkpoint.failed` | WARNING | `checkpoint_id`, `on_fail`, `retry_count` | Checkpoint failed |
| `planner_v2.context.check` | INFO | `score`, `is_sufficient` | Context completeness check |
| `planner_v2.clarification.requesting` | INFO | `missing_fields` | Asking Pali to get clarification |
| `planner_v2.progress` | INFO | `stage`, `progress` | Progress update within orchestration |
| `planner_v2.delegate.genplan` | INFO | `job_id` | **Delegating to GenPlanAgent (NEW)** |
| `planner_v2.genplan.failed` | ERROR | `error` | GenPlan delegation failed |
| `planner_v2.genplan.error` | ERROR | `error` | GenPlan execution error |
| `planner_v2.delegate.react_prompt` | INFO | `phase`, `has_generation_plan` | Delegating to ReactPromptAgent |
| `planner_v2.react_prompt.failed` | ERROR | `error` | ReactPrompt delegation failed |
| `planner_v2.react_prompt.error` | ERROR | `error` | ReactPrompt execution error |
| `planner_v2.delegate.evaluator` | INFO | `phase`, `has_image` | Delegating to EvaluatorAgentV2 |
| `planner_v2.evaluator.error` | ERROR | `error` | Evaluator execution error |
| `planner_v2.generation.start` | INFO | `job_id`, `model_id`, `pipeline_type` | Starting image generation |
| `planner_v2.generation.complete` | INFO | `images_count` | Image generation complete |
| `planner_v2.generation.failed` | ERROR | `error` | Image generation failed |
| `planner_v2.generation.error` | ERROR | `error` | Generation execution error |
| `planner_v2.result.sending_to_pali` | INFO | `job_id` | Sending result to Pali |
| `planner_v2.orchestration.retry` | WARNING | `reason`, `retry_count` | Retrying due to evaluation failure |
| `planner_v2.accepting_with_warning` | WARNING | `checkpoint_id` | Accepting with warning |
| `planner_v2.orchestration.complete` | INFO | `job_id`, `images_count`, `retry_count` | Orchestration finished successfully |
| `planner_v2.orchestration.max_retries` | ERROR | `retry_count` | Max retries exceeded |
| `planner_v2.orchestration.error` | ERROR | `error`, `error_type` | Orchestration failed |
| `planner_v2.legacy_run` | WARNING | `phase`, `message` | Legacy run() method called |

---

## Evaluator Agent V2 Logs

**File**: `/palet8_agents/agents/evaluator_agent_v2.py`

| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `evaluator_v2.run.start` | INFO | `phase`, `has_image_data` | Evaluator execution started |
| `evaluator_v2.run.error` | ERROR | `phase`, `error`, `error_type` | Evaluator execution failed |
| `evaluator_v2.prompt_eval.start` | INFO | `mode`, `prompt_length`, `product_type` | Prompt evaluation started |
| `evaluator_v2.prompt_eval.scored` | INFO | `overall`, `decision`, `threshold`, `failed_dimensions` | Prompt quality scored |
| `evaluator_v2.prompt_eval.policy_fail` | ERROR | `failed_dimensions` | Policy violation in prompt |
| `evaluator_v2.prompt_eval.fix_required` | WARNING | `failed_dimensions`, `feedback_count` | Prompt needs revision |
| `evaluator_v2.prompt_eval.passed` | INFO | `overall_score` | Prompt passed evaluation |
| `evaluator_v2.result_eval.start` | INFO | `mode`, `has_plan` | Result evaluation started |
| `evaluator_v2.result_eval.scored` | INFO | `overall`, `decision`, `threshold`, `failed_dimensions` | Result quality scored |
| `evaluator_v2.result_eval.policy_fail` | ERROR | `failed_dimensions` | Policy violation in result |
| `evaluator_v2.result_eval.rejected` | WARNING | `overall`, `failed_dimensions`, `feedback_count`, `should_retry` | Result rejected |
| `evaluator_v2.result_eval.approved` | INFO | `overall`, `threshold` | Result approved |
| `evaluator_v2.tool.prompt_quality_failed` | WARNING | `error` | Prompt quality tool failed |
| `evaluator_v2.tool.image_evaluation_failed` | WARNING | `error` | Image evaluation tool failed |

---

## GenPlan Agent Logs (NEW)

**File**: `/palet8_agents/agents/genplan_agent.py`

**GenPlan is the generation planning agent that determines complexity, genflow, model selection, and parameters.**

| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `genplan.run.start` | INFO | `job_id`, `has_requirements` | GenPlan ReAct loop started |
| `genplan.step.action` | DEBUG | `step`, `action` | Action selected in ReAct loop |
| `genplan.complexity.start` | INFO | `job_id` | Complexity analysis started |
| `genplan.complexity.determined` | INFO | `job_id`, `complexity`, `rationale`, `source`, `triggers_found` | Complexity level determined |
| `genplan.user_info.start` | INFO | `job_id` | User info parsing started |
| `genplan.user_info.parsed` | INFO | `job_id`, `subject`, `style`, `product_type`, `has_reference`, `has_text_content`, `intents` | User requirements parsed |
| `genplan.genflow.start` | INFO | `job_id` | Genflow determination started |
| `genplan.genflow.selected` | INFO | `job_id`, `flow_type`, `flow_name`, `rationale`, `triggered_by` | Genflow (single/dual) selected |
| `genplan.model.start` | INFO | `job_id` | Model selection started |
| `genplan.model.selected` | INFO | `job_id`, `model_id`, `source`, `scenario`, `rationale`, `alternatives`, `pipeline` | Model selected |
| `genplan.parameters.start` | INFO | `job_id` | Parameter extraction started |
| `genplan.parameters.extracted` | INFO | `job_id`, `width`, `height`, `steps`, `guidance_scale`, `has_provider_params` | Generation parameters extracted |
| `genplan.validate.start` | INFO | `job_id` | Plan validation started |
| `genplan.validate.complete` | INFO | `job_id`, `is_valid`, `error_count`, `errors` | Plan validation completed |
| `genplan.run.complete` | INFO | `job_id`, `complexity`, `genflow_type`, `model_id`, `steps`, `is_valid` | GenPlan ReAct loop completed |
| `genplan.run.error` | ERROR | `error`, `error_type` | GenPlan execution failed |
| `genplan.clarification.requested` | INFO | `job_id`, `field` | User clarification requested |

---

## Genflow Service Logs

**File**: `/palet8_agents/services/genflow_service.py`

| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `genflow_service.config.loaded` | INFO | `config_path` | Genflow config loaded |
| `genflow_service.config.load_failed` | WARNING | `error`, `error_type` | Genflow config load failed |
| `genflow_service.decision.dual` | INFO | `pipeline_name`, `triggers`, `rationale` | Dual pipeline selected |

---

## React Prompt Agent Logs

**File**: `/palet8_agents/agents/react_prompt_agent.py`

**ReactPrompt now receives GenerationPlan from GenPlan agent via context.metadata["generation_plan"].**

| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `react_prompt.run.start` | INFO | `job_id`, `phase`, `has_previous_plan`, `has_generation_plan` | ReAct loop started |
| `react_prompt.init_state.from_generation_plan` | DEBUG | `mode`, `provider_params_keys` | State initialized from GenerationPlan |
| `react_prompt.run.complete` | INFO | `job_id`, `quality_score`, `quality_acceptable`, `steps`, `revision_count`, `mode` | ReAct loop completed |
| `react_prompt.run.error` | ERROR | `error`, `error_type` | ReAct loop failed |
| `react_prompt.step.action` | DEBUG | `step`, `action` | Action selected in loop |
| `react_prompt.context.start` | INFO | `job_id`, `user_id` | Context building started |
| `react_prompt.context.complete` | INFO | `job_id`, `history_count`, `art_refs_count`, `web_search_used` | Context building completed |
| `react_prompt.context.web_search_triggered` | INFO | `job_id`, `history_count`, `art_refs_count` | RAG insufficient, web search triggered |
| `react_prompt.web_search.complete` | INFO | `results_count`, `has_answer`, `provider` | Web search completed |
| `react_prompt.web_search.failed` | WARNING | `error`, `error_type` | Web search failed |
| `react_prompt.dimensions.start` | INFO | `job_id`, `mode` | Dimension selection started |
| `react_prompt.dimensions.complete` | INFO | `job_id`, `subject`, `aesthetic`, `used_fallback` | Dimensions selected |
| `react_prompt.dimensions.fallback` | WARNING | `job_id`, `subject` | Using fallback dimensions |
| `react_prompt.compose.start` | INFO | `job_id`, `mode` | Prompt composition started |
| `react_prompt.compose.complete` | INFO | `job_id`, `prompt_length`, `negative_prompt_length`, `used_fallback` | Prompt composed |
| `react_prompt.compose.fallback` | WARNING | `job_id`, `error`, `error_type` | Using fallback prompt |
| `react_prompt.quality.start` | INFO | `job_id`, `prompt_length` | Quality evaluation started |
| `react_prompt.quality.scored` | INFO | `job_id`, `overall`, `decision`, `threshold`, `failed_dimensions` | Quality scored |
| `react_prompt.refine.start` | INFO | `job_id`, `revision`, `failed_dimensions` | Prompt refinement started |
| `react_prompt.refine.complete` | INFO | `job_id`, `revision`, `new_prompt_length` | Prompt refinement completed |

---

## Assembly Service Logs

**File**: `/palet8_agents/services/assembly_service.py`

| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `assembly.execution.start` | INFO | `job_id`, `task_id`, `pipeline_type`, `model_id`, `prompt_length`, `negative_prompt_length`, `num_images`, `width`, `height`, `has_reference` | Generation started |
| `assembly.progress` | INFO | `job_id`, `task_id`, `stage`, `progress_pct`, `message` | Progress update (0-100%) |
| `assembly.execution.complete` | INFO | `job_id`, `task_id`, `duration_ms`, `actual_cost`, `images_count`, `pipeline_type`, `model_used` | Generation completed |
| `assembly.execution.error` | ERROR | `job_id`, `task_id`, `error`, `error_code`, `duration_ms`, `stage_failed`, `partial_cost` | Generation failed |
| `assembly.execution.timeout` | ERROR | `job_id`, `task_id`, `timeout_seconds`, `duration_ms`, `stage_at_timeout` | Generation timed out |
| `assembly.execution.unexpected_error` | ERROR | `job_id`, `task_id`, `error`, `error_type`, `duration_ms` | Unexpected error |
| `assembly.single.generation.sending` | INFO | `model`, `prompt`, `negative_prompt`, `width`, `height`, `num_images`, `steps`, `guidance_scale`, `seed` | Single pipeline request sent |
| `assembly.single.generation.received` | INFO | `images_count`, `cost_usd`, `provider`, `model_used` | Single pipeline response received |
| `assembly.dual.stage1.sending` | INFO | `model`, `purpose`, `prompt`, `negative_prompt`, `width`, `height`, `steps` | Dual pipeline stage 1 sent |
| `assembly.dual.stage1.complete` | INFO | `model`, `cost_usd`, `provider`, `has_image`, `image_url` | Dual pipeline stage 1 done |
| `assembly.dual.stage2.sending` | INFO | `model`, `purpose`, `prompt`, `input_image_url`, `width`, `height` | Dual pipeline stage 2 sent |
| `assembly.dual.stage2.complete` | INFO | `model`, `cost_usd`, `provider`, `images_count` | Dual pipeline stage 2 done |

### Progress Stages

The `assembly.progress` event tracks pipeline execution with `progress_pct` values:

**Single Pipeline:**
| Stage | Progress % | Description |
|-------|------------|-------------|
| `generation_start` | 0% | Starting image generation |
| `single_pipeline_start` | 20% | Generating with single model |
| `waiting_for_provider` | 40% | Waiting for provider response |
| `processing_results` | 80% | Processing results |
| `complete` | 100% | Generation complete |

**Dual Pipeline:**
| Stage | Progress % | Description |
|-------|------------|-------------|
| `generation_start` | 0% | Starting image generation |
| `dual_stage1_start` | 10% | Stage 1 started (initial generation) |
| `dual_generating_initial` | 20% | Generating initial image |
| `dual_stage2_start` | 50% | Stage 2 started (refinement) |
| `dual_refining_image` | 70% | Refining image |
| `finalizing_results` | 90% | Finalizing results |
| `complete` | 100% | Generation complete |

---

## Safety Agent Logs

**File**: `/palet8_agents/agents/safety_agent.py`

| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `safety.run.start` | INFO | `job_id`, `has_user_input`, `has_requirements`, `has_plan` | Safety check started |
| `safety.run.error` | ERROR | `job_id`, `error`, `error_type` | Safety check failed |
| `safety.monitoring.start` | INFO | `job_id` | Background monitoring started |
| `safety.event.received` | DEBUG | `event_type`, `job_id`, `data_length` | Safety event received |
| `safety.event.error` | ERROR | `job_id`, `event_type`, `error`, `error_type` | Event processing failed |
| `safety.flag.detected` | WARNING | `job_id`, `category`, `severity`, `score`, `source` | Safety flag raised |
| `safety.violation.critical` | ERROR | `job_id`, `category`, `description` | Critical violation detected |
| `safety.verdict.generated` | INFO | `job_id`, `is_safe`, `overall_score`, `flags_count`, `blocked_categories` | Final verdict generated |
| `safety.ip.detected` | WARNING | `term`, `source`, `action` | IP/trademark detected |
| `safety.config.not_found` | WARNING | `config_path` | Config file not found |
| `safety.config.load_error` | ERROR | `config_path`, `error`, `error_type` | Config load failed |
| `safety.llm_check.failed` | WARNING | `error`, `error_type` | LLM safety check failed |

---

## Flow Detection

### Forward Flow (Happy Path) - UPDATED Architecture

**Pali is always on as communication layer. Planner is pure orchestrator that delegates to GenPlan and ReactPrompt.**

```
api.request.start
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PALI (always on - communication layer)                              │
│   pali.generate.start                                               │
│   pali.generate.delegating                                          │
│   │                                                                 │
│   ▼                                                                 │
│ ┌─────────────────────────────────────────────────────────────────┐ │
│ │ PLANNER (pure orchestrator - follows pipeline_methods.yaml)     │ │
│ │   planner_v2.orchestration.start                                │ │
│ │   planner_v2.pipeline.selected                                  │ │
│ │   planner_v2.checkpoint.start (context_check)                   │ │
│ │   planner_v2.checkpoint.start (safety_check)                    │ │
│ │   │                                                             │ │
│ │   ├─→ planner_v2.delegate.genplan  (CHECKPOINT 1: NEW)          │ │
│ │   │     └─→ genplan.run.start                                   │ │
│ │   │         └─→ genplan.complexity.determined                   │ │
│ │   │             └─→ genplan.user_info.parsed                    │ │
│ │   │                 └─→ genplan.genflow.selected                │ │
│ │   │                     └─→ genplan.model.selected              │ │
│ │   │                         └─→ genplan.parameters.extracted    │ │
│ │   │                             └─→ genplan.validate.complete   │ │
│ │   │                                 └─→ genplan.run.complete    │ │
│ │   │                                                             │ │
│ │   ├─→ planner_v2.delegate.react_prompt  (CHECKPOINT 2)          │ │
│ │   │     └─→ react_prompt.run.start (receives GenerationPlan)    │ │
│ │   │         └─→ react_prompt.context.complete                   │ │
│ │   │             └─→ react_prompt.dimensions.complete            │ │
│ │   │                 └─→ react_prompt.compose.complete           │ │
│ │   │                     └─→ react_prompt.quality.scored         │ │
│ │   │                         └─→ react_prompt.run.complete       │ │
│ │   │                                                             │ │
│ │   ├─→ planner_v2.delegate.evaluator (phase=create_plan)         │ │
│ │   │     └─→ evaluator_v2.prompt_eval.start                      │ │
│ │   │         └─→ evaluator_v2.prompt_eval.passed                 │ │
│ │   │                                                             │ │
│ │   ├─→ planner_v2.generation.start  (CHECKPOINT 4)               │ │
│ │   │     └─→ assembly.execution.start                            │ │
│ │   │         └─→ assembly.progress (0% → 100%)                   │ │
│ │   │             └─→ planner_v2.generation.complete              │ │
│ │   │                                                             │ │
│ │   ├─→ planner_v2.delegate.evaluator (phase=execute)             │ │
│ │   │     └─→ evaluator_v2.result_eval.start                      │ │
│ │   │         └─→ evaluator_v2.result_eval.approved               │ │
│ │   │                                                             │ │
│ │   planner_v2.result.sending_to_pali                             │ │
│ │   planner_v2.orchestration.complete                             │ │
│ └─────────────────────────────────────────────────────────────────┘ │
│   │                                                                 │
│   pali.result.presenting                                            │
│   pali.user.confirmation_pending                                    │
│   pali.user.confirmed                                               │
│   pali.session.complete                                             │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
api.request.complete
```

### Clarification Flow

```
planner_v2.orchestration.start
  └─> planner_v2.context.check (insufficient)
      └─> planner_v2.clarification.requesting
          └─> pali.clarification.from_planner
              └─> [User provides more info]
                  └─> pali.generate.start (retry)
                      └─> planner_v2.orchestration.start
```

### Retry Flow

```
planner_v2.delegate.evaluator (phase=execute)
  └─> evaluator_v2.result_eval.rejected
      └─> planner_v2.orchestration.retry (reason=post_eval_rejected)
          └─> planner_v2.delegate.react_prompt (retry_count=1)
              └─> [Loop back through generation]
```

### Reverse Flow Indicators

| Event | Meaning |
|-------|---------|
| `planner_v2.context.insufficient` | Needs user clarification |
| `planner_v2.fix_plan.start` | Retry loop entered |
| `evaluator_v2.prompt_eval.fix_required` | Prompt needs revision |
| `evaluator_v2.result_eval.rejected` | Result failed, retrying |
| `pali.clarification.requested` | Needs more user input |

### Error/Block Indicators

| Event | Meaning |
|-------|---------|
| `planner_v2.safety.blocked` | Content blocked for safety |
| `evaluator_v2.prompt_eval.policy_fail` | Policy violation |
| `evaluator_v2.result_eval.policy_fail` | Policy violation |
| `safety.violation.critical` | Critical safety issue |
| `assembly.execution.error` | Generation failed |
| `assembly.execution.timeout` | Generation timed out |

---

## Querying Logs

### Google Cloud Logging Filters

**All agent logs:**
```
resource.type="cloud_run_revision"
resource.labels.service_name="palet8-agents"
jsonPayload.event=~"pali\.|planner_v2\.|genplan\.|evaluator_v2\.|react_prompt\.|assembly\.|safety\."
```

**GenPlan events only:**
```
resource.type="cloud_run_revision"
resource.labels.service_name="palet8-agents"
jsonPayload.event=~"genplan\."
```

**By job_id:**
```
resource.type="cloud_run_revision"
resource.labels.service_name="palet8-agents"
jsonPayload.job_id="YOUR_JOB_ID"
```

**Errors only:**
```
resource.type="cloud_run_revision"
resource.labels.service_name="palet8-agents"
severity>=ERROR
```

**Safety flags:**
```
resource.type="cloud_run_revision"
resource.labels.service_name="palet8-agents"
jsonPayload.event=~"safety\.flag|safety\.violation"
```

**Slow requests (>1s):**
```
resource.type="cloud_run_revision"
resource.labels.service_name="palet8-agents"
jsonPayload.event="api.request.complete"
jsonPayload.duration_ms>1000
```

**Generation progress:**
```
resource.type="cloud_run_revision"
resource.labels.service_name="palet8-agents"
jsonPayload.event="assembly.progress"
jsonPayload.job_id="YOUR_JOB_ID"
```

---

## Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier |
| `task_id` | string | Task within job |
| `user_id` | string | User identifier |
| `conversation_id` | string | Chat conversation ID |
| `request_id` | string | HTTP request ID |
| `duration_ms` | int | Operation duration in milliseconds |
| `progress_pct` | int | Progress percentage (0-100) |
| `score` / `overall` | float | Quality/safety score (0.0-1.0) |
| `decision` | string | Outcome (PASS, APPROVE, REJECT, FIX_REQUIRED) |
| `phase` | string | Execution phase (initial, post_prompt, fix_plan, edit) |
| `mode` | string | Prompt mode (RELAX, STANDARD, COMPLEX) |
| `cost_usd` | float | Cost in USD |
| `error` | string | Error message |
| `error_type` | string | Error class name |
| `pipeline_type` | string | single or dual |
| `model_id` / `model_used` | string | Model identifier |
| `is_safe` | boolean | Safety check result |
| `is_complete` | boolean | Requirements completeness |
| `next_agent` | string | Agent to delegate to |

---

## Implementation Files

| Component | File |
|-----------|------|
| Logger Infrastructure | `/src/utils/logger.py` |
| API Middleware | `/src/api/middleware/logging_middleware.py` |
| Pali Agent | `/palet8_agents/agents/pali_agent.py` |
| Planner Agent V2 | `/palet8_agents/agents/planner_agent_v2.py` |
| **GenPlan Agent** (NEW) | `/palet8_agents/agents/genplan_agent.py` |
| Evaluator Agent V2 | `/palet8_agents/agents/evaluator_agent_v2.py` |
| React Prompt Agent | `/palet8_agents/agents/react_prompt_agent.py` |
| Assembly Service | `/palet8_agents/services/assembly_service.py` |
| **Genflow Service** (NEW) | `/palet8_agents/services/genflow_service.py` |
| Safety Agent | `/palet8_agents/agents/safety_agent.py` |
| Pipeline Methods Config | `/config/pipeline_methods.yaml` |
| GenPlan Data Models | `/palet8_agents/models/genplan.py` |
