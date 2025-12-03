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

---

## Planner Agent V2 Logs

**File**: `/palet8_agents/agents/planner_agent_v2.py`

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

## Assembly Service Logs

**File**: `/palet8_agents/services/assembly_service.py`

| Event | Level | Fields | Description |
|-------|-------|--------|-------------|
| `assembly.execution.start` | INFO | `job_id`, `task_id`, `pipeline_type`, `model_id`, `prompt_length`, `negative_prompt_length`, `num_images`, `width`, `height`, `has_reference` | Generation started |
| `assembly.execution.complete` | INFO | `job_id`, `task_id`, `duration_ms`, `actual_cost`, `images_count`, `pipeline_type`, `model_used` | Generation completed |
| `assembly.execution.error` | ERROR | `job_id`, `task_id`, `error`, `error_code`, `duration_ms`, `stage_failed`, `partial_cost` | Generation failed |
| `assembly.execution.timeout` | ERROR | `job_id`, `task_id`, `timeout_seconds`, `duration_ms`, `stage_at_timeout` | Generation timed out |
| `assembly.single.generation.sending` | INFO | `model`, `prompt`, `negative_prompt`, `width`, `height`, `num_images`, `steps`, `guidance_scale`, `seed` | Single pipeline request sent |
| `assembly.single.generation.received` | INFO | `images_count`, `cost_usd`, `provider`, `model_used` | Single pipeline response received |
| `assembly.dual.stage1.sending` | INFO | `model`, `purpose`, `prompt`, `negative_prompt`, `width`, `height`, `steps` | Dual pipeline stage 1 sent |
| `assembly.dual.stage1.complete` | INFO | `model`, `cost_usd`, `provider`, `has_image`, `image_url` | Dual pipeline stage 1 done |
| `assembly.dual.stage2.sending` | INFO | `model`, `purpose`, `prompt`, `input_image_url`, `width`, `height` | Dual pipeline stage 2 sent |
| `assembly.dual.stage2.complete` | INFO | `model`, `cost_usd`, `provider`, `images_count` | Dual pipeline stage 2 done |

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

### Forward Flow (Happy Path)

```
api.request.start
  └─> pali.session.start
      └─> pali.input.validated
          └─> pali.requirements.analyzed
              └─> pali.delegation.triggered (next_agent=planner)
                  └─> planner_v2.run.start (phase=initial)
                      └─> planner_v2.context.evaluated
                          └─> planner_v2.safety.classified
                              └─> planner_v2.complexity.classified
                                  └─> planner_v2.model.selected
                                      └─> planner_v2.pipeline.selected
                                          └─> planner_v2.assembly_request.created
                                              └─> evaluator_v2.prompt_eval.start
                                                  └─> evaluator_v2.prompt_eval.passed
                                                      └─> assembly.execution.start
                                                          └─> assembly.single.generation.sending
                                                              └─> assembly.single.generation.received
                                                                  └─> assembly.execution.complete
                                                                      └─> evaluator_v2.result_eval.start
                                                                          └─> evaluator_v2.result_eval.approved
api.request.complete
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
jsonPayload.event=~"pali\.|planner_v2\.|evaluator_v2\.|assembly\.|safety\."
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
| Evaluator Agent V2 | `/palet8_agents/agents/evaluator_agent_v2.py` |
| Assembly Service | `/palet8_agents/services/assembly_service.py` |
| Safety Agent | `/palet8_agents/agents/safety_agent.py` |
