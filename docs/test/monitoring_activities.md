# Pipeline Monitoring Activities

This document describes all monitored activities from user click to output confirmation, with corresponding log checkpoints.

## Architecture Overview

**NEW (Inline Orchestration):**
- **Pali** is always on as the communication layer (user ↔ system)
- **Planner** stays inline as the central orchestrator (doesn't exit until complete)
- **Specialized agents** (GenPlan, ReactPrompt, Evaluator, Safety) are invoked only at checkpoints

```
User ←→ /chat/generate ←→ PALI (always on)
                              │
                              ▼
                         PLANNER (pure orchestrator)
                              │
    ┌─────────────┬───────────┼───────────┬─────────────┐
    ▼             ▼           ▼           ▼             ▼
 GenPlan    ReactPrompt  AssemblyService  Evaluator   Safety
(planning)   (prompt)     (generation)   (quality)   (safety)
```

**Agent Responsibilities:**
- **Planner**: Pure orchestrator - reads pipeline_methods.yaml, calls agents at checkpoints, handles retries, routes clarification requests
- **GenPlan**: Generation planning - complexity, genflow (single/dual), model selection, parameters
- **ReactPrompt**: Context evaluation, clarification questions, context enrichment, dimension selection, prompt composition
- **Evaluator**: Quality gates - pre-gen prompt evaluation, post-gen result evaluation
- **Safety**: Content safety checks

---

## Main Generation Flow (Happy Path)

**Key Roles:**
- **Pali (always on)**: Communication layer - stays active throughout the session
- **Planner (pure orchestrator)**: Coordinates pipeline - delegates to specialized agents at checkpoints, routes clarification requests
- **GenPlan**: Generation planning - complexity, genflow, model selection, parameters
- **ReactPrompt**: Context evaluation, clarification questions, context enrichment, prompt composition
- **Evaluator**: Quality gates - prompt and result evaluation

| Step | Activity (user/system action) | Agent/Service | Log checkpoint |
|------|------------------------------|---------------|----------------|
| 1 | User clicks **Generate**; the front-end sends a generation request to the API | API middleware | `api.request.start` |
| 2 | **Pali session starts (stays on throughout)** | Pali agent | `pali.session.start` |
| 3 | Pali receives generation request | Pali agent | `pali.generate.start` |
| 4 | Pali validates the user's input and conversation context | Pali agent | `pali.input.validated` |
| 5 | Pali analyses requirements (subject, aesthetic, etc.) | Pali agent | `pali.requirements.analyzed` |
| 6 | **Pali delegates to Planner for inline orchestration** | Pali agent → Planner agent | `pali.generate.delegating` |
| 7 | **Planner starts inline orchestration (stays active until complete)** | Planner agent | `planner_v2.orchestration.start` |
| 8 | Planner selects pipeline method from config | Planner agent | `planner_v2.pipeline.selected` |
| 9 | Planner performs basic context gate check | Planner agent | `planner_v2.checkpoint.start` (context_check) |
| 10 | Planner performs safety classification on the inputs | Planner agent / Safety tool | `planner_v2.checkpoint.start` (safety_check) |
| 11 | **Planner delegates generation planning to GenPlan agent (checkpoint)** | Planner agent → GenPlan | `planner_v2.delegate.genplan` |
| 12 | GenPlan starts ReAct loop | GenPlan agent | `genplan.run.start` |
| 13 | GenPlan analyzes complexity level (simple/standard/complex) | GenPlan agent | `genplan.complexity.determined` |
| 14 | GenPlan parses user requirements into structured info | GenPlan agent | `genplan.user_info.parsed` |
| 15 | GenPlan determines genflow (single vs dual pipeline) | GenPlan agent | `genplan.genflow.selected` |
| 16 | GenPlan selects optimal model based on scenario | GenPlan agent | `genplan.model.selected` |
| 17 | GenPlan extracts generation parameters (steps, guidance, dimensions) | GenPlan agent | `genplan.parameters.extracted` |
| 18 | GenPlan validates the complete plan | GenPlan agent | `genplan.validate.complete` |
| 19 | GenPlan returns GenerationPlan to Planner | GenPlan agent | `genplan.run.complete` |
| 20 | **Planner delegates prompt building to ReactPrompt agent (checkpoint)** | Planner agent → ReactPrompt | `planner_v2.delegate.react_prompt` |
| 21 | ReactPrompt starts ReAct loop (receives GenerationPlan from context) | ReactPrompt agent | `react_prompt.run.start` |
| 22 | ReactPrompt evaluates context sufficiency based on complexity | ReactPrompt agent | `react_prompt.context.evaluate.complete` |
| 23 | ReactPrompt generates clarification questions if context insufficient | ReactPrompt agent | `react_prompt.questions.generate.complete` (if needed) |
| 24 | ReactPrompt builds context (art refs, history, web search if needed) | ReactPrompt agent | `react_prompt.context.complete` |
| 25 | ReactPrompt selects dimensions (subject, aesthetic, style) | ReactPrompt agent | `react_prompt.dimensions.complete` |
| 26 | ReactPrompt composes the final prompt and negative prompt | ReactPrompt agent | `react_prompt.compose.complete` |
| 27 | ReactPrompt evaluates prompt quality | ReactPrompt agent | `react_prompt.quality.scored` |
| 28 | ReactPrompt returns PromptPlan to Planner | ReactPrompt agent | `react_prompt.run.complete` |
| 29 | **Planner delegates to Evaluator for pre-gen check (checkpoint)** | Planner agent → Evaluator agent | `planner_v2.delegate.evaluator` |
| 30 | Prompt quality is scored against thresholds | Evaluator agent | `evaluator_v2.prompt_eval.start` |
| 31 | Prompt evaluation scored | Evaluator agent | `evaluator_v2.prompt_eval.scored` |
| 32 | Prompt evaluation passes, returns to Planner | Evaluator agent | `evaluator_v2.prompt_eval.passed` |
| 33 | **Planner executes generation via Assembly service (checkpoint)** | Planner agent → Assembly service | `planner_v2.generation.start` |
| 34 | Progress: Starting image generation (0%) | Assembly service | `assembly.progress` (stage=generation_start) |
| 35 | Progress: Single pipeline started (20%) | Assembly service | `assembly.progress` (stage=single_pipeline_start) |
| 36 | Generation request is sent to the provider | Assembly service | `assembly.single.generation.sending` |
| 37 | Progress: Waiting for provider response (40%) | Assembly service | `assembly.progress` (stage=waiting_for_provider) |
| 38 | Single-pipeline response is received from the provider | Assembly service | `assembly.single.generation.received` |
| 39 | Progress: Processing results (80%) | Assembly service | `assembly.progress` (stage=processing_results) |
| 40 | Progress: Generation complete (100%) | Assembly service | `assembly.progress` (stage=complete) |
| 41 | Assembly service returns images to Planner | Planner agent | `planner_v2.generation.complete` |
| 42 | **Planner delegates to Evaluator for post-gen check (checkpoint)** | Planner agent → Evaluator agent | `planner_v2.delegate.evaluator` |
| 43 | Result quality is scored | Evaluator agent | `evaluator_v2.result_eval.start` |
| 44 | Result evaluation scored | Evaluator agent | `evaluator_v2.result_eval.scored` |
| 45 | Result evaluation approves the image, returns to Planner | Evaluator agent | `evaluator_v2.result_eval.approved` |
| 46 | **Planner sends result to Pali for user presentation** | Planner agent → Pali agent | `planner_v2.result.sending_to_pali` |
| 47 | **Pali presents result to user** | Pali agent | `pali.result.presenting` |
| 48 | **Pali waits for user confirmation** | Pali agent | `pali.user.confirmation_pending` |
| 49 | User confirms result | Pali agent | `pali.user.confirmed` |
| 50 | **Planner orchestration completes** | Planner agent | `planner_v2.orchestration.complete` |
| 51 | **Pali session completes** | Pali agent | `pali.session.complete` |
| 52 | The API request completes; response returned to front-end | API middleware | `api.request.complete` |

---

## Dual Pipeline Flow (Steps 24-31 Alternative)

| Step | Activity (user/system action) | Agent/Service | Log checkpoint |
|------|------------------------------|---------------|----------------|
| 24 | Progress: Starting image generation (0%) | Assembly service | `assembly.progress` (stage=generation_start) |
| 25 | Progress: Stage 1 started (10%) | Assembly service | `assembly.progress` (stage=dual_stage1_start) |
| 26 | Stage 1 request sent to provider | Assembly service | `assembly.dual.stage1.sending` |
| 27 | Progress: Generating initial image (20%) | Assembly service | `assembly.progress` (stage=dual_generating_initial) |
| 28 | Stage 1 response received | Assembly service | `assembly.dual.stage1.complete` |
| 29 | Progress: Stage 2 started (50%) | Assembly service | `assembly.progress` (stage=dual_stage2_start) |
| 30 | Stage 2 request sent to provider | Assembly service | `assembly.dual.stage2.sending` |
| 31 | Progress: Refining image (70%) | Assembly service | `assembly.progress` (stage=dual_refining_image) |
| 32 | Stage 2 response received | Assembly service | `assembly.dual.stage2.complete` |
| 33 | Progress: Finalizing results (90%) | Assembly service | `assembly.progress` (stage=finalizing_results) |
| 34 | Progress: Complete (100%) | Assembly service | `assembly.progress` (stage=complete) |
| 35 | Assembly execution complete | Assembly service | `assembly.execution.complete` |

---

## Edit/Revision Flow (User Requests Changes)

| Step | Activity (user/system action) | Agent/Service | Log checkpoint |
|------|------------------------------|---------------|----------------|
| E1 | User sends edit request (e.g., "make it more colorful") | API middleware | `api.request.start` |
| E2 | Pali agent receives the edit message in existing conversation | Pali agent | `pali.chat_turn.start` |
| E3 | Pali agent validates the edit input | Pali agent | `pali.input.validated` |
| E4 | Pali agent analyses updated requirements with edit context | Pali agent | `pali.requirements.analyzed` |
| E5 | Pali agent delegates edit task to Planner | Pali agent → Planner agent | `pali.delegation.triggered` |
| E6 | Pali chat turn completes | Pali agent | `pali.chat_turn.complete` |
| E7 | Planner starts fix-plan phase with user feedback | Planner agent | `planner_v2.fix_plan.start` |
| E8 | Planner re-evaluates context with edit requirements | Planner agent | `planner_v2.context.evaluated` |
| E9 | React Prompt starts refinement with feedback | React Prompt agent | `react_prompt.refine.start` |
| E10 | React Prompt incorporates edit feedback into prompt | React Prompt agent | `react_prompt.refine.complete` |
| E11 | React Prompt re-evaluates refined prompt quality | React Prompt agent | `react_prompt.quality.scored` |
| E12 | React Prompt completes with updated prompt plan | React Prompt agent | `react_prompt.run.complete` |
| E13 | Planner updates model/pipeline selection if needed | Planner agent | `planner_v2.model.selected` |
| E14 | Planner creates new AssemblyRequest with refined prompt | Planner agent | `planner_v2.assembly_request.created` |
| E15 | Prompt re-evaluation begins | Evaluator agent | `evaluator_v2.prompt_eval.start` |
| E16 | Prompt passes evaluation | Evaluator agent | `evaluator_v2.prompt_eval.passed` |
| E17 | New generation execution starts | Assembly service | `assembly.execution.start` |
| E18-E25 | *(Same as steps 24-31 in main flow)* | Assembly service | `assembly.progress` → `assembly.execution.complete` |
| E26 | Result evaluation on new image | Evaluator agent | `evaluator_v2.result_eval.start` |
| E27 | Result approved | Evaluator agent | `evaluator_v2.result_eval.approved` |
| E28 | API request completes with new result | API middleware | `api.request.complete` |

---

## Auto-Revision Flow (System Rejects & Retries)

| Step | Activity (user/system action) | Agent/Service | Log checkpoint |
|------|------------------------------|---------------|----------------|
| R1 | Prompt evaluation fails quality threshold | Evaluator agent | `evaluator_v2.prompt_eval.fix_required` |
| R2 | Planner enters fix-plan phase automatically | Planner agent | `planner_v2.fix_plan.start` |
| R3 | React Prompt refines the prompt based on feedback | React Prompt agent | `react_prompt.refine.start` |
| R4 | React Prompt completes refinement | React Prompt agent | `react_prompt.refine.complete` |
| R5 | *(Loop back to step 20 - prompt evaluation)* | Evaluator agent | `evaluator_v2.prompt_eval.start` |
| — | | | |
| R6 | Result evaluation rejects generated image | Evaluator agent | `evaluator_v2.result_eval.rejected` |
| R7 | System decides whether to retry generation | Evaluator agent | (check `should_retry` field) |
| R8 | *(If retry: loop back to step 23 - assembly.execution.start)* | Assembly service | `assembly.execution.start` |

---

## Clarification Flow (Insufficient Context)

When ReactPrompt determines that the gathered context is insufficient (based on complexity level from GenPlan), it generates clarification questions and returns early to Planner. Planner routes the clarification request to Pali, who asks the user. Pali determines the appropriate message type (`ui_selector` or `general`) and generates a natural language response.

### Message Types

| Type | When | Example |
|------|------|---------|
| `ui_selector` | Template UI exists for missing field | "Pick a style from the options!" |
| `general` | No template found | "What mood do you want?" |

### Frontend Contract

Message event structure when clarification is needed:

```json
{
  "type": "message",
  "role": "assistant",
  "content": "Pick a style from the options!",
  "metadata": {
    "requires_input": true,
    "message_type": "ui_selector" | "general",
    "selector_id": "aesthetic_style" | null,
    "missing_fields": ["style"]
  }
}
```

### Selector IDs

| selector_id | UI Component | Field |
|-------------|--------------|-------|
| `product_category` | Product type picker | product_type |
| `aesthetic_style` | Style selector | style |
| `aspect_ratio` | Dimension picker | dimensions |
| `system_character` | Character selector | character |
| `reference_image` | Image upload | reference_image |
| `text_in_image` | Text input form | text_content |

### Field Priority (Multiple Missing)

When multiple fields are missing, Pali asks the highest priority first:

| Priority | Field | Reason |
|----------|-------|--------|
| 1 | `subject` | Core requirement - can't proceed without it |
| 2 | `product_type` | Affects all downstream choices |
| 3 | `style` | Major visual decision |
| 4 | `dimensions` | Layout decision |
| 5 | `character` | Character selection |
| 6 | `mood` | Emotional direction |
| 7 | `colors` | Color preferences |
| 8 | `reference_image` | Visual reference |
| 9 | `text_content` | Text overlay content |

### Clarification Flow Steps

| Step | Activity (user/system action) | Agent/Service | Log checkpoint |
|------|------------------------------|---------------|----------------|
| C1 | ReactPrompt evaluates context sufficiency based on complexity | ReactPrompt agent | `react_prompt.context.evaluate.complete` |
| C2 | ReactPrompt finds context insufficient, generates questions | ReactPrompt agent | `react_prompt.questions.generate.complete` |
| C3 | ReactPrompt returns clarification result to Planner | ReactPrompt agent | `react_prompt.clarification.needed` |
| C4 | Planner routes clarification request to Pali | Planner agent | `planner_v2.clarification.routing` |
| C5 | **Pali generates clarification response (with message type)** | Pali agent | `pali.clarification.generated` |
| C6 | **Pali sends clarification to frontend** | Chat route | `pali.clarification.sent` |
| C7 | Job status updated to AWAITING_CLARIFICATION | Chat route | `job.status.updated` |
| C8 | Front-end displays message + UI selector (if applicable) | Front-end | — |
| C9 | User answers questions or selects options | Front-end | — |
| C10 | User submits response; new API request starts | API middleware | `api.request.start` |
| C11 | Pali receives user's answer in chat turn | Pali agent | `pali.chat_turn.start` |
| C12 | Pali validates the new input | Pali agent | `pali.input.validated` |
| C13 | Pali re-analyses requirements with new context | Pali agent | `pali.requirements.analyzed` |
| C14 | **Auto-resume triggered (if was awaiting clarification)** | Chat route | `pali.generation.auto_resuming` |
| C15 | **Internal generation call** | Chat route | `pali.generation.internal_trigger` |
| C16 | *(Continue from main generation flow)* | Planner agent | `planner_v2.orchestration.start` |

### Checkpoint Data Reference

| Checkpoint | Data Fields |
|------------|-------------|
| `react_prompt.context.evaluate.complete` | `available_count`, `missing_count`, `score`, `needs_clarification`, `priority_field` |
| `react_prompt.questions.generate.complete` | `question_count`, `priority_field` |
| `react_prompt.clarification.needed` | `missing_fields`, `question_count` |
| `planner_v2.clarification.routing` | `missing_fields`, `priority_field`, `question_count` |
| `pali.clarification.generated` | `message_type`, `selector_id`, `prioritized_field`, `total_missing_fields` |
| `pali.clarification.llm_failed` | `error`, `message_type`, `using_fallback: true` |
| `pali.clarification.from_planner` | `message_type`, `selector_id`, `missing_fields` |
| `pali.clarification.sent` | `conversation_id`, `message_type`, `selector_id` |
| `pali.generation.auto_resuming` | `conversation_id` |
| `pali.generation.internal_trigger` | `job_id`, `conversation_id` |

### Templated Question Types (UI Selectors)

Pali can trigger UI selector components for common clarification needs:

| Question Type | selector_id | Description |
|--------------|-------------|-------------|
| Product Type Selector | `product_category` | Grid of product categories |
| Style Picker | `aesthetic_style` | Visual style options |
| Dimension Picker | `aspect_ratio` | Image dimension options |
| Character Selector | `system_character` | Character/persona options |
| Reference Image Upload | `reference_image` | Drag & drop or browse |
| Text Input Form | `text_in_image` | Text content for overlays |

---

## Error/Block Flow

| Step | Activity (user/system action) | Agent/Service | Log checkpoint |
|------|------------------------------|---------------|----------------|
| X1 | Safety check blocks content | Planner agent | `planner_v2.safety.blocked` |
| X2 | Prompt violates policy | Evaluator agent | `evaluator_v2.prompt_eval.policy_fail` |
| X3 | Result violates policy | Evaluator agent | `evaluator_v2.result_eval.policy_fail` |
| X4 | Critical safety violation detected | Safety agent | `safety.violation.critical` |
| X5 | Generation fails with error | Assembly service | `assembly.execution.error` |
| X6 | Generation times out | Assembly service | `assembly.execution.timeout` |

---

## Stop Flow (User Cancels Generation)

When a user requests to stop an in-progress generation, the system gracefully cancels while preserving the conversation.

| Step | Activity (user/system action) | Agent/Service | Log checkpoint |
|------|------------------------------|---------------|----------------|
| S1 | User clicks **Stop** during generation | Front-end | — |
| S2 | Stop request sent to `/chat/stop/{conversation_id}` | API middleware | `api.request.start` |
| S3 | **Pali handles stop request** | Pali agent | `pali.stop.requested` |
| S4 | Job status updated to CANCELLED | Chat route | `job.status.updated` |
| S5 | Pali generates friendly stop confirmation | Pali agent | — |
| S6 | **Generation stopped, conversation preserved** | Chat route | `pali.generation.stopped` |
| S7 | API request completes with confirmation | API middleware | `api.request.complete` |

### Stop Checkpoint Data

| Checkpoint | Data Fields |
|------------|-------------|
| `pali.stop.requested` | `job_id`, `reason` |
| `pali.generation.stopped` | `conversation_id`, `job_id` |

### Valid Job Statuses for Stop

Stop is only valid when job status is one of:
- `GENERATING` - Active image generation
- `PLANNING` - Planner orchestrating
- `AWAITING_CLARIFICATION` - Waiting for user input

---

## Flow Diagram Summary

```
User Click Generate
        │
        ▼
┌─────────────────┐
│ api.request.start│
└────────┬────────┘
         │
         ▼
╔═══════════════════════════════════════════════════════════════════════════╗
║                    PALI AGENT (ALWAYS ON)                                 ║
║                    Communication Layer                                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  pali.session.start                                                       ║
║  pali.generate.start                                                      ║
║  pali.input.validated                                                     ║
║  pali.requirements.analyzed                                               ║
║  pali.generate.delegating                                                 ║
║         │                                                                 ║
║         ▼                                                                 ║
║  ┌─────────────────────────────────────────────────────────────────────┐  ║
║  │              PLANNER V2 (PURE ORCHESTRATOR)                         │  ║
║  │              Follows pipeline_methods.yaml checkpoints              │  ║
║  ├─────────────────────────────────────────────────────────────────────┤  ║
║  │  planner_v2.orchestration.start                                     │  ║
║  │  planner_v2.pipeline.selected                                       │  ║
║  │  planner_v2.checkpoint.start (context_check)                        │  ║
║  │         │                                                           │  ║
║  │         ├──────────▶ [Insufficient?] ──▶ pali.clarification.requested  ║
║  │         │                                    │                      │  ║
║  │         │            ┌───────────────────────┘                      │  ║
║  │         │            ▼                                              │  ║
║  │         │     User answers via Pali                                 │  ║
║  │         │            │                                              │  ║
║  │         │◀───────────┘ (loop back with context)                     │  ║
║  │         │                                                           │  ║
║  │  planner_v2.checkpoint.start (safety_check)                         │  ║
║  │         │                                                           │  ║
║  │         ▼ (CHECKPOINT 1: Generation Planning)                       │  ║
║  │  ┌─────────────────────────────────────────┐                        │  ║
║  │  │ → GENPLAN AGENT (NEW)                   │                        │  ║
║  │  │   genplan.run.start                     │                        │  ║
║  │  │   genplan.complexity.determined         │                        │  ║
║  │  │   genplan.user_info.parsed              │                        │  ║
║  │  │   genplan.genflow.selected              │                        │  ║
║  │  │   genplan.model.selected                │                        │  ║
║  │  │   genplan.parameters.extracted          │                        │  ║
║  │  │   genplan.validate.complete             │                        │  ║
║  │  │   genplan.run.complete                  │                        │  ║
║  │  └──────────────────┬──────────────────────┘                        │  ║
║  │                     │                                               │  ║
║  │         ▼ (CHECKPOINT 2: Prompt Building)                           │  ║
║  │  ┌─────────────────────────────────────────┐                        │  ║
║  │  │ → REACT PROMPT AGENT                    │                        │  ║
║  │  │   (receives GenerationPlan from GenPlan)│                        │  ║
║  │  │   react_prompt.run.start                │                        │  ║
║  │  │   react_prompt.context.evaluate.complete│────▶ [INSUFFICIENT?]   │  ║
║  │  │   react_prompt.questions.generate.complete     │                 │  ║
║  │  │         │                               │      │                 │  ║
║  │  │         │◀──────────────────────────────│──────┘ (clarification) │  ║
║  │  │   react_prompt.context.complete         │                        │  ║
║  │  │   react_prompt.dimensions.complete      │                        │  ║
║  │  │   react_prompt.compose.complete         │                        │  ║
║  │  │   react_prompt.quality.scored           │                        │  ║
║  │  │   react_prompt.run.complete             │                        │  ║
║  │  └──────────────────┬──────────────────────┘                        │  ║
║  │                     │                                               │  ║
║  │         ▼ (CHECKPOINT 3: Pre-Gen Evaluation)                        │  ║
║  │  ┌─────────────────────────────────────────┐                        │  ║
║  │  │ → EVALUATOR AGENT (phase=create_plan)   │                        │  ║
║  │  │   evaluator_v2.prompt_eval.start        │                        │  ║
║  │  │   evaluator_v2.prompt_eval.scored       │────▶ [FIX_REQUIRED?]   │  ║
║  │  │   evaluator_v2.prompt_eval.passed       │           │            │  ║
║  │  └──────────────────┬──────────────────────┘           │            │  ║
║  │                     │                                  │            │  ║
║  │                     │◀─────────────────────────────────┘ (retry)    │  ║
║  │                     │                                               │  ║
║  │         ▼ (CHECKPOINT 4: Generation)                                │  ║
║  │  ┌─────────────────────────────────────────┐                        │  ║
║  │  │ → ASSEMBLY SERVICE                      │                        │  ║
║  │  │   planner_v2.generation.start           │                        │  ║
║  │  │   assembly.progress (0%→100%)           │                        │  ║
║  │  │   assembly.single.generation.sending    │                        │  ║
║  │  │   assembly.single.generation.received   │                        │  ║
║  │  │   planner_v2.generation.complete        │                        │  ║
║  │  └──────────────────┬──────────────────────┘                        │  ║
║  │                     │                                               │  ║
║  │         ▼ (CHECKPOINT 5: Post-Gen Evaluation)                       │  ║
║  │  ┌─────────────────────────────────────────┐                        │  ║
║  │  │ → EVALUATOR AGENT (phase=execute)       │                        │  ║
║  │  │   evaluator_v2.result_eval.start        │                        │  ║
║  │  │   evaluator_v2.result_eval.scored       │────▶ [REJECTED?]       │  ║
║  │  │   evaluator_v2.result_eval.approved     │           │            │  ║
║  │  └──────────────────┬──────────────────────┘           │            │  ║
║  │                     │                                  │            │  ║
║  │                     │◀─────────────────────────────────┘ (retry)    │  ║
║  │                     │                                               │  ║
║  │  planner_v2.result.sending_to_pali                                  │  ║
║  │  planner_v2.orchestration.complete                                  │  ║
║  └─────────────────────┬───────────────────────────────────────────────┘  ║
║                        │                                                  ║
║  pali.result.presenting                                                   ║
║  pali.user.confirmation_pending                                           ║
║         │                                                                 ║
║         ├───────────────────▶ [User Edits?] ──▶ pali.edit.received        ║
║         │                                            │                    ║
║         │                     ┌──────────────────────┘                    ║
║         │                     ▼                                           ║
║         │              Back to Planner (edit loop)                        ║
║         │                                                                 ║
║         ▼                                                                 ║
║  pali.user.confirmed                                                      ║
║  pali.session.complete                                                    ║
╚═════════════════════════════════════════════════════════════════════════╦═╝
                  │
                  ▼
         ┌─────────────────┐
         │api.request.complete│
         └────────┬────────┘
                  │
                  ▼
            User Sees Result
```

### Key Design Principles

1. **Pali Always On**: Pali wraps the entire session as the communication layer between user and system
2. **Planner Pure Orchestrator**: Planner reads pipeline_methods.yaml and delegates to specialized agents at checkpoints
3. **Planner Internal Todo List**: Planner maintains an internal todo list to track checkpoint progress, with status changes logged for visibility
4. **GenPlan for Planning**: GenPlan agent handles all generation planning (complexity, genflow, model, parameters)
5. **ReactPrompt for Prompts & Context**: ReactPrompt evaluates context sufficiency, generates clarification questions, builds context, and composes prompts
6. **Checkpoint Architecture**: Specialized agents (GenPlan, ReactPrompt, Evaluator) are invoked only at specific checkpoints
7. **Clarification via ReactPrompt**: When context is insufficient, ReactPrompt generates questions and returns to Planner for routing through Pali
8. **Clarification via GenPlan**: When complexity is ambiguous, GenPlan generates a complexity selector question routed through Planner/Pali
9. **Retry Loops**: Planner handles evaluation failures with automatic retry logic (max 3 retries)
10. **User Confirmation**: Session doesn't complete until user explicitly confirms or requests edits

---

## Planner Internal Todo List

The Planner maintains an internal todo list to track checkpoint progress throughout the orchestration:

### Todo List Checkpoints

| Checkpoint | Description | Data Fields |
|------------|-------------|-------------|
| `planner_v2.todo.initialized` | Todo list created from pipeline checkpoints | `job_id`, `todo_count`, `todos` |
| `planner_v2.todo.in_progress` | Checkpoint started | `job_id`, `checkpoint_id`, `description`, `todos` |
| `planner_v2.todo.completed` | Checkpoint completed successfully | `job_id`, `checkpoint_id`, `description`, `progress`, `todos` |
| `planner_v2.todo.failed` | Checkpoint failed | `job_id`, `checkpoint_id`, `description`, `error`, `todos` |
| `planner_v2.todo.all_complete` | All checkpoints completed | `job_id`, `progress`, `todos` |

### Todo Item Structure

```json
{
  "id": "checkpoint_id",
  "description": "Human-readable description",
  "status": "pending|in_progress|completed|failed|skipped",
  "result": {},
  "error": null
}
```

### Progress Tracking

The todo list provides real-time progress tracking:
- **progress_pct**: Percentage of completed items (0.0 - 1.0)
- **total**: Total number of checkpoints
- **completed**: Number of successfully completed checkpoints
- **failed**: Number of failed checkpoints
- **pending**: Number of remaining checkpoints

---

## GenPlan Agent Checkpoints

The GenPlan agent uses a ReAct (Reason-Act-Observe) loop for generation planning:

| Checkpoint | Description | Data Fields |
|------------|-------------|-------------|
| `genplan.run.start` | GenPlan ReAct loop starts | `job_id`, `has_requirements` |
| `genplan.step.action` | Next action determined | `step`, `action` |
| `genplan.complexity.determined` | Complexity level determined | `complexity`, `rationale`, `source`, `triggers_found` |
| `genplan.user_info.parsed` | User requirements parsed | `subject`, `style`, `product_type`, `has_reference`, `intents` |
| `genplan.genflow.selected` | Genflow (single/dual) selected | `flow_type`, `flow_name`, `rationale`, `triggered_by` |
| `genplan.model.selected` | Model selected | `model_id`, `scenario`, `rationale`, `alternatives` |
| `genplan.parameters.extracted` | Generation parameters extracted | `width`, `height`, `steps`, `guidance_scale`, `has_provider_params` |
| `genplan.validate.complete` | Plan validated | `is_valid`, `error_count`, `errors` |
| `genplan.run.complete` | GenPlan returns GenerationPlan | `complexity`, `genflow_type`, `model_id`, `steps`, `is_valid` |
| `genplan.run.error` | GenPlan error | `error`, `error_type` |
| `genplan.clarification.requested` | User clarification needed | `field` |

### GenPlan Complexity Clarification

When complexity is not provided in requirements and cannot be confidently determined, GenPlan asks the user via a selector question:

| Step | Activity | Agent | Checkpoint |
|------|----------|-------|------------|
| G1 | GenPlan checks if complexity clarification needed | GenPlan | (internal check) |
| G2 | GenPlan generates complexity selector question | GenPlan | `genplan.clarification.requested` |
| G3 | Planner routes clarification to Pali | Planner | `planner_v2.clarification.routing` |
| G4 | Pali presents selector to user | Pali | `pali.clarification.sent` |
| G5 | User selects complexity (Quick/Standard/Complex) | Frontend | — |
| G6 | Resume with user's complexity choice | Planner | `planner_v2.orchestration.start` |

**Complexity Selector Options:**
| Value | Label | Description |
|-------|-------|-------------|
| `simple` | Quick/Relax | Fast generation, less detail |
| `standard` | Standard | Balanced quality and speed |
| `complex` | Complex/Pro | Maximum detail and quality |

**selector_id:** `generation_mode`

### GenPlan Output: GenerationPlan

The GenerationPlan passed to ReactPrompt contains:
- `complexity`: simple | standard | complex
- `complexity_rationale`: Why this complexity was chosen
- `user_info`: Parsed user requirements (UserParseResult)
- `genflow`: Pipeline selection (GenflowConfig with flow_type, flow_name)
- `model_id`: Selected model identifier
- `model_rationale`: Why this model was chosen
- `model_alternatives`: Other viable model options
- `model_input_params`: Standard params (width, height, steps, guidance_scale)
- `provider_params`: Model-specific params (quality, stylize, etc.)
- `pipeline`: PipelineConfig with stage details
- `is_valid`: Whether the plan passed validation

---

## How to Query Checkpoint Logs

Logs are emitted to GCP Cloud Logging using structlog with an `event` field for checkpoint names.

### Basic Query (Recent Logs)

```bash
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="palet8-agents"' \
  --project=palet8-system \
  --limit=50 \
  --freshness=15m \
  | grep -A2 -B2 "event"
```

### Filter by Checkpoint Event

```bash
# All Pali events
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="palet8-agents" jsonPayload.event=~"pali.*"' \
  --project=palet8-system \
  --limit=50 \
  --freshness=15m

# All Planner events
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="palet8-agents" jsonPayload.event=~"planner_v2.*"' \
  --project=palet8-system \
  --limit=50 \
  --freshness=15m

# All GenPlan events (NEW)
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="palet8-agents" jsonPayload.event=~"genplan.*"' \
  --project=palet8-system \
  --limit=50 \
  --freshness=15m

# All ReactPrompt events
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="palet8-agents" jsonPayload.event=~"react_prompt.*"' \
  --project=palet8-system \
  --limit=50 \
  --freshness=15m

# All Assembly events
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="palet8-agents" jsonPayload.event=~"assembly.*"' \
  --project=palet8-system \
  --limit=50 \
  --freshness=15m
```

### Find Errors

```bash
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="palet8-agents" severity>=ERROR' \
  --project=palet8-system \
  --limit=50 \
  --freshness=30m
```

### Formatted Table Output

```bash
gcloud logging read 'resource.type="cloud_run_revision" resource.labels.service_name="palet8-agents"' \
  --project=palet8-system \
  --limit=100 \
  --freshness=5m \
  --format="table(timestamp,severity,jsonPayload.event,jsonPayload.message)"
```

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--freshness=15m` | Only logs from last 15 minutes |
| `--limit=50` | Maximum number of log entries |
| `--format="table(...)"` | Custom output format |
| `jsonPayload.event` | The checkpoint event name (structlog field) |
| `jsonPayload.message` | Human-readable log message |

### Example Output

```
event: pali.session.start
event: pali.input.validated
event: pali.requirements.analyzed
  is_complete: true
event: pali.generate.delegating
event: planner_v2.orchestration.start
event: planner_v2.context.check
  is_sufficient: false          ← Context insufficient, triggers clarification
event: planner_v2.clarification.requesting
event: api.request.complete
```
