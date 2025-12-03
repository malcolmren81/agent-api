"""
Interactive Agent API routes.
"""
from uuid import uuid4
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from src.models.schemas import InteractiveRequest, InteractiveResponse, FullPipelineResponse
from src.api.orchestrator import AgentOrchestrator
from src.services.task_aggregator import TaskAggregator
from src.utils import get_logger
from typing import Any
import numpy as np
import json

logger = get_logger(__name__)
router = APIRouter()

# Global orchestrator instance
orchestrator = AgentOrchestrator()


def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert NumPy types and other non-JSON-serializable types to native Python types.

    Args:
        obj: Object to sanitize

    Returns:
        JSON-serializable version of the object
    """
    if obj is None:
        return None
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, 'item'):  # Other NumPy scalars
        return obj.item()
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # For other types, try to convert to string or return as-is
        try:
            return str(obj)
        except:
            return obj


async def aggregate_task_in_background(
    task_id: str,
    shop: str,
    original_prompt: str,
    orchestrator_result: dict = None
):
    """
    Background task to aggregate AgentLog records into Task record.

    This runs AFTER the HTTP response is sent to ensure zero impact on workflow timing.
    All errors are caught and logged to prevent any disruption to the main flow.

    Args:
        task_id: Workflow task ID
        shop: Shop domain
        original_prompt: Original user prompt
        orchestrator_result: Full result from orchestrator (contains credit cost, image URLs, etc.)
    """
    import asyncio

    try:
        print(f"🔥 BACKGROUND TASK STARTED: task_id={task_id}, shop={shop}")

        # CRITICAL FIX: Wait 2 seconds to ensure AgentLog records are committed to database
        # This prevents race condition where aggregator queries before Prisma commits are complete
        print(f"⏳ Waiting 2 seconds for database commits to complete...")
        await asyncio.sleep(2)

        logger.info(
            "Starting background task aggregation",
            task_id=task_id,
            shop=shop
        )

        # Extract credit cost and image data from orchestrator result
        actual_credit_cost = None
        generated_image_url = None
        mockup_urls = []

        if orchestrator_result:
            # Extract actual credit cost from credits object
            credits = orchestrator_result.get("credits", {})
            if isinstance(credits, dict):
                # Calculate actual deduction from balance difference
                balance_before = credits.get("balance_before")
                balance_after = credits.get("balance_after")
                if balance_before is not None and balance_after is not None:
                    actual_credit_cost = balance_before - balance_after
                else:
                    # Fallback to cost if balances not available
                    actual_credit_cost = credits.get("cost")

            # Extract generated image URL from best_image
            best_image = orchestrator_result.get("best_image")
            if best_image and isinstance(best_image, dict):
                generated_image_url = best_image.get("url")

            # Extract mockup URLs from products
            products = orchestrator_result.get("products", [])
            if products:
                for product in products:
                    if isinstance(product, dict) and "mockup_url" in product:
                        mockup_urls.append(product["mockup_url"])

        aggregator = TaskAggregator()
        print(f"🔥 Calling aggregator.create_task_from_logs()...")
        result = await aggregator.create_task_from_logs(
            task_id=task_id,
            shop=shop,
            original_prompt=original_prompt,
            actual_credit_cost=actual_credit_cost,
            generated_image_url=generated_image_url,
            mockup_urls=mockup_urls
        )

        if result:
            # STRUCTURED SUCCESS LOGGING: Cloud Run can track success metrics
            print(f"✅ BACKGROUND_TASK_SUCCESS task_id={task_id} shop={shop} status=completed")
            logger.info(
                "Background task aggregation completed successfully",
                task_id=task_id,
                shop=shop,
                status="completed",
                metric_name="background_task_success"
            )
        else:
            # STRUCTURED WARNING: aggregator returned None (could indicate timing issue)
            print(f"⚠️ BACKGROUND_TASK_WARNING task_id={task_id} shop={shop} status=no_task_created reason=aggregator_returned_none")
            logger.warning(
                "Background task aggregation returned None - no Task record created",
                task_id=task_id,
                shop=shop,
                status="no_task_created",
                metric_name="background_task_no_result"
            )

    except Exception as e:
        # Catch ALL errors to prevent disrupting the main workflow
        # Log but never raise - this is fire-and-forget
        # STRUCTURED ERROR LOGGING: Cloud Run can alert on these patterns
        error_type = type(e).__name__
        error_message = str(e)

        print(f"❌ BACKGROUND_TASK_FAILURE task_id={task_id} shop={shop} error_type={error_type} error=\"{error_message}\"")
        print(f"   Full error details:")
        import traceback
        print(traceback.format_exc())

        logger.error(
            "Background task aggregation failed - main workflow unaffected but Task record not created",
            task_id=task_id,
            shop=shop,
            error_type=error_type,
            error=error_message,
            status="failed",
            metric_name="background_task_failure",
            exc_info=True
        )


@router.post("/run")
async def run_interactive_agent(request: InteractiveRequest):
    """
    Submit a user prompt to the interactive agent (SSE STREAMING).

    Processes request synchronously while streaming real-time progress updates
    via Server-Sent Events (SSE). Connection stays open until completion.

    Args:
        request: Interactive agent request

    Returns:
        StreamingResponse with SSE events for progress and final result
    """
    async def event_stream():
        """Generate SSE events for pipeline execution."""
        import asyncio

        task_id = str(uuid4())

        try:
            # DEBUG: Log that route handler is called
            print(f"🔍 ROUTE HANDLER CALLED: /interactive-agent/run for task {task_id}")
            logger.info(
                "Interactive agent SSE streaming request received",
                task_id=task_id,
                user_id=request.user_id,
                prompt_length=len(request.prompt),
                reasoning_model=request.reasoning_model,
                image_model=request.image_model,
            )

            # Send initial event with task_id
            yield f"data: {json.dumps({'type': 'started', 'task_id': task_id, 'status': 'processing'})}\n\n"

            # DEBUG: Log before orchestrator call
            print(f"🔍 ABOUT TO CALL orchestrator.run_full_pipeline() for task {task_id}")

            # Run pipeline as async task
            pipeline_task = asyncio.create_task(
                orchestrator.run_full_pipeline(
                    user_prompt=request.prompt,
                    user_id=request.user_id,
                    customer_id=request.customer_id,
                    email=request.email,
                    shop_domain=getattr(request, 'shop_domain', None),
                    reasoning_model=request.reasoning_model,
                    image_model=request.image_model,
                    num_images=request.num_images or 2,
                    task_id=task_id,  # Pass task_id so orchestrator uses same ID
                )
            )

            # Send keepalive pings while pipeline runs
            while not pipeline_task.done():
                yield f": keepalive\n\n"
                await asyncio.sleep(5)  # Send ping every 5 seconds

            # Get result
            print(f"🔍 PIPELINE TASK DONE for task {task_id}")
            result = await pipeline_task
            print(f"🔍 PIPELINE RESULT RECEIVED for task {task_id}, type: {type(result)}")
            print(f"🔍 RESULT KEYS: {list(result.keys()) if isinstance(result, dict) else 'NOT A DICT'}")
            print(f"🔍 SUCCESS VALUE: {result.get('success') if isinstance(result, dict) else 'N/A'}")

            # Sanitize result
            sanitized_result = sanitize_for_json(result)
            sanitized_result["task_id"] = task_id
            print(f"🔍 SANITIZED RESULT SUCCESS: {sanitized_result.get('success')}")

            # Send final result event
            print(f"🔍 ABOUT TO CHECK success flag...")
            if sanitized_result.get("success"):
                print(f"🔍 SUCCESS=True, sending 'complete' event...")
                yield f"data: {json.dumps({'type': 'complete', **sanitized_result})}\n\n"
                print(f"🔍 'complete' event yielded")
                logger.info(f"[SSE] Task {task_id} completed successfully")

                # Aggregate task logs in background (fire-and-forget)
                shop = getattr(request, 'shop_domain', None) or "unknown"
                asyncio.create_task(
                    aggregate_task_in_background(
                        task_id=task_id,
                        shop=shop,
                        original_prompt=request.prompt,
                        orchestrator_result=sanitized_result
                    )
                )
            else:
                print(f"🔍 SUCCESS=False, sending 'error' event. Error: {sanitized_result.get('error')}")
                yield f"data: {json.dumps({'type': 'error', 'task_id': task_id, 'error': sanitized_result.get('error', 'Unknown error'), 'stage': sanitized_result.get('stage', 'unknown')})}\n\n"
                logger.error(f"[SSE] Task {task_id} failed: {sanitized_result.get('error')}")

                # Aggregate failed task logs too
                shop = getattr(request, 'shop_domain', None) or "unknown"
                asyncio.create_task(
                    aggregate_task_in_background(
                        task_id=task_id,
                        shop=shop,
                        original_prompt=request.prompt,
                        orchestrator_result=sanitized_result
                    )
                )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[SSE] Task {task_id} exception: {error_msg}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'task_id': task_id, 'error': error_msg, 'stage': 'exception'})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.post("/run-full", response_model=FullPipelineResponse)
async def run_full_pipeline(
    request: InteractiveRequest,
    background_tasks: BackgroundTasks
) -> FullPipelineResponse:
    """
    Execute the complete agent pipeline synchronously.

    This endpoint runs all 7 agents in sequence:
    1. Interactive Agent - Validates input
    2. Planner Agent - Creates execution plan
    3. Prompt Manager - Optimizes prompts
    4. Model-Selection - Chooses optimal models
    5. Generation Agent - Creates images
    6. Evaluation Agent - Scores images
    7. Product Generator - Creates mockups

    Args:
        request: Interactive agent request
        background_tasks: FastAPI background tasks for post-processing

    Returns:
        Complete pipeline result with products
    """
    try:
        logger.info(
            "Full pipeline request received",
            user_id=request.user_id,
            prompt_length=len(request.prompt),
        )

        # Execute full pipeline
        result = await orchestrator.run_full_pipeline(
            user_prompt=request.prompt,
            user_id=request.user_id,
            customer_id=request.customer_id,
            email=request.email,
            shop_domain=getattr(request, 'shop_domain', None),
            reasoning_model=request.reasoning_model,
            image_model=request.image_model,
            num_images=request.num_images or 2,
        )

        if not result.get("success"):
            # Extract only JSON-serializable fields from evaluations
            # Convert NumPy types to native Python types
            def to_python_type(obj):
                """Convert NumPy/other types to native Python types."""
                if hasattr(obj, 'item'):  # NumPy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: to_python_type(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [to_python_type(item) for item in obj]
                return obj

            evaluations = result.get("all_evaluations", [])
            clean_evaluations = []
            for eval_data in evaluations:
                clean_eval = {
                    "image_index": to_python_type(eval_data.get("image_index")),
                    "overall_score": to_python_type(eval_data.get("overall_score")),
                    "total_score": to_python_type(eval_data.get("total_score")),
                    "score": to_python_type(eval_data.get("score")),
                    "approved": to_python_type(eval_data.get("approved")),
                    "objective_scores": to_python_type(eval_data.get("objective_scores", {})),
                    "subjective_scores": to_python_type(eval_data.get("subjective_scores", {})),
                    "feedback": eval_data.get("feedback", ""),
                    "base64_data": eval_data.get("base64_data", ""),  # Include image for debugging
                }
                clean_evaluations.append(clean_eval)

            raise HTTPException(
                status_code=400,
                detail={
                    "error": result.get("error"),
                    "stage": result.get("stage"),
                    "all_evaluations": clean_evaluations,
                    "best_score": to_python_type(result.get("best_score", 0)),
                    "refinement_attempts": result.get("refinement_attempts", 0),
                },
            )

        # Sanitize result to remove NumPy types before Pydantic serialization
        sanitized_result = sanitize_for_json(result)

        # Schedule background task aggregation (runs AFTER response is sent)
        # This ensures zero impact on workflow timing
        task_id = sanitized_result.get("task_id")
        shop = getattr(request, 'shop_domain', None) or "unknown"

        if task_id:
            background_tasks.add_task(
                aggregate_task_in_background,
                task_id=task_id,
                shop=shop,
                original_prompt=request.prompt,
                orchestrator_result=sanitized_result
            )
            logger.info(
                "Background task aggregation scheduled",
                task_id=task_id,
                shop=shop
            )
        else:
            logger.warning(
                "No task_id in result, skipping background aggregation",
                result_keys=list(sanitized_result.keys())
            )

        return FullPipelineResponse(**sanitized_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Full pipeline error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result/{task_id}")
async def get_task_result(task_id: str):
    """
    Legacy endpoint for backward compatibility.

    The /run endpoint now uses SSE streaming and returns results directly.
    This endpoint is no longer used but kept for backward compatibility.

    Args:
        task_id: Task ID

    Returns:
        Message indicating SSE streaming is used
    """
    return {
        "task_id": task_id,
        "status": "not_available",
        "message": "Results are now streamed via SSE. Connect to /run endpoint which returns a streaming response with real-time progress and final results."
    }
