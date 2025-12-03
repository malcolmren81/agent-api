"""
Task API routes.

Provides endpoints for querying comprehensive task execution logs.
Tasks aggregate data from AgentLog records for complete pipeline visibility.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from src.database import prisma
from src.utils import get_logger
from src.models.schemas import (
    TaskResponse,
    TaskListResponse,
    CreateTaskRequest
)
# TODO: Restore task_aggregator when src/services is created
# from src.services.task_aggregator import task_aggregator

logger = get_logger(__name__)
router = APIRouter()


# Component name mapping for log checkpoints
STAGE_TO_COMPONENT = {
    "interactive": "pali",
    "planner": "planner_v2",
    "prompt_manager": "react_prompt",
    "model_selection": "planner_v2",
    "generation": "assembly",
    "evaluation": "evaluator_v2",
    "product_generator": "assembly",
}


def generate_log_checkpoints(stages: List[Dict[str, Any]], task_created_at: datetime) -> List[Dict[str, Any]]:
    """
    Generate log checkpoints from stage data.

    Creates structured log checkpoints that represent the pipeline execution flow.
    """
    checkpoints = []
    current_time = task_created_at

    for stage in stages:
        stage_name = stage.get("stage", "unknown")
        component = STAGE_TO_COMPONENT.get(stage_name, stage_name)
        duration = stage.get("duration", 0)
        status = stage.get("status", "unknown")

        # Start checkpoint
        checkpoints.append({
            "event": f"{component}.run.start",
            "level": "INFO",
            "timestamp": current_time.isoformat(),
            "fields": {
                "stage": stage_name,
                "model": stage.get("modelName"),
            },
            "component": component,
        })

        # End checkpoint based on status
        from datetime import timedelta
        end_time = current_time + timedelta(milliseconds=duration)

        if status == "success":
            checkpoints.append({
                "event": f"{component}.run.complete",
                "level": "INFO",
                "timestamp": end_time.isoformat(),
                "fields": {
                    "stage": stage_name,
                    "duration_ms": duration,
                    "credits_used": stage.get("creditsUsed", 0),
                    "llm_tokens": stage.get("llmTokens"),
                },
                "component": component,
            })
        elif status == "failed":
            checkpoints.append({
                "event": f"{component}.run.error",
                "level": "ERROR",
                "timestamp": end_time.isoformat(),
                "fields": {
                    "stage": stage_name,
                    "duration_ms": duration,
                    "error": stage.get("keyOutput", {}).get("error", "Unknown error"),
                },
                "component": component,
            })

        current_time = end_time

    return checkpoints


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    shop: Optional[str] = None,
    status: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    task_id: Optional[str] = None,
) -> TaskListResponse:
    """
    List tasks with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        shop: Filter by shop domain
        status: Filter by status (completed/failed/processing)
        start_date: Filter by creation date (start)
        end_date: Filter by creation date (end)
        task_id: Filter by task ID (partial match)

    Returns:
        Paginated list of tasks
    """
    try:
        # Build where clause
        where = {}

        if shop:
            where["shop"] = shop

        if status:
            where["status"] = status

        if task_id:
            where["taskId"] = {"contains": task_id}

        if start_date or end_date:
            where["createdAt"] = {}
            if start_date:
                where["createdAt"]["gte"] = start_date
            if end_date:
                where["createdAt"]["lte"] = end_date

        # Get total count
        total = await prisma.task.count(where=where)

        # Get paginated results
        skip = (page - 1) * page_size
        tasks = await prisma.task.find_many(
            where=where,
            skip=skip,
            take=page_size,
            order={"createdAt": "desc"}
        )

        # Convert to response models
        task_responses = [
            TaskResponse(
                id=task.id,
                taskId=task.taskId,
                shop=task.shop,
                originalPrompt=task.originalPrompt,
                userRequest=task.userRequest,
                stages=task.stages,
                promptJourney=task.promptJourney,
                totalDuration=task.totalDuration,
                creditsCost=task.creditsCost,
                performanceBreakdown=task.performanceBreakdown,
                evaluationResults=task.evaluationResults,
                logCheckpoints=generate_log_checkpoints(task.stages or [], task.createdAt),
                generatedImageUrl=task.generatedImageUrl,
                mockupUrls=task.mockupUrls,
                finalPrompt=task.finalPrompt,
                status=task.status,
                errorMessage=task.errorMessage,
                createdAt=task.createdAt,
                completedAt=task.completedAt,
                updatedAt=task.updatedAt
            )
            for task in tasks
        ]

        has_more = (skip + page_size) < total

        return TaskListResponse(
            tasks=task_responses,
            total=total,
            page=page,
            pageSize=page_size,
            hasMore=has_more
        )

    except Exception as e:
        logger.error("Error fetching tasks", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    """
    Get comprehensive task details by task ID.

    Args:
        task_id: Task identifier

    Returns:
        Complete task record with all pipeline stages and results
    """
    try:
        task = await prisma.task.find_unique(
            where={"taskId": task_id}
        )

        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )

        response = TaskResponse(
            id=task.id,
            taskId=task.taskId,
            shop=task.shop,
            originalPrompt=task.originalPrompt,
            userRequest=task.userRequest,
            stages=task.stages,
            promptJourney=task.promptJourney,
            totalDuration=task.totalDuration,
            creditsCost=task.creditsCost,
            performanceBreakdown=task.performanceBreakdown,
            evaluationResults=task.evaluationResults,
            logCheckpoints=generate_log_checkpoints(task.stages or [], task.createdAt),
            generatedImageUrl=task.generatedImageUrl,
            mockupUrls=task.mockupUrls,
            finalPrompt=task.finalPrompt,
            status=task.status,
            errorMessage=task.errorMessage,
            createdAt=task.createdAt,
            completedAt=task.completedAt,
            updatedAt=task.updatedAt
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching task", task_id=task_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tasks", response_model=TaskResponse)
async def create_task(
    request: CreateTaskRequest,
    background_tasks: BackgroundTasks
) -> TaskResponse:
    """
    Create a new task record.

    This endpoint is used internally by the workflow service
    to create comprehensive task logs after workflow completion.

    Args:
        request: Task creation request
        background_tasks: FastAPI background tasks

    Returns:
        Created task record
    """
    try:
        logger.info(f"Creating task record for task_id={request.taskId}")

        task = await prisma.task.create(
            data={
                "taskId": request.taskId,
                "shop": request.shop,
                "originalPrompt": request.originalPrompt,
                "userRequest": request.userRequest or {},
                "stages": request.stages,
                "promptJourney": request.promptJourney,
                "totalDuration": request.totalDuration,
                "creditsCost": request.creditsCost,
                "performanceBreakdown": request.performanceBreakdown,
                "evaluationResults": request.evaluationResults,
                "generatedImageUrl": request.generatedImageUrl,
                "mockupUrls": request.mockupUrls or [],
                "finalPrompt": request.finalPrompt,
                "status": request.status,
                "errorMessage": request.errorMessage,
                "completedAt": request.completedAt or datetime.now()
            }
        )

        logger.info(f"Successfully created task record for task_id={request.taskId}")

        response = TaskResponse(
            id=task.id,
            taskId=task.taskId,
            shop=task.shop,
            originalPrompt=task.originalPrompt,
            userRequest=task.userRequest,
            stages=task.stages,
            promptJourney=task.promptJourney,
            totalDuration=task.totalDuration,
            creditsCost=task.creditsCost,
            performanceBreakdown=task.performanceBreakdown,
            evaluationResults=task.evaluationResults,
            logCheckpoints=generate_log_checkpoints(task.stages or [], task.createdAt),
            generatedImageUrl=task.generatedImageUrl,
            mockupUrls=task.mockupUrls,
            finalPrompt=task.finalPrompt,
            status=task.status,
            errorMessage=task.errorMessage,
            createdAt=task.createdAt,
            completedAt=task.completedAt,
            updatedAt=task.updatedAt
        )

        return response

    except Exception as e:
        logger.error("Error creating task", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# TODO: Restore when task_aggregator is available
# @router.post("/tasks/aggregate/{task_id}")
# async def aggregate_task_from_logs(
#     task_id: str,
#     background_tasks: BackgroundTasks,
#     shop: str = Query("default.myshopify.com"),
#     original_prompt: str = Query(""),
#     generated_image_url: Optional[str] = None
# ) -> dict:
#     """
#     Trigger task aggregation from AgentLog records.
#
#     This endpoint aggregates AgentLog records into a comprehensive Task record.
#     The aggregation runs in the background to avoid blocking the response.
#
#     Args:
#         task_id: Task identifier
#         background_tasks: FastAPI background tasks
#         shop: Shop domain
#         original_prompt: User's original prompt
#         generated_image_url: URL of generated image
#
#     Returns:
#         Status message
#     """
#     try:
#         logger.info(f"Triggering task aggregation for task_id={task_id}")
#
#         # Queue aggregation in background
#         background_tasks.add_task(
#             task_aggregator.create_task_from_logs,
#             task_id=task_id,
#             shop=shop,
#             original_prompt=original_prompt,
#             generated_image_url=generated_image_url
#         )
#
#         return {
#             "status": "queued",
#             "message": f"Task aggregation queued for task_id={task_id}",
#             "task_id": task_id
#         }
#
#     except Exception as e:
#         logger.error("Error queuing task aggregation", error=str(e), exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def delete_task(task_id: str) -> dict:
    """
    Soft delete a task (marks as deleted, doesn't remove from database).

    Args:
        task_id: Task identifier

    Returns:
        Deletion confirmation
    """
    try:
        task = await prisma.task.find_unique(
            where={"taskId": task_id}
        )

        if not task:
            raise HTTPException(
                status_code=404,
                detail=f"Task {task_id} not found"
            )

        # Update status to indicate deletion
        await prisma.task.update(
            where={"taskId": task_id},
            data={"status": "deleted"}
        )

        logger.info(f"Soft deleted task task_id={task_id}")

        return {
            "status": "success",
            "message": f"Task {task_id} deleted successfully",
            "task_id": task_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting task", task_id=task_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
