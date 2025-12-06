"""
AgentLog API routes.

Provides endpoints for querying and analyzing agent execution logs.
"""
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from src.database import prisma
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()



# Response models
class AgentLogResponse(BaseModel):
    """Single agent log entry"""
    id: str
    shop: str
    taskId: str
    agentName: str
    input: dict
    output: dict
    reasoning: Optional[str]
    executionTime: int
    status: str
    routingMode: Optional[str]
    usedLlm: bool
    confidence: Optional[float]
    fallbackUsed: bool
    creditsUsed: int
    llmTokens: Optional[int]
    createdAt: datetime


class AgentLogListResponse(BaseModel):
    """Paginated list of agent logs"""
    logs: List[AgentLogResponse]
    total: int
    page: int
    pageSize: int
    hasMore: bool


class RoutingStatsResponse(BaseModel):
    """Routing statistics"""
    totalRequests: int
    ruleBasedCount: int
    llmBasedCount: int
    hybridCount: int
    averageExecutionTime: float
    totalCreditsUsed: int
    averageConfidence: float


@router.get("/agent-logs", response_model=AgentLogListResponse)
async def list_agent_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    agent_name: Optional[str] = None,
    status: Optional[str] = None,
    routing_mode: Optional[str] = None,
    task_id: Optional[str] = None,
    shop: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
) -> AgentLogListResponse:
    """
    List agent logs with filtering and pagination.

    Args:
        page: Page number (1-indexed)
        page_size: Number of results per page
        agent_name: Filter by agent name
        status: Filter by status (success/failed/skipped)
        routing_mode: Filter by routing mode (rule/llm/hybrid)
        task_id: Filter by task ID
        shop: Filter by shop domain
        start_date: Filter by creation date (start)
        end_date: Filter by creation date (end)

    Returns:
        Paginated list of agent logs
    """
    try:

        # Build where clause
        where = {}
        if agent_name:
            where["agentName"] = agent_name
        if status:
            where["status"] = status
        if routing_mode:
            where["routingMode"] = routing_mode
        if task_id:
            where["taskId"] = task_id
        if shop:
            where["shop"] = shop
        if start_date or end_date:
            where["createdAt"] = {}
            if start_date:
                where["createdAt"]["gte"] = start_date
            if end_date:
                where["createdAt"]["lte"] = end_date

        # Get total count
        total = await prisma.agentlog.count(where=where)

        # Get paginated results
        skip = (page - 1) * page_size
        logs = await prisma.agentlog.find_many(
            where=where,
            skip=skip,
            take=page_size,
            order={"createdAt": "desc"}
        )

        # Convert to response models
        log_responses = [
            AgentLogResponse(
                id=log.id,
                shop=log.shop,
                taskId=log.taskId,
                agentName=log.agentName,
                input=log.input,
                output=log.output,
                reasoning=log.reasoning,
                executionTime=log.executionTime,
                status=log.status,
                routingMode=log.routingMode,
                usedLlm=log.usedLlm,
                confidence=log.confidence,
                fallbackUsed=log.fallbackUsed,
                creditsUsed=log.creditsUsed,
                llmTokens=log.llmTokens,
                createdAt=log.createdAt
            )
            for log in logs
        ]

        has_more = (skip + page_size) < total


        return AgentLogListResponse(
            logs=log_responses,
            total=total,
            page=page,
            pageSize=page_size,
            hasMore=has_more
        )

    except Exception as e:
        logger.error("Error fetching agent logs", error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-logs/task/{task_id}", response_model=List[AgentLogResponse])
async def get_logs_by_task(task_id: str) -> List[AgentLogResponse]:
    """
    Get all agent logs for a specific task.

    Args:
        task_id: Task ID to query

    Returns:
        List of agent logs for the task, ordered by creation time
    """
    try:

        logs = await prisma.agentlog.find_many(
            where={"taskId": task_id},
            order={"createdAt": "asc"}
        )

        if not logs:
            raise HTTPException(
                status_code=404,
                detail=f"No logs found for task {task_id}"
            )

        log_responses = [
            AgentLogResponse(
                id=log.id,
                shop=log.shop,
                taskId=log.taskId,
                agentName=log.agentName,
                input=log.input,
                output=log.output,
                reasoning=log.reasoning,
                executionTime=log.executionTime,
                status=log.status,
                routingMode=log.routingMode,
                usedLlm=log.usedLlm,
                confidence=log.confidence,
                fallbackUsed=log.fallbackUsed,
                creditsUsed=log.creditsUsed,
                llmTokens=log.llmTokens,
                createdAt=log.createdAt
            )
            for log in logs
        ]

        return log_responses

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching logs by task", task_id=task_id, error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-logs/{log_id}", response_model=AgentLogResponse)
async def get_agent_log(log_id: str) -> AgentLogResponse:
    """
    Get a single agent log by ID.

    Args:
        log_id: Agent log ID

    Returns:
        Agent log entry
    """
    try:

        log = await prisma.agentlog.find_unique(where={"id": log_id})

        if not log:
            raise HTTPException(
                status_code=404,
                detail=f"Agent log {log_id} not found"
            )

        response = AgentLogResponse(
            id=log.id,
            shop=log.shop,
            taskId=log.taskId,
            agentName=log.agentName,
            input=log.input,
            output=log.output,
            reasoning=log.reasoning,
            executionTime=log.executionTime,
            status=log.status,
            routingMode=log.routingMode,
            usedLlm=log.usedLlm,
            confidence=log.confidence,
            fallbackUsed=log.fallbackUsed,
            creditsUsed=log.creditsUsed,
            llmTokens=log.llmTokens,
            createdAt=log.createdAt
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching agent log", log_id=log_id, error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agent-logs/stats/routing", response_model=RoutingStatsResponse)
async def get_routing_stats(
    days: int = Query(7, ge=1, le=90),
    shop: Optional[str] = None
) -> RoutingStatsResponse:
    """
    Get routing statistics.

    Args:
        days: Number of days to analyze (default 7)
        shop: Optional shop filter

    Returns:
        Routing statistics
    """
    try:

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Build where clause
        where = {
            "createdAt": {
                "gte": start_date,
                "lte": end_date
            }
        }
        if shop:
            where["shop"] = shop

        # Get all logs in date range
        logs = await prisma.agentlog.find_many(where=where)

        if not logs:
            return RoutingStatsResponse(
                totalRequests=0,
                ruleBasedCount=0,
                llmBasedCount=0,
                hybridCount=0,
                averageExecutionTime=0.0,
                totalCreditsUsed=0,
                averageConfidence=0.0
            )

        # Calculate statistics
        total_requests = len(logs)
        rule_based_count = sum(1 for log in logs if log.routingMode == "rule")
        llm_based_count = sum(1 for log in logs if log.routingMode == "llm")
        hybrid_count = sum(1 for log in logs if log.routingMode == "hybrid")

        total_execution_time = sum(log.executionTime for log in logs)
        avg_execution_time = total_execution_time / total_requests if total_requests > 0 else 0.0

        total_credits = sum(log.creditsUsed for log in logs)

        confidences = [log.confidence for log in logs if log.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0


        return RoutingStatsResponse(
            totalRequests=total_requests,
            ruleBasedCount=rule_based_count,
            llmBasedCount=llm_based_count,
            hybridCount=hybrid_count,
            averageExecutionTime=avg_execution_time,
            totalCreditsUsed=total_credits,
            averageConfidence=avg_confidence
        )

    except Exception as e:
        logger.error("Error calculating routing stats", error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
