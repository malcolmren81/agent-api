"""
Workflow API routes.

Provides endpoints for aggregating multi-agent workflow execution data.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from prisma import Prisma
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Initialize Prisma client
prisma = Prisma()


# Response models
class AgentExecutionData(BaseModel):
    """Single agent execution in workflow"""
    agentName: str
    startTime: datetime
    endTime: datetime
    status: str
    input: dict
    output: dict
    routingMetadata: Dict[str, Any]
    cost: int
    duration: int
    errorMessage: Optional[str] = None


class WorkflowResponse(BaseModel):
    """Complete workflow data"""
    requestId: str
    userId: str
    timestamp: datetime
    status: str
    agents: Dict[str, AgentExecutionData]
    totalCost: int
    totalDuration: int
    finalResult: str


class ActiveWorkflowResponse(BaseModel):
    """Simplified active workflow data"""
    requestId: str
    userId: str
    startTime: datetime
    completedAgents: List[str]
    currentAgent: Optional[str]
    status: str


@router.get("/workflow/{task_id}", response_model=WorkflowResponse)
async def get_workflow(task_id: str) -> WorkflowResponse:
    """
    Get complete workflow execution data for a task.

    Aggregates all agent logs for a task into a single workflow view.

    Args:
        task_id: Task ID to query

    Returns:
        Complete workflow execution data
    """
    try:
        await prisma.connect()

        # Get all agent logs for this task
        logs = await prisma.agentlog.find_many(
            where={"taskId": task_id},
            order={"createdAt": "asc"}
        )

        if not logs:
            await prisma.disconnect()
            raise HTTPException(
                status_code=404,
                detail=f"No workflow found for task {task_id}"
            )

        # Extract user ID from first log's input
        user_id = logs[0].input.get("user_id", "unknown")

        # Build agents dict
        agents_dict = {}
        total_cost = 0
        total_duration = 0

        for log in logs:
            # Calculate start and end times
            start_time = log.createdAt
            end_time = log.createdAt  # Approximation; will add timedelta if needed

            # Build routing metadata
            routing_metadata = {
                "mode": log.routingMode,
                "used_llm": log.usedLlm,
                "confidence": log.confidence,
                "fallback_used": log.fallbackUsed,
            }

            # Add reasoning if present
            if log.reasoning:
                routing_metadata["reasoning"] = log.reasoning

            # Add agent-specific metadata from output
            if "routing_metadata" in log.output:
                routing_metadata.update(log.output["routing_metadata"])

            # Create agent execution data
            agent_execution = AgentExecutionData(
                agentName=log.agentName,
                startTime=start_time,
                endTime=end_time,
                status=log.status,
                input=log.input,
                output=log.output,
                routingMetadata=routing_metadata,
                cost=log.creditsUsed,
                duration=log.executionTime,
                errorMessage=log.output.get("error") if log.status == "failed" else None
            )

            agents_dict[log.agentName] = agent_execution
            total_cost += log.creditsUsed
            total_duration += log.executionTime

        # Determine final result
        final_result = "approved"
        if any(log.status == "failed" for log in logs):
            final_result = "error"
        elif "evaluation" in agents_dict:
            eval_output = agents_dict["evaluation"].output
            if not eval_output.get("success", False):
                final_result = "rejected"

        # Determine overall status
        if any(log.status == "failed" for log in logs):
            status = "failed"
        else:
            status = "completed"

        await prisma.disconnect()

        return WorkflowResponse(
            requestId=task_id,
            userId=user_id,
            timestamp=logs[0].createdAt,
            status=status,
            agents=agents_dict,
            totalCost=total_cost,
            totalDuration=total_duration,
            finalResult=final_result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fetching workflow", task_id=task_id, error_detail=str(e), exc_info=True)
        await prisma.disconnect()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/active", response_model=List[ActiveWorkflowResponse])
async def get_active_workflows(
    limit: int = Query(10, ge=1, le=50)
) -> List[ActiveWorkflowResponse]:
    """
    Get currently executing workflows.

    A workflow is considered active if it has logs created in the last 5 minutes
    and doesn't have all 7 agents completed.

    Args:
        limit: Maximum number of active workflows to return

    Returns:
        List of active workflows
    """
    try:
        await prisma.connect()

        # Get recent logs (last 5 minutes)
        from datetime import timedelta
        recent_time = datetime.now() - timedelta(minutes=5)

        recent_logs = await prisma.agentlog.find_many(
            where={
                "createdAt": {"gte": recent_time}
            },
            order={"createdAt": "desc"}
        )

        # Group by task ID
        task_logs: Dict[str, List[Any]] = {}
        for log in recent_logs:
            if log.taskId not in task_logs:
                task_logs[log.taskId] = []
            task_logs[log.taskId].append(log)

        # Expected agents in complete workflow
        expected_agents = {
            "interactive", "planner", "prompt_manager",
            "model_selection", "generation", "evaluation", "product_generator"
        }

        active_workflows = []

        for task_id, logs in task_logs.items():
            completed_agents = [log.agentName for log in logs]
            completed_set = set(completed_agents)

            # Only include if not all agents completed
            if not expected_agents.issubset(completed_set):
                # Determine current agent (last one executed)
                current_agent = completed_agents[-1] if completed_agents else None

                # Determine status
                if any(log.status == "failed" for log in logs):
                    status = "failed"
                else:
                    status = "running"

                # Get user ID
                user_id = logs[0].input.get("user_id", "unknown")

                active_workflows.append(
                    ActiveWorkflowResponse(
                        requestId=task_id,
                        userId=user_id,
                        startTime=logs[0].createdAt,
                        completedAgents=completed_agents,
                        currentAgent=current_agent,
                        status=status
                    )
                )

                if len(active_workflows) >= limit:
                    break

        await prisma.disconnect()
        return active_workflows

    except Exception as e:
        logger.error("Error fetching active workflows", error_detail=str(e), exc_info=True)
        await prisma.disconnect()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/workflow/list", response_model=List[WorkflowResponse])
async def list_workflows(
    limit: int = Query(20, ge=1, le=100),
    shop: Optional[str] = None,
    status: Optional[str] = None
) -> List[WorkflowResponse]:
    """
    List recent workflows.

    Args:
        limit: Maximum number of workflows to return
        shop: Optional shop filter
        status: Optional status filter

    Returns:
        List of recent workflows
    """
    try:
        await prisma.connect()

        # Build where clause
        where = {}
        if shop:
            where["shop"] = shop

        # Get recent logs
        logs = await prisma.agentlog.find_many(
            where=where,
            order={"createdAt": "desc"},
            take=limit * 7  # Get enough logs to cover limit workflows (7 agents each)
        )

        # Group by task ID
        task_logs: Dict[str, List[Any]] = {}
        for log in logs:
            if log.taskId not in task_logs:
                task_logs[log.taskId] = []
            task_logs[log.taskId].append(log)

        # Build workflows
        workflows = []
        for task_id, task_log_list in list(task_logs.items())[:limit]:
            try:
                # Sort logs by creation time
                task_log_list.sort(key=lambda x: x.createdAt)

                # Extract user ID
                user_id = task_log_list[0].input.get("user_id", "unknown")

                # Build agents dict
                agents_dict = {}
                total_cost = 0
                total_duration = 0

                for log in task_log_list:
                    routing_metadata = {
                        "mode": log.routingMode,
                        "used_llm": log.usedLlm,
                        "confidence": log.confidence,
                        "fallback_used": log.fallbackUsed,
                    }

                    if log.reasoning:
                        routing_metadata["reasoning"] = log.reasoning

                    if "routing_metadata" in log.output:
                        routing_metadata.update(log.output["routing_metadata"])

                    agent_execution = AgentExecutionData(
                        agentName=log.agentName,
                        startTime=log.createdAt,
                        endTime=log.createdAt,
                        status=log.status,
                        input=log.input,
                        output=log.output,
                        routingMetadata=routing_metadata,
                        cost=log.creditsUsed,
                        duration=log.executionTime,
                        errorMessage=log.output.get("error") if log.status == "failed" else None
                    )

                    agents_dict[log.agentName] = agent_execution
                    total_cost += log.creditsUsed
                    total_duration += log.executionTime

                # Determine final result
                final_result = "approved"
                if any(log.status == "failed" for log in task_log_list):
                    final_result = "error"
                elif "evaluation" in agents_dict:
                    eval_output = agents_dict["evaluation"].output
                    if not eval_output.get("success", False):
                        final_result = "rejected"

                # Determine overall status
                workflow_status = "completed"
                if any(log.status == "failed" for log in task_log_list):
                    workflow_status = "failed"

                # Apply status filter if provided
                if status and workflow_status != status:
                    continue

                workflows.append(
                    WorkflowResponse(
                        requestId=task_id,
                        userId=user_id,
                        timestamp=task_log_list[0].createdAt,
                        status=workflow_status,
                        agents=agents_dict,
                        totalCost=total_cost,
                        totalDuration=total_duration,
                        finalResult=final_result
                    )
                )
            except Exception as e:
                logger.warning(f"Error processing workflow {task_id}", error_detail=str(e))
                continue

        await prisma.disconnect()
        return workflows

    except Exception as e:
        logger.error("Error listing workflows", error_detail=str(e), exc_info=True)
        await prisma.disconnect()
        raise HTTPException(status_code=500, detail=str(e))
