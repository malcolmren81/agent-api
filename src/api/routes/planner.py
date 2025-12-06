"""
Planner Agent API routes.
"""
from fastapi import APIRouter, HTTPException
from src.models.schemas import PlannerResponse, InteractiveRequest
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/plan", response_model=PlannerResponse)
async def create_plan(request: InteractiveRequest) -> PlannerResponse:
    """
    Generate an execution plan for the given prompt.

    Args:
        request: Interactive request with prompt

    Returns:
        Execution plan with steps and cost estimates
    """
    try:
        logger.info("Planning request received", prompt_length=len(request.prompt))

        # TODO: Implement actual planner agent logic
        # This is a stub response
        return PlannerResponse(
            task_id="stub-task-id",
            steps=[],
            total_estimated_cost=0.0,
            total_estimated_time=0.0,
            selected_reasoning_model=request.reasoning_model or "gemini",
            selected_image_model=request.image_model or "flux",
        )

    except Exception as e:
        logger.error("Planner error", error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
