"""
Evaluation Agent API routes.
"""
from fastapi import APIRouter, HTTPException
from src.models.schemas import EvaluationRequest, EvaluationResponse, EvaluationScore
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_image(request: EvaluationRequest) -> EvaluationResponse:
    """
    Evaluate an image based on specified criteria.

    Args:
        request: Evaluation request

    Returns:
        Evaluation scores and feedback
    """
    try:
        logger.info(
            "Evaluation request received",
            image_id=request.image_id,
            reasoning_model=request.reasoning_model,
        )

        # TODO: Implement actual evaluation agent logic

        return EvaluationResponse(
            image_id=request.image_id,
            scores=EvaluationScore(
                prompt_adherence=0.0,
                aesthetics=0.0,
                product_suitability=0.0,
                safety=0.0,
                overall=0.0,
            ),
            feedback="Stub evaluation response",
            approved=False,
            reasoning_model_used=request.reasoning_model,
        )

    except Exception as e:
        logger.error("Evaluation error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
