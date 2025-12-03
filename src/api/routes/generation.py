"""
Generation Agent API routes.
"""
from fastapi import APIRouter, HTTPException
from src.models.schemas import GenerationRequest, GenerationResponse
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/generate", response_model=GenerationResponse)
async def generate_images(request: GenerationRequest) -> GenerationResponse:
    """
    Generate images using the specified model.

    Args:
        request: Generation request

    Returns:
        Generated images with metadata
    """
    try:
        logger.info(
            "Generation request received",
            task_id=request.task_id,
            model=request.image_model,
            num_images=request.num_images,
        )

        # TODO: Implement actual generation agent logic

        return GenerationResponse(
            task_id=request.task_id,
            images=[],
            total_cost=0.0,
            total_time=0.0,
        )

    except Exception as e:
        logger.error("Generation error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
