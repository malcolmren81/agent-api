"""
Product Generator API routes.
"""
from fastapi import APIRouter, HTTPException
from src.models.schemas import ProductRequest, ProductResponse
from src.utils import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("/create", response_model=ProductResponse)
async def create_products(request: ProductRequest) -> ProductResponse:
    """
    Composite approved image onto product mock-ups.

    Args:
        request: Product generation request

    Returns:
        Generated product composites
    """
    try:
        logger.info(
            "Product generation request received",
            task_id=request.task_id,
            image_id=request.image_id,
            product_types=request.product_types,
        )

        # TODO: Implement actual product generator logic

        return ProductResponse(
            task_id=request.task_id,
            products=[],
            total_time=0.0,
        )

    except Exception as e:
        logger.error("Product generation error", error_detail=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
