"""
Product Generator Microservice - Separate FastAPI service for GPU operations.

This service handles image compositing on GPU-enabled infrastructure.
Runs independently from the main backend service.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List
from src.agents.product_generator_agent import ProductGeneratorAgent
from src.utils import get_logger
from config import settings

logger = get_logger(__name__)

# Create FastAPI app for Product Generator microservice
app = FastAPI(
    title="Product Generator Microservice",
    description="GPU-enabled image compositing service (A2A)",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Product Generator Agent
product_agent = ProductGeneratorAgent()


class ProductGenerationRequest(BaseModel):
    """Request for product generation."""
    best_image: Dict[str, Any] = Field(..., description="Approved image data")
    context: Dict[str, Any] = Field(..., description="Execution context")


class ProductGenerationResponse(BaseModel):
    """Response from product generation."""
    success: bool
    products: List[Dict[str, Any]] = Field(default_factory=list)
    product_metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "product-generator",
        "version": "0.1.0",
    }


@app.post("/generate", response_model=ProductGenerationResponse)
async def generate_products(
    request: ProductGenerationRequest,
) -> ProductGenerationResponse:
    """
    Generate product mockups from approved image.

    This endpoint runs on GPU-enabled infrastructure for efficient
    image compositing operations.

    Args:
        request: Product generation request

    Returns:
        Generated products
    """
    try:
        logger.info(
            "Product generation request received",
            image_id=request.best_image.get("image_id"),
            num_product_types=len(request.context.get("product_types", [])),
        )

        # Run Product Generator Agent
        result = await product_agent.run({
            "best_image": request.best_image,
            "context": request.context,
        })

        return ProductGenerationResponse(**result)

    except Exception as e:
        logger.error(
            "Product generation failed",
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Product Generator Microservice",
        "version": "0.1.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    # Product Generator runs on a different port (8081 by default)
    uvicorn.run(
        "src.api.product_service:app",
        host="0.0.0.0",
        port=8081,
        reload=settings.environment == "development",
    )
