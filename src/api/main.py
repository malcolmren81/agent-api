"""
FastAPI main application.
"""
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from config import settings
from src.utils import get_logger, registry
from src.models.schemas import HealthResponse, ErrorResponse, ErrorDetail
# NOTE: interactive route removed - uses old src.agents orchestrator (deprecated)
# New chat API uses palet8_agents directly
from src.api.routes import planner, generation, evaluation, product, agent_logs, workflow, templates, tasks, debug, migrations, chat
from src.api import websocket
from src.database import connect_db, disconnect_db

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan manager."""
    print("\n" + "=" * 80)
    print("🚀 APPLICATION STARTUP: Lifespan manager starting")
    print("=" * 80 + "\n")

    logger.info(
        "Starting Phase 2 Agent Backend",
        environment=settings.environment,
        version="0.1.0",
    )

    try:
        print("Attempting to connect to database...")
        # Connect to database on startup
        await connect_db()
        print("✅ Database connection completed in lifespan")

        # Run schema migrations (add new columns if they don't exist)
        try:
            from src.database import prisma
            print("Running database schema migrations...")
            await prisma.execute_raw('ALTER TABLE "AgentLog" ADD COLUMN IF NOT EXISTS "modelName" TEXT')
            await prisma.execute_raw('ALTER TABLE "AgentLog" ADD COLUMN IF NOT EXISTS "llmTokens" INTEGER')
            print("✅ Schema migrations completed successfully")
        except Exception as migration_error:
            print(f"⚠️ Migration warning: {type(migration_error).__name__}: {str(migration_error)}")
            print("Continuing startup - columns may already exist")

        # Initialize selector cache from Admin API
        try:
            from palet8_agents.services.selector_service import SelectorService
            print("Initializing selector cache from Admin API...")
            selector_service = SelectorService.get_instance()
            await selector_service.initialize()
            stats = selector_service.get_cache_stats()
            print(f"✅ Selector cache initialized: {stats['selector_count']} selectors cached")
        except Exception as selector_error:
            print(f"⚠️ Selector cache warning: {type(selector_error).__name__}: {str(selector_error)}")
            print("Continuing startup - selector cache may not be available")

    except Exception as e:
        print(f"⚠️ WARNING: Database connection failed in lifespan: {type(e).__name__}: {str(e)}")
        print("Application will start but database operations may fail")
        # Don't raise - allow app to start so we can investigate

    print("\n" + "=" * 80)
    print("✅ APPLICATION READY: All startup tasks completed")
    print("=" * 80 + "\n")

    yield

    print("\n" + "=" * 80)
    print("🛑 APPLICATION SHUTDOWN: Lifespan manager shutting down")
    print("=" * 80 + "\n")

    # Disconnect from database on shutdown
    await disconnect_db()
    logger.info("Shutting down Phase 2 Agent Backend")

    print("\n" + "=" * 80)
    print("✅ SHUTDOWN COMPLETE")
    print("=" * 80 + "\n")


# Create FastAPI app
app = FastAPI(
    title="Phase 2 Agent Backend",
    description="Multi-agent system with Google ADK, Gemini, ChatGPT, Flux, and Gemini Image",
    version="0.1.0",
    lifespan=lifespan,
    root_path="/agents",  # Load balancer routes /agents/* to this service
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/response logging middleware with correlation ID tracking
from src.api.middleware import LoggingMiddleware
app.add_middleware(LoggingMiddleware)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        path=request.url.path,
        exc_info=True,
    )

    error_response = ErrorResponse(
        error=ErrorDetail(
            code="INTERNAL_ERROR",
            message="An internal error occurred",
            details={"type": type(exc).__name__} if settings.environment != "production" else None,
        )
    )

    return JSONResponse(
        status_code=500,
        content=error_response.model_dump(),
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with real service checks."""
    from src.utils.health_check import perform_health_checks

    # Perform actual health checks
    checks = await perform_health_checks()

    # Convert to boolean status for response
    services = {
        service: result.get("status", False)
        for service, result in checks.items()
    }

    # Determine overall status
    all_healthy = all(services.values())
    status = "healthy" if all_healthy else "degraded"

    # Add detailed info for debugging
    details = {
        service: result.get("message") or result.get("error", "Unknown")
        for service, result in checks.items()
    }

    logger.info(f"Health check: {status}", **details)

    return HealthResponse(
        status=status,
        services=services,
    )


# Metrics endpoint
@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    if not settings.enable_metrics:
        return JSONResponse(
            status_code=404,
            content={"error": "Metrics are disabled"},
        )

    data = generate_latest(registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


# Include routers
# NOTE: interactive-agent route removed - uses deprecated src.agents orchestrator
# Use /chat/* endpoints instead which use palet8_agents
app.include_router(planner.router, prefix="/planner", tags=["Planner"])
app.include_router(generation.router, prefix="/generation", tags=["Generation"])
app.include_router(evaluation.router, prefix="/evaluation", tags=["Evaluation"])
app.include_router(product.router, prefix="/product-generator", tags=["Product"])

# Chat API for Pali agent conversations (ChatUI integration)
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

# Phase 4: Workflow and Agent Log APIs
app.include_router(agent_logs.router, prefix="/api", tags=["Agent Logs"])
app.include_router(workflow.router, prefix="/api", tags=["Workflow"])
app.include_router(websocket.router, prefix="/api", tags=["WebSocket"])
app.include_router(templates.router, prefix="/api", tags=["Templates"])
app.include_router(tasks.router, prefix="/api", tags=["Tasks"])

# Debug routes (test data insertion)
app.include_router(debug.router, prefix="/debug", tags=["Debug"])

# Migration routes (temporary for schema changes)
app.include_router(migrations.router, prefix="/api", tags=["Migrations"])


# Root endpoint
@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint."""
    return {
        "service": "Phase 2 Agent Backend",
        "version": "0.1.0",
        "status": "running",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.environment == "development",
    )
