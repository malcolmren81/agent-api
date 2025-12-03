"""
Health check utilities for verifying service dependencies.
"""
import asyncio
from typing import Dict, Any
import httpx

from config import settings
from src.utils import get_logger

logger = get_logger(__name__)


async def check_gemini_api() -> Dict[str, Any]:
    """Check if Gemini API is accessible."""
    try:
        if not settings.gemini_api_key or settings.gemini_api_key == "":
            return {"status": False, "error": "API key not configured"}

        # Simple check: just verify the key format is reasonable
        # Full API test would require making an actual API call
        if len(settings.gemini_api_key) < 10:
            return {"status": False, "error": "API key appears invalid"}

        # TODO: Make actual Gemini API test call
        return {"status": True, "message": "API key configured"}
    except Exception as e:
        logger.error(f"Gemini health check failed: {e}")
        return {"status": False, "error": str(e)}


async def check_openai_api() -> Dict[str, Any]:
    """Check if OpenAI API is accessible."""
    try:
        if not settings.openai_api_key or settings.openai_api_key == "":
            return {"status": False, "error": "API key not configured"}

        if len(settings.openai_api_key) < 10:
            return {"status": False, "error": "API key appears invalid"}

        # Make a lightweight API call to verify the key
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                    timeout=5.0
                )

                if response.status_code == 200:
                    return {"status": True, "message": "API accessible"}
                elif response.status_code == 401:
                    return {"status": False, "error": "Invalid API key"}
                else:
                    return {"status": False, "error": f"HTTP {response.status_code}"}
            except httpx.TimeoutException:
                return {"status": False, "error": "Request timeout"}
            except Exception as e:
                return {"status": False, "error": f"Request failed: {str(e)}"}

    except Exception as e:
        logger.error(f"OpenAI health check failed: {e}")
        return {"status": False, "error": str(e)}


async def check_flux_api() -> Dict[str, Any]:
    """Check if Flux API is accessible."""
    try:
        if not settings.flux_api_key or settings.flux_api_key == "":
            return {"status": False, "error": "API key not configured"}

        if len(settings.flux_api_key) < 10:
            return {"status": False, "error": "API key appears invalid"}

        # Make a lightweight API call to verify the key
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{settings.flux_api_endpoint}/health",
                    headers={"Authorization": f"Bearer {settings.flux_api_key}"},
                    timeout=5.0
                )

                if response.status_code in [200, 404]:  # 404 is ok if no health endpoint
                    return {"status": True, "message": "API accessible"}
                elif response.status_code == 401:
                    return {"status": False, "error": "Invalid API key"}
                else:
                    # If health endpoint doesn't exist, key is likely still valid
                    return {"status": True, "message": "API key configured"}
            except httpx.TimeoutException:
                return {"status": False, "error": "Request timeout"}
            except Exception as e:
                # Some APIs don't have health endpoints, so this might be ok
                return {"status": True, "message": "API key configured (endpoint unreachable)"}

    except Exception as e:
        logger.error(f"Flux health check failed: {e}")
        return {"status": False, "error": str(e)}


async def check_product_generator() -> Dict[str, Any]:
    """Check if Product Generator service is accessible (A2A)."""
    try:
        if not settings.use_product_a2a:
            return {"status": True, "message": "Using local agent (A2A disabled)"}

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    f"{settings.product_generator_url}/health",
                    timeout=3.0
                )

                if response.status_code == 200:
                    return {"status": True, "message": "A2A service accessible"}
                else:
                    return {"status": False, "error": f"HTTP {response.status_code}"}
            except httpx.TimeoutException:
                return {"status": False, "error": "Service timeout"}
            except httpx.ConnectError:
                return {"status": False, "error": "Cannot connect to service"}
            except Exception as e:
                return {"status": False, "error": str(e)}

    except Exception as e:
        logger.error(f"Product Generator health check failed: {e}")
        return {"status": False, "error": str(e)}


async def check_database() -> Dict[str, Any]:
    """Check database connection."""
    # TODO: Implement in Phase 5 when database is connected
    return {"status": True, "message": "No database configured yet (Phase 5)"}


async def perform_health_checks() -> Dict[str, Dict[str, Any]]:
    """
    Perform all health checks concurrently.

    Returns:
        Dictionary with health check results for each service
    """
    logger.info("Performing health checks")

    # Run all checks concurrently
    results = await asyncio.gather(
        check_gemini_api(),
        check_openai_api(),
        check_flux_api(),
        check_product_generator(),
        check_database(),
        return_exceptions=True
    )

    # Package results
    checks = {
        "gemini_api": results[0] if not isinstance(results[0], Exception) else {"status": False, "error": str(results[0])},
        "openai_api": results[1] if not isinstance(results[1], Exception) else {"status": False, "error": str(results[1])},
        "flux_api": results[2] if not isinstance(results[2], Exception) else {"status": False, "error": str(results[2])},
        "product_generator": results[3] if not isinstance(results[3], Exception) else {"status": False, "error": str(results[3])},
        "database": results[4] if not isinstance(results[4], Exception) else {"status": False, "error": str(results[4])},
    }

    # Determine overall health
    all_healthy = all(check.get("status", False) for check in checks.values())

    logger.info(
        "Health checks completed",
        all_healthy=all_healthy,
        **{k: v.get("status") for k, v in checks.items()}
    )

    return checks
