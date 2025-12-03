"""
A2A (Agent-to-Agent) Client for Product Generator microservice.

Handles communication with the separate GPU-enabled Product Generator service.
Includes circuit breaker, retry logic, and graceful degradation.
"""
import httpx
import time
from typing import Any, Dict, Optional
from enum import Enum
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from src.utils import get_logger
from config import settings

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


class A2AClient:
    """
    Client for Agent-to-Agent communication with Product Generator.

    The Product Generator runs as a separate microservice with GPU access
    for image compositing operations.

    Features:
    - Retry logic with exponential backoff
    - Circuit breaker pattern (opens after 5 failures, half-open after 30s)
    - Graceful degradation support
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> None:
        """
        Initialize A2A client with circuit breaker.

        Args:
            base_url: Product Generator service URL
            timeout: Request timeout in seconds
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds before attempting recovery (half-open)
        """
        self.base_url = base_url or getattr(
            settings, "product_generator_url", "http://localhost:8081"
        )
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

        # Circuit breaker state
        self.circuit_state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.last_failure_time: Optional[float] = None
        self.recovery_timeout = recovery_timeout

        logger.info(
            "A2A Client initialized with circuit breaker",
            product_generator_url=self.base_url,
            timeout=timeout,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )

    def _check_circuit_state(self) -> bool:
        """
        Check and update circuit breaker state.

        Returns:
            True if circuit is closed/half-open (request allowed), False if open
        """
        if self.circuit_state == CircuitState.CLOSED:
            return True

        if self.circuit_state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_failure_time is not None:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.recovery_timeout:
                    logger.info(
                        "Circuit transitioning to HALF_OPEN",
                        elapsed=elapsed,
                        recovery_timeout=self.recovery_timeout,
                    )
                    self.circuit_state = CircuitState.HALF_OPEN
                    return True
            return False

        # HALF_OPEN state - allow request to test recovery
        return True

    def _record_success(self) -> None:
        """Record a successful request, closing circuit if needed."""
        if self.circuit_state == CircuitState.HALF_OPEN:
            logger.info("Circuit recovered, transitioning to CLOSED")
            self.circuit_state = CircuitState.CLOSED
            self.failure_count = 0
            self.last_failure_time = None
        elif self.circuit_state == CircuitState.CLOSED:
            # Reset failure count on success
            if self.failure_count > 0:
                logger.debug("Resetting failure count after success")
                self.failure_count = 0

    def _record_failure(self) -> None:
        """Record a failed request, opening circuit if threshold reached."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.circuit_state == CircuitState.HALF_OPEN:
            logger.warning("Circuit test failed, reopening")
            self.circuit_state = CircuitState.OPEN
        elif self.failure_count >= self.failure_threshold:
            logger.error(
                "Circuit breaker threshold reached, OPENING circuit",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )
            self.circuit_state = CircuitState.OPEN

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            httpx.TimeoutException,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        )),
        reraise=True,
    )
    async def generate_products(
        self,
        best_image: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Request product generation from the microservice with circuit breaker.

        Args:
            best_image: Approved image data with base64
            context: Execution context with product types

        Returns:
            Product generation result with success flag

        Note:
            Returns error dict if circuit is open or request fails.
            Caller should check 'success' field and fallback to local generator.
        """
        # Check circuit breaker state
        if not self._check_circuit_state():
            logger.warning(
                "A2A circuit breaker OPEN, blocking request",
                circuit_state=self.circuit_state.value,
                failure_count=self.failure_count,
            )
            return {
                "success": False,
                "error": "Product Generator service unavailable (circuit breaker open)",
                "circuit_open": True,
                "context": context,
            }

        try:
            logger.info(
                "Sending A2A request to Product Generator",
                image_id=best_image.get("image_id"),
                num_product_types=len(context.get("product_types", [])),
                circuit_state=self.circuit_state.value,
            )

            response = await self.client.post(
                "/generate",
                json={
                    "best_image": best_image,
                    "context": context,
                },
            )

            # Handle specific HTTP errors
            if response.status_code == 503:
                logger.warning("Product Generator service unavailable (503)")
                self._record_failure()
                raise httpx.HTTPStatusError(
                    "Service unavailable",
                    request=response.request,
                    response=response
                )

            response.raise_for_status()
            result = response.json()

            # Record success - circuit breaker
            self._record_success()

            logger.info(
                "A2A request successful",
                num_products=len(result.get("products", [])),
                circuit_state=self.circuit_state.value,
            )

            return {
                "success": True,
                **result,
            }

        except httpx.TimeoutException as e:
            logger.warning(
                "A2A request timeout, retrying",
                error=str(e),
                timeout=self.timeout,
            )
            self._record_failure()
            raise  # Will be retried by @retry decorator

        except httpx.NetworkError as e:
            logger.warning(
                "A2A network error, retrying",
                error=str(e),
            )
            self._record_failure()
            raise  # Will be retried by @retry decorator

        except httpx.HTTPStatusError as e:
            logger.error(
                "A2A HTTP error",
                error=str(e),
                status_code=e.response.status_code if hasattr(e, 'response') else None,
            )
            self._record_failure()
            return {
                "success": False,
                "error": f"Product Generator HTTP error: {e.response.status_code}",
                "context": context,
            }

        except Exception as e:
            logger.error(
                "Unexpected A2A error",
                error=str(e),
                exc_info=True,
            )
            self._record_failure()
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "context": context,
            }

    async def health_check(self) -> bool:
        """
        Check if Product Generator service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = await self.client.get("/health", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.warning(
                "Product Generator health check failed",
                error=str(e),
            )
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
