"""
Assembly Service

Orchestrates image generation execution based on AssemblyRequest.
Handles single and dual pipelines, provider API calls, and result assembly.

Documentation Reference: Section 4.5
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Awaitable
from uuid import uuid4
import asyncio
import base64
import time

from src.utils.logger import get_logger

from palet8_agents.models import (
    AssemblyRequest,
    ExecutionResult,
    ExecutionStatus,
    GeneratedImageData,
    PipelineConfig,
)
from palet8_agents.services.image_generation_service import (
    ImageGenerationService,
    ImageGenerationRequest,
    ImageGenerationResult,
    ImageGenerationError,
)

logger = get_logger(__name__)


class AssemblyError(Exception):
    """Base exception for AssemblyService errors."""
    pass


class PipelineError(AssemblyError):
    """Raised when pipeline execution fails."""
    pass


@dataclass
class ProgressCallback:
    """Callback for reporting progress during execution."""
    callback: Optional[Callable[[str, float, Optional[str]], Awaitable[None]]] = None

    async def report(self, stage: str, progress: float, message: Optional[str] = None) -> None:
        """Report progress if callback is set."""
        if self.callback:
            try:
                await self.callback(stage, progress, message)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")


class AssemblyService:
    """
    Service for orchestrating image generation execution.

    Takes an AssemblyRequest from the Planner and executes the generation
    pipeline, handling single or dual model workflows.

    Features:
    - Single and dual pipeline support
    - Provider abstraction via ImageGenerationService
    - Progress callbacks for SSE streaming
    - Cost and timing tracking
    - Base64 encoding for image data
    """

    def __init__(
        self,
        image_service: Optional[ImageGenerationService] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ):
        """
        Initialize AssemblyService.

        Args:
            image_service: ImageGenerationService instance (created if not provided)
            timeout: Overall timeout for execution in seconds
            max_retries: Maximum retry attempts for failed generations
        """
        self._image_service = image_service
        self._timeout = timeout
        self._max_retries = max_retries
        self._owns_image_service = image_service is None

    async def _get_image_service(self) -> ImageGenerationService:
        """Get or create the image generation service."""
        if self._image_service is None:
            self._image_service = ImageGenerationService(
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        return self._image_service

    async def execute(
        self,
        request: AssemblyRequest,
        progress_callback: Optional[Callable[[str, float, Optional[str]], Awaitable[None]]] = None,
    ) -> ExecutionResult:
        """
        Execute image generation based on AssemblyRequest.

        Args:
            request: AssemblyRequest with prompt, model, and pipeline config
            progress_callback: Optional async callback for progress updates
                              Called with (stage, progress, message)

        Returns:
            ExecutionResult with generated images and metadata
        """
        task_id = str(uuid4())
        start_time = time.time()
        progress = ProgressCallback(callback=progress_callback)

        logger.info(
            "assembly.execution.start",
            job_id=request.job_id,
            task_id=task_id,
            pipeline_type=request.pipeline.pipeline_type,
            model_id=request.model_id or request.pipeline.stage_1_model,
            prompt_length=len(request.prompt),
            negative_prompt_length=len(request.negative_prompt or ""),
            num_images=request.parameters.num_images,
            width=request.parameters.width,
            height=request.parameters.height,
            has_reference=request.reference_image_url is not None,
        )

        try:
            # Report start
            await progress.report("generation_start", 0.0, "Starting image generation")

            # Determine pipeline type and execute
            pipeline = request.pipeline
            if pipeline.pipeline_type == "dual" and pipeline.stage_2_model:
                result = await self._execute_dual_pipeline(request, task_id, progress)
            else:
                result = await self._execute_single_pipeline(request, task_id, progress)

            # Calculate duration
            result.duration_ms = int((time.time() - start_time) * 1000)
            result.job_id = request.job_id
            result.task_id = task_id

            # Report completion
            await progress.report("complete", 1.0, "Generation complete")

            logger.info(
                "assembly.execution.complete",
                job_id=request.job_id,
                task_id=task_id,
                duration_ms=result.duration_ms,
                actual_cost=result.actual_cost,
                images_count=len(result.images),
                pipeline_type=result.pipeline_type,
                model_used=result.model_used,
            )

            return result

        except ImageGenerationError as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "assembly.execution.error",
                job_id=request.job_id,
                task_id=task_id,
                error=str(e),
                error_code="GENERATION_ERROR",
                duration_ms=duration_ms,
                stage_failed="generation",
                partial_cost=0.0,
            )
            await progress.report("error", 0.0, str(e))
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                success=False,
                error=str(e),
                error_code="GENERATION_ERROR",
                job_id=request.job_id,
                task_id=task_id,
                duration_ms=int((time.time() - start_time) * 1000),
            )
        except asyncio.TimeoutError:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "assembly.execution.timeout",
                job_id=request.job_id,
                task_id=task_id,
                timeout_seconds=self._timeout,
                duration_ms=duration_ms,
                stage_at_timeout="generation",
            )
            await progress.report("error", 0.0, "Generation timed out")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                success=False,
                error="Generation timed out",
                error_code="TIMEOUT",
                job_id=request.job_id,
                task_id=task_id,
                duration_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "assembly.execution.unexpected_error",
                job_id=request.job_id,
                task_id=task_id,
                error=str(e),
                error_type=type(e).__name__,
                duration_ms=duration_ms,
                exc_info=True,
            )
            await progress.report("error", 0.0, str(e))
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                success=False,
                error=str(e),
                error_code="UNEXPECTED_ERROR",
                job_id=request.job_id,
                task_id=task_id,
                duration_ms=int((time.time() - start_time) * 1000),
            )

    async def _execute_single_pipeline(
        self,
        request: AssemblyRequest,
        task_id: str,
        progress: ProgressCallback,
    ) -> ExecutionResult:
        """Execute single model pipeline."""
        await progress.report("generation_progress", 0.2, "Generating with single model")

        image_service = await self._get_image_service()

        # Build generation request
        gen_request = ImageGenerationRequest(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            model=request.model_id or request.pipeline.stage_1_model,
            width=request.parameters.width,
            height=request.parameters.height,
            num_images=request.parameters.num_images,
            steps=request.parameters.steps,
            guidance_scale=request.parameters.guidance_scale,
            seed=request.parameters.seed,
            reference_image_url=request.reference_image_url,
            reference_strength=request.reference_strength,
            provider_settings=request.parameters.provider_settings,
            metadata={"job_id": request.job_id, "task_id": task_id},
        )

        # Log request details
        logger.info(
            "assembly.single.generation.sending",
            model=gen_request.model,
            prompt=gen_request.prompt[:200] if gen_request.prompt else None,
            negative_prompt=gen_request.negative_prompt[:100] if gen_request.negative_prompt else None,
            width=gen_request.width,
            height=gen_request.height,
            num_images=gen_request.num_images,
            steps=gen_request.steps,
            guidance_scale=gen_request.guidance_scale,
            seed=gen_request.seed,
        )

        # Execute generation
        await progress.report("generation_progress", 0.4, "Waiting for provider")
        gen_result = await image_service.generate_images(gen_request)

        logger.info(
            "assembly.single.generation.received",
            images_count=len(gen_result.images),
            cost_usd=gen_result.cost_usd,
            provider=gen_result.provider,
            model_used=gen_result.model_used,
        )

        await progress.report("generation_progress", 0.8, "Processing results")

        # Convert to ExecutionResult
        images = []
        for img in gen_result.images:
            images.append(GeneratedImageData(
                url=img.url,
                base64_data=img.base64_data,
                seed=img.seed,
                revised_prompt=img.revised_prompt,
                provider=gen_result.provider,
                model_used=gen_result.model_used,
                metadata=img.metadata,
            ))

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            success=True,
            images=images,
            model_used=gen_result.model_used,
            provider=gen_result.provider,
            pipeline_type="single",
            actual_cost=gen_result.cost_usd,
            metadata=gen_result.metadata,
        )

    async def _execute_dual_pipeline(
        self,
        request: AssemblyRequest,
        task_id: str,
        progress: ProgressCallback,
    ) -> ExecutionResult:
        """Execute dual model pipeline (generate then refine)."""
        pipeline = request.pipeline

        await progress.report("generation_progress", 0.1, f"Stage 1: {pipeline.stage_1_purpose}")

        image_service = await self._get_image_service()

        # Stage 1: Initial generation
        stage1_request = ImageGenerationRequest(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            model=pipeline.stage_1_model,
            width=request.parameters.width,
            height=request.parameters.height,
            num_images=1,  # Generate one image for refinement
            steps=request.parameters.steps,
            guidance_scale=request.parameters.guidance_scale,
            seed=request.parameters.seed,
            reference_image_url=request.reference_image_url,
            reference_strength=request.reference_strength,
            provider_settings=request.parameters.provider_settings,
            metadata={"job_id": request.job_id, "task_id": task_id, "stage": 1},
        )

        # Log stage 1 request
        logger.info(
            "assembly.dual.stage1.sending",
            model=pipeline.stage_1_model,
            purpose=pipeline.stage_1_purpose,
            prompt=stage1_request.prompt[:200] if stage1_request.prompt else None,
            negative_prompt=stage1_request.negative_prompt[:100] if stage1_request.negative_prompt else None,
            width=stage1_request.width,
            height=stage1_request.height,
            steps=stage1_request.steps,
        )

        await progress.report("generation_progress", 0.2, "Generating initial image")
        stage1_result = await image_service.generate_images(stage1_request)

        if not stage1_result.images:
            raise PipelineError("Stage 1 produced no images")

        stage1_image = stage1_result.images[0]
        stage1_data = {
            "model": stage1_result.model_used,
            "provider": stage1_result.provider,
            "cost": stage1_result.cost_usd,
            "image_url": stage1_image.url,
        }

        logger.info(
            "assembly.dual.stage1.complete",
            model=stage1_result.model_used,
            cost_usd=stage1_result.cost_usd,
            provider=stage1_result.provider,
            has_image=len(stage1_result.images) > 0,
            image_url=stage1_image.url[:100] if stage1_image.url else None,
        )

        await progress.report("generation_progress", 0.5, f"Stage 2: {pipeline.stage_2_purpose}")

        # Stage 2: Refinement
        stage2_request = ImageGenerationRequest(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt or None,
            model=pipeline.stage_2_model,
            width=request.parameters.width,
            height=request.parameters.height,
            num_images=request.parameters.num_images,
            steps=request.parameters.steps,
            guidance_scale=request.parameters.guidance_scale,
            reference_image_url=stage1_image.url,  # Use stage 1 output as reference
            metadata={"job_id": request.job_id, "task_id": task_id, "stage": 2},
        )

        # Log stage 2 request
        logger.info(
            "assembly.dual.stage2.sending",
            model=pipeline.stage_2_model,
            purpose=pipeline.stage_2_purpose,
            prompt=stage2_request.prompt[:200] if stage2_request.prompt else None,
            input_image_url=stage1_image.url[:100] if stage1_image.url else None,
            width=stage2_request.width,
            height=stage2_request.height,
        )

        await progress.report("generation_progress", 0.7, "Refining image")
        stage2_result = await image_service.generate_images(stage2_request)

        logger.info(
            "assembly.dual.stage2.complete",
            model=stage2_result.model_used,
            cost_usd=stage2_result.cost_usd,
            provider=stage2_result.provider,
            images_count=len(stage2_result.images),
        )

        stage2_data = {
            "model": stage2_result.model_used,
            "provider": stage2_result.provider,
            "cost": stage2_result.cost_usd,
        }

        await progress.report("generation_progress", 0.9, "Finalizing results")

        # Convert final images
        images = []
        for img in stage2_result.images:
            images.append(GeneratedImageData(
                url=img.url,
                base64_data=img.base64_data,
                seed=img.seed,
                revised_prompt=img.revised_prompt,
                provider=stage2_result.provider,
                model_used=stage2_result.model_used,
                metadata=img.metadata,
            ))

        # Calculate total cost
        total_cost = stage1_result.cost_usd + stage2_result.cost_usd

        return ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            success=True,
            images=images,
            model_used=f"{pipeline.stage_1_model} + {pipeline.stage_2_model}",
            provider=stage2_result.provider,
            pipeline_type="dual",
            actual_cost=total_cost,
            stage_1_result=stage1_data,
            stage_2_result=stage2_data,
            metadata={
                "pipeline_name": pipeline.pipeline_name,
                "decision_rationale": pipeline.decision_rationale,
            },
        )

    async def download_and_encode(self, image_url: str) -> Optional[str]:
        """
        Download an image and convert to base64.

        Args:
            image_url: URL of the image to download

        Returns:
            Base64 encoded image data, or None if download fails
        """
        try:
            import httpx

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(image_url)
                if response.status_code == 200:
                    return base64.b64encode(response.content).decode("utf-8")
        except Exception as e:
            logger.warning(f"Failed to download image: {e}")

        return None

    async def close(self) -> None:
        """Close resources."""
        if self._owns_image_service and self._image_service:
            await self._image_service.close()
            self._image_service = None

    async def __aenter__(self) -> "AssemblyService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
