"""
Tests for AssemblyService.

Tests single and dual pipeline execution with mocked image generation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Import models directly
import sys
import importlib.util


def load_module(name, path):
    """Load a Python module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load models
models_module = load_module(
    'palet8_agents.models',
    'palet8_agents/models/__init__.py'
)

# Import model classes from module
AssemblyRequest = models_module.AssemblyRequest
GenerationParameters = models_module.GenerationParameters
PipelineConfig = models_module.PipelineConfig
ExecutionResult = models_module.ExecutionResult
ExecutionStatus = models_module.ExecutionStatus
GeneratedImageData = models_module.GeneratedImageData

# Load image generation service for mocking
image_gen_module = load_module(
    'palet8_agents.services.image_generation_service',
    'palet8_agents/services/image_generation_service.py'
)
ImageGenerationResult = image_gen_module.ImageGenerationResult
GeneratedImage = image_gen_module.GeneratedImage
ImageGenerationRequest = image_gen_module.ImageGenerationRequest

# Load assembly service
assembly_module = load_module(
    'palet8_agents.services.assembly_service',
    'palet8_agents/services/assembly_service.py'
)
AssemblyService = assembly_module.AssemblyService
AssemblyError = assembly_module.AssemblyError
PipelineError = assembly_module.PipelineError


class TestAssemblyServiceBasic:
    """Basic AssemblyService tests."""

    def test_assembly_request_creation(self):
        """Test creating an AssemblyRequest."""
        request = AssemblyRequest(
            prompt="A cute cat",
            negative_prompt="blurry, low quality",
            mode="STANDARD",
            model_id="runware:101@1",
            parameters=GenerationParameters(
                width=1024,
                height=1024,
                num_images=1,
            ),
            job_id="job-123",
            user_id="user-456",
        )

        assert request.prompt == "A cute cat"
        assert request.negative_prompt == "blurry, low quality"
        assert request.model_id == "runware:101@1"
        assert request.parameters.width == 1024
        assert request.parameters.height == 1024

    def test_execution_result_creation(self):
        """Test creating an ExecutionResult."""
        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            success=True,
            images=[
                GeneratedImageData(
                    url="https://example.com/image.png",
                    provider="runware",
                    model_used="runware:101@1",
                )
            ],
            model_used="runware:101@1",
            provider="runware",
            actual_cost=0.03,
            duration_ms=5000,
            job_id="job-123",
        )

        assert result.success is True
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.images) == 1
        assert result.first_image_url == "https://example.com/image.png"
        assert result.actual_cost == 0.03

    def test_execution_result_serialization(self):
        """Test ExecutionResult to_dict and from_dict."""
        result = ExecutionResult(
            status=ExecutionStatus.COMPLETED,
            success=True,
            images=[
                GeneratedImageData(
                    url="https://example.com/image.png",
                    seed=12345,
                    provider="runware",
                    model_used="test-model",
                )
            ],
            model_used="test-model",
            provider="runware",
            pipeline_type="single",
            actual_cost=0.05,
            duration_ms=3000,
            job_id="job-abc",
            task_id="task-xyz",
        )

        data = result.to_dict()
        assert data["status"] == "completed"
        assert data["success"] is True
        assert len(data["images"]) == 1
        assert data["actual_cost"] == 0.05

        # Reconstruct
        restored = ExecutionResult.from_dict(data)
        assert restored.success is True
        assert restored.status == ExecutionStatus.COMPLETED
        assert len(restored.images) == 1


class TestAssemblyServiceSinglePipeline:
    """Tests for single pipeline execution."""

    @pytest.fixture
    def mock_image_service(self):
        """Create a mock ImageGenerationService."""
        service = MagicMock()
        service.generate_images = AsyncMock(return_value=ImageGenerationResult(
            images=[
                GeneratedImage(
                    url="https://cdn.runware.ai/image123.png",
                    seed=42,
                )
            ],
            model_used="runware:101@1",
            provider="runware",
            cost_usd=0.03,
        ))
        service.close = AsyncMock()
        return service

    @pytest.fixture
    def assembly_request(self):
        """Create a test AssemblyRequest."""
        return AssemblyRequest(
            prompt="A beautiful sunset over the ocean",
            negative_prompt="blurry, dark",
            mode="STANDARD",
            model_id="runware:101@1",
            parameters=GenerationParameters(
                width=1024,
                height=1024,
                num_images=1,
                steps=30,
                guidance_scale=7.5,
            ),
            pipeline=PipelineConfig(
                pipeline_type="single",
                stage_1_model="runware:101@1",
            ),
            job_id="test-job-001",
            user_id="test-user-001",
            product_type="poster",
        )

    @pytest.mark.asyncio
    async def test_single_pipeline_execution(self, mock_image_service, assembly_request):
        """Test successful single pipeline execution."""
        service = AssemblyService(image_service=mock_image_service)

        result = await service.execute(assembly_request)

        assert result.success is True
        assert result.status == ExecutionStatus.COMPLETED
        assert len(result.images) == 1
        assert result.images[0].url == "https://cdn.runware.ai/image123.png"
        assert result.provider == "runware"
        assert result.pipeline_type == "single"
        assert result.actual_cost == 0.03

        # Verify image service was called correctly
        mock_image_service.generate_images.assert_called_once()
        call_args = mock_image_service.generate_images.call_args[0][0]
        assert call_args.prompt == "A beautiful sunset over the ocean"
        assert call_args.width == 1024
        assert call_args.height == 1024

    @pytest.mark.asyncio
    async def test_single_pipeline_with_provider_settings(self, mock_image_service):
        """Test single pipeline passes provider_settings."""
        request = AssemblyRequest(
            prompt="A test prompt",
            mode="STANDARD",
            model_id="runware:101@1",
            parameters=GenerationParameters(
                width=512,
                height=512,
                provider_settings={
                    "scheduler": "euler",
                    "clip_skip": 2,
                },
            ),
            pipeline=PipelineConfig(pipeline_type="single"),
            job_id="job-123",
            user_id="user-123",
        )

        service = AssemblyService(image_service=mock_image_service)
        result = await service.execute(request)

        assert result.success is True

        # Verify provider_settings were passed
        call_args = mock_image_service.generate_images.call_args[0][0]
        assert call_args.provider_settings == {"scheduler": "euler", "clip_skip": 2}

    @pytest.mark.asyncio
    async def test_single_pipeline_with_reference_image(self, mock_image_service):
        """Test single pipeline with reference image."""
        request = AssemblyRequest(
            prompt="A test prompt",
            mode="STANDARD",
            model_id="runware:101@1",
            reference_image_url="https://example.com/ref.png",
            reference_strength=0.8,
            parameters=GenerationParameters(width=1024, height=1024),
            pipeline=PipelineConfig(pipeline_type="single"),
            job_id="job-123",
            user_id="user-123",
        )

        service = AssemblyService(image_service=mock_image_service)
        result = await service.execute(request)

        assert result.success is True

        # Verify reference was passed
        call_args = mock_image_service.generate_images.call_args[0][0]
        assert call_args.reference_image_url == "https://example.com/ref.png"
        assert call_args.reference_strength == 0.8

    @pytest.mark.asyncio
    async def test_single_pipeline_failure(self, mock_image_service, assembly_request):
        """Test single pipeline handles generation failure."""
        mock_image_service.generate_images = AsyncMock(
            side_effect=Exception("Provider API error")
        )

        service = AssemblyService(image_service=mock_image_service)
        result = await service.execute(assembly_request)

        assert result.success is False
        assert result.status == ExecutionStatus.FAILED
        assert "Provider API error" in result.error
        assert result.error_code == "UNEXPECTED_ERROR"


class TestAssemblyServiceDualPipeline:
    """Tests for dual pipeline execution."""

    @pytest.fixture
    def mock_image_service_dual(self):
        """Create mock for dual pipeline (called twice)."""
        service = MagicMock()

        # First call returns stage 1 image
        stage1_result = ImageGenerationResult(
            images=[GeneratedImage(url="https://cdn.runware.ai/stage1.png", seed=100)],
            model_used="flux-schnell",
            provider="runware",
            cost_usd=0.02,
        )

        # Second call returns refined image
        stage2_result = ImageGenerationResult(
            images=[GeneratedImage(url="https://cdn.runware.ai/stage2.png", seed=200)],
            model_used="nano-banana-2-pro",
            provider="runware",
            cost_usd=0.04,
        )

        service.generate_images = AsyncMock(side_effect=[stage1_result, stage2_result])
        service.close = AsyncMock()
        return service

    @pytest.fixture
    def dual_pipeline_request(self):
        """Create a dual pipeline AssemblyRequest."""
        return AssemblyRequest(
            prompt="A creative poster with text",
            negative_prompt="blurry",
            mode="COMPLEX",
            model_id="flux-schnell",
            parameters=GenerationParameters(
                width=1024,
                height=1024,
                num_images=1,
            ),
            pipeline=PipelineConfig(
                pipeline_type="dual",
                pipeline_name="layout_poster",
                stage_1_model="flux-schnell",
                stage_1_purpose="Initial layout",
                stage_2_model="nano-banana-2-pro",
                stage_2_purpose="Text refinement",
            ),
            job_id="dual-job-001",
            user_id="user-001",
        )

    @pytest.mark.asyncio
    async def test_dual_pipeline_execution(self, mock_image_service_dual, dual_pipeline_request):
        """Test successful dual pipeline execution."""
        service = AssemblyService(image_service=mock_image_service_dual)

        result = await service.execute(dual_pipeline_request)

        assert result.success is True
        assert result.status == ExecutionStatus.COMPLETED
        assert result.pipeline_type == "dual"
        assert len(result.images) == 1
        # Should have final (stage 2) image
        assert result.images[0].url == "https://cdn.runware.ai/stage2.png"
        # Total cost should be sum of both stages
        assert result.actual_cost == 0.06  # 0.02 + 0.04

        # Verify stage data
        assert result.stage_1_result is not None
        assert result.stage_1_result["image_url"] == "https://cdn.runware.ai/stage1.png"
        assert result.stage_2_result is not None
        assert result.stage_2_result["model"] == "nano-banana-2-pro"

        # Verify image service called twice
        assert mock_image_service_dual.generate_images.call_count == 2

    @pytest.mark.asyncio
    async def test_dual_pipeline_stage1_failure(self, mock_image_service_dual, dual_pipeline_request):
        """Test dual pipeline handles stage 1 failure."""
        mock_image_service_dual.generate_images = AsyncMock(
            side_effect=Exception("Stage 1 failed")
        )

        service = AssemblyService(image_service=mock_image_service_dual)
        result = await service.execute(dual_pipeline_request)

        assert result.success is False
        assert result.status == ExecutionStatus.FAILED
        assert "Stage 1 failed" in result.error

    @pytest.mark.asyncio
    async def test_dual_pipeline_empty_stage1(self, mock_image_service_dual, dual_pipeline_request):
        """Test dual pipeline handles empty stage 1 result."""
        mock_image_service_dual.generate_images = AsyncMock(
            return_value=ImageGenerationResult(
                images=[],  # No images
                model_used="flux-schnell",
                provider="runware",
                cost_usd=0.02,
            )
        )

        service = AssemblyService(image_service=mock_image_service_dual)
        result = await service.execute(dual_pipeline_request)

        assert result.success is False
        assert result.status == ExecutionStatus.FAILED
        assert "Stage 1 produced no images" in result.error


class TestAssemblyServiceProgressCallback:
    """Tests for progress callback functionality."""

    @pytest.fixture
    def mock_image_service(self):
        """Create a mock ImageGenerationService."""
        service = MagicMock()
        service.generate_images = AsyncMock(return_value=ImageGenerationResult(
            images=[GeneratedImage(url="https://cdn.runware.ai/test.png")],
            model_used="test-model",
            provider="runware",
            cost_usd=0.03,
        ))
        service.close = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_progress_callback_called(self, mock_image_service):
        """Test that progress callback is called during execution."""
        request = AssemblyRequest(
            prompt="Test",
            model_id="test-model",
            parameters=GenerationParameters(),
            pipeline=PipelineConfig(pipeline_type="single"),
            job_id="job-1",
            user_id="user-1",
        )

        progress_events = []

        async def capture_progress(stage, progress, message):
            progress_events.append((stage, progress, message))

        service = AssemblyService(image_service=mock_image_service)
        result = await service.execute(request, progress_callback=capture_progress)

        assert result.success is True
        # Should have multiple progress events
        assert len(progress_events) >= 2
        # Should start with generation_start or generation_progress
        stages = [e[0] for e in progress_events]
        assert "generation_start" in stages or "generation_progress" in stages
        # Should end with complete
        assert progress_events[-1][0] == "complete"


class TestAssemblyServiceContextManager:
    """Tests for context manager functionality."""

    @pytest.mark.asyncio
    async def test_context_manager_closes_service(self):
        """Test that context manager properly closes resources."""
        mock_image_service = MagicMock()
        mock_image_service.generate_images = AsyncMock(return_value=ImageGenerationResult(
            images=[GeneratedImage(url="https://example.com/test.png")],
            model_used="test",
            provider="runware",
            cost_usd=0.01,
        ))
        mock_image_service.close = AsyncMock()

        request = AssemblyRequest(
            prompt="Test",
            model_id="test",
            parameters=GenerationParameters(),
            pipeline=PipelineConfig(pipeline_type="single"),
            job_id="job",
            user_id="user",
        )

        async with AssemblyService(image_service=mock_image_service) as service:
            result = await service.execute(request)
            assert result.success is True

        # close() should NOT be called on injected service (not owned)
        # This is the expected behavior
        mock_image_service.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager_owns_service(self):
        """Test context manager closes self-created service."""
        with patch.object(AssemblyService, '_get_image_service') as mock_get:
            mock_service = MagicMock()
            mock_service.generate_images = AsyncMock(return_value=ImageGenerationResult(
                images=[GeneratedImage(url="https://example.com/test.png")],
                model_used="test",
                provider="runware",
                cost_usd=0.01,
            ))
            mock_service.close = AsyncMock()
            mock_get.return_value = mock_service

            request = AssemblyRequest(
                prompt="Test",
                model_id="test",
                parameters=GenerationParameters(),
                pipeline=PipelineConfig(pipeline_type="single"),
                job_id="job",
                user_id="user",
            )

            # Create service without injecting image_service (will own it)
            service = AssemblyService()
            service._image_service = mock_service
            service._owns_image_service = True

            async with service:
                result = await service.execute(request)
                assert result.success is True

            # close() should be called on owned service
            mock_service.close.assert_called_once()
