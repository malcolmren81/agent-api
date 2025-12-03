"""Tests for palet8_agents.services.model_selection_service module."""

import pytest
from palet8_agents.services.model_selection_service import (
    ModelSelectionService,
    ModelSelectionError,
    NoCompatibleModelError,
    ModelSelectionConfig,
)
from palet8_agents.models import PipelineConfig


class TestModelSelectionConfig:
    """Tests for ModelSelectionConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = ModelSelectionConfig()
        assert config.default_model == "flux-1-kontext-pro"
        assert "text_in_image" in config.dual_pipeline_triggers
        assert "creative_art" in config.dual_pipelines
        assert "photorealistic" in config.dual_pipelines
        assert "layout_poster" in config.dual_pipelines
        assert config.default_cost_per_image == 0.04

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ModelSelectionConfig(
            default_model="custom-model",
            default_cost_per_image=0.10,
        )
        assert config.default_model == "custom-model"
        assert config.default_cost_per_image == 0.10


class TestModelSelectionService:
    """Tests for ModelSelectionService."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        service = ModelSelectionService()
        assert service._model_info_service is None
        assert service._config is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = ModelSelectionConfig(default_model="test-model")
        service = ModelSelectionService(config=config)
        assert service._config.default_model == "test-model"


class TestPipelineSelection:
    """Tests for pipeline selection logic."""

    def test_single_pipeline_no_triggers(self):
        """Test single pipeline when no triggers detected."""
        service = ModelSelectionService()
        pipeline = service.select_pipeline(
            requirements={"subject": "A simple cat"},
            prompt="A simple cat illustration",
        )

        assert pipeline.pipeline_type == "single"
        assert pipeline.pipeline_name is None
        assert "no dual triggers" in pipeline.decision_rationale

    def test_dual_pipeline_text_trigger(self):
        """Test dual pipeline triggered by text keywords."""
        service = ModelSelectionService()
        pipeline = service.select_pipeline(
            requirements={"subject": "Logo with text"},
            prompt="A professional logo with typography and lettering",
        )

        assert pipeline.pipeline_type == "dual"
        assert pipeline.pipeline_name is not None
        assert "text_in_image" in pipeline.decision_rationale

    def test_dual_pipeline_production_trigger(self):
        """Test dual pipeline triggered by production keywords."""
        service = ModelSelectionService()
        pipeline = service.select_pipeline(
            requirements={"subject": "Print-ready poster design"},
            prompt="4k poster design for billboard",
        )

        assert pipeline.pipeline_type == "dual"

    def test_dual_pipeline_character_trigger(self):
        """Test dual pipeline triggered by character refinement."""
        service = ModelSelectionService()
        pipeline = service.select_pipeline(
            requirements={"subject": "Character with face fix needed"},
            prompt="Character pose adjust with expression change",
        )

        assert pipeline.pipeline_type == "dual"

    def test_dual_pipeline_photorealistic_style(self):
        """Test photorealistic pipeline selection."""
        service = ModelSelectionService()
        pipeline = service.select_pipeline(
            requirements={
                "subject": "Product photo",
                "style": "photorealistic",
            },
            prompt="Product photo with text overlay",
        )

        assert pipeline.pipeline_type == "dual"
        assert pipeline.pipeline_name == "photorealistic"

    def test_dual_pipeline_layout_content(self):
        """Test layout pipeline selection for posters."""
        service = ModelSelectionService()
        pipeline = service.select_pipeline(
            requirements={"subject": "Event poster"},
            prompt="Poster design with headline text",
        )

        assert pipeline.pipeline_type == "dual"
        assert pipeline.pipeline_name == "layout_poster"


class TestPipelineConfig:
    """Tests for pipeline configuration retrieval."""

    def test_get_available_pipelines(self):
        """Test getting available pipeline configurations."""
        service = ModelSelectionService()
        pipelines = service.get_available_pipelines()

        assert "creative_art" in pipelines
        assert "photorealistic" in pipelines
        assert "layout_poster" in pipelines
        assert "stage_1_model" in pipelines["creative_art"]
        assert "stage_2_model" in pipelines["creative_art"]

    def test_get_pipeline_triggers(self):
        """Test getting pipeline triggers."""
        service = ModelSelectionService()
        triggers = service.get_pipeline_triggers()

        assert "text_in_image" in triggers
        assert "character_refinement" in triggers
        assert "multi_element" in triggers
        assert "production_quality" in triggers
        assert "text" in triggers["text_in_image"]


class TestCostEstimation:
    """Tests for cost and latency estimation."""

    def test_estimate_cost_single_pipeline(self):
        """Test cost estimation for single pipeline."""
        service = ModelSelectionService()
        pipeline = PipelineConfig(pipeline_type="single")

        cost, latency = service.estimate_cost(pipeline, num_images=1)

        assert cost == 0.04  # Default cost
        assert latency == 15000  # Default latency

    def test_estimate_cost_dual_pipeline(self):
        """Test cost estimation for dual pipeline."""
        service = ModelSelectionService()
        pipeline = PipelineConfig(pipeline_type="dual")

        cost, latency = service.estimate_cost(pipeline, num_images=1)

        assert cost == 0.08  # Double for dual
        assert latency == 37500  # 15000 * 2.5

    def test_estimate_cost_multiple_images(self):
        """Test cost estimation for multiple images."""
        service = ModelSelectionService()
        pipeline = PipelineConfig(pipeline_type="single")

        cost, latency = service.estimate_cost(pipeline, num_images=4)

        assert cost == 0.16  # 0.04 * 4

    def test_estimate_cost_with_model_specs(self):
        """Test cost estimation with model specs."""
        service = ModelSelectionService()
        pipeline = PipelineConfig(pipeline_type="single")
        model_specs = {"cost": {"per_image": 0.10}}

        cost, latency = service.estimate_cost(
            pipeline, model_specs=model_specs, num_images=1
        )

        assert cost == 0.10


class TestModelSelection:
    """Tests for model selection logic."""

    @pytest.mark.asyncio
    async def test_select_model_no_context(self):
        """Test model selection with no context."""
        service = ModelSelectionService()

        model_id, rationale, alternatives, specs = await service.select_model(
            mode="STANDARD",
            requirements={},
        )

        assert model_id == "flux-1-kontext-pro"  # Default
        assert "no compatible models" in rationale.lower()

    @pytest.mark.asyncio
    async def test_select_model_with_models(self):
        """Test model selection with available models."""
        service = ModelSelectionService()
        context = {
            "available_models": [
                {
                    "model_id": "model-a",
                    "display_name": "Model A",
                    "quality_score": 0.9,
                    "capabilities": ["fast"],
                },
                {
                    "model_id": "model-b",
                    "display_name": "Model B",
                    "quality_score": 0.7,
                },
            ],
            "compatibility_results": {
                "model-a": {"compatible": True, "score": 0.8},
                "model-b": {"compatible": True, "score": 0.6},
            },
        }

        model_id, rationale, alternatives, specs = await service.select_model(
            mode="STANDARD",
            requirements={},
            model_info_context=context,
        )

        assert model_id == "model-a"
        assert "Model A" in rationale
        assert "model-b" in alternatives

    @pytest.mark.asyncio
    async def test_select_model_speed_priority(self):
        """Test model selection with speed priority."""
        service = ModelSelectionService()
        context = {
            "available_models": [
                {
                    "model_id": "fast-model",
                    "display_name": "Fast Model",
                    "quality_score": 0.7,
                    "capabilities": ["fast"],
                },
                {
                    "model_id": "slow-model",
                    "display_name": "Slow Model",
                    "quality_score": 0.9,
                },
            ],
            "compatibility_results": {
                "fast-model": {"compatible": True, "score": 0.7},
                "slow-model": {"compatible": True, "score": 0.7},
            },
        }

        model_id, rationale, _, _ = await service.select_model(
            mode="STANDARD",
            requirements={"priority": "fast"},
            model_info_context=context,
        )

        assert model_id == "fast-model"
        assert "fast generation" in rationale

    @pytest.mark.asyncio
    async def test_select_model_quality_priority(self):
        """Test model selection with quality priority."""
        service = ModelSelectionService()
        context = {
            "available_models": [
                {
                    "model_id": "low-quality",
                    "display_name": "Low Quality",
                    "quality_score": 0.5,
                },
                {
                    "model_id": "high-quality",
                    "display_name": "High Quality",
                    "quality_score": 0.95,
                },
            ],
            "compatibility_results": {
                "low-quality": {"compatible": True, "score": 0.7},
                "high-quality": {"compatible": True, "score": 0.7},
            },
        }

        model_id, rationale, _, _ = await service.select_model(
            mode="STANDARD",
            requirements={"quality_level": "premium"},
            model_info_context=context,
        )

        assert model_id == "high-quality"
        assert "premium quality" in rationale


class TestAsyncOperations:
    """Async tests for ModelSelectionService."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with ModelSelectionService() as service:
            assert service is not None

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing service."""
        service = ModelSelectionService()
        await service.close()  # Should not raise
