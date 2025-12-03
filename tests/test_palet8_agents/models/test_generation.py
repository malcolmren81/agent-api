"""Tests for palet8_agents.models.generation module."""

import pytest
from palet8_agents.models.generation import (
    GenerationParameters,
    PipelineConfig,
    AssemblyRequest,
)
from palet8_agents.models.safety import SafetyClassification


class TestGenerationParameters:
    """Tests for GenerationParameters dataclass."""

    def test_init_defaults(self):
        """Test default initialization."""
        params = GenerationParameters()
        assert params.width == 1024
        assert params.height == 1024
        assert params.steps == 30
        assert params.guidance_scale == 7.5
        assert params.seed is None
        assert params.num_images == 1
        assert params.provider_settings == {}

    def test_init_custom(self):
        """Test custom initialization."""
        params = GenerationParameters(
            width=512,
            height=768,
            steps=50,
            guidance_scale=9.0,
            seed=42,
            num_images=4,
            provider_settings={"quality": "high"},
        )
        assert params.width == 512
        assert params.height == 768
        assert params.steps == 50
        assert params.seed == 42
        assert params.provider_settings == {"quality": "high"}

    def test_to_dict(self):
        """Test to_dict serialization."""
        params = GenerationParameters(
            width=2048,
            height=2048,
            provider_settings={"air_id": "test-123"},
        )
        data = params.to_dict()

        assert data["width"] == 2048
        assert data["height"] == 2048
        assert data["steps"] == 30
        assert data["provider_settings"]["air_id"] == "test-123"

    def test_to_dict_no_provider_settings(self):
        """Test to_dict without provider_settings."""
        params = GenerationParameters()
        data = params.to_dict()

        assert "provider_settings" not in data or data["provider_settings"] == {}

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "width": 768,
            "height": 1024,
            "steps": 40,
            "guidance_scale": 8.0,
            "seed": 123,
            "num_images": 2,
            "provider_settings": {"model": "flux"},
        }
        params = GenerationParameters.from_dict(data)

        assert params.width == 768
        assert params.height == 1024
        assert params.steps == 40
        assert params.seed == 123
        assert params.provider_settings["model"] == "flux"


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_init_defaults(self):
        """Test default initialization (single pipeline)."""
        config = PipelineConfig()
        assert config.pipeline_type == "single"
        assert config.pipeline_name is None
        assert config.stage_1_model == ""
        assert config.stage_1_purpose == ""
        assert config.stage_2_model is None
        assert config.stage_2_purpose is None
        assert config.decision_rationale == ""

    def test_init_single_pipeline(self):
        """Test single pipeline initialization."""
        config = PipelineConfig(
            pipeline_type="single",
            stage_1_model="flux-pro",
            stage_1_purpose="Generate final image",
            decision_rationale="Simple request, single pipeline sufficient",
        )
        assert config.pipeline_type == "single"
        assert config.stage_1_model == "flux-pro"
        assert config.stage_2_model is None

    def test_init_dual_pipeline(self):
        """Test dual pipeline initialization."""
        config = PipelineConfig(
            pipeline_type="dual",
            pipeline_name="creative_art",
            stage_1_model="midjourney-v7",
            stage_1_purpose="Creative generation",
            stage_2_model="nano-banana-2-pro",
            stage_2_purpose="Text refinement",
            decision_rationale="Text in image detected",
        )
        assert config.pipeline_type == "dual"
        assert config.pipeline_name == "creative_art"
        assert config.stage_2_model == "nano-banana-2-pro"

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = PipelineConfig(
            pipeline_type="dual",
            pipeline_name="photorealistic",
            stage_1_model="imagen-4",
            stage_1_purpose="Base generation",
            stage_2_model="editor-pro",
            stage_2_purpose="Refinement",
        )
        data = config.to_dict()

        assert data["pipeline_type"] == "dual"
        assert data["pipeline_name"] == "photorealistic"
        assert data["stage_2_model"] == "editor-pro"

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "pipeline_type": "dual",
            "pipeline_name": "layout_poster",
            "stage_1_model": "flux-flex",
            "stage_1_purpose": "Layout",
            "stage_2_model": "qwen-edit",
            "stage_2_purpose": "Text placement",
            "decision_rationale": "Poster design",
        }
        config = PipelineConfig.from_dict(data)

        assert config.pipeline_type == "dual"
        assert config.pipeline_name == "layout_poster"
        assert config.stage_2_model == "qwen-edit"


class TestAssemblyRequest:
    """Tests for AssemblyRequest dataclass."""

    def test_init_defaults(self):
        """Test default initialization."""
        request = AssemblyRequest()
        assert request.prompt == ""
        assert request.negative_prompt == ""
        assert request.mode == "STANDARD"
        assert request.dimensions == {}
        assert isinstance(request.pipeline, PipelineConfig)
        assert request.model_id == ""
        assert isinstance(request.parameters, GenerationParameters)
        assert request.reference_image_url is None
        assert request.reference_strength == 0.75
        assert request.prompt_quality_score == 0.0
        assert request.quality_acceptable is False
        assert isinstance(request.safety, SafetyClassification)
        assert request.estimated_cost == 0.0
        assert request.job_id == ""

    def test_init_full(self):
        """Test full initialization."""
        pipeline = PipelineConfig(
            pipeline_type="single",
            stage_1_model="flux-pro",
        )
        params = GenerationParameters(width=2048, height=2048)
        safety = SafetyClassification(
            is_safe=True,
            requires_review=False,
            risk_level="low",
            categories=[],
        )

        request = AssemblyRequest(
            prompt="A beautiful mountain landscape",
            negative_prompt="blurry, low quality",
            mode="COMPLEX",
            dimensions={"subject": "mountain", "background": "sky"},
            pipeline=pipeline,
            model_id="flux-pro-1.1",
            model_rationale="Best for landscapes",
            model_alternatives=["imagen-4", "sdxl"],
            parameters=params,
            reference_image_url="https://example.com/ref.jpg",
            reference_strength=0.5,
            prompt_quality_score=0.85,
            quality_acceptable=True,
            safety=safety,
            estimated_cost=0.08,
            estimated_time_ms=20000,
            job_id="job-xyz",
            user_id="user-123",
            product_type="poster",
            print_method="digital",
            revision_count=0,
        )

        assert request.prompt == "A beautiful mountain landscape"
        assert request.mode == "COMPLEX"
        assert request.model_id == "flux-pro-1.1"
        assert request.parameters.width == 2048
        assert request.safety.is_safe is True
        assert request.job_id == "job-xyz"

    def test_to_dict(self):
        """Test to_dict serialization."""
        request = AssemblyRequest(
            prompt="Test prompt",
            mode="RELAX",
            model_id="test-model",
            job_id="job-test",
        )
        data = request.to_dict()

        assert data["prompt"] == "Test prompt"
        assert data["mode"] == "RELAX"
        assert data["model_id"] == "test-model"
        assert data["job_id"] == "job-test"
        assert "pipeline" in data
        assert "parameters" in data
        assert "safety" in data

    def test_from_dict(self):
        """Test from_dict deserialization."""
        data = {
            "prompt": "A cat",
            "negative_prompt": "dog",
            "mode": "STANDARD",
            "dimensions": {"subject": "cat"},
            "pipeline": {
                "pipeline_type": "single",
                "pipeline_name": None,
                "stage_1_model": "flux",
                "stage_1_purpose": "Generate",
                "stage_2_model": None,
                "stage_2_purpose": None,
                "decision_rationale": "",
            },
            "model_id": "flux-pro",
            "model_rationale": "Fast and good",
            "model_alternatives": [],
            "parameters": {
                "width": 1024,
                "height": 1024,
                "steps": 30,
                "guidance_scale": 7.5,
                "seed": None,
                "num_images": 1,
                "provider_settings": {},
            },
            "reference_image_url": None,
            "reference_strength": 0.75,
            "prompt_quality_score": 0.8,
            "quality_acceptable": True,
            "safety": {
                "is_safe": True,
                "requires_review": False,
                "risk_level": "low",
                "categories": [],
                "flags": {},
                "reason": "",
            },
            "estimated_cost": 0.04,
            "estimated_time_ms": 15000,
            "context_used": None,
            "job_id": "job-fromdict",
            "user_id": "user-1",
            "product_type": "tshirt",
            "print_method": None,
            "revision_count": 0,
            "metadata": {},
        }
        request = AssemblyRequest.from_dict(data)

        assert request.prompt == "A cat"
        assert request.mode == "STANDARD"
        assert request.model_id == "flux-pro"
        assert request.pipeline.stage_1_model == "flux"
        assert request.parameters.width == 1024
        assert request.safety.is_safe is True

    def test_roundtrip(self):
        """Test to_dict and from_dict roundtrip."""
        pipeline = PipelineConfig(
            pipeline_type="dual",
            pipeline_name="creative_art",
            stage_1_model="midjourney",
            stage_1_purpose="Create",
            stage_2_model="nano",
            stage_2_purpose="Refine",
        )
        params = GenerationParameters(width=512, height=768, seed=42)

        original = AssemblyRequest(
            prompt="Original prompt",
            negative_prompt="bad quality",
            mode="COMPLEX",
            pipeline=pipeline,
            parameters=params,
            model_id="test-model",
            job_id="roundtrip-job",
        )

        data = original.to_dict()
        restored = AssemblyRequest.from_dict(data)

        assert restored.prompt == original.prompt
        assert restored.mode == original.mode
        assert restored.pipeline.pipeline_name == original.pipeline.pipeline_name
        assert restored.parameters.seed == original.parameters.seed
        assert restored.job_id == original.job_id
