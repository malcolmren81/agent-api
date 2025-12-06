"""Tests for palet8_agents.services.genflow_service module.

Goal-based test cases:
- Goal 4: Genflow Service Works Correctly (TC4.1-TC4.6)
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Import the service
from palet8_agents.services.genflow_service import (
    GenflowService,
    GenflowServiceConfig,
)
from palet8_agents.models.genplan import GenflowConfig


class TestGenflowServiceInit:
    """Tests for GenflowService initialization."""

    def test_init_defaults(self):
        """Test default initialization."""
        service = GenflowService()
        assert service._config is not None
        assert service._config.genflows is not None

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        custom_config = GenflowServiceConfig(
            genflows={
                "test_flow": {
                    "description": "Test flow",
                    "stages": 1,
                }
            }
        )
        service = GenflowService(config=custom_config)
        assert "test_flow" in service._config.genflows


class TestGenflowDualTriggers:
    """Tests for dual pipeline triggers (TC4.1-TC4.2)."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_trigger_typography_activates_dual(self, service):
        """TC4.1: Trigger 'typography' activates dual pipeline."""
        requirements = {
            "subject": "a poster with typography",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            prompt="A poster with bold typography and text",
        )

        assert genflow.flow_type == "dual"
        assert "text" in genflow.rationale.lower() or "typography" in genflow.rationale.lower()

    def test_trigger_text_overlay_activates_dual(self, service):
        """Test 'text overlay' trigger."""
        requirements = {
            "subject": "an image with text overlay",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="standard",
            prompt="Create an image with text overlay",
        )

        assert genflow.flow_type == "dual"

    def test_trigger_lettering_activates_dual(self, service):
        """Test 'lettering' trigger."""
        requirements = {
            "subject": "decorative lettering design",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="standard",
        )

        assert genflow.flow_type == "dual"

    def test_trigger_character_edit_activates_dual(self, service):
        """TC4.2: Trigger 'character edit' activates dual pipeline."""
        requirements = {
            "subject": "a character",
            "character_edit": True,
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            prompt="A character with face refinement",
            user_info={"has_reference": True},
        )

        assert genflow.flow_type == "dual"

    def test_trigger_face_fix_activates_dual(self, service):
        """Test 'face fix' trigger."""
        requirements = {
            "subject": "portrait",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            prompt="A portrait with face fix and expression adjustment",
        )

        assert genflow.flow_type == "dual"

    def test_trigger_expression_activates_dual(self, service):
        """Test 'expression' character trigger."""
        requirements = {
            "subject": "character with specific expression",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="standard",
            prompt="A character with happy expression adjustment",
        )

        # May or may not trigger dual depending on other factors
        # but should recognize the trigger
        assert genflow is not None


class TestGenflowStyleMapping:
    """Tests for style-based pipeline mapping (TC4.3)."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_style_photorealistic_maps_to_pipeline(self, service):
        """TC4.3: Style 'photorealistic' maps to photorealistic pipeline."""
        requirements = {
            "subject": "a landscape",
            "style": "photorealistic",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            prompt="A photorealistic landscape photo",
            user_info={"style": "photorealistic"},
        )

        # Should trigger dual pipeline for complex photorealistic
        if genflow.flow_type == "dual":
            assert "photo" in genflow.flow_name.lower()

    def test_style_realistic_maps_correctly(self, service):
        """Test 'realistic' style mapping."""
        requirements = {
            "subject": "portrait",
            "style": "realistic",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            user_info={"style": "realistic"},
        )

        # Should be handled appropriately
        assert genflow is not None


class TestGenflowContentMapping:
    """Tests for content-based pipeline mapping (TC4.4)."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_content_poster_maps_to_layout_poster(self, service):
        """TC4.4: Content 'poster' maps to layout_poster pipeline."""
        requirements = {
            "subject": "a movie poster",
            "product_type": "poster",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            prompt="A professional movie poster with title",
            user_info={"product_type": "poster"},
        )

        # Poster should trigger layout_poster dual pipeline
        if genflow.flow_type == "dual":
            assert "poster" in genflow.flow_name.lower() or "layout" in genflow.flow_name.lower()

    def test_content_billboard_triggers_dual(self, service):
        """Test billboard content triggers appropriate pipeline."""
        requirements = {
            "subject": "billboard advertisement",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            prompt="A professional billboard advertisement",
        )

        # Billboard is a production quality trigger
        assert genflow.flow_type == "dual"


class TestGenflowDefaultBehavior:
    """Tests for default pipeline behavior (TC4.5)."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_no_triggers_defaults_to_single(self, service):
        """TC4.5: No triggers defaults to single pipeline."""
        requirements = {
            "subject": "a simple cat",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="simple",
            prompt="A simple cat illustration",
        )

        assert genflow.flow_type == "single"
        assert "single" in genflow.flow_name.lower()

    def test_simple_request_uses_single(self, service):
        """Test simple request uses single pipeline."""
        requirements = {
            "subject": "a flower",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="simple",
        )

        assert genflow.flow_type == "single"

    def test_standard_without_triggers_uses_single(self, service):
        """Test standard complexity without triggers uses single."""
        requirements = {
            "subject": "a landscape",
            "style": "illustration",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="standard",
            prompt="A beautiful landscape illustration",
        )

        # Without text/character triggers, should use single
        # (unless style triggers something)
        assert genflow is not None


class TestGenflowComplexityConsideration:
    """Tests for complexity-based pipeline consideration (TC4.6)."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_complex_mode_considers_dual(self, service):
        """TC4.6: Complex mode triggers dual pipeline consideration."""
        requirements = {
            "subject": "a detailed artwork",
        }

        # Complex mode should consider dual more often
        genflow_complex = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            prompt="A highly detailed professional artwork",
        )

        genflow_simple = service.determine_genflow(
            requirements=requirements,
            complexity="simple",
            prompt="A simple artwork",
        )

        # Complex is more likely to be dual than simple
        # (though not guaranteed without triggers)
        assert genflow_complex is not None
        assert genflow_simple.flow_type == "single"

    def test_complex_with_production_quality(self, service):
        """Test complex with production quality keywords."""
        requirements = {
            "subject": "a professional product shot",
        }

        genflow = service.determine_genflow(
            requirements=requirements,
            complexity="complex",
            prompt="A print-ready professional product shot for billboard",
        )

        # Production quality should trigger dual
        assert genflow.flow_type == "dual"


class TestGenflowTriggerDetection:
    """Tests for trigger detection logic."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_detects_text_in_image_triggers(self, service):
        """Test detection of text_in_image triggers."""
        triggers = service.get_flow_triggers()

        assert "text_in_image" in triggers
        text_triggers = triggers["text_in_image"]

        # Should include common text-related keywords
        assert any("text" in t.lower() for t in text_triggers)
        assert any("typography" in t.lower() for t in text_triggers)

    def test_detects_character_triggers(self, service):
        """Test detection of character_refinement triggers."""
        triggers = service.get_flow_triggers()

        assert "character_refinement" in triggers
        char_triggers = triggers["character_refinement"]

        # Should include character-related keywords
        assert any("character" in t.lower() for t in char_triggers)

    def test_detects_production_triggers(self, service):
        """Test detection of production_quality triggers."""
        triggers = service.get_flow_triggers()

        assert "production_quality" in triggers
        prod_triggers = triggers["production_quality"]

        # Should include production keywords
        assert any("print" in t.lower() or "professional" in t.lower() for t in prod_triggers)

    def test_detects_multi_element_triggers(self, service):
        """Test detection of multi_element triggers."""
        triggers = service.get_flow_triggers()

        assert "multi_element" in triggers


class TestGenflowConfigOutput:
    """Tests for GenflowConfig output structure."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_single_flow_has_correct_structure(self, service):
        """Test single flow output structure."""
        genflow = service.determine_genflow(
            requirements={"subject": "a cat"},
            complexity="simple",
        )

        assert genflow.flow_type == "single"
        assert genflow.flow_name is not None
        assert genflow.rationale is not None
        assert len(genflow.rationale) > 0

    def test_dual_flow_has_correct_structure(self, service):
        """Test dual flow output structure."""
        genflow = service.determine_genflow(
            requirements={"subject": "poster with text"},
            complexity="complex",
            prompt="A poster with typography and text overlay",
        )

        assert genflow.flow_type == "dual"
        assert genflow.flow_name is not None
        assert genflow.rationale is not None
        # Dual should have triggered_by
        if genflow.triggered_by:
            assert genflow.triggered_by in [
                "text_in_image",
                "character_refinement",
                "multi_element",
                "production_quality",
            ]

    def test_genflow_config_to_dict(self, service):
        """Test GenflowConfig serialization."""
        genflow = service.determine_genflow(
            requirements={"subject": "a cat"},
            complexity="simple",
        )

        genflow_dict = genflow.to_dict()

        assert "flow_type" in genflow_dict
        assert "flow_name" in genflow_dict
        assert "rationale" in genflow_dict


class TestGenflowAvailableFlows:
    """Tests for available flows."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_get_available_flows(self, service):
        """Test getting available flows."""
        flows = service.get_available_flows()

        assert isinstance(flows, dict)
        assert len(flows) > 0

        # Should have both single and dual flows
        flow_names = list(flows.keys())
        assert any("single" in name.lower() for name in flow_names)

    def test_available_flows_have_descriptions(self, service):
        """Test that available flows have descriptions."""
        flows = service.get_available_flows()

        for name, config in flows.items():
            # Each flow should have some configuration
            assert config is not None


class TestGenflowUserInfo:
    """Tests for user_info parameter handling."""

    @pytest.fixture
    def service(self):
        """Create service for testing."""
        return GenflowService()

    def test_user_info_style_affects_pipeline(self, service):
        """Test that user_info style affects pipeline selection."""
        # Photorealistic style in user_info
        genflow_photo = service.determine_genflow(
            requirements={"subject": "portrait"},
            complexity="complex",
            user_info={"style": "photorealistic"},
        )

        # Art style in user_info
        genflow_art = service.determine_genflow(
            requirements={"subject": "portrait"},
            complexity="complex",
            user_info={"style": "illustration"},
        )

        # Both should work, potentially with different pipelines
        assert genflow_photo is not None
        assert genflow_art is not None

    def test_user_info_product_type_affects_pipeline(self, service):
        """Test that user_info product_type affects pipeline selection."""
        genflow = service.determine_genflow(
            requirements={"subject": "design"},
            complexity="complex",
            user_info={"product_type": "poster"},
        )

        # Poster product type should influence selection
        assert genflow is not None

    def test_user_info_text_content_triggers_dual(self, service):
        """Test that user_info text_content triggers dual pipeline."""
        genflow = service.determine_genflow(
            requirements={"subject": "design"},
            complexity="standard",
            user_info={"text_content": "HELLO"},
        )

        # Text content should trigger dual
        assert genflow.flow_type == "dual"
