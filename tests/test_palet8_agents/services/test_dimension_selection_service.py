"""Tests for palet8_agents.services.dimension_selection_service module."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from palet8_agents.services.dimension_selection_service import (
    DimensionSelectionService,
    DimensionSelectionError,
    DimensionFillError,
    DimensionConfig,
)
from palet8_agents.models import PromptDimensions


class TestDimensionConfig:
    """Tests for DimensionConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = DimensionConfig()
        assert "subject" in config.dimension_mapping
        assert "style" in config.dimension_mapping
        assert "RELAX" in config.required_by_mode
        assert "STANDARD" in config.required_by_mode
        assert "COMPLEX" in config.required_by_mode
        assert config.fill_temperature == 0.5

    def test_custom_values(self):
        """Test custom configuration values."""
        config = DimensionConfig(
            fill_temperature=0.3,
            required_by_mode={"SIMPLE": ["subject"]},
        )
        assert config.fill_temperature == 0.3
        assert "SIMPLE" in config.required_by_mode


class TestDimensionSelectionService:
    """Tests for DimensionSelectionService."""

    def test_init_with_defaults(self):
        """Test initialization with defaults."""
        service = DimensionSelectionService()
        assert service._text_service is None
        assert service._owns_service is True
        assert service._config is not None

    def test_init_with_config(self):
        """Test initialization with custom config."""
        config = DimensionConfig(fill_temperature=0.2)
        service = DimensionSelectionService(config=config)
        assert service._config.fill_temperature == 0.2


class TestRequirementsMapping:
    """Tests for requirements to dimensions mapping."""

    def test_map_requirements_basic(self):
        """Test basic requirements mapping."""
        service = DimensionSelectionService()
        requirements = {
            "subject": "A cat",
            "style": "realistic",
            "colors": ["blue", "green"],
            "mood": "playful",
        }
        dimensions = service.map_requirements(requirements)

        assert dimensions.subject == "A cat"
        assert dimensions.aesthetic == "realistic"
        assert dimensions.color == "blue, green"
        assert dimensions.mood == "playful"

    def test_map_requirements_list_conversion(self):
        """Test that list values are converted to strings."""
        service = DimensionSelectionService()
        requirements = {
            "colors": ["red", "blue", "green"],
        }
        dimensions = service.map_requirements(requirements)

        assert dimensions.color == "red, blue, green"

    def test_map_requirements_empty(self):
        """Test mapping with empty requirements."""
        service = DimensionSelectionService()
        dimensions = service.map_requirements({})

        assert dimensions.subject is None
        assert dimensions.aesthetic is None

    def test_map_requirements_alternate_keys(self):
        """Test mapping with alternate key names."""
        service = DimensionSelectionService()
        requirements = {
            "aesthetic": "cartoon",
            "color_palette": "warm",
            "layout": "centered",
            "emotion": "joyful",
        }
        dimensions = service.map_requirements(requirements)

        assert dimensions.aesthetic == "cartoon"
        assert dimensions.color == "warm"
        assert dimensions.composition == "centered"
        assert dimensions.mood == "joyful"


class TestRequiredDimensions:
    """Tests for required dimensions by mode."""

    def test_get_required_relax(self):
        """Test required dimensions for RELAX mode."""
        service = DimensionSelectionService()
        required = service.get_required_dimensions("RELAX")

        assert "subject" in required
        assert len(required) == 1

    def test_get_required_standard(self):
        """Test required dimensions for STANDARD mode."""
        service = DimensionSelectionService()
        required = service.get_required_dimensions("STANDARD")

        assert "subject" in required
        assert "aesthetic" in required
        assert "background" in required

    def test_get_required_complex(self):
        """Test required dimensions for COMPLEX mode."""
        service = DimensionSelectionService()
        required = service.get_required_dimensions("COMPLEX")

        assert "subject" in required
        assert "aesthetic" in required
        assert "background" in required
        assert "composition" in required
        assert "lighting" in required

    def test_get_required_case_insensitive(self):
        """Test that mode is case insensitive."""
        service = DimensionSelectionService()
        required_lower = service.get_required_dimensions("relax")
        required_upper = service.get_required_dimensions("RELAX")

        assert required_lower == required_upper


class TestMissingDimensions:
    """Tests for missing dimension detection."""

    def test_get_missing_all_missing(self):
        """Test when all dimensions are missing."""
        service = DimensionSelectionService()
        dimensions = PromptDimensions()
        missing = service.get_missing_dimensions(dimensions, "STANDARD")

        assert "subject" in missing
        assert "aesthetic" in missing
        assert "background" in missing

    def test_get_missing_some_present(self):
        """Test when some dimensions are present."""
        service = DimensionSelectionService()
        dimensions = PromptDimensions()
        dimensions.subject = "A sunset"
        dimensions.aesthetic = "realistic"
        missing = service.get_missing_dimensions(dimensions, "STANDARD")

        assert "subject" not in missing
        assert "aesthetic" not in missing
        assert "background" in missing

    def test_get_missing_all_present(self):
        """Test when all dimensions are present."""
        service = DimensionSelectionService()
        dimensions = PromptDimensions()
        dimensions.subject = "A cat"
        missing = service.get_missing_dimensions(dimensions, "RELAX")

        assert len(missing) == 0


class TestTechnicalSpecs:
    """Tests for technical specifications building."""

    def test_build_technical_empty(self):
        """Test building with empty context."""
        service = DimensionSelectionService()
        specs = service._build_technical_specs(
            requirements={},
            product_type=None,
            print_method=None,
            planning_context={},
        )

        assert isinstance(specs, dict)

    def test_build_technical_screen_print(self):
        """Test building for screen print method."""
        service = DimensionSelectionService()
        specs = service._build_technical_specs(
            requirements={},
            product_type="apparel",
            print_method="screen_print",
            planning_context={},
        )

        assert specs.get("color_separation") == "spot colors, max 6"
        assert specs.get("halftone") == "required for gradients"

    def test_build_technical_embroidery(self):
        """Test building for embroidery method."""
        service = DimensionSelectionService()
        specs = service._build_technical_specs(
            requirements={},
            product_type="apparel",
            print_method="embroidery",
            planning_context={},
        )

        assert specs.get("stitch_type") == "fill and satin"
        assert specs.get("max_colors") == "12"

    def test_build_technical_with_overrides(self):
        """Test that requirements override defaults."""
        service = DimensionSelectionService()
        specs = service._build_technical_specs(
            requirements={"dpi": "600"},
            product_type=None,
            print_method=None,
            planning_context={},
        )

        assert specs.get("dpi") == "600"

    def test_build_technical_with_product_spec(self):
        """Test building with product specifications."""
        service = DimensionSelectionService()
        specs = service._build_technical_specs(
            requirements={},
            product_type="poster",
            print_method=None,
            planning_context={
                "product_spec": {
                    "print_area": {
                        "width": 12,
                        "height": 18,
                        "dpi": 300,
                    }
                }
            },
        )

        assert specs.get("size") == "12x18 inch"
        assert specs.get("dpi") == "300 DPI"


class TestAsyncOperations:
    """Async tests for DimensionSelectionService."""

    @pytest.mark.asyncio
    async def test_select_dimensions_basic(self):
        """Test basic dimension selection."""
        service = DimensionSelectionService()
        requirements = {
            "subject": "A mountain landscape",
            "style": "realistic",
        }

        dimensions = await service.select_dimensions(
            mode="RELAX",
            requirements=requirements,
        )

        assert dimensions.subject == "A mountain landscape"
        assert dimensions.aesthetic == "realistic"

    @pytest.mark.asyncio
    async def test_select_dimensions_complex_mode(self):
        """Test dimension selection with COMPLEX mode."""
        mock_text_service = AsyncMock()
        mock_text_service.generate_text = AsyncMock(
            return_value=MagicMock(
                content="background: gradient sky\ncomposition: centered\nlighting: natural"
            )
        )

        service = DimensionSelectionService(text_service=mock_text_service)
        requirements = {
            "subject": "A portrait",
            "style": "oil painting",
        }

        dimensions = await service.select_dimensions(
            mode="COMPLEX",
            requirements=requirements,
            product_type="canvas",
            print_method=None,
        )

        assert dimensions.subject == "A portrait"
        assert dimensions.technical is not None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with DimensionSelectionService() as service:
            assert service is not None

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing service releases resources."""
        service = DimensionSelectionService()
        mock_text = AsyncMock()
        service._text_service = mock_text
        service._owns_service = True

        await service.close()

        mock_text.close.assert_called_once()
        assert service._text_service is None
