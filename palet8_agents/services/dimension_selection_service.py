"""
Dimension Selection Service.

Handles selection and population of prompt dimensions based on mode,
product type, and user requirements.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from palet8_agents.models import PromptDimensions

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class DimensionConfig:
    """Configuration for dimension selection service."""

    # Dimension mapping from requirements to dimension fields
    dimension_mapping: Dict[str, str] = field(default_factory=lambda: {
        "subject": "subject",
        "style": "aesthetic",
        "aesthetic": "aesthetic",
        "colors": "color",
        "color_palette": "color",
        "color": "color",
        "composition": "composition",
        "layout": "composition",
        "background": "background",
        "lighting": "lighting",
        "texture": "texture",
        "detail": "detail_level",
        "mood": "mood",
        "emotion": "mood",
        "expression": "expression",
        "pose": "pose",
        "art_style": "art_movement",
        "reference_style": "reference_style",
    })

    # Required dimensions by mode
    required_by_mode: Dict[str, List[str]] = field(default_factory=lambda: {
        "RELAX": ["subject"],
        "STANDARD": ["subject", "aesthetic", "background"],
        "COMPLEX": ["subject", "aesthetic", "background", "composition", "lighting"],
    })

    # LLM temperature for dimension filling
    fill_temperature: float = 0.5

    # Dimension definitions for LLM prompts
    dimension_definitions: Dict[str, str] = field(default_factory=lambda: {
        "subject": "The main focus or subject of the image",
        "aesthetic": "Visual style (realistic, cartoon, minimalist, etc.)",
        "color": "Color palette or dominant colors",
        "composition": "Layout and arrangement of elements",
        "background": "Background type (solid, gradient, scene, etc.)",
        "lighting": "Lighting style (natural, studio, dramatic, etc.)",
        "texture": "Surface texture qualities",
        "detail_level": "Level of detail (high, medium, stylized)",
        "mood": "Emotional tone or atmosphere",
        "expression": "Character expression if applicable",
        "pose": "Character pose if applicable",
        "art_movement": "Art movement or historical style",
        "reference_style": "Reference to known style or artist",
    })


# =============================================================================
# EXCEPTIONS
# =============================================================================


class DimensionSelectionError(Exception):
    """Base exception for dimension selection errors."""
    pass


class DimensionFillError(DimensionSelectionError):
    """Error filling missing dimensions."""
    pass


# =============================================================================
# SERVICE
# =============================================================================


class DimensionSelectionService:
    """
    Service for selecting and populating prompt dimensions.

    This service handles:
    - Mapping user requirements to dimension fields
    - Determining required dimensions based on mode
    - Filling missing dimensions using LLM
    - Building technical specs for COMPLEX mode
    """

    def __init__(
        self,
        text_service: Optional[Any] = None,
        config: Optional[DimensionConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize dimension selection service.

        Args:
            text_service: Optional TextLLMService for dimension filling
            config: Optional custom configuration
            config_path: Optional path to config file
        """
        self._text_service = text_service
        self._owns_service = text_service is None
        self._config = config or DimensionConfig()

        if config_path:
            self._load_config(config_path)

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            if data.get("dimension_mapping"):
                self._config.dimension_mapping = data["dimension_mapping"]
            if data.get("required_by_mode"):
                self._config.required_by_mode = data["required_by_mode"]
            if data.get("fill_temperature"):
                self._config.fill_temperature = data["fill_temperature"]
            if data.get("dimension_definitions"):
                self._config.dimension_definitions = data["dimension_definitions"]

        except Exception as e:
            logger.warning(f"Failed to load dimension config: {e}")

    async def _get_text_service(self) -> Any:
        """Get or create text service."""
        if self._text_service is None:
            from palet8_agents.llm import TextLLMService
            self._text_service = TextLLMService()
            self._owns_service = True
        return self._text_service

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def select_dimensions(
        self,
        mode: str,
        requirements: Dict[str, Any],
        product_type: Optional[str] = None,
        print_method: Optional[str] = None,
        planning_context: Optional[Dict[str, Any]] = None,
    ) -> PromptDimensions:
        """
        Select and populate dimensions for prompt composition.

        Args:
            mode: Generation mode (RELAX, STANDARD, COMPLEX)
            requirements: User requirements dict
            product_type: Target product (affects technical specs)
            print_method: Print method (screen print, DTG, etc.)
            planning_context: Additional context from planner

        Returns:
            PromptDimensions with selected values
        """
        dimensions = PromptDimensions()

        # Map requirements to dimensions
        dimensions = self._map_requirements_to_dimensions(requirements, dimensions)

        # Get required dimensions for mode
        mode_upper = mode.upper()
        required_dims = set(self._config.required_by_mode.get(
            mode_upper,
            self._config.required_by_mode["STANDARD"]
        ))

        # Find missing required dimensions
        missing_required = self._get_missing_dimensions(dimensions, required_dims)

        # Fill missing required dimensions using LLM
        if missing_required:
            try:
                await self._fill_missing_dimensions(
                    dimensions=dimensions,
                    missing=missing_required,
                    planning_context=planning_context or {},
                )
            except Exception as e:
                logger.warning(f"Failed to fill missing dimensions: {e}")

        # Build technical specs for COMPLEX mode
        if mode_upper == "COMPLEX":
            dimensions.technical = self._build_technical_specs(
                requirements=requirements,
                product_type=product_type,
                print_method=print_method,
                planning_context=planning_context or {},
            )

        return dimensions

    def map_requirements(
        self,
        requirements: Dict[str, Any],
    ) -> PromptDimensions:
        """
        Map requirements to dimensions without LLM filling.

        Args:
            requirements: User requirements dict

        Returns:
            PromptDimensions with mapped values (may have None fields)
        """
        dimensions = PromptDimensions()
        return self._map_requirements_to_dimensions(requirements, dimensions)

    def get_required_dimensions(self, mode: str) -> List[str]:
        """
        Get list of required dimensions for a mode.

        Args:
            mode: Generation mode

        Returns:
            List of required dimension names
        """
        mode_upper = mode.upper()
        return self._config.required_by_mode.get(
            mode_upper,
            self._config.required_by_mode["STANDARD"]
        )

    def get_missing_dimensions(
        self,
        dimensions: PromptDimensions,
        mode: str,
    ) -> List[str]:
        """
        Get missing required dimensions for a mode.

        Args:
            dimensions: Current dimensions
            mode: Generation mode

        Returns:
            List of missing dimension names
        """
        required = set(self.get_required_dimensions(mode))
        return self._get_missing_dimensions(dimensions, required)

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _map_requirements_to_dimensions(
        self,
        requirements: Dict[str, Any],
        dimensions: PromptDimensions,
    ) -> PromptDimensions:
        """Map requirements to dimension fields."""
        for req_key, dim_key in self._config.dimension_mapping.items():
            if req_key in requirements and requirements[req_key]:
                value = requirements[req_key]
                # Convert lists to comma-separated strings
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                # Only set if dimension exists and value is non-empty
                if hasattr(dimensions, dim_key) and value:
                    setattr(dimensions, dim_key, str(value))
        return dimensions

    def _get_missing_dimensions(
        self,
        dimensions: PromptDimensions,
        required: set,
    ) -> List[str]:
        """Get list of missing required dimensions."""
        missing = []
        for dim in required:
            # Skip technical (handled separately)
            if dim == "technical":
                continue
            value = getattr(dimensions, dim, None)
            if value is None or value == "":
                missing.append(dim)
        return missing

    async def _fill_missing_dimensions(
        self,
        dimensions: PromptDimensions,
        missing: List[str],
        planning_context: Dict[str, Any],
    ) -> None:
        """Fill missing dimensions using LLM reasoning."""
        text_service = await self._get_text_service()

        system_prompt = """You are an expert at describing images for AI generation.
Given a subject and context, suggest appropriate values for missing dimensions.
Return values in format: dimension: value (one per line).
Be specific and descriptive. Avoid vague terms."""

        # Build dimension definitions for missing fields
        dim_definitions = {
            dim: self._config.dimension_definitions.get(dim, dim)
            for dim in missing
        }

        user_prompt = f"""Subject: {dimensions.subject or 'unknown'}
Current dimensions: {dimensions.to_dict()}
Missing dimensions to fill: {missing}
Dimension definitions: {dim_definitions}

Suggest values for the missing dimensions:"""

        try:
            result = await text_service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=self._config.fill_temperature,
            )

            # Parse response
            for line in result.content.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip()
                    if hasattr(dimensions, key) and value:
                        setattr(dimensions, key, value)

        except Exception as e:
            logger.warning(f"Failed to fill missing dimensions: {e}")
            raise DimensionFillError(f"LLM dimension fill failed: {e}") from e

    def _build_technical_specs(
        self,
        requirements: Dict[str, Any],
        product_type: Optional[str],
        print_method: Optional[str],
        planning_context: Dict[str, Any],
    ) -> Dict[str, str]:
        """Build technical specifications for COMPLEX mode."""
        technical = {}

        # Get template from planning context
        template = planning_context.get("technical_template", {})
        product_spec = planning_context.get("product_spec", {})

        # Apply template defaults
        if template:
            for key, value in template.items():
                if isinstance(value, str):
                    technical[key] = value

        # Add product specifications
        print_area = product_spec.get("print_area", {})
        if print_area.get("width") and print_area.get("height"):
            technical["size"] = f"{print_area['width']}x{print_area['height']} inch"
            technical["dpi"] = f"{print_area.get('dpi', 300)} DPI"

        # Print method specific settings
        if print_method == "screen_print":
            technical["color_separation"] = "spot colors, max 6"
            technical["halftone"] = "required for gradients"
        elif print_method == "embroidery":
            technical["stitch_type"] = "fill and satin"
            technical["max_colors"] = "12"
        elif print_method == "sublimation":
            technical["color_mode"] = "full color"
            technical["bleed"] = "0.125 inch"

        # Override with explicit requirements
        for key in ["dpi", "color_separation", "bleed", "safe_zone", "resolution"]:
            if requirements.get(key):
                technical[key] = str(requirements[key])

        return technical

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    async def close(self) -> None:
        """Close service and release resources."""
        if self._text_service and self._owns_service:
            await self._text_service.close()
            self._text_service = None

    async def __aenter__(self) -> "DimensionSelectionService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
