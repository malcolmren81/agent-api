"""
Model Selection Service.

Handles selection of optimal image generation model and pipeline configuration
based on requirements, mode, and task scope.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from palet8_agents.models import PipelineConfig

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ModelSelectionConfig:
    """Configuration for model selection service."""

    # Default model if no compatible models found (loaded from config)
    default_model: Optional[str] = None

    # Dual pipeline triggers
    dual_pipeline_triggers: Dict[str, List[str]] = field(default_factory=lambda: {
        "text_in_image": [
            "text", "typography", "lettering", "font", "words",
            "title", "headline", "quote", "caption", "label",
        ],
        "character_refinement": [
            "character edit", "face fix", "expression change",
            "pose adjust", "facial expression", "character pose",
        ],
        "multi_element": [
            "multiple subjects", "complex composition", "layered design",
            "multiple characters", "group scene",
        ],
        "production_quality": [
            "print-ready", "production", "high accuracy",
            "4k", "poster", "billboard", "large format",
        ],
    })

    # Dual pipeline configurations (models loaded from image_models_config.yaml)
    dual_pipelines: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "creative_art": {
            "name": "High-Creative Art Pipeline",
            "stage_1_model": None,
            "stage_1_purpose": "Generate creative, non-realistic composition",
            "stage_2_model": None,
            "stage_2_purpose": "Refine characters, add/correct text, adjust elements",
        },
        "photorealistic": {
            "name": "Photorealistic Pipeline",
            "stage_1_model": None,
            "stage_1_purpose": "Generate photorealistic base with exceptional detail",
            "stage_2_model": None,
            "stage_2_purpose": "Character edits, text overlays, stylistic adjustments",
        },
        "layout_poster": {
            "name": "Layout & Poster Design Pipeline",
            "stage_1_model": None,
            "stage_1_purpose": "Generate layout with accurate text placement",
            "stage_2_model": None,
            "stage_2_purpose": "Targeted edits, text correction, color adjustments",
        },
    })

    # Style to pipeline mapping
    style_mapping: Dict[str, str] = field(default_factory=lambda: {
        "photorealistic": "photorealistic",
        "photo": "photorealistic",
        "realistic": "photorealistic",
        "product": "photorealistic",
        "lifestyle": "photorealistic",
    })

    # Content to pipeline mapping
    content_mapping: Dict[str, str] = field(default_factory=lambda: {
        "poster": "layout_poster",
        "banner": "layout_poster",
        "infographic": "layout_poster",
        "layout": "layout_poster",
    })

    # Default pipeline when dual is triggered but no specific match
    default_dual_pipeline: str = "creative_art"

    # Cost estimation
    default_cost_per_image: float = 0.04
    default_latency_ms: int = 15000
    dual_pipeline_latency_multiplier: float = 2.5


# =============================================================================
# EXCEPTIONS
# =============================================================================


class ModelSelectionError(Exception):
    """Base exception for model selection errors."""
    pass


class NoCompatibleModelError(ModelSelectionError):
    """No compatible model found for requirements."""
    pass


# =============================================================================
# SERVICE
# =============================================================================


class ModelSelectionService:
    """
    Service for selecting optimal image generation model and pipeline.

    This service handles:
    - Selecting best model based on requirements and mode
    - Deciding between single and dual pipeline
    - Building pipeline configurations
    - Estimating costs and latency
    """

    # Default path to image models config
    DEFAULT_MODELS_CONFIG = Path(__file__).parent.parent.parent / "config" / "image_models_config.yaml"

    def __init__(
        self,
        model_info_service: Optional[Any] = None,
        config: Optional[ModelSelectionConfig] = None,
        config_path: Optional[Path] = None,
        models_config_path: Optional[Path] = None,
    ):
        """
        Initialize model selection service.

        Args:
            model_info_service: Optional ModelInfoService for model data
            config: Optional custom configuration
            config_path: Optional path to config file
            models_config_path: Optional path to image_models_config.yaml
        """
        self._model_info_service = model_info_service
        self._config = config or ModelSelectionConfig()

        if config_path:
            self._load_config(config_path)

        # Auto-load models from image_models_config.yaml
        models_path = models_config_path or self.DEFAULT_MODELS_CONFIG
        self._load_models_config(models_path)

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            if data.get("dual_pipeline_triggers"):
                self._config.dual_pipeline_triggers = data["dual_pipeline_triggers"]
            if data.get("dual_pipelines"):
                self._config.dual_pipelines = data["dual_pipelines"]
            if data.get("pipeline_selection", {}).get("style_mapping"):
                self._config.style_mapping = data["pipeline_selection"]["style_mapping"]
            if data.get("pipeline_selection", {}).get("content_mapping"):
                self._config.content_mapping = data["pipeline_selection"]["content_mapping"]
            if data.get("pipeline_selection", {}).get("default"):
                self._config.default_dual_pipeline = data["pipeline_selection"]["default"]
            if data.get("cost_estimation"):
                ce = data["cost_estimation"]
                if "default_per_image" in ce:
                    self._config.default_cost_per_image = ce["default_per_image"]
                if "default_latency_ms" in ce:
                    self._config.default_latency_ms = ce["default_latency_ms"]
                if "dual_pipeline_latency_multiplier" in ce:
                    self._config.dual_pipeline_latency_multiplier = ce["dual_pipeline_latency_multiplier"]

        except Exception as e:
            logger.warning(f"Failed to load pipeline config: {e}")

    def _load_models_config(self, config_path: Path) -> None:
        """Load models from image_models_config.yaml."""
        try:
            if not config_path.exists():
                logger.warning(
                    f"model_selection_service.models_config.not_found: {config_path}"
                )
                return

            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            # Load default model from scenario_selection (first model in art_no_reference)
            scenario_selection = data.get("scenario_selection", {})
            art_no_ref = scenario_selection.get("art_no_reference", {})
            priority_models = art_no_ref.get("priority_models", {})
            if priority_models and not self._config.default_model:
                # Get first priority model as default
                first_priority = min(priority_models.keys())
                self._config.default_model = priority_models[first_priority]

            # Load dual pipeline models
            dual_pipeline = data.get("dual_pipeline", {})
            for pipeline_name, pipeline_config in dual_pipeline.items():
                if pipeline_name in self._config.dual_pipelines:
                    stage_1 = pipeline_config.get("stage_1", {})
                    stage_2 = pipeline_config.get("stage_2", {})

                    self._config.dual_pipelines[pipeline_name]["stage_1_model"] = stage_1.get("model")
                    self._config.dual_pipelines[pipeline_name]["stage_1_purpose"] = stage_1.get("purpose")
                    self._config.dual_pipelines[pipeline_name]["stage_2_model"] = stage_2.get("model")
                    self._config.dual_pipelines[pipeline_name]["stage_2_purpose"] = stage_2.get("purpose")

            logger.info(
                f"model_selection_service.models_config.loaded: default={self._config.default_model}, "
                f"pipelines={list(dual_pipeline.keys())}"
            )

        except Exception as e:
            logger.warning(f"Failed to load models config: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    async def select_model(
        self,
        mode: str,
        requirements: Dict[str, Any],
        model_info_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, str, List[str], Dict[str, Any]]:
        """
        Select optimal model based on requirements.

        Args:
            mode: Generation mode (RELAX, STANDARD, COMPLEX)
            requirements: User requirements
            model_info_context: Available models and compatibility info

        Returns:
            Tuple of (model_id, rationale, alternatives, model_specs)
        """
        if not model_info_context:
            model_info_context = {}

        # Get requirements
        needs_speed = requirements.get("priority") == "fast"
        needs_quality = requirements.get("quality_level") == "premium"

        # Score models
        compatible_models = []
        available_models = model_info_context.get("available_models", [])
        compatibility_results = model_info_context.get("compatibility_results", {})

        for model_data in available_models:
            model_id = model_data.get("model_id")
            compat = compatibility_results.get(model_id, {})

            # Skip incompatible models
            if not compat.get("compatible", True):
                continue

            score = compat.get("score", 0.5)

            # Adjust score based on requirements
            if needs_speed and "fast" in str(model_data.get("capabilities", [])):
                score += 0.1
            if needs_quality:
                score += model_data.get("quality_score", 0.5) * 0.2
            if mode.upper() == "COMPLEX":
                # Prefer higher quality for complex modes
                score += model_data.get("quality_score", 0.5) * 0.1

            compatible_models.append((model_id, score, model_data))

        # Sort by score
        compatible_models.sort(key=lambda x: x[1], reverse=True)

        if not compatible_models:
            # Fallback to default
            return (
                self._config.default_model,
                "Default model (no compatible models found)",
                [],
                {},
            )

        best_model_id, best_score, best_model_data = compatible_models[0]
        alternatives = [m[0] for m in compatible_models[1:4]]

        # Generate rationale
        rationale = self._build_rationale(
            best_model_data, mode, needs_speed, needs_quality
        )

        # Extract model specs for provider settings
        model_specs = {
            "air_id": best_model_data.get("air_id"),
            "provider": best_model_data.get("provider"),
            "specs": best_model_data.get("specs", {}),
            "provider_params": best_model_data.get("provider_params", []),
            "workflows": best_model_data.get("workflows", []),
            "cost": best_model_data.get("cost", {}),
        }

        return best_model_id, rationale, alternatives, model_specs

    def select_pipeline(
        self,
        requirements: Dict[str, Any],
        prompt: str = "",
    ) -> PipelineConfig:
        """
        Decide single vs dual pipeline based on requirements.

        Decision factors:
        - Text accuracy needed → Dual
        - Character refinement → Dual
        - Multi-element composition → Dual
        - Simple request → Single

        Args:
            requirements: User requirements
            prompt: Generated prompt text (for additional analysis)

        Returns:
            PipelineConfig with pipeline type and model info
        """
        # Combine all text content for analysis
        text_content = " ".join([
            str(requirements.get("subject", "")),
            str(requirements.get("prompt", "")),
            str(requirements.get("description", "")),
            str(requirements.get("style", "")),
            prompt,
        ]).lower()

        # Check triggers for dual pipeline
        triggers_found = []

        for trigger_type, keywords in self._config.dual_pipeline_triggers.items():
            for keyword in keywords:
                if keyword in text_content:
                    triggers_found.append((trigger_type, keyword))
                    break

        if not triggers_found:
            # No triggers - use single pipeline
            return PipelineConfig(
                pipeline_type="single",
                pipeline_name=None,
                stage_1_model="",  # Will be filled by select_model
                stage_1_purpose="Generate final image",
                decision_rationale="Single pipeline: no dual triggers detected",
            )

        # Determine which dual pipeline to use
        pipeline_intent = self._determine_pipeline_intent(
            triggers_found, requirements, text_content
        )

        # Build pipeline config
        pipeline_def = self._config.dual_pipelines.get(
            pipeline_intent,
            self._config.dual_pipelines[self._config.default_dual_pipeline]
        )

        rationale = f"Dual pipeline selected: {', '.join([f'{t[0]}:{t[1]}' for t in triggers_found])}"
        logger.info(f"Pipeline decision: DUAL ({pipeline_intent}) - {rationale}")

        return PipelineConfig(
            pipeline_type="dual",
            pipeline_name=pipeline_intent,
            stage_1_model=pipeline_def["stage_1_model"],
            stage_1_purpose=pipeline_def["stage_1_purpose"],
            stage_2_model=pipeline_def.get("stage_2_model"),
            stage_2_purpose=pipeline_def.get("stage_2_purpose"),
            decision_rationale=rationale,
        )

    def estimate_cost(
        self,
        pipeline: PipelineConfig,
        model_specs: Optional[Dict[str, Any]] = None,
        num_images: int = 1,
    ) -> Tuple[float, int]:
        """
        Estimate cost and latency for generation.

        Args:
            pipeline: Pipeline configuration
            model_specs: Model specifications with cost info
            num_images: Number of images to generate

        Returns:
            Tuple of (estimated_cost, estimated_latency_ms)
        """
        # Get base cost
        if model_specs and model_specs.get("cost"):
            base_cost = model_specs["cost"].get(
                "per_image",
                self._config.default_cost_per_image
            )
        else:
            base_cost = self._config.default_cost_per_image

        # Calculate total cost
        total_cost = base_cost * num_images
        if pipeline.pipeline_type == "dual":
            # Dual pipeline costs roughly double (two generations)
            total_cost *= 2

        # Calculate latency
        base_latency = self._config.default_latency_ms
        if pipeline.pipeline_type == "dual":
            base_latency = int(
                base_latency * self._config.dual_pipeline_latency_multiplier
            )

        return total_cost, base_latency

    def get_available_pipelines(self) -> Dict[str, Dict[str, str]]:
        """
        Get all available dual pipeline configurations.

        Returns:
            Dict of pipeline name to configuration
        """
        return self._config.dual_pipelines.copy()

    def get_pipeline_triggers(self) -> Dict[str, List[str]]:
        """
        Get all dual pipeline triggers.

        Returns:
            Dict of trigger type to keywords
        """
        return self._config.dual_pipeline_triggers.copy()

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _build_rationale(
        self,
        model_data: Dict[str, Any],
        mode: str,
        needs_speed: bool,
        needs_quality: bool,
    ) -> str:
        """Build selection rationale string."""
        rationale_parts = [
            f"Selected {model_data.get('display_name', model_data.get('model_id', 'model'))}"
        ]

        if needs_quality:
            rationale_parts.append("for premium quality")
        if needs_speed:
            rationale_parts.append("for fast generation")
        if mode.upper() == "COMPLEX":
            rationale_parts.append("suitable for complex prompts")

        return " ".join(rationale_parts)

    def _determine_pipeline_intent(
        self,
        triggers_found: List[Tuple[str, str]],
        requirements: Dict[str, Any],
        text_content: str,
    ) -> str:
        """Determine which dual pipeline to use based on triggers and context."""
        has_text_need = any(t[0] == "text_in_image" for t in triggers_found)
        has_production_need = any(t[0] == "production_quality" for t in triggers_found)
        has_character_need = any(t[0] == "character_refinement" for t in triggers_found)
        has_multi_element = any(t[0] == "multi_element" for t in triggers_found)

        # Check style
        style = requirements.get("style", "").lower()
        is_photorealistic = any(
            w in style for w in self._config.style_mapping.keys()
        )

        # Check content type
        is_layout = any(
            w in text_content for w in self._config.content_mapping.keys()
        )

        # Determine intent
        if has_text_need or has_production_need:
            if is_layout:
                return "layout_poster"
            elif is_photorealistic:
                return "photorealistic"
            else:
                return "creative_art"
        elif has_character_need or has_multi_element:
            if is_photorealistic:
                return "photorealistic"
            else:
                return "creative_art"

        return self._config.default_dual_pipeline

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    async def close(self) -> None:
        """Close service and release resources."""
        pass  # No resources to release

    async def __aenter__(self) -> "ModelSelectionService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
