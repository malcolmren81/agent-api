"""
Genflow Service.

Handles determination of generation flow (genflow) method.
Consolidates pipeline selection logic to determine single vs dual pipeline.

This service is used by the GenPlan Agent to determine the optimal
generation strategy based on requirements and complexity.
"""

import structlog
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from palet8_agents.models.genplan import GenflowConfig

logger = structlog.get_logger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class GenflowServiceConfig:
    """Configuration for Genflow service."""

    # Dual pipeline triggers - keywords that activate dual pipeline
    dual_pipeline_triggers: Dict[str, List[str]] = field(default_factory=lambda: {
        "text_in_image": [
            "text", "typography", "lettering", "font", "words",
            "title", "headline", "quote", "caption", "label",
            "text overlay", "text content",
        ],
        "character_refinement": [
            "character edit", "face fix", "expression change",
            "pose adjust", "facial expression", "character pose",
            "character refinement",
        ],
        "multi_element": [
            "multiple subjects", "complex composition", "layered design",
            "multiple characters", "group scene", "multi-element",
        ],
        "production_quality": [
            "print-ready", "production", "high accuracy",
            "4k", "poster", "billboard", "large format",
            "professional quality",
        ],
    })

    # Genflow definitions
    genflows: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        # Single pipeline flows
        "single_standard": {
            "type": "single",
            "name": "Standard Single Generation",
            "description": "Single model output for quick results",
            "use_cases": ["RELAX mode", "simple compositions", "quick iterations"],
        },
        "single_quality": {
            "type": "single",
            "name": "Quality Single Generation",
            "description": "Single model with quality priority",
            "use_cases": ["STANDARD mode with quality focus", "production-ready single pass"],
        },
        # Dual pipeline flows - loaded from image_models_config.yaml
        # These are defaults that will be overwritten by config file
        "creative_art": {
            "type": "dual",
            "name": "High-Creative Art Pipeline",
            "description": "Painterly concept art with character refinement and text correction",
            "use_cases": ["Concept art", "stylized artwork with text", "creative compositions"],
            # Models loaded from config/image_models_config.yaml
            "stage_1_model": None,
            "stage_1_purpose": "Generate creative, non-realistic composition",
            "stage_2_model": None,
            "stage_2_purpose": "Refine characters, add/correct text, adjust elements",
        },
        "photorealistic": {
            "type": "dual",
            "name": "Photorealistic Pipeline",
            "description": "High-quality photo-real base with character and style editing",
            "use_cases": ["Product photography", "realistic scenes", "images needing text overlays"],
            # Models loaded from config/image_models_config.yaml
            "stage_1_model": None,
            "stage_1_purpose": "Generate photorealistic base with exceptional detail",
            "stage_2_model": None,
            "stage_2_purpose": "Character edits, text overlays, stylistic adjustments",
        },
        "layout_poster": {
            "type": "dual",
            "name": "Layout & Poster Design Pipeline",
            "description": "Complex layouts with precise typography, then localized edits",
            "use_cases": ["Marketing posters", "infographics", "bilingual text designs"],
            # Models loaded from config/image_models_config.yaml
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
        "photography": "photorealistic",
    })

    # Content to pipeline mapping
    content_mapping: Dict[str, str] = field(default_factory=lambda: {
        "poster": "layout_poster",
        "banner": "layout_poster",
        "infographic": "layout_poster",
        "layout": "layout_poster",
        "flyer": "layout_poster",
        "brochure": "layout_poster",
    })

    # Default single flow by complexity
    complexity_to_single_flow: Dict[str, str] = field(default_factory=lambda: {
        "simple": "single_standard",
        "standard": "single_quality",
        "complex": "single_quality",  # Complex may still use single if no triggers
    })

    # Default dual pipeline when triggered but no specific match
    default_dual_pipeline: str = "creative_art"


# =============================================================================
# SERVICE
# =============================================================================


class GenflowService:
    """
    Service for determining generation flow (genflow) method.

    Consolidates pipeline selection logic to determine:
    - Single vs dual pipeline
    - Which specific pipeline configuration to use
    - Rationale for the selection

    Used by GenPlan Agent in the DETERMINE_GENFLOW action.
    """

    # Default path to image models config
    DEFAULT_MODELS_CONFIG = Path(__file__).parent.parent.parent / "config" / "image_models_config.yaml"

    def __init__(
        self,
        config: Optional[GenflowServiceConfig] = None,
        config_path: Optional[Path] = None,
        models_config_path: Optional[Path] = None,
    ):
        """
        Initialize Genflow service.

        Args:
            config: Optional custom configuration
            config_path: Optional path to config file (genflow.yaml or pipeline.yaml)
            models_config_path: Optional path to image_models_config.yaml
        """
        self._config = config or GenflowServiceConfig()

        # Load genflow/pipeline config if provided
        if config_path:
            self._load_config(config_path)

        # Auto-load dual pipeline models from image_models_config.yaml
        models_path = models_config_path or self.DEFAULT_MODELS_CONFIG
        self._load_models_config(models_path)

    def _load_config(self, config_path: Path) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            # Load triggers
            if data.get("dual_pipeline_triggers"):
                self._config.dual_pipeline_triggers = data["dual_pipeline_triggers"]

            # Load genflows (may be under 'genflows' or 'dual_pipelines')
            if data.get("genflows"):
                self._config.genflows.update(data["genflows"])
            if data.get("dual_pipelines"):
                # Convert old format to new
                for name, pipeline_def in data["dual_pipelines"].items():
                    self._config.genflows[name] = {
                        "type": "dual",
                        "name": pipeline_def.get("name", name),
                        "description": pipeline_def.get("description", ""),
                        "stage_1_model": pipeline_def.get("stage_1_model"),
                        "stage_1_purpose": pipeline_def.get("stage_1_purpose"),
                        "stage_2_model": pipeline_def.get("stage_2_model"),
                        "stage_2_purpose": pipeline_def.get("stage_2_purpose"),
                    }

            # Load mappings
            if data.get("pipeline_selection", {}).get("style_mapping"):
                self._config.style_mapping = data["pipeline_selection"]["style_mapping"]
            if data.get("pipeline_selection", {}).get("content_mapping"):
                self._config.content_mapping = data["pipeline_selection"]["content_mapping"]
            if data.get("pipeline_selection", {}).get("default"):
                self._config.default_dual_pipeline = data["pipeline_selection"]["default"]

            logger.info(
                "genflow_service.config.loaded",
                config_path=str(config_path),
            )

        except Exception as e:
            logger.warning(
                "genflow_service.config.load_failed",
                error_detail=str(e),
                error_type=type(e).__name__,
            )

    def _load_models_config(self, config_path: Path) -> None:
        """Load dual pipeline models from image_models_config.yaml."""
        try:
            if not config_path.exists():
                logger.warning(
                    "genflow_service.models_config.not_found",
                    config_path=str(config_path),
                )
                return

            with open(config_path, "r") as f:
                data = yaml.safe_load(f)

            dual_pipeline = data.get("dual_pipeline", {})
            if not dual_pipeline:
                logger.warning(
                    "genflow_service.models_config.no_dual_pipeline",
                    config_path=str(config_path),
                )
                return

            # Update genflow configs with models from image_models_config.yaml
            for pipeline_name, pipeline_config in dual_pipeline.items():
                if pipeline_name in self._config.genflows:
                    stage_1 = pipeline_config.get("stage_1", {})
                    stage_2 = pipeline_config.get("stage_2", {})

                    self._config.genflows[pipeline_name]["stage_1_model"] = stage_1.get("model")
                    self._config.genflows[pipeline_name]["stage_1_purpose"] = stage_1.get("purpose")
                    self._config.genflows[pipeline_name]["stage_2_model"] = stage_2.get("model")
                    self._config.genflows[pipeline_name]["stage_2_purpose"] = stage_2.get("purpose")

                    logger.debug(
                        "genflow_service.models_config.pipeline_loaded",
                        pipeline_name=pipeline_name,
                        stage_1_model=stage_1.get("model"),
                        stage_2_model=stage_2.get("model"),
                    )

            logger.info(
                "genflow_service.models_config.loaded",
                cfg_path=str(config_path),
                pipelines_loaded=list(dual_pipeline.keys()),
            )

        except Exception as e:
            logger.warning(
                "genflow_service.models_config.load_failed",
                error_detail=str(e),
                exception_type=type(e).__name__,
            )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def determine_genflow(
        self,
        requirements: Dict[str, Any],
        complexity: str,
        prompt: str = "",
        user_info: Optional[Dict[str, Any]] = None,
    ) -> GenflowConfig:
        """
        Determine the generation flow based on requirements and complexity.

        This is the main entry point for genflow determination.
        Called by GenPlan Agent in the DETERMINE_GENFLOW action.

        Args:
            requirements: User requirements dictionary
            complexity: Complexity level ("simple", "standard", "complex")
            prompt: Generated prompt text (for additional trigger detection)
            user_info: Parsed user info (from UserParseResult.to_dict())

        Returns:
            GenflowConfig with flow_type, flow_name, and rationale
        """
        # Build text content for trigger analysis
        text_content = self._build_text_content(requirements, prompt, user_info)

        # Check for dual pipeline triggers
        triggers_found = self._detect_triggers(text_content)

        # Determine if we should use dual pipeline
        should_use_dual = self._should_use_dual(triggers_found, complexity, user_info)

        if not should_use_dual:
            # Use single pipeline
            flow_name = self._config.complexity_to_single_flow.get(
                complexity, "single_standard"
            )
            flow_def = self._config.genflows.get(flow_name, {})

            return GenflowConfig(
                flow_type="single",
                flow_name=flow_name,
                description=flow_def.get("description", "Single model generation"),
                rationale=f"Single pipeline: complexity={complexity}, no dual triggers detected",
                triggered_by=None,
            )

        # Determine which dual pipeline to use
        pipeline_name = self._select_dual_pipeline(
            triggers_found, requirements, text_content, user_info
        )

        flow_def = self._config.genflows.get(
            pipeline_name,
            self._config.genflows.get(self._config.default_dual_pipeline, {})
        )

        # Build rationale
        trigger_summary = ", ".join([f"{t[0]}:{t[1]}" for t in triggers_found[:3]])
        rationale = f"Dual pipeline ({pipeline_name}): triggers=[{trigger_summary}]"

        logger.info(
            "genflow_service.decision.dual",
            pipeline_name=pipeline_name,
            triggers=[t[0] for t in triggers_found[:3]],
            rationale=rationale,
        )

        return GenflowConfig(
            flow_type="dual",
            flow_name=pipeline_name,
            description=flow_def.get("description", ""),
            rationale=rationale,
            triggered_by=triggers_found[0][0] if triggers_found else None,
        )

    def get_available_flows(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all available genflow configurations.

        Returns:
            Dict of flow name to configuration
        """
        return self._config.genflows.copy()

    def get_flow_triggers(self) -> Dict[str, List[str]]:
        """
        Get trigger keywords for dual pipeline activation.

        Returns:
            Dict of trigger type to keywords
        """
        return self._config.dual_pipeline_triggers.copy()

    def get_flow_by_name(self, flow_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific flow configuration by name.

        Args:
            flow_name: Name of the flow (e.g., "creative_art", "single_standard")

        Returns:
            Flow configuration dict or None if not found
        """
        return self._config.genflows.get(flow_name)

    def get_dual_pipeline_config(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """
        Get dual pipeline configuration including stage models.

        Args:
            pipeline_name: Name of the dual pipeline

        Returns:
            Pipeline config with stage_1/stage_2 model info, or None
        """
        flow = self._config.genflows.get(pipeline_name)
        if flow and flow.get("type") == "dual":
            return {
                "pipeline_name": pipeline_name,
                "stage_1_model": flow.get("stage_1_model"),
                "stage_1_purpose": flow.get("stage_1_purpose"),
                "stage_2_model": flow.get("stage_2_model"),
                "stage_2_purpose": flow.get("stage_2_purpose"),
            }
        return None

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _build_text_content(
        self,
        requirements: Dict[str, Any],
        prompt: str,
        user_info: Optional[Dict[str, Any]],
    ) -> str:
        """Build combined text content for trigger analysis."""
        parts = [
            str(requirements.get("subject", "")),
            str(requirements.get("prompt", "")),
            str(requirements.get("description", "")),
            str(requirements.get("style", "")),
            str(requirements.get("product_type", "")),
            prompt,
        ]

        if user_info:
            parts.extend([
                str(user_info.get("subject", "")),
                str(user_info.get("style", "")),
                str(user_info.get("text_content", "")),
                str(user_info.get("product_type", "")),
                " ".join(user_info.get("extracted_intents", [])),
            ])

        return " ".join(parts).lower()

    def _detect_triggers(
        self,
        text_content: str,
    ) -> List[Tuple[str, str]]:
        """
        Detect dual pipeline triggers in text content.

        Returns:
            List of (trigger_type, matched_keyword) tuples
        """
        triggers_found = []

        for trigger_type, keywords in self._config.dual_pipeline_triggers.items():
            for keyword in keywords:
                if keyword in text_content:
                    triggers_found.append((trigger_type, keyword))
                    break  # Only one match per trigger type

        return triggers_found

    def _should_use_dual(
        self,
        triggers_found: List[Tuple[str, str]],
        complexity: str,
        user_info: Optional[Dict[str, Any]],
    ) -> bool:
        """
        Determine if dual pipeline should be used.

        Args:
            triggers_found: Detected triggers
            complexity: Complexity level
            user_info: Parsed user info

        Returns:
            True if dual pipeline should be used
        """
        # Check for explicit triggers
        if triggers_found:
            return True

        # Check for text content (strong indicator for dual)
        if user_info and user_info.get("text_content"):
            return True

        # Complex mode may benefit from dual pipeline
        # But only if there's some indication it's needed
        if complexity == "complex":
            # Check for production/quality indicators
            if user_info:
                intents = user_info.get("extracted_intents", [])
                if any(i in intents for i in ["poster", "production", "print", "professional"]):
                    return True

        return False

    def _select_dual_pipeline(
        self,
        triggers_found: List[Tuple[str, str]],
        requirements: Dict[str, Any],
        text_content: str,
        user_info: Optional[Dict[str, Any]],
    ) -> str:
        """
        Select which dual pipeline to use.

        Returns:
            Pipeline name (e.g., "creative_art", "photorealistic", "layout_poster")
        """
        # Analyze triggers
        has_text_need = any(t[0] == "text_in_image" for t in triggers_found)
        has_production_need = any(t[0] == "production_quality" for t in triggers_found)
        has_character_need = any(t[0] == "character_refinement" for t in triggers_found)
        has_multi_element = any(t[0] == "multi_element" for t in triggers_found)

        # Check style for photorealistic
        style = requirements.get("style", "").lower()
        if user_info:
            style = style or user_info.get("style", "").lower()

        is_photorealistic = any(
            w in style for w in self._config.style_mapping.keys()
        )

        # Check content type for layout
        is_layout = any(
            w in text_content for w in self._config.content_mapping.keys()
        )

        # Also check product_type
        product_type = requirements.get("product_type", "").lower()
        if user_info:
            product_type = product_type or user_info.get("product_type", "").lower()

        if product_type in self._config.content_mapping:
            is_layout = True

        # Decision logic
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

        # Check style/content mappings directly
        for style_key, pipeline in self._config.style_mapping.items():
            if style_key in style:
                return pipeline

        for content_key, pipeline in self._config.content_mapping.items():
            if content_key in text_content:
                return pipeline

        return self._config.default_dual_pipeline

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    async def close(self) -> None:
        """Close service and release resources."""
        pass  # No resources to release

    async def __aenter__(self) -> "GenflowService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
