"""
Model Info Service - Data Provider

Loads model information from image_models_config.yaml and provides
model selection and routing based on task requirements.

Architecture:
- Single source of truth: config/image_models_config.yaml
- This service is a DATA PROVIDER, not a decision maker
- Planner Agent makes the actual model selection decision

Documentation Reference: Section 2.1.6 (Swimlane: Model Info Service)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import yaml

from palet8_agents.core.config import get_config

logger = logging.getLogger(__name__)

# Config file path relative to project root
CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "image_models_config.yaml"


class ModelInfoError(Exception):
    """Base exception for ModelInfoService errors."""
    pass


class ModelCapability(Enum):
    """Model capability categories."""
    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    ANIME = "anime"
    PRODUCT = "product"
    CHARACTER = "character"
    ABSTRACT = "abstract"
    LOGO = "logo"
    TEXT_RENDERING = "text_rendering"
    FINE_DETAIL = "fine_detail"
    FAST = "fast"
    HIGH_QUALITY = "high_quality"
    REFERENCE_IMAGE = "reference_image"
    TEXT_TO_IMAGE = "text_to_image"
    IMAGE_TO_IMAGE = "image_to_image"


@dataclass
class ModelInfo:
    """Information about an image generation model."""
    model_id: str
    display_name: str
    provider: str
    air_id: str  # Runware AIR format
    capabilities: List[ModelCapability]
    workflows: List[str]  # text-to-image, image-to-image, etc.
    cost_per_image: float = 0.0
    cost_tier: str = "standard"  # economy, standard, premium
    average_latency_ms: int = 15000
    max_resolution: int = 1024
    reference_images: int = 0  # Max reference images supported
    prompt_length: str = ""  # e.g., "1-2000 characters"
    quality_score: float = 0.8  # 0.0 to 1.0
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    specs: Dict[str, Any] = field(default_factory=dict)  # Full specs from config
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "provider": self.provider,
            "air_id": self.air_id,
            "capabilities": [c.value for c in self.capabilities],
            "workflows": self.workflows,
            "cost_per_image": self.cost_per_image,
            "cost_tier": self.cost_tier,
            "average_latency_ms": self.average_latency_ms,
            "max_resolution": self.max_resolution,
            "reference_images": self.reference_images,
            "prompt_length": self.prompt_length,
            "quality_score": self.quality_score,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "specs": self.specs,
            "metadata": self.metadata,
        }

    def has_capability(self, capability: ModelCapability) -> bool:
        """Check if model has a specific capability."""
        return capability in self.capabilities

    @property
    def supports_reference_image(self) -> bool:
        """Check if model supports reference images."""
        return self.reference_images > 0

    @property
    def supports_negative_prompt(self) -> bool:
        """Check if model supports negative prompts."""
        return self.specs.get("negative_prompt") is not None


@dataclass
class ModelCapabilities:
    """Detailed capabilities of a model."""
    model_id: str
    strengths: List[str]
    weaknesses: List[str]
    best_for: List[str]
    avoid_for: List[str]
    sample_prompts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "best_for": self.best_for,
            "avoid_for": self.avoid_for,
            "sample_prompts": self.sample_prompts,
        }


@dataclass
class ModelSelection:
    """Result of model selection."""
    model_id: str
    rationale: str
    score: float = 0.0  # Selection confidence
    parameters: Dict[str, Any] = field(default_factory=dict)  # Recommended parameters
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "rationale": self.rationale,
            "score": self.score,
            "parameters": self.parameters,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
        }


class ModelInfoService:
    """
    Data Provider for model information.

    Loads from config/image_models_config.yaml (single source of truth).

    Features:
    - Model registry loaded from YAML config
    - Model capability lookup
    - Scenario-based model suggestions
    - Cost estimation
    - Provider specs for generation
    """

    # Capability weights for scoring
    CAPABILITY_WEIGHTS = {
        ModelCapability.PHOTOREALISTIC: 1.0,
        ModelCapability.ARTISTIC: 0.9,
        ModelCapability.PRODUCT: 1.0,
        ModelCapability.CHARACTER: 0.95,
        ModelCapability.FINE_DETAIL: 0.85,
        ModelCapability.HIGH_QUALITY: 1.0,
        ModelCapability.FAST: 0.7,
        ModelCapability.TEXT_RENDERING: 0.9,
        ModelCapability.REFERENCE_IMAGE: 0.8,
    }

    # Quality score mapping by cost tier
    TIER_QUALITY_SCORES = {
        "economy": 0.75,
        "standard": 0.85,
        "premium": 0.95,
    }

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize ModelInfoService by loading from YAML config."""
        self._config = get_config()
        self._config_path = config_path or CONFIG_PATH
        self._model_registry: Dict[str, ModelInfo] = {}
        self._scenario_selection: Dict[str, Dict] = {}
        self._dual_pipelines: Dict[str, Dict] = {}
        self._cost_tiers: Dict[str, Dict] = {}
        self._raw_config: Dict[str, Any] = {}

        self._load_config()

    def _load_config(self) -> None:
        """Load model configuration from YAML file."""
        try:
            if not self._config_path.exists():
                logger.warning(f"Config file not found: {self._config_path}")
                return

            with open(self._config_path, "r") as f:
                self._raw_config = yaml.safe_load(f)

            # Load model registry
            model_registry = self._raw_config.get("model_registry", {})
            for model_id, model_data in model_registry.items():
                self._model_registry[model_id] = self._parse_model_info(model_id, model_data)

            # Load scenario selection
            self._scenario_selection = self._raw_config.get("scenario_selection", {})

            # Load dual pipelines
            self._dual_pipelines = self._raw_config.get("dual_pipeline", {})

            # Load cost tiers
            self._cost_tiers = self._raw_config.get("cost_tiers", {})

            logger.info(f"Loaded {len(self._model_registry)} models from config")

        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            raise ModelInfoError(f"Config loading failed: {e}")

    def _parse_model_info(self, model_id: str, data: Dict[str, Any]) -> ModelInfo:
        """Parse model data from config into ModelInfo."""
        specs = data.get("specs", {})
        cost = data.get("cost", {})
        workflows = data.get("workflows", [])

        # Determine capabilities from data
        capabilities = self._infer_capabilities(data, workflows)

        # Extract cost per image
        cost_per_image = cost.get("per_image", 0.0)
        if not cost_per_image and cost.get("per_mp"):
            # Estimate for 1MP (1024x1024)
            cost_per_image = cost["per_mp"]

        # Extract max resolution from dimensions
        max_resolution = self._extract_max_resolution(specs.get("dimensions", []))

        return ModelInfo(
            model_id=model_id,
            display_name=data.get("display_name", model_id),
            provider=data.get("provider", "unknown"),
            air_id=data.get("air_id", ""),
            capabilities=capabilities,
            workflows=workflows,
            cost_per_image=cost_per_image,
            cost_tier=cost.get("cost_tier", "standard"),
            max_resolution=max_resolution,
            reference_images=specs.get("reference_images", 0),
            prompt_length=specs.get("prompt_length", ""),
            quality_score=self.TIER_QUALITY_SCORES.get(cost.get("cost_tier", "standard"), 0.85),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            specs=specs,
            metadata={
                "description": data.get("description", ""),
                "alias": data.get("alias"),
                "role": data.get("role"),
            },
        )

    def _infer_capabilities(self, data: Dict[str, Any], workflows: List[str]) -> List[ModelCapability]:
        """Infer capabilities from model data."""
        capabilities = []
        description = data.get("description", "").lower()
        strengths = [s.lower() for s in data.get("strengths", [])]
        specs = data.get("specs", {})

        # Workflow-based capabilities
        if "text-to-image" in workflows:
            capabilities.append(ModelCapability.TEXT_TO_IMAGE)
        if "image-to-image" in workflows or "reference-to-image" in workflows:
            capabilities.append(ModelCapability.IMAGE_TO_IMAGE)
            capabilities.append(ModelCapability.REFERENCE_IMAGE)

        # Reference image support
        if specs.get("reference_images", 0) > 0:
            capabilities.append(ModelCapability.REFERENCE_IMAGE)

        # Strength-based inference
        for strength in strengths:
            if "text" in strength or "typography" in strength:
                capabilities.append(ModelCapability.TEXT_RENDERING)
            if "photo" in strength or "realistic" in strength:
                capabilities.append(ModelCapability.PHOTOREALISTIC)
            if "artistic" in strength or "creative" in strength:
                capabilities.append(ModelCapability.ARTISTIC)
            if "detail" in strength:
                capabilities.append(ModelCapability.FINE_DETAIL)
            if "fast" in strength or "quick" in strength:
                capabilities.append(ModelCapability.FAST)
            if "quality" in strength or "production" in strength:
                capabilities.append(ModelCapability.HIGH_QUALITY)
            if "anime" in strength:
                capabilities.append(ModelCapability.ANIME)
            if "product" in strength:
                capabilities.append(ModelCapability.PRODUCT)

        # Description-based inference
        if "realism" in description or "photorealistic" in description:
            capabilities.append(ModelCapability.PHOTOREALISTIC)
        if "artistic" in description or "creative" in description:
            capabilities.append(ModelCapability.ARTISTIC)
        if "text" in description or "typography" in description:
            capabilities.append(ModelCapability.TEXT_RENDERING)

        return list(set(capabilities))  # Deduplicate

    def _extract_max_resolution(self, dimensions: Any) -> int:
        """Extract maximum resolution from dimensions spec."""
        if not dimensions:
            return 1024

        # Handle dict format (min/max)
        if isinstance(dimensions, dict):
            return dimensions.get("max", 1024)

        # Handle list format
        if isinstance(dimensions, list):
            max_res = 1024
            for dim in dimensions:
                if isinstance(dim, str):
                    # Parse "1024x1024 (1:1)" format
                    parts = dim.split()[0].split("x")
                    if len(parts) == 2:
                        try:
                            max_res = max(max_res, int(parts[0]), int(parts[1]))
                        except ValueError:
                            pass
            return max_res

        return 1024

    def get_available_models(self) -> List[ModelInfo]:
        """Get list of all available models."""
        return list(self._model_registry.values())

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        return self._model_registry.get(model_id)

    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Alias for get_model (backward compatibility)."""
        return self.get_model(model_id)

    def get_model_specs(self, model_id: str) -> Dict[str, Any]:
        """
        Get full specs for a model (for provider settings).

        Returns the raw specs dict from config for building provider settings.
        """
        model = self._model_registry.get(model_id)
        if not model:
            return {}

        return {
            "air_id": model.air_id,
            "provider": model.provider,
            "specs": model.specs,
            "provider_params": model.specs.get("provider_params", {}),
            "workflows": model.workflows,
            "cost": {
                "per_image": model.cost_per_image,
                "cost_tier": model.cost_tier,
            },
        }

    def get_models_for_scenario(self, scenario: str) -> List[str]:
        """
        Get priority-ordered model list for a scenario.

        Scenarios: art_no_reference, art_with_reference, photo_no_reference, photo_with_reference
        """
        scenario_config = self._scenario_selection.get(scenario, {})
        priority_models = scenario_config.get("priority_models", {})

        # Return sorted by priority key
        return [priority_models[k] for k in sorted(priority_models.keys())]

    def get_dual_pipeline(self, pipeline_name: str) -> Optional[Dict[str, Any]]:
        """Get dual pipeline configuration."""
        return self._dual_pipelines.get(pipeline_name)

    def get_models_by_cost_tier(self, tier: str) -> List[str]:
        """Get models in a specific cost tier (economy, standard, premium)."""
        tier_config = self._cost_tiers.get(tier, {})
        return tier_config.get("models", [])

    def get_selection_context(
        self,
        product_type: Optional[str] = None,
        requirements: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get context for model selection (used by Planner Agent).

        Returns available models with compatibility info for requirements.
        """
        requirements = requirements or {}
        needs_reference = requirements.get("reference_image") is not None
        max_cost = requirements.get("max_cost_per_image")

        available_models = []
        compatibility_results = {}

        for model_id, model in self._model_registry.items():
            model_data = model.to_dict()

            # Check compatibility
            compatible = True
            score = model.quality_score
            reasons = []

            # Reference image requirement
            if needs_reference and not model.supports_reference_image:
                compatible = False
                reasons.append("No reference image support")

            # Cost constraint
            if max_cost and model.cost_per_image > max_cost:
                compatible = False
                reasons.append(f"Exceeds budget (${model.cost_per_image} > ${max_cost})")

            compatibility_results[model_id] = {
                "compatible": compatible,
                "score": score if compatible else 0,
                "reasons": reasons,
            }

            available_models.append(model_data)

        return {
            "available_models": available_models,
            "model_count": len(available_models),
            "compatibility_results": compatibility_results,
            "scenario_selection": self._scenario_selection,
            "cost_tiers": self._cost_tiers,
        }

    def estimate_cost(
        self,
        model_id: str,
        num_images: int = 1,
        steps: int = 30,
        width: int = 1024,
        height: int = 1024,
    ) -> Optional[float]:
        """Estimate generation cost for a model."""
        model = self._model_registry.get(model_id)
        if not model:
            return None

        # Base cost
        cost = model.cost_per_image * num_images

        # Some models charge per megapixel
        raw_cost = self._raw_config.get("model_registry", {}).get(model_id, {}).get("cost", {})
        if raw_cost.get("per_mp"):
            mp = (width * height) / 1_000_000
            cost = raw_cost["per_mp"] * mp * num_images

        return cost

    def get_model_capabilities(self, model_id: str) -> Optional[ModelCapabilities]:
        """Get detailed capabilities of a model."""
        model = self._model_registry.get(model_id)
        if not model:
            return None

        # Use strengths/weaknesses from config
        best_for = []
        avoid_for = []

        for cap in model.capabilities:
            if cap == ModelCapability.PHOTOREALISTIC:
                best_for.append("Product photography, realistic scenes")
            elif cap == ModelCapability.ARTISTIC:
                best_for.append("Artistic interpretations, stylized images")
            elif cap == ModelCapability.TEXT_RENDERING:
                best_for.append("Images with text, logos, posters")
            elif cap == ModelCapability.REFERENCE_IMAGE:
                best_for.append("Style transfer, character consistency")

        if not model.supports_reference_image:
            avoid_for.append("Style transfer from reference images")

        return ModelCapabilities(
            model_id=model_id,
            strengths=model.strengths,
            weaknesses=model.weaknesses,
            best_for=best_for,
            avoid_for=avoid_for,
        )

    def _calculate_model_score(
        self,
        model: ModelInfo,
        prompt: str,
        product_type: str,
        quality_level: str,
        needs_reference: bool,
        needs_speed: bool,
        max_cost: Optional[float],
    ) -> float:
        """Calculate selection score for a model."""
        score = model.quality_score

        # Capability matching
        capability_score = 0.0
        relevant_capabilities = self._get_relevant_capabilities(product_type, prompt)

        for cap in relevant_capabilities:
            if model.has_capability(cap):
                capability_score += self.CAPABILITY_WEIGHTS.get(cap, 0.5)

        if relevant_capabilities:
            capability_score /= len(relevant_capabilities)
            score = (score + capability_score) / 2

        # Quality level adjustment
        if quality_level == "premium":
            if ModelCapability.HIGH_QUALITY in model.capabilities:
                score += 0.1
            else:
                score -= 0.1
        elif quality_level == "draft":
            if ModelCapability.FAST in model.capabilities:
                score += 0.1

        # Reference image requirement
        if needs_reference and not model.supports_reference_image:
            score -= 0.3

        # Speed requirement
        if needs_speed:
            if ModelCapability.FAST in model.capabilities:
                score += 0.15
            elif model.average_latency_ms > 15000:
                score -= 0.1

        # Cost constraint
        if max_cost and model.cost_per_image > max_cost:
            score -= 0.5  # Heavy penalty for exceeding budget

        return min(max(score, 0.0), 1.0)  # Clamp to 0-1

    def _get_relevant_capabilities(
        self,
        product_type: str,
        prompt: str,
    ) -> List[ModelCapability]:
        """Determine relevant capabilities for the task."""
        capabilities = []
        prompt_lower = prompt.lower()

        # Product type mapping
        if product_type in ["apparel", "drinkware", "accessories"]:
            capabilities.append(ModelCapability.PRODUCT)
            capabilities.append(ModelCapability.PHOTOREALISTIC)
        elif product_type == "wall_art":
            capabilities.append(ModelCapability.ARTISTIC)

        # Prompt analysis
        if any(word in prompt_lower for word in ["realistic", "photo", "photograph"]):
            capabilities.append(ModelCapability.PHOTOREALISTIC)
        if any(word in prompt_lower for word in ["artistic", "painting", "illustration"]):
            capabilities.append(ModelCapability.ARTISTIC)
        if any(word in prompt_lower for word in ["anime", "manga", "cartoon"]):
            capabilities.append(ModelCapability.ANIME)
        if any(word in prompt_lower for word in ["character", "person", "portrait"]):
            capabilities.append(ModelCapability.CHARACTER)
        if any(word in prompt_lower for word in ["detailed", "intricate", "fine"]):
            capabilities.append(ModelCapability.FINE_DETAIL)
        if any(word in prompt_lower for word in ["text", "words", "letters", "typography"]):
            capabilities.append(ModelCapability.TEXT_RENDERING)

        return list(set(capabilities))  # Deduplicate

    def _generate_rationale(
        self,
        model: ModelInfo,
        prompt: str,
        requirements: Dict[str, Any],
    ) -> str:
        """Generate human-readable rationale for model selection."""
        reasons = []

        if ModelCapability.PRODUCT in model.capabilities:
            reasons.append("optimized for product images")
        if ModelCapability.PHOTOREALISTIC in model.capabilities:
            reasons.append("excellent photorealistic quality")
        if ModelCapability.HIGH_QUALITY in model.capabilities:
            reasons.append("premium output quality")
        if ModelCapability.FAST in model.capabilities:
            reasons.append("fast generation time")
        if model.supports_reference_image and requirements.get("reference_image"):
            reasons.append("supports reference image styling")

        if not reasons:
            reasons.append("general-purpose model suitable for the task")

        return f"Selected {model.display_name}: " + ", ".join(reasons)

    def _get_recommended_parameters(
        self,
        model: ModelInfo,
        requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get recommended generation parameters for the model."""
        params = {
            "steps": 30,
            "guidance_scale": 7.5,
        }

        quality_level = requirements.get("quality_level", "standard")

        if quality_level == "premium":
            params["steps"] = 50
            params["guidance_scale"] = 8.0
        elif quality_level == "draft":
            params["steps"] = 20
            params["guidance_scale"] = 7.0

        # Model-specific adjustments
        if "schnell" in model.model_id.lower():
            params["steps"] = min(params["steps"], 25)

        return params
