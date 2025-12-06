"""
Configuration loading and management for agents.

This module handles loading agent configurations from YAML files,
including model profiles, routing policies, and agent settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import yaml


@dataclass
class ModelProfile:
    """
    Configuration profile for an LLM model.

    Defines which model to use for a specific agent or task,
    along with its parameters and fallback options.

    Values loaded from config/agent_routing_policy.yaml
    """
    name: str
    primary_model: str = ""
    fallback_model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

    # Cost tracking (from agent_routing_policy.yaml model_profiles)
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "primary_model": self.primary_model,
            "fallback_model": self.fallback_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "max_retries": self.max_retries,
            "retry_delay_seconds": self.retry_delay_seconds,
        }

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ModelProfile":
        """Create from dictionary."""
        return cls(
            name=name,
            primary_model=data.get("primary_model", ""),
            fallback_model=data.get("fallback_model", ""),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 1000),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            cost_per_1k_input=data.get("cost_per_1k_input", 0.0),
            cost_per_1k_output=data.get("cost_per_1k_output", 0.0),
            max_retries=data.get("max_retries", 3),
            retry_delay_seconds=data.get("retry_delay_seconds", 1.0),
        )


@dataclass
class ImageModelConfig:
    """
    Configuration for image generation models.

    Values loaded from config/image_models_config.yaml
    """
    # Models loaded from image_models_config.yaml - no hardcoded defaults
    primary: Optional[str] = None
    fallback: Optional[str] = None
    default_dimensions: str = "1024x1024"
    default_steps: int = 30


@dataclass
class EmbeddingModelConfig:
    """
    Configuration for embedding models.

    Values from agent_routing_policy.yaml embedding_models section
    Uses Google Vertex AI for both text and image embeddings.
    """
    # Text embeddings (prompts, summaries, user history)
    text_model: str = "gemini-embedding-001"
    text_dimensions: int = 768

    # Image embeddings (art library, generated images)
    image_model: str = "multimodalembedding@001"
    image_dimensions: int = 1408

    # Provider
    provider: str = "google"


@dataclass
class SafetyConfig:
    """
    Safety agent configuration.

    Full config in config/safety_config.yaml
    """
    # Categories to check (from safety_config.yaml blocking_behavior)
    categories: List[str] = field(default_factory=lambda: [
        "nsfw",
        "violence",
        "hate",
        "ip_trademark",
        "illegal",
    ])

    # Tagging thresholds per category (from safety_config.yaml)
    # Note: Only NSFW blocks, others tag only
    blocking_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "nsfw": 0.5,         # Lower threshold - any moderate NSFW = block
        "violence": 0.7,     # Tag at 70%+ confidence
        "hate": 0.7,         # Tag at 70%+ confidence
        "ip_trademark": 0.5, # Lower threshold - detect IP early
        "illegal": 0.8,      # Tag at 80%+ confidence
    })

    # IP blocklist loaded from safety_config.yaml ip_detection.known_entities
    ip_blocklist: List[str] = field(default_factory=list)


@dataclass
class EvaluationConfig:
    """
    Evaluation agent configuration.

    Full config in config/evaluation_config.yaml
    """
    # Quality threshold for approval (from evaluation_config.yaml)
    quality_threshold: float = 0.45  # acceptance_threshold in legacy config

    # Scoring weights (from evaluation_config.yaml result_weights)
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "prompt_fidelity": 0.20,
        "product_readiness": 0.20,
        "technical_quality": 0.15,
        "background_composition": 0.15,
        "aesthetic": 0.15,
        "text_legibility": 0.05,
        "safety": 0.10,
    })

    # Objective check thresholds (from evaluation_config.yaml)
    min_resolution: int = 512  # image_size.min_resolution
    min_coverage_percent: float = 0.5  # lowered from 0.7 per config
    max_retries: int = 3  # from evaluation_config.yaml


@dataclass
class AgentConfig:
    """
    Complete configuration for the agent system.

    This is the main configuration class that aggregates all settings
    for models, agents, safety, and evaluation.
    """
    # Model profiles for each agent type
    model_profiles: Dict[str, ModelProfile] = field(default_factory=dict)

    # Image and embedding model configs
    image_models: ImageModelConfig = field(default_factory=ImageModelConfig)
    embedding_models: EmbeddingModelConfig = field(default_factory=EmbeddingModelConfig)

    # Safety and evaluation configs
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # RAG configuration
    rag_retrieval_limit: int = 10
    rag_similarity_threshold: float = 0.7

    # General settings
    max_conversation_turns: int = 20
    max_generation_retries: int = 3

    def get_model_profile(self, agent_name: str) -> Optional[ModelProfile]:
        """Get model profile for a specific agent."""
        return self.model_profiles.get(agent_name)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_profiles": {
                name: profile.to_dict()
                for name, profile in self.model_profiles.items()
            },
            "image_models": {
                "primary": self.image_models.primary,
                "fallback": self.image_models.fallback,
                "default_dimensions": self.image_models.default_dimensions,
                "default_steps": self.image_models.default_steps,
            },
            "embedding_models": {
                "provider": self.embedding_models.provider,
                "text": {
                    "model": self.embedding_models.text_model,
                    "dimensions": self.embedding_models.text_dimensions,
                },
                "image": {
                    "model": self.embedding_models.image_model,
                    "dimensions": self.embedding_models.image_dimensions,
                },
            },
            "safety": {
                "categories": self.safety.categories,
                "blocking_thresholds": self.safety.blocking_thresholds,
                "ip_blocklist": self.safety.ip_blocklist,
            },
            "evaluation": {
                "quality_threshold": self.evaluation.quality_threshold,
                "scoring_weights": self.evaluation.scoring_weights,
                "min_resolution": self.evaluation.min_resolution,
                "min_coverage_percent": self.evaluation.min_coverage_percent,
                "max_retries": self.evaluation.max_retries,
            },
            "rag_retrieval_limit": self.rag_retrieval_limit,
            "rag_similarity_threshold": self.rag_similarity_threshold,
            "max_conversation_turns": self.max_conversation_turns,
            "max_generation_retries": self.max_generation_retries,
        }


# Global config instance
_config: Optional[AgentConfig] = None


def load_config(config_path: Optional[str] = None) -> AgentConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Loaded AgentConfig instance.
    """
    global _config

    if config_path is None:
        # Default config path
        config_path = os.environ.get(
            "AGENT_CONFIG_PATH",
            str(Path(__file__).parent.parent.parent / "config" / "agent_routing_policy.yaml"),
        )

    config_file = Path(config_path)

    if not config_file.exists():
        # Return default config if file doesn't exist
        _config = AgentConfig()
        return _config

    with open(config_file, "r") as f:
        data = yaml.safe_load(f) or {}

    # Parse model profiles
    model_profiles = {}
    for name, profile_data in data.get("model_profiles", {}).items():
        model_profiles[name] = ModelProfile.from_dict(name, profile_data)

    # Parse image models
    image_data = data.get("image_models", {})
    primary_model = image_data.get("primary")
    fallback_model = image_data.get("fallback")

    # Load models from image_models_config.yaml if not set in main config
    if not primary_model or not fallback_model:
        image_models_config_path = config_file.parent / "image_models_config.yaml"
        if image_models_config_path.exists():
            with open(image_models_config_path, "r") as f:
                img_config = yaml.safe_load(f) or {}

            # Get primary from scenario_selection (first priority in art_no_reference)
            scenario_selection = img_config.get("scenario_selection", {})
            art_no_ref = scenario_selection.get("art_no_reference", {})
            priority_models = art_no_ref.get("priority_models", {})
            if priority_models and not primary_model:
                first_priority = min(priority_models.keys())
                primary_model = priority_models[first_priority]

            # Get fallback from second priority
            if priority_models and not fallback_model:
                priorities = sorted(priority_models.keys())
                if len(priorities) > 1:
                    fallback_model = priority_models[priorities[1]]

    image_models = ImageModelConfig(
        primary=primary_model,
        fallback=fallback_model,
        default_dimensions=image_data.get("default_dimensions", "1024x1024"),
        default_steps=image_data.get("default_steps", 30),
    )

    # Parse embedding models
    embedding_data = data.get("embedding_models", {})
    text_config = embedding_data.get("text", {})
    image_config = embedding_data.get("image", {})
    embedding_models = EmbeddingModelConfig(
        text_model=text_config.get("model", "gemini-embedding-001"),
        text_dimensions=text_config.get("dimensions", 768),
        image_model=image_config.get("model", "multimodalembedding@001"),
        image_dimensions=image_config.get("dimensions", 1408),
        provider=text_config.get("provider", "google"),
    )

    # Parse safety config
    safety_data = data.get("safety", {})
    safety = SafetyConfig(
        categories=safety_data.get("categories", SafetyConfig().categories),
        blocking_thresholds=safety_data.get(
            "blocking_thresholds",
            SafetyConfig().blocking_thresholds,
        ),
        ip_blocklist=safety_data.get("ip_blocklist", []),
    )

    # Parse evaluation config
    eval_data = data.get("evaluation", {})
    evaluation = EvaluationConfig(
        quality_threshold=eval_data.get("quality_threshold", 0.45),
        scoring_weights=eval_data.get(
            "scoring_weights",
            EvaluationConfig().scoring_weights,
        ),
        min_resolution=eval_data.get("min_resolution", 512),
        min_coverage_percent=eval_data.get("min_coverage_percent", 0.8),
        max_retries=eval_data.get("max_retries", 3),
    )

    _config = AgentConfig(
        model_profiles=model_profiles,
        image_models=image_models,
        embedding_models=embedding_models,
        safety=safety,
        evaluation=evaluation,
        rag_retrieval_limit=data.get("rag_retrieval_limit", 10),
        rag_similarity_threshold=data.get("rag_similarity_threshold", 0.7),
        max_conversation_turns=data.get("max_conversation_turns", 20),
        max_generation_retries=data.get("max_generation_retries", 3),
    )

    return _config


def get_config() -> AgentConfig:
    """
    Get the current configuration.

    Loads default config if not already loaded.

    Returns:
        Current AgentConfig instance.
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_model_profile(agent_name: str) -> Optional[ModelProfile]:
    """
    Get model profile for a specific agent.

    Args:
        agent_name: Name of the agent (pali, planner, evaluator, safety)

    Returns:
        ModelProfile if found, None otherwise.
    """
    config = get_config()
    return config.get_model_profile(agent_name)


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None
