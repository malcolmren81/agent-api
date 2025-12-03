"""
Application settings and configuration management.
"""
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Server Configuration
    server_host: str = Field(default="0.0.0.0", description="Server host")
    server_port: int = Field(default=8000, description="Server port")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Environment"
    )
    log_level: str = Field(default="INFO", description="Log level")

    # GCP Configuration
    gcp_project_id: str = Field(default="palet8-system", description="GCP Project ID")
    gcp_region: str = Field(default="us-central1", description="GCP Region")

    # Google Gemini API
    gemini_api_key: str = Field(description="Gemini API key")
    gemini_model_base: str = Field(
        default="gemini-2.5-flash-lite", description="Gemini base LLM (fast, cost-efficient)"
    )
    gemini_model_reasoning: str = Field(
        default="gemini-2.5-flash", description="Gemini reasoning model (advanced thinking)"
    )
    gemini_model_image: str = Field(
        default="imagen-3.0-generate-001", description="Imagen 3 model via Vertex AI"
    )
    gemini_thinking_budget: str = Field(
        default="default", description="Gemini thinking budget"
    )

    # OpenAI API
    openai_api_key: str = Field(description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="OpenAI model")
    openai_org_id: str = Field(default="", description="OpenAI organization ID")

    # Runware API (Image Generation)
    runware_api_key: str = Field(default="", description="Runware API key for image generation")

    # Flux 1 Kontext API
    flux_api_key: str = Field(default="", description="Flux API key")
    flux_api_endpoint: str = Field(
        default="https://api.bfl.ai/v1", description="Flux API endpoint"
    )
    flux_model_pro: str = Field(
        default="flux-1-kontext-pro", description="Flux Pro model (lower cost)"
    )
    flux_model_max: str = Field(
        default="flux-1-kontext-max", description="Flux Max model (premium quality)"
    )
    flux_default_model: str = Field(
        default="flux-1-kontext-pro", description="Default Flux model to use"
    )

    # Model Selection Defaults
    default_reasoning_model: Literal["gemini", "chatgpt"] = Field(
        default="gemini", description="Default reasoning model"
    )
    default_image_model: Literal["flux", "gemini"] = Field(
        default="flux", description="Default image model"
    )

    # Cost Configuration (USD per request)
    gemini_base_cost: float = Field(
        default=0.00005, description="Gemini 2.5 Flash Lite base LLM cost per request"
    )
    gemini_reasoning_cost: float = Field(
        default=0.0001, description="Gemini 2.5 Flash reasoning cost per request"
    )
    gemini_image_cost: float = Field(
        default=0.04, description="Gemini 2.5 Flash Image generation cost per image"
    )
    chatgpt_reasoning_cost: float = Field(
        default=0.005, description="GPT-4o reasoning cost per request"
    )
    flux_pro_cost: float = Field(
        default=0.04, description="Flux Pro image generation cost per image"
    )
    flux_max_cost: float = Field(
        default=0.055, description="Flux Max image generation cost per image"
    )

    # Performance Thresholds
    max_request_timeout: int = Field(
        default=300, description="Max request timeout in seconds"
    )
    gpu_enabled: bool = Field(default=True, description="Enable GPU support")
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")

    # A2A Microservices
    use_product_a2a: bool = Field(
        default=False, description="Use A2A microservice for Product Generator"
    )
    product_generator_url: str = Field(
        default="http://localhost:8081", description="Product Generator service URL"
    )
    a2a_auth_token: str = Field(default="", description="A2A authentication token")
    a2a_enable_tls: bool = Field(default=False, description="Enable TLS for A2A")

    # Monitoring
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics port")
    enable_structured_logging: bool = Field(
        default=True, description="Enable structured JSON logging"
    )

    # Language Constraints
    allowed_languages: list[str] = Field(
        default=["en"], description="Allowed languages"
    )
    default_language: str = Field(default="en", description="Default language")

    # Phase 7.1.2: Admin API for Product Templates
    admin_api_url: str = Field(
        default="https://palet8-admin-api-702210710671.us-central1.run.app",
        description="Admin API URL for fetching product templates"
    )


# Global settings instance
settings = Settings()
