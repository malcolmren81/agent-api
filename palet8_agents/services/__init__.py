"""
Services layer for business logic and cross-cutting concerns.

This layer sits between the API/orchestrator and domain agents, handling:
- Requirements analysis and context evaluation
- Safety classification
- Long-term memory storage (RAG)
- Text and image generation
- Embeddings and similarity search
- Cross-cutting concerns that span multiple agents
"""
from palet8_agents.services.requirements_analysis_service import (
    RequirementsAnalysisService,
    RequirementsAnalysisError,
    ExtractionError,
    RequirementsConfig,
)
from palet8_agents.services.context_analysis_service import (
    ContextAnalysisService,
    ContextAnalysisError,
    ContextConfig,
)
from palet8_agents.services.safety_classification_service import (
    SafetyClassificationService,
    SafetyClassificationError,
    SafetyConfig,
)
from palet8_agents.services.memory_service import (
    MemoryService,
    MemoryServiceError,
    DatabaseConnectionError,
    StorageError,
    RetrievalError,
    MemorySearchResult,
)
from palet8_agents.services.dimension_selection_service import (
    DimensionSelectionService,
    DimensionSelectionError,
    DimensionFillError,
    DimensionConfig,
)
from palet8_agents.services.model_selection_service import (
    ModelSelectionService,
    ModelSelectionError,
    NoCompatibleModelError,
    ModelSelectionConfig,
)
from palet8_agents.services.prompt_evaluation_service import (
    PromptEvaluationService,
    PromptEvaluationError,
    PromptEvaluationConfig,
)
from palet8_agents.services.result_evaluation_service import (
    ResultEvaluationService,
    ResultEvaluationError,
    ResultEvaluationConfig,
)
from palet8_agents.services.assembly_service import (
    AssemblyService,
    AssemblyError,
    PipelineError,
)

__all__ = [
    # Requirements Analysis
    "RequirementsAnalysisService",
    "RequirementsAnalysisError",
    "ExtractionError",
    "RequirementsConfig",
    # Context Analysis
    "ContextAnalysisService",
    "ContextAnalysisError",
    "ContextConfig",
    # Safety Classification
    "SafetyClassificationService",
    "SafetyClassificationError",
    "SafetyConfig",
    # Memory
    "MemoryService",
    "MemoryServiceError",
    "DatabaseConnectionError",
    "StorageError",
    "RetrievalError",
    "MemorySearchResult",
    # Dimension Selection
    "DimensionSelectionService",
    "DimensionSelectionError",
    "DimensionFillError",
    "DimensionConfig",
    # Model Selection
    "ModelSelectionService",
    "ModelSelectionError",
    "NoCompatibleModelError",
    "ModelSelectionConfig",
    # Prompt Evaluation
    "PromptEvaluationService",
    "PromptEvaluationError",
    "PromptEvaluationConfig",
    # Result Evaluation
    "ResultEvaluationService",
    "ResultEvaluationError",
    "ResultEvaluationConfig",
    # Assembly / Execution
    "AssemblyService",
    "AssemblyError",
    "PipelineError",
]
