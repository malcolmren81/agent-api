"""
Palet8 Agents Models Package

Shared data classes and enums for the agent system.
Extracted from agent implementations for reusability across services and tools.
"""

# Enums
from .enums import (
    PlannerPhase,
    EvaluationPhase,
)

# Requirements (from PaliAgent)
from .requirements import (
    RequirementsStatus,
)

# Context (from PlannerAgent)
from .context import (
    ContextCompleteness,
)

# Safety (from SafetyAgent and PlannerAgent)
from .safety import (
    SafetyCategory,
    SafetySeverity,
    SafetyClassification,
    SafetyFlag,
    SafetyResult,
)

# Prompt (from PlannerAgent and EvaluatorAgent)
from .prompt import (
    PromptQualityDimension,
    PromptDimensions,
    PromptQualityResult,
)

# Generation (from PlannerAgent and AssemblyService)
from .generation import (
    ExecutionStatus,
    GenerationParameters,
    PipelineConfig,
    AssemblyRequest,
    GeneratedImageData,
    ExecutionResult,
)

# Evaluation (from EvaluatorAgent)
from .evaluation import (
    EvaluationDecision,
    ResultQualityDimension,
    RetrySuggestion,
    ResultQualityResult,
    EvaluationPlan,
    EvaluationFeedback,
)

# Planning (agent coordination contracts)
from .planning import (
    PlanningTask,
    ContextSummary,
    PromptPlan,
)

__all__ = [
    # Enums
    "PlannerPhase",
    "EvaluationPhase",
    # Requirements
    "RequirementsStatus",
    # Context
    "ContextCompleteness",
    # Safety
    "SafetyCategory",
    "SafetySeverity",
    "SafetyClassification",
    "SafetyFlag",
    "SafetyResult",
    # Prompt
    "PromptQualityDimension",
    "PromptDimensions",
    "PromptQualityResult",
    # Generation
    "ExecutionStatus",
    "GenerationParameters",
    "PipelineConfig",
    "AssemblyRequest",
    "GeneratedImageData",
    "ExecutionResult",
    # Evaluation
    "EvaluationDecision",
    "ResultQualityDimension",
    "RetrySuggestion",
    "ResultQualityResult",
    "EvaluationPlan",
    "EvaluationFeedback",
    # Planning
    "PlanningTask",
    "ContextSummary",
    "PromptPlan",
]
