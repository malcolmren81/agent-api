"""
Tools for agent execution.

This module provides the tool framework that agents use to perform
specific actions like retrieving context, generating images, and
managing job state.
"""

from palet8_agents.tools.base import BaseTool, ToolParameter, ToolSchema, ToolResult, ParameterType
from palet8_agents.tools.registry import ToolRegistry, get_registry, register_tool

# New tools (PR 4) - import first as they have fewer dependencies
from palet8_agents.tools.requirements_tool import RequirementsTool
from palet8_agents.tools.dimension_tool import DimensionTool
from palet8_agents.tools.model_selector_tool import ModelSelectorTool
from palet8_agents.tools.prompt_quality_tool import PromptQualityTool
from palet8_agents.tools.image_evaluation_tool import ImageEvaluationTool
from palet8_agents.tools.safety_tool import SafetyTool
from palet8_agents.tools.memory_tool import MemoryTool

# Existing tools - wrapped in try-except for optional dependencies (Prisma, etc.)
try:
    from palet8_agents.tools.context_tool import ContextTool
except ImportError:
    ContextTool = None  # type: ignore

try:
    from palet8_agents.tools.image_tool import ImageTool
except ImportError:
    ImageTool = None  # type: ignore

try:
    from palet8_agents.tools.job_tool import JobTool
except ImportError:
    JobTool = None  # type: ignore

try:
    from palet8_agents.tools.search_tool import SearchTool
except ImportError:
    SearchTool = None  # type: ignore

__all__ = [
    # Base classes
    "BaseTool",
    "ToolParameter",
    "ToolSchema",
    "ToolResult",
    "ParameterType",
    # Registry
    "ToolRegistry",
    "get_registry",
    "register_tool",
    # Existing tools (may be None if dependencies missing)
    "ContextTool",
    "ImageTool",
    "JobTool",
    "SearchTool",
    # New tools
    "RequirementsTool",
    "DimensionTool",
    "ModelSelectorTool",
    "PromptQualityTool",
    "ImageEvaluationTool",
    "SafetyTool",
    "MemoryTool",
]
