"""
Tool registry for managing available tools.

This module provides a singleton registry for registering and
retrieving tools that agents can use.
"""

from typing import Any, Dict, List, Optional, Type
from palet8_agents.tools.base import BaseTool
from palet8_agents.core.exceptions import ToolNotFoundError


class ToolRegistry:
    """
    Singleton registry for managing tools.

    The registry maintains a collection of available tools that agents
    can use during execution. Tools are registered by name and can be
    retrieved individually or as a group.
    """

    _instance: Optional["ToolRegistry"] = None
    _tools: Dict[str, BaseTool]

    def __new__(cls) -> "ToolRegistry":
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool with the registry.

        Args:
            tool: Tool instance to register

        Raises:
            ValueError: If a tool with the same name already exists
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def register_class(self, tool_class: Type[BaseTool], **kwargs) -> BaseTool:
        """
        Register a tool class by instantiating it.

        Args:
            tool_class: Tool class to instantiate and register
            **kwargs: Arguments to pass to the tool constructor

        Returns:
            The registered tool instance
        """
        tool = tool_class(**kwargs)
        self.register(tool)
        return tool

    def unregister(self, name: str) -> None:
        """
        Unregister a tool by name.

        Args:
            name: Name of the tool to unregister
        """
        if name in self._tools:
            del self._tools[name]

    def get(self, name: str) -> BaseTool:
        """
        Get a tool by name.

        Args:
            name: Name of the tool to retrieve

        Returns:
            The requested tool

        Raises:
            ToolNotFoundError: If the tool is not found
        """
        if name not in self._tools:
            raise ToolNotFoundError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def get_optional(self, name: str) -> Optional[BaseTool]:
        """
        Get a tool by name, returning None if not found.

        Args:
            name: Name of the tool to retrieve

        Returns:
            The tool if found, None otherwise
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """
        Check if a tool is registered.

        Args:
            name: Name of the tool to check

        Returns:
            True if the tool is registered
        """
        return name in self._tools

    def list_tools(self) -> List[str]:
        """
        Get list of all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all(self) -> List[BaseTool]:
        """
        Get all registered tools.

        Returns:
            List of all tool instances
        """
        return list(self._tools.values())

    def get_tools_for_agent(self, tool_names: List[str]) -> List[BaseTool]:
        """
        Get a subset of tools by name.

        Args:
            tool_names: List of tool names to retrieve

        Returns:
            List of tool instances (skips tools not found)
        """
        tools = []
        for name in tool_names:
            if name in self._tools:
                tools.append(self._tools[name])
        return tools

    def get_openai_schemas(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get OpenAI-compatible tool schemas.

        Args:
            tool_names: Optional list of specific tools to include.
                       If None, includes all registered tools.

        Returns:
            List of tool schemas in OpenAI format
        """
        if tool_names is None:
            tools = self._tools.values()
        else:
            tools = [self._tools[name] for name in tool_names if name in self._tools]

        return [tool.get_openai_schema() for tool in tools]

    def clear(self) -> None:
        """Clear all registered tools (useful for testing)."""
        self._tools.clear()

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """
    Get the global tool registry instance.

    Returns:
        The singleton ToolRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def register_tool(tool: BaseTool) -> BaseTool:
    """
    Register a tool with the global registry.

    Can be used as a decorator for tool classes.

    Args:
        tool: Tool instance to register

    Returns:
        The registered tool
    """
    get_registry().register(tool)
    return tool


def tool(name: str, description: str):
    """
    Decorator for registering tool classes.

    Usage:
        @tool("my_tool", "Description of my tool")
        class MyTool(BaseTool):
            async def execute(self, **kwargs):
                ...

    Args:
        name: Tool name
        description: Tool description

    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseTool]) -> Type[BaseTool]:
        # Override name and description
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.name = name
            self.description = description

        cls.__init__ = new_init
        return cls

    return decorator


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _registry
    if _registry:
        _registry.clear()
    _registry = None
