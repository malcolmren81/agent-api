"""
Base tool class for agent tools.

This module defines the abstract base class for all tools that agents
can use, along with the schema definitions for tool parameters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json


class ParameterType(Enum):
    """Supported parameter types for tool definitions."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """
    Definition of a single tool parameter.

    Used to build the JSON Schema for tool definitions that are
    sent to LLM providers for function calling.
    """
    name: str
    type: ParameterType
    description: str
    required: bool = True
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None
    items_type: Optional[ParameterType] = None  # For array types

    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema format."""
        schema: Dict[str, Any] = {
            "type": self.type.value,
            "description": self.description,
        }

        if self.enum:
            schema["enum"] = self.enum

        if self.default is not None:
            schema["default"] = self.default

        if self.type == ParameterType.ARRAY and self.items_type:
            schema["items"] = {"type": self.items_type.value}

        return schema


@dataclass
class ToolSchema:
    """
    Complete schema for a tool definition.

    This represents the full tool definition that gets sent to
    LLM providers for function calling support.
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)

    def to_openai_format(self) -> Dict[str, Any]:
        """
        Convert to OpenAI function calling format.

        Returns a dict suitable for the 'tools' parameter in chat completion.
        """
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = param.to_schema()
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


@dataclass
class ToolResult:
    """
    Result from tool execution.

    Contains the output data along with success status and
    any error information.
    """
    success: bool
    data: Any
    error: Optional[str] = None
    error_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "error_code": self.error_code,
        }

    def to_llm_response(self) -> str:
        """
        Format result for LLM consumption.

        Returns a string representation suitable for including
        in the conversation as a tool result.
        """
        if self.success:
            if isinstance(self.data, (dict, list)):
                return json.dumps(self.data, indent=2)
            return str(self.data)
        else:
            return f"Error: {self.error}"


class BaseTool(ABC):
    """
    Abstract base class for all agent tools.

    Each tool represents a specific capability that agents can invoke,
    such as retrieving context, generating images, or managing jobs.

    Tools define their parameters using ToolParameter and implement
    the execute method to perform their action.
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Optional[List[ToolParameter]] = None,
    ):
        """
        Initialize the tool.

        Args:
            name: Unique identifier for the tool
            description: Human-readable description of what the tool does
            parameters: List of parameter definitions
        """
        self.name = name
        self.description = description
        self.parameters = parameters or []
        self._schema: Optional[ToolSchema] = None

    @property
    def schema(self) -> ToolSchema:
        """Get the tool schema."""
        if self._schema is None:
            self._schema = ToolSchema(
                name=self.name,
                description=self.description,
                parameters=self.parameters,
            )
        return self._schema

    def get_openai_schema(self) -> Dict[str, Any]:
        """Get OpenAI-compatible tool definition."""
        return self.schema.to_openai_format()

    def validate_input(self, **kwargs) -> Optional[str]:
        """
        Validate input parameters.

        Args:
            **kwargs: Parameters to validate

        Returns:
            Error message if validation fails, None if valid
        """
        for param in self.parameters:
            if param.required and param.name not in kwargs:
                if param.default is None:
                    return f"Missing required parameter: {param.name}"

            if param.name in kwargs:
                value = kwargs[param.name]

                # Type validation
                if param.type == ParameterType.STRING and not isinstance(value, str):
                    return f"Parameter {param.name} must be a string"
                elif param.type == ParameterType.INTEGER and not isinstance(value, int):
                    return f"Parameter {param.name} must be an integer"
                elif param.type == ParameterType.NUMBER and not isinstance(value, (int, float)):
                    return f"Parameter {param.name} must be a number"
                elif param.type == ParameterType.BOOLEAN and not isinstance(value, bool):
                    return f"Parameter {param.name} must be a boolean"
                elif param.type == ParameterType.ARRAY and not isinstance(value, list):
                    return f"Parameter {param.name} must be an array"
                elif param.type == ParameterType.OBJECT and not isinstance(value, dict):
                    return f"Parameter {param.name} must be an object"

                # Enum validation
                if param.enum and value not in param.enum:
                    return f"Parameter {param.name} must be one of: {param.enum}"

        return None

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult containing the execution outcome
        """
        pass

    async def __call__(self, **kwargs) -> ToolResult:
        """
        Call the tool with validation.

        This method validates input before executing the tool.

        Args:
            **kwargs: Tool parameters

        Returns:
            ToolResult from execution
        """
        # Validate input
        error = self.validate_input(**kwargs)
        if error:
            return ToolResult(
                success=False,
                data=None,
                error=error,
                error_code="VALIDATION_ERROR",
            )

        # Apply defaults
        for param in self.parameters:
            if param.name not in kwargs and param.default is not None:
                kwargs[param.name] = param.default

        # Execute
        try:
            return await self.execute(**kwargs)
        except Exception as e:
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                error_code="EXECUTION_ERROR",
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
