"""
Message definitions for agent conversations.

This module provides the data structures for representing messages in
multi-turn conversations with agents.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import json


class MessageRole(Enum):
    """Role of the message sender."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """
    Represents a tool call requested by the assistant.

    When an agent decides to use a tool, it creates a ToolCall that
    specifies which tool to invoke and with what arguments.
    """
    id: str
    name: str
    arguments: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            arguments=data.get("arguments", {}),
        )


@dataclass
class ToolResult:
    """
    Represents the result of a tool execution.

    After a tool is invoked, this structure captures its output
    to be included in the conversation history.
    """
    tool_call_id: str
    name: str
    content: Any
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tool_call_id": self.tool_call_id,
            "name": self.name,
            "content": self.content,
            "is_error": self.is_error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolResult":
        """Create from dictionary."""
        return cls(
            tool_call_id=data["tool_call_id"],
            name=data["name"],
            content=data["content"],
            is_error=data.get("is_error", False),
        )


@dataclass
class Message:
    """
    A single message in a conversation.

    Messages can be from the system, user, assistant, or tool.
    Assistant messages may include tool calls, and tool messages
    contain the results of tool executions.
    """
    role: MessageRole
    content: Optional[str] = None

    # For assistant messages that include tool calls
    tool_calls: List[ToolCall] = field(default_factory=list)

    # For tool result messages
    tool_result: Optional[ToolResult] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Optional message ID (set when persisted)
    id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization and LLM API calls."""
        result: Dict[str, Any] = {
            "role": self.role.value,
        }

        if self.content is not None:
            result["content"] = self.content

        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        if self.tool_result:
            result["tool_result"] = self.tool_result.to_dict()

        if self.metadata:
            result["metadata"] = self.metadata

        if self.id:
            result["id"] = self.id

        result["created_at"] = self.created_at.isoformat()

        return result

    def to_llm_format(self) -> Dict[str, Any]:
        """
        Convert to format expected by LLM APIs (OpenAI-compatible).

        Returns a simplified dict suitable for sending to LLM providers.
        """
        if self.role == MessageRole.TOOL:
            return {
                "role": "tool",
                "tool_call_id": self.tool_result.tool_call_id if self.tool_result else "",
                "content": json.dumps(self.tool_result.content) if self.tool_result else "",
            }

        result: Dict[str, Any] = {
            "role": self.role.value,
            "content": self.content or "",
        }

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dictionary."""
        role = MessageRole(data["role"])

        tool_calls = []
        if "tool_calls" in data:
            tool_calls = [ToolCall.from_dict(tc) for tc in data["tool_calls"]]

        tool_result = None
        if "tool_result" in data:
            tool_result = ToolResult.from_dict(data["tool_result"])

        created_at = datetime.utcnow()
        if "created_at" in data:
            if isinstance(data["created_at"], str):
                created_at = datetime.fromisoformat(data["created_at"])
            else:
                created_at = data["created_at"]

        return cls(
            role=role,
            content=data.get("content"),
            tool_calls=tool_calls,
            tool_result=tool_result,
            metadata=data.get("metadata", {}),
            created_at=created_at,
            id=data.get("id"),
        )

    @classmethod
    def system(cls, content: str, **kwargs) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content, **kwargs)

    @classmethod
    def user(cls, content: str, **kwargs) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content, **kwargs)

    @classmethod
    def assistant(
        cls,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        **kwargs,
    ) -> "Message":
        """Create an assistant message."""
        return cls(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls or [],
            **kwargs,
        )

    @classmethod
    def tool(cls, tool_result: ToolResult, **kwargs) -> "Message":
        """Create a tool result message."""
        return cls(
            role=MessageRole.TOOL,
            tool_result=tool_result,
            **kwargs,
        )


@dataclass
class Conversation:
    """
    A conversation containing multiple messages.

    Conversations track the full history of interactions between
    the user and agents, including tool calls and their results.
    """
    id: Optional[str] = None
    user_id: str = ""
    job_id: Optional[str] = None
    messages: List[Message] = field(default_factory=list)
    status: str = "active"  # active, completed, abandoned
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()

    def add_user_message(self, content: str, **kwargs) -> Message:
        """Add a user message and return it."""
        message = Message.user(content, **kwargs)
        self.add_message(message)
        return message

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[ToolCall]] = None,
        **kwargs,
    ) -> Message:
        """Add an assistant message and return it."""
        message = Message.assistant(content, tool_calls, **kwargs)
        self.add_message(message)
        return message

    def add_tool_result(self, tool_result: ToolResult, **kwargs) -> Message:
        """Add a tool result message and return it."""
        message = Message.tool(tool_result, **kwargs)
        self.add_message(message)
        return message

    def get_last_user_message(self) -> Optional[Message]:
        """Get the most recent user message."""
        for message in reversed(self.messages):
            if message.role == MessageRole.USER:
                return message
        return None

    def get_last_assistant_message(self) -> Optional[Message]:
        """Get the most recent assistant message."""
        for message in reversed(self.messages):
            if message.role == MessageRole.ASSISTANT:
                return message
        return None

    def to_llm_messages(self) -> List[Dict[str, Any]]:
        """
        Convert conversation to format expected by LLM APIs.

        Returns a list of messages suitable for sending to LLM providers.
        """
        return [msg.to_llm_format() for msg in self.messages]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "job_id": self.job_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "status": self.status,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Conversation":
        """Create from dictionary."""
        messages = [Message.from_dict(m) for m in data.get("messages", [])]

        created_at = datetime.utcnow()
        if "created_at" in data:
            if isinstance(data["created_at"], str):
                created_at = datetime.fromisoformat(data["created_at"])
            else:
                created_at = data["created_at"]

        updated_at = datetime.utcnow()
        if "updated_at" in data:
            if isinstance(data["updated_at"], str):
                updated_at = datetime.fromisoformat(data["updated_at"])
            else:
                updated_at = data["updated_at"]

        return cls(
            id=data.get("id"),
            user_id=data.get("user_id", ""),
            job_id=data.get("job_id"),
            messages=messages,
            status=data.get("status", "active"),
            metadata=data.get("metadata", {}),
            created_at=created_at,
            updated_at=updated_at,
        )

    def __len__(self) -> int:
        return len(self.messages)
