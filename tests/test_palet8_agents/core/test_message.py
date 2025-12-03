"""
Unit tests for palet8_agents.core.message module.
"""

import pytest
from datetime import datetime

from palet8_agents.core.message import (
    MessageRole,
    ToolCall,
    ToolResult,
    Message,
    Conversation,
)


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_role_values(self):
        """Test all expected role values exist."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test basic tool call creation."""
        tc = ToolCall(
            id="tc_123",
            name="search",
            arguments={"query": "test"},
        )
        assert tc.id == "tc_123"
        assert tc.name == "search"
        assert tc.arguments == {"query": "test"}

    def test_to_dict(self):
        """Test tool call serialization."""
        tc = ToolCall(
            id="tc_123",
            name="search",
            arguments={"query": "test"},
        )
        data = tc.to_dict()
        assert data["id"] == "tc_123"
        assert data["name"] == "search"
        assert data["arguments"] == {"query": "test"}

    def test_from_dict(self):
        """Test tool call deserialization."""
        data = {
            "id": "tc_123",
            "name": "search",
            "arguments": {"query": "test"},
        }
        tc = ToolCall.from_dict(data)
        assert tc.id == "tc_123"
        assert tc.name == "search"


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_success(self):
        """Test successful tool result."""
        tr = ToolResult(
            tool_call_id="tc_123",
            name="search",
            content={"results": ["a", "b", "c"]},
        )
        assert tr.tool_call_id == "tc_123"
        assert tr.is_error is False

    def test_tool_result_error(self):
        """Test error tool result."""
        tr = ToolResult(
            tool_call_id="tc_123",
            name="search",
            content="Error: not found",
            is_error=True,
        )
        assert tr.is_error is True


class TestMessage:
    """Tests for Message dataclass."""

    def test_system_message(self):
        """Test system message creation."""
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are a helpful assistant."

    def test_user_message(self):
        """Test user message creation."""
        msg = Message.user("Hello, how are you?")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello, how are you?"

    def test_assistant_message(self):
        """Test assistant message creation."""
        msg = Message.assistant("I'm doing well, thank you!")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "I'm doing well, thank you!"

    def test_assistant_message_with_tool_calls(self):
        """Test assistant message with tool calls."""
        tool_calls = [
            ToolCall(id="tc_1", name="search", arguments={"q": "test"})
        ]
        msg = Message.assistant(tool_calls=tool_calls)
        assert msg.role == MessageRole.ASSISTANT
        assert len(msg.tool_calls) == 1

    def test_tool_message(self):
        """Test tool result message creation."""
        tr = ToolResult(
            tool_call_id="tc_1",
            name="search",
            content={"results": []},
        )
        msg = Message.tool(tr)
        assert msg.role == MessageRole.TOOL
        assert msg.tool_result == tr

    def test_to_dict(self):
        """Test message serialization."""
        msg = Message.user("Hello")
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Hello"

    def test_from_dict(self):
        """Test message deserialization."""
        data = {
            "role": "user",
            "content": "Hello",
        }
        msg = Message.from_dict(data)
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"

    def test_to_llm_format(self):
        """Test conversion to LLM API format."""
        msg = Message.user("Hello")
        llm_format = msg.to_llm_format()
        assert llm_format["role"] == "user"
        assert llm_format["content"] == "Hello"


class TestConversation:
    """Tests for Conversation dataclass."""

    def test_conversation_creation(self):
        """Test basic conversation creation."""
        conv = Conversation(
            user_id="user123",
        )
        assert conv.user_id == "user123"
        assert len(conv.messages) == 0
        assert conv.status == "active"

    def test_add_message(self):
        """Test adding messages to conversation."""
        conv = Conversation(user_id="user123")
        msg = Message.user("Hello")
        conv.add_message(msg)
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hello"

    def test_add_user_message(self):
        """Test convenience method for adding user message."""
        conv = Conversation(user_id="user123")
        msg = conv.add_user_message("Hello")
        assert msg.role == MessageRole.USER
        assert len(conv.messages) == 1

    def test_add_assistant_message(self):
        """Test convenience method for adding assistant message."""
        conv = Conversation(user_id="user123")
        msg = conv.add_assistant_message("Hi there!")
        assert msg.role == MessageRole.ASSISTANT
        assert len(conv.messages) == 1

    def test_get_last_user_message(self):
        """Test getting last user message."""
        conv = Conversation(user_id="user123")
        conv.add_user_message("First")
        conv.add_assistant_message("Response")
        conv.add_user_message("Second")

        last = conv.get_last_user_message()
        assert last.content == "Second"

    def test_get_last_assistant_message(self):
        """Test getting last assistant message."""
        conv = Conversation(user_id="user123")
        conv.add_user_message("Question")
        conv.add_assistant_message("Answer 1")
        conv.add_assistant_message("Answer 2")

        last = conv.get_last_assistant_message()
        assert last.content == "Answer 2"

    def test_to_llm_messages(self):
        """Test conversion to LLM message format."""
        conv = Conversation(user_id="user123")
        conv.add_user_message("Hello")
        conv.add_assistant_message("Hi!")

        llm_messages = conv.to_llm_messages()
        assert len(llm_messages) == 2
        assert llm_messages[0]["role"] == "user"
        assert llm_messages[1]["role"] == "assistant"

    def test_to_dict(self):
        """Test conversation serialization."""
        conv = Conversation(
            id="conv123",
            user_id="user123",
        )
        conv.add_user_message("Hello")
        data = conv.to_dict()
        assert data["id"] == "conv123"
        assert len(data["messages"]) == 1

    def test_from_dict(self):
        """Test conversation deserialization."""
        data = {
            "id": "conv123",
            "user_id": "user123",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
        }
        conv = Conversation.from_dict(data)
        assert conv.id == "conv123"
        assert len(conv.messages) == 1

    def test_len(self):
        """Test conversation length."""
        conv = Conversation(user_id="user123")
        assert len(conv) == 0
        conv.add_user_message("Hello")
        assert len(conv) == 1
