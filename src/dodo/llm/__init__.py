"""LLM abstractions for DoDo."""

from .llm import LLM
from .message import (
    Message,
    Content,
    TextContent,
    ImageContent,
    ToolResultContent,
    ToolCall,
    ToolResultStatus,
    SystemMessage,
    UserMessage,
    ModelMessage,
)

__all__ = [
    "LLM",
    # Content types
    "Message",
    "Content",
    "TextContent",
    "ImageContent",
    "ToolResultContent",
    "ToolCall",
    "ToolResultStatus",
    # Message types (3 only)
    "SystemMessage",
    "UserMessage",
    "ModelMessage",
]
