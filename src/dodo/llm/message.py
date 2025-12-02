"""Message type for conversational LLM history."""

from typing import List
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from .content import Content, Text, Image, ToolResult, ToolCall


class Role(str, Enum):
    """Message role."""

    SYSTEM = "system"
    USER = "user"
    MODEL = "model"


class Message(BaseModel):
    """A message in the conversation history."""

    role: Role
    content: List[Content] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def text(self) -> str:
        """Get all text content concatenated."""
        texts = [c.text for c in self.content if isinstance(c, Text)]
        return " ".join(texts)

    @property
    def images(self) -> List[Image]:
        """Get all image content."""
        return [c for c in self.content if isinstance(c, Image)]

    @property
    def tool_calls(self) -> List[ToolCall]:
        """Get all tool calls."""
        return [c for c in self.content if isinstance(c, ToolCall)]

    @property
    def tool_results(self) -> List[ToolResult]:
        """Get all tool results."""
        return [c for c in self.content if isinstance(c, ToolResult)]

    def __str__(self) -> str:
        parts = []

        # Text
        if self.text:
            text_str = self.text
            if len(text_str) > 100:
                text_str = text_str[:100] + "..."
            parts.append(f"text={text_str}")

        # Images
        if self.images:
            parts.append(f"images={len(self.images)}")

        # Tool calls
        if self.tool_calls:
            tool_names = [tc.name for tc in self.tool_calls]
            parts.append(f"tool_calls=[{', '.join(tool_names)}]")

        # Tool results
        if self.tool_results:
            result_names = [tr.name for tr in self.tool_results]
            parts.append(f"tool_results=[{', '.join(result_names)}]")

        if parts:
            return f"Message({self.role.value}, {', '.join(parts)})"
        return f"Message({self.role.value}, empty)"
