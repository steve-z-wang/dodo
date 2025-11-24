"""Message types for conversational LLM history with tool calling support."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

from dodo.content import Content, Text, Image, ToolResult, ToolResultStatus


class ToolCall(BaseModel):
    """Tool call from LLM."""

    id: Optional[str] = None  # OpenAI/Anthropic provide ID, Gemini doesn't
    name: str
    arguments: Dict[str, Any]

    def __str__(self) -> str:
        args_str = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        if len(args_str) > 100:
            args_str = args_str[:100] + "..."
        return f"ToolCall({self.name}, {args_str})"




class Message(BaseModel):
    """Base message with automatic timestamp."""

    timestamp: datetime = Field(default_factory=datetime.now)
    content: Optional[List[Content]] = None


class SystemMessage(Message):
    """System instruction message."""

    def __str__(self) -> str:
        if not self.content:
            return "SystemMessage(empty)"
        texts = [part.text for part in self.content if hasattr(part, "text")]
        combined = " ".join(texts)
        if len(combined) > 200:
            combined = combined[:200] + "..."
        return f"SystemMessage({combined})"


class UserMessage(Message):
    """User message with text/images."""

    def __str__(self) -> str:
        if not self.content:
            return "UserMessage(empty)"
        text_parts = [
            part.text for part in self.content if isinstance(part, Text)
        ]
        image_count = sum(1 for part in self.content if isinstance(part, Image))

        text_str = " ".join(text_parts)
        if len(text_str) > 200:
            text_str = text_str[:200] + "..."

        if image_count > 0:
            return f"UserMessage(text={text_str}, images={image_count})"
        return f"UserMessage({text_str})"


class ModelMessage(Message):
    """Model response with optional tool calls."""

    tool_calls: Optional[List[ToolCall]] = None

    def __str__(self) -> str:
        parts = []

        if self.content:
            text_parts = [
                part.text for part in self.content if isinstance(part, Text)
            ]
            image_count = sum(
                1 for part in self.content if isinstance(part, Image)
            )

            if text_parts:
                text_str = " ".join(text_parts)
                if len(text_str) > 100:
                    text_str = text_str[:100] + "..."
                parts.append(f"text={text_str}")

            if image_count > 0:
                parts.append(f"images={image_count}")

        if self.tool_calls:
            tool_names = [tc.name for tc in self.tool_calls]
            parts.append(f"tools=[{', '.join(tool_names)}]")

        if parts:
            return f"ModelMessage({', '.join(parts)})"
        return "ModelMessage(empty)"


