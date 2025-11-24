"""Content types for messages."""

from typing import Optional
from enum import Enum
from pydantic import BaseModel


class ImageMimeType(str, Enum):
    """Supported image MIME types."""

    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    GIF = "image/gif"


class ToolResultStatus(str, Enum):
    """Tool execution result status."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class Content(BaseModel):
    """Base class for message content parts."""

    tag: Optional[str] = None  # Tag for filtering (e.g., "observation", "context")
    lifespan: Optional[int] = None  # How many iterations to keep this content


class Text(Content):
    """Text content part."""

    text: str

    def __str__(self) -> str:
        if len(self.text) > 100:
            return f"Text({self.text[:100]}...)"
        return f"Text({self.text})"


class Image(Content):
    """Image content part (base64-encoded)."""

    data: str  # base64-encoded
    mime_type: ImageMimeType = ImageMimeType.PNG

    def __str__(self) -> str:
        return f"Image(mime_type={self.mime_type.value}, size={len(self.data)} bytes)"


class ToolResult(Content):
    """Result of tool execution - can be included in UserMessage content."""

    tool_call_id: Optional[str] = None  # Match by ID if available
    name: str  # Tool name
    status: ToolResultStatus  # Execution status
    error: Optional[str] = None  # Error message if status is ERROR
    description: str = ""  # Human-readable description of action taken
    terminal: bool = False  # If True, task should stop after this tool

    def __str__(self) -> str:
        parts = [f"{self.name}: {self.status.value}"]
        if self.error:
            parts.append(f"error={self.error}")
        if self.terminal:
            parts.append("terminal")
        return f"ToolResult({', '.join(parts)})"
