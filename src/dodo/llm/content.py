"""Content types for messages."""

import base64
from typing import Optional, Union, Dict, Any
from enum import Enum
from pathlib import Path
from pydantic import BaseModel


# =============================================================================
# Base
# =============================================================================


class Content(BaseModel):
    """Base class for message content parts."""

    tag: Optional[str] = None  # Tag for filtering (e.g., "observation", "context")
    lifespan: Optional[int] = None  # How many iterations to keep this content


# =============================================================================
# Text
# =============================================================================


class Text(Content):
    """Text content part."""

    text: str

    def __str__(self) -> str:
        if len(self.text) > 100:
            return f"Text({self.text[:100]}...)"
        return f"Text({self.text})"


# =============================================================================
# Image
# =============================================================================


class ImageMimeType(str, Enum):
    """Supported image MIME types."""

    PNG = "image/png"
    JPEG = "image/jpeg"
    WEBP = "image/webp"
    GIF = "image/gif"


class Image(Content):
    """Image content part (base64-encoded)."""

    base64: str
    mime_type: ImageMimeType = ImageMimeType.PNG

    def __str__(self) -> str:
        return f"Image({self.mime_type.value}, {len(self.base64)} bytes)"

    @classmethod
    def from_bytes(cls, data: bytes, **kwargs) -> "Image":
        """Create Image from raw bytes, auto-detecting mime type."""
        mime_type = cls._detect_mime_type(data)
        encoded = base64.b64encode(data).decode("utf-8")
        return cls(base64=encoded, mime_type=mime_type, **kwargs)

    @classmethod
    def from_file(cls, path: Union[str, Path], **kwargs) -> "Image":
        """Create Image from file path."""
        path = Path(path)
        data = path.read_bytes()
        return cls.from_bytes(data, **kwargs)

    @classmethod
    def from_base64(
        cls,
        data: str,
        mime_type: Union[ImageMimeType, str] = ImageMimeType.PNG,
        **kwargs,
    ) -> "Image":
        """Create Image from base64 string."""
        if isinstance(mime_type, str):
            mime_type = cls._parse_mime_type(mime_type)
        return cls(base64=data, mime_type=mime_type, **kwargs)

    @staticmethod
    def _detect_mime_type(data: bytes) -> ImageMimeType:
        """Detect image mime type from magic bytes."""
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return ImageMimeType.PNG
        elif data.startswith(b"\xff\xd8\xff"):
            return ImageMimeType.JPEG
        elif data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return ImageMimeType.GIF
        elif data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ImageMimeType.WEBP
        else:
            raise ValueError("Unknown image format")

    @staticmethod
    def _parse_mime_type(mime_type: str) -> ImageMimeType:
        """Parse mime type string to enum."""
        mime_type = mime_type.lower()
        if mime_type in ("png", "image/png"):
            return ImageMimeType.PNG
        elif mime_type in ("jpg", "jpeg", "image/jpeg"):
            return ImageMimeType.JPEG
        elif mime_type in ("gif", "image/gif"):
            return ImageMimeType.GIF
        elif mime_type in ("webp", "image/webp"):
            return ImageMimeType.WEBP
        else:
            raise ValueError(f"Unknown mime type: {mime_type}")


# =============================================================================
# ToolResult
# =============================================================================


class ToolResultStatus(str, Enum):
    """Tool execution result status."""

    SUCCESS = "success"
    ERROR = "error"
    SKIPPED = "skipped"


class ToolResult(Content):
    """Result of tool execution."""

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


# =============================================================================
# ToolCall
# =============================================================================


class ToolCall(Content):
    """Tool call from model."""

    id: Optional[str] = None  # OpenAI/Anthropic provide ID, Gemini doesn't
    name: str
    arguments: Dict[str, Any]

    def __str__(self) -> str:
        args_str = ", ".join(f"{k}={v}" for k, v in self.arguments.items())
        if len(args_str) > 100:
            args_str = args_str[:100] + "..."
        return f"ToolCall({self.name}, {args_str})"
