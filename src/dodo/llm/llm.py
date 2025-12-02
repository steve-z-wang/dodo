"""LLM base class for conversation-based generation with tool calling support."""

import logging
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from .message import Message, Role

if TYPE_CHECKING:
    from dodo.tool import Tool


class LLM(ABC):
    """Abstract base class for Large Language Models with conversation support.

    Implementations must provide:
    - call_tools(): Generate response with tool calling

    Example implementation:
        class MyLLM(LLM):
            async def call_tools(self, messages, tools):
                # Convert to your LLM's format
                # Make API call
                # Return Message with role=Role.MODEL and tool calls in content
                pass
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def call_tools(
        self,
        messages: List[Message],
        tools: List["Tool"],
    ) -> Message:
        """Generate response with tool calling.

        Args:
            messages: Conversation history as list of Message objects.
            tools: List of tools available for the LLM to call.

        Returns:
            Message with role=Role.MODEL containing text and/or ToolCall content.

        Example:
            >>> messages = [
            ...     Message(role=Role.SYSTEM, content=[Text(text="You are an agent")]),
            ...     Message(role=Role.USER, content=[Text(text="Do something")]),
            ... ]
            >>> response = await llm.call_tools(messages, tools=[MyTool()])
            >>> print(response.tool_calls[0].name)
            "my_tool"
        """
        pass
