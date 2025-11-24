"""LLM base class for conversation-based generation with tool calling support."""

import logging
from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from .message import Message, ModelMessage

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
                # Return ModelMessage with tool_calls
                pass
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    async def call_tools(
        self,
        messages: List[Message],
        tools: List["Tool"],
    ) -> ModelMessage:
        """Generate response with tool calling.

        Args:
            messages: Conversation history as list of Message objects.
                Can include SystemMessage, UserMessage, ModelMessage.
            tools: List of tools available for the LLM to call.

        Returns:
            ModelMessage with content (text/images) and/or tool_calls.

        Example:
            >>> messages = [
            ...     SystemMessage(content=[TextContent(text="You are an agent")]),
            ...     UserMessage(content=[TextContent(text="Do something")]),
            ... ]
            >>> response = await llm.call_tools(messages, tools=[MyTool()])
            >>> print(response.tool_calls[0].name)
            "my_tool"
        """
        pass
