"""Gemini LLM provider."""

import logging
from typing import List, Optional, TYPE_CHECKING

from google import genai
from google.genai import types

from dodo.llm import LLM, Message
from .gemini_mapper import (
    messages_to_gemini_content,
    gemini_response_to_message,
    build_tool_config,
)

if TYPE_CHECKING:
    from dodo.tools import Tool


class Gemini(LLM):
    """Gemini LLM provider.

    Example:
        >>> from dodo.llm.providers import Gemini
        >>> llm = Gemini(model="gemini-2.5-flash")
        >>> # Uses GEMINI_API_KEY or GOOGLE_API_KEY env var
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        temperature: float = 0.5,
    ):
        """Initialize Gemini.

        Args:
            model: Gemini model name (e.g., "gemini-2.5-flash", "gemini-2.5-pro")
            api_key: Optional API key (if not set via environment variable)
            temperature: Sampling temperature (0.0 to 1.0)
        """
        super().__init__()

        # Create client - uses GEMINI_API_KEY or GOOGLE_API_KEY env var if not provided
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

        self.model_name = model
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)

    async def call_tools(
        self,
        messages: List[Message],
        tools: List["Tool"],
    ) -> Message:
        """Generate response with tool calling."""
        gemini_content, system_instruction = messages_to_gemini_content(messages)
        tool_config = build_tool_config(tools)

        config = types.GenerateContentConfig(
            temperature=self.temperature,
            system_instruction=system_instruction,
            tools=[tool_config],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True  # We handle function calling ourselves
            ),
        )

        # Use async client
        response = await self._client.aio.models.generate_content(
            model=self.model_name,
            contents=gemini_content,
            config=config,
        )

        # Log token usage
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            self.logger.info(
                f"Token usage - Prompt: {usage.prompt_token_count}, "
                f"Response: {usage.candidates_token_count}, "
                f"Total: {usage.total_token_count}"
            )

        return gemini_response_to_message(response)
