"""Integration tests for Gemini LLM provider.

These tests require a valid GEMINI_API_KEY environment variable.
They will be skipped if the API key is not set.
"""

import os
import pytest
from pydantic import BaseModel, Field

from dodo.llm import (
    Gemini,
    Message,
    Role,
    Text,
    ToolResult,
    ToolResultStatus,
    ToolCall,
)
from dodo.tools import Tool


# Skip all tests if GEMINI_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="GEMINI_API_KEY environment variable not set",
)


# --- Test Tools ---


class CalculatorTool(Tool):
    """Simple calculator tool for testing."""

    name = "calculator"
    description = "Perform basic arithmetic calculations"

    class Params(BaseModel):
        expression: str = Field(description="Math expression to evaluate (e.g., '2 + 2')")

    async def execute(self, params: Params) -> ToolResult:
        try:
            # Simple eval for basic math (safe for testing)
            result = eval(params.expression, {"__builtins__": {}}, {})
            return ToolResult(
                name=self.name,
                status=ToolResultStatus.SUCCESS,
                description=f"Result: {result}",
            )
        except Exception as e:
            return ToolResult(
                name=self.name,
                status=ToolResultStatus.ERROR,
                error=str(e),
            )


class GreetTool(Tool):
    """Simple greeting tool for testing."""

    name = "greet"
    description = "Greet a person by name"

    class Params(BaseModel):
        name: str = Field(description="Name of person to greet")

    async def execute(self, params: Params) -> ToolResult:
        return ToolResult(
            name=self.name,
            status=ToolResultStatus.SUCCESS,
            description=f"Hello, {params.name}!",
        )


# --- Integration Tests ---


class TestGeminiIntegration:
    """Integration tests for Gemini LLM provider."""

    @pytest.fixture
    def gemini(self):
        """Create Gemini client."""
        return Gemini(model="gemini-2.5-flash", temperature=0.0)

    @pytest.mark.asyncio
    async def test_simple_response(self, gemini):
        """Test that Gemini returns a response."""
        messages = [
            Message(role=Role.SYSTEM, content=[Text(text="You are a helpful assistant. Be concise.")]),
            Message(role=Role.USER, content=[Text(text="What is 2 + 2? Reply with just the number.")]),
        ]

        result = await gemini.call_tools(messages, tools=[])

        assert isinstance(result, Message)
        assert result.role == Role.MODEL
        assert result.text is not None
        assert "4" in result.text

    @pytest.mark.asyncio
    async def test_tool_calling(self, gemini):
        """Test that Gemini can call tools."""
        messages = [
            Message(
                role=Role.SYSTEM,
                content=[Text(text="You are a helpful assistant. Use tools when needed.")]
            ),
            Message(role=Role.USER, content=[Text(text="Please greet Alice.")]),
        ]

        result = await gemini.call_tools(messages, tools=[GreetTool()])

        assert isinstance(result, Message)
        assert result.role == Role.MODEL
        assert result.tool_calls is not None
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0].name == "greet"
        assert result.tool_calls[0].arguments.get("name") == "Alice"

    @pytest.mark.asyncio
    async def test_tool_result_handling(self, gemini):
        """Test that Gemini can process tool results."""
        messages = [
            Message(
                role=Role.SYSTEM,
                content=[Text(text="You are a helpful assistant. Use tools when needed.")]
            ),
            Message(role=Role.USER, content=[Text(text="What is 5 * 7?")]),
            Message(
                role=Role.MODEL,
                content=[
                    Text(text="I'll calculate this."),
                    ToolCall(name="calculator", arguments={"expression": "5 * 7"})
                ],
            ),
            Message(
                role=Role.USER,
                content=[
                    ToolResult(
                        name="calculator",
                        status=ToolResultStatus.SUCCESS,
                        description="Result: 35",
                    )
                ]
            ),
        ]

        result = await gemini.call_tools(messages, tools=[CalculatorTool()])

        assert isinstance(result, Message)
        assert result.role == Role.MODEL
        assert result.text is not None
        assert "35" in result.text

    @pytest.mark.asyncio
    async def test_multiple_tools_available(self, gemini):
        """Test that Gemini can choose from multiple tools."""
        messages = [
            Message(
                role=Role.SYSTEM,
                content=[Text(text="You are a helpful assistant. Use the appropriate tool.")]
            ),
            Message(role=Role.USER, content=[Text(text="Calculate 10 + 20.")]),
        ]

        result = await gemini.call_tools(
            messages, tools=[GreetTool(), CalculatorTool()]
        )

        assert isinstance(result, Message)
        assert result.role == Role.MODEL
        # Should choose calculator, not greet
        if result.tool_calls:
            assert result.tool_calls[0].name == "calculator"

    @pytest.mark.asyncio
    async def test_text_captured(self, gemini):
        """Test that model text response is captured."""
        messages = [
            Message(
                role=Role.SYSTEM,
                content=[Text(text="You are a helpful assistant. Be concise.")]
            ),
            Message(role=Role.USER, content=[Text(text="Say hello.")]),
        ]

        # No tools provided - model must respond with text
        result = await gemini.call_tools(messages, tools=[])

        assert isinstance(result, Message)
        assert result.role == Role.MODEL
        # Without tools, model must produce text
        assert result.text is not None
        assert len(result.text) > 0
