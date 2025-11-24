"""Unit tests for RedoRunner."""

import pytest
from unittest.mock import AsyncMock, Mock
from pydantic import BaseModel, Field

from dodo.runner import RedoRunner, Run, TaskResult, TaskStatus
from dodo.tools import Tool
from dodo.llm import ModelMessage, UserMessage, ToolCall, Text, ToolResult, ToolResultStatus


class TestParams(BaseModel):
    """Test tool parameters."""

    value: str = Field(description="Test value")


class TestTool(Tool):
    """Test tool for redo."""

    name = "test_tool"
    description = "Test tool"
    Params = TestParams

    def __init__(self):
        super().__init__()
        self.execute_mock = AsyncMock()

    async def execute(self, params: BaseModel):
        return await self.execute_mock(params)


@pytest.mark.asyncio
async def test_redo_basic_replay():
    """Test basic replay of tool calls."""
    # Setup tool
    tool = TestTool()
    tool.execute_mock.return_value = ToolResult(
        name="test_tool",
        status=ToolResultStatus.SUCCESS,
        description="Success",
    )

    # Create run with tool calls
    run = Run(
        _result=TaskResult(status=TaskStatus.COMPLETED, feedback="Done"),
        action_log="- test_tool(value='hello')",
        messages=[
            ModelMessage(
                thoughts="Testing",
                tool_calls=[
                    ToolCall(name="test_tool", arguments={"value": "hello"}),
                ],
            ),
            UserMessage(
                content=[
                    ToolResult(
                        name="test_tool",
                        status=ToolResultStatus.SUCCESS,
                        description="Success",
                    )
                ]
            ),
        ],
        task_description="Test task",
        steps_used=1,
        max_steps=10,
    )

    # Replay
    observe = AsyncMock(return_value=[Text(text="context")])
    redo_runner = RedoRunner(tools=[tool], observe=observe)
    await redo_runner.replay(run)

    # Verify tool was called
    assert tool.execute_mock.called
    call_args = tool.execute_mock.call_args[0][0]
    assert call_args.value == "hello"


@pytest.mark.asyncio
async def test_redo_multiple_tool_calls():
    """Test replay with multiple tool calls."""
    tool = TestTool()
    tool.execute_mock.return_value = ToolResult(
        name="test_tool",
        status=ToolResultStatus.SUCCESS,
        description="Success",
    )

    # Run with multiple tool calls
    run = Run(
        _result=TaskResult(status=TaskStatus.COMPLETED, feedback="Done"),
        action_log="- test_tool multiple times",
        messages=[
            ModelMessage(
                thoughts="First",
                tool_calls=[
                    ToolCall(name="test_tool", arguments={"value": "first"}),
                ],
            ),
            UserMessage(content=[ToolResult(name="test_tool", status=ToolResultStatus.SUCCESS)]),
            ModelMessage(
                thoughts="Second",
                tool_calls=[
                    ToolCall(name="test_tool", arguments={"value": "second"}),
                ],
            ),
            UserMessage(content=[ToolResult(name="test_tool", status=ToolResultStatus.SUCCESS)]),
        ],
        task_description="Test",
        steps_used=2,
        max_steps=10,
    )

    observe = AsyncMock(return_value=[])
    redo_runner = RedoRunner(tools=[tool], observe=observe)
    await redo_runner.replay(run)

    # Verify called twice
    assert tool.execute_mock.call_count == 2


@pytest.mark.asyncio
async def test_redo_tool_not_found():
    """Test error when tool not found."""
    tool = TestTool()

    run = Run(
        _result=TaskResult(status=TaskStatus.COMPLETED),
        action_log="",
        messages=[
            ModelMessage(
                thoughts="Test",
                tool_calls=[
                    ToolCall(name="unknown_tool", arguments={"value": "test"}),
                ],
            ),
        ],
        task_description="Test",
        steps_used=1,
        max_steps=10,
    )

    observe = AsyncMock(return_value=[])
    redo_runner = RedoRunner(tools=[tool], observe=observe)

    with pytest.raises(ValueError, match="Tool 'unknown_tool' not found"):
        await redo_runner.replay(run)


@pytest.mark.asyncio
async def test_redo_tool_execution_fails():
    """Test error when tool execution fails."""
    tool = TestTool()
    tool.execute_mock.return_value = ToolResult(
        name="test_tool",
        status=ToolResultStatus.ERROR,
        error="Something went wrong",
    )

    run = Run(
        _result=TaskResult(status=TaskStatus.COMPLETED),
        action_log="",
        messages=[
            ModelMessage(
                thoughts="Test",
                tool_calls=[
                    ToolCall(name="test_tool", arguments={"value": "test"}),
                ],
            ),
        ],
        task_description="Test",
        steps_used=1,
        max_steps=10,
    )

    observe = AsyncMock(return_value=[])
    redo_runner = RedoRunner(tools=[tool], observe=observe)

    with pytest.raises(RuntimeError, match="Tool 'test_tool' failed"):
        await redo_runner.replay(run)


@pytest.mark.asyncio
async def test_redo_empty_run():
    """Test replay of run with no tool calls."""
    tool = TestTool()

    run = Run(
        _result=TaskResult(status=TaskStatus.COMPLETED),
        action_log="",
        messages=[
            UserMessage(content=[Text(text="No tool calls")]),
        ],
        task_description="Test",
        steps_used=0,
        max_steps=10,
    )

    observe = AsyncMock(return_value=[])
    redo_runner = RedoRunner(tools=[tool], observe=observe)

    # Should not raise, just does nothing
    await redo_runner.replay(run)
    assert not tool.execute_mock.called


@pytest.mark.asyncio
async def test_redo_extract_tool_calls():
    """Test tool call extraction from messages."""
    tool = TestTool()

    run = Run(
        _result=TaskResult(status=TaskStatus.COMPLETED),
        action_log="",
        messages=[
            UserMessage(content=[Text(text="User message")]),  # No tool calls
            ModelMessage(
                thoughts="Model 1",
                tool_calls=[
                    ToolCall(name="test_tool", arguments={"value": "a"}),
                    ToolCall(name="test_tool", arguments={"value": "b"}),
                ],
            ),
            UserMessage(content=[ToolResult(name="test_tool", status=ToolResultStatus.SUCCESS)]),
            ModelMessage(thoughts="Model 2", tool_calls=None),  # No tool calls
            ModelMessage(
                thoughts="Model 3",
                tool_calls=[
                    ToolCall(name="test_tool", arguments={"value": "c"}),
                ],
            ),
        ],
        task_description="Test",
        steps_used=3,
        max_steps=10,
    )

    observe = AsyncMock(return_value=[])
    redo_runner = RedoRunner(tools=[tool], observe=observe)

    # Extract tool calls
    tool_calls = redo_runner._extract_tool_calls(run)

    # Should have 3 tool calls (a, b, c)
    assert len(tool_calls) == 3
    assert tool_calls[0].arguments["value"] == "a"
    assert tool_calls[1].arguments["value"] == "b"
    assert tool_calls[2].arguments["value"] == "c"
