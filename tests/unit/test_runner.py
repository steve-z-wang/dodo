"""Tests for TaskRunner."""

import pytest
from typing import List
from pydantic import BaseModel, Field

from dodo.runner import TaskRunner, TaskStatus, MemoryConfig
from dodo.llm import LLM, Message, Role, Text, ToolResult, ToolResultStatus, ToolCall
from dodo.tools import Tool


# =============================================================================
# Helpers
# =============================================================================


async def no_observation():
    """Empty observation callback."""
    return []


# =============================================================================
# Mock LLM
# =============================================================================


class MockLLM(LLM):
    """Mock LLM that returns predefined responses."""

    def __init__(self, responses: List[Message]):
        self.responses = responses
        self.call_count = 0
        self.received_messages: List[List[Message]] = []

    async def call_tools(self, messages: List[Message], tools: List[Tool]) -> Message:
        self.received_messages.append(messages)
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


# =============================================================================
# Mock Tools
# =============================================================================


class AddTool(Tool):
    """Simple tool that adds two numbers."""

    name = "add"
    description = "Add two numbers"

    class Params(BaseModel):
        a: int = Field(description="First number")
        b: int = Field(description="Second number")

    async def execute(self, params: Params) -> ToolResult:
        result = params.a + params.b
        return ToolResult(
            name=self.name,
            status=ToolResultStatus.SUCCESS,
            description=f"Added {params.a} + {params.b} = {result}",
        )


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.asyncio
async def test_runner_completes_task():
    """Test that runner completes when complete_work is called."""
    # LLM calls complete_work on first iteration
    llm = MockLLM(
        responses=[
            Message(
                role=Role.MODEL,
                content=[
                    Text(text="I'll complete the task"),
                    ToolCall(
                        id="1",
                        name="complete_work",
                        arguments={"feedback": "Task done successfully"},
                    )
                ],
            )
        ]
    )

    runner = TaskRunner(
        llm=llm,
        tools=[],
        observe=no_observation,
    )

    run = await runner.run(task="Do something", max_iterations=10)

    assert run._result.status == TaskStatus.COMPLETED
    assert run._result.feedback == "Task done successfully"
    assert run.steps_used == 1


@pytest.mark.asyncio
async def test_runner_aborts_task():
    """Test that runner aborts when abort_work is called."""
    llm = MockLLM(
        responses=[
            Message(
                role=Role.MODEL,
                content=[
                    Text(text="I cannot do this"),
                    ToolCall(
                        id="1",
                        name="abort_work",
                        arguments={"reason": "Impossible task"},
                    )
                ],
            )
        ]
    )

    runner = TaskRunner(
        llm=llm,
        tools=[],
        observe=no_observation,
    )

    run = await runner.run(task="Do impossible thing", max_iterations=10)

    assert run._result.status == TaskStatus.ABORTED
    assert run._result.feedback == "Impossible task"
    assert run.steps_used == 1


@pytest.mark.asyncio
async def test_runner_max_iterations():
    """Test that runner aborts when max iterations reached."""
    # LLM never calls complete_work or abort_work
    llm = MockLLM(
        responses=[
            Message(
                role=Role.MODEL,
                content=[
                    Text(text="Still working..."),
                    ToolCall(id=str(i), name="add", arguments={"a": 1, "b": 2})
                ],
            )
            for i in range(5)
        ]
    )

    runner = TaskRunner(
        llm=llm,
        tools=[AddTool()],
        observe=no_observation,
    )

    run = await runner.run(task="Keep adding", max_iterations=3)

    assert run._result.status == TaskStatus.ABORTED
    assert run._result.feedback == "Reached maximum iterations"
    assert run.steps_used == 3


@pytest.mark.asyncio
async def test_runner_uses_tools():
    """Test that runner executes tools and passes results to LLM."""
    llm = MockLLM(
        responses=[
            Message(
                role=Role.MODEL,
                content=[
                    Text(text="Let me add those"),
                    ToolCall(id="1", name="add", arguments={"a": 5, "b": 3})
                ],
            ),
            Message(
                role=Role.MODEL,
                content=[
                    Text(text="Done!"),
                    ToolCall(
                        id="2",
                        name="complete_work",
                        arguments={"feedback": "Added 5 + 3 = 8"},
                    )
                ],
            ),
        ]
    )

    runner = TaskRunner(
        llm=llm,
        tools=[AddTool()],
        observe=no_observation,
    )

    run = await runner.run(task="Add 5 and 3", max_iterations=10)

    assert run._result.status == TaskStatus.COMPLETED
    assert run.steps_used == 2
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_runner_with_observation():
    """Test that observe callback results are included in messages."""
    observations = [Text(text="Current state: ready")]

    llm = MockLLM(
        responses=[
            Message(
                role=Role.MODEL,
                content=[
                    ToolCall(
                        id="1",
                        name="complete_work",
                        arguments={"feedback": "Observed state"},
                    )
                ],
            )
        ]
    )

    async def observe():
        return observations

    runner = TaskRunner(
        llm=llm,
        tools=[],
        observe=observe,
    )

    run = await runner.run(task="Check state", max_iterations=10)

    # Verify observation was passed to LLM
    assert llm.call_count == 1
    messages = llm.received_messages[0]
    # Find UserMessage content
    user_msg = messages[1]  # Second message is UserMessage
    assert any(
        isinstance(c, Text) and "Current state: ready" in c.text
        for c in user_msg.content
    )


@pytest.mark.asyncio
async def test_runner_with_output_schema():
    """Test that runner handles structured output."""

    class OutputData(BaseModel):
        value: int

    llm = MockLLM(
        responses=[
            Message(
                role=Role.MODEL,
                content=[
                    ToolCall(
                        id="1",
                        name="complete_work",
                        arguments={
                            "feedback": "Extracted value",
                            "output": {"value": 42},
                        },
                    )
                ],
            )
        ]
    )

    runner = TaskRunner(
        llm=llm,
        tools=[],
        observe=no_observation,
    )

    run = await runner.run(
        task="Extract value",
        max_iterations=10,
        output_schema=OutputData,
    )

    assert run._result.status == TaskStatus.COMPLETED
    assert run._result.output.value == 42


@pytest.mark.asyncio
async def test_runner_memory_compacting():
    """Test that old iterations are compacted into summary when exceeding recent_window."""
    # Run 5 iterations with recent_window=2
    # Iterations 1-3 should be summarized, 4-5 should be in full detail
    llm = MockLLM(
        responses=[
            Message(
                role=Role.MODEL,
                content=[
                    Text(text=f"Iteration {i}"),
                    ToolCall(id=str(i), name="add", arguments={"a": i, "b": 1})
                ],
            )
            for i in range(1, 5)
        ]
        + [
            Message(
                role=Role.MODEL,
                content=[
                    ToolCall(
                        id="5",
                        name="complete_work",
                        arguments={"feedback": "Done after 5 iterations"},
                    )
                ],
            )
        ]
    )

    runner = TaskRunner(
        llm=llm,
        tools=[AddTool()],
        observe=no_observation,
        memory=MemoryConfig(recent_window=2),
    )

    run = await runner.run(task="Do 5 iterations", max_iterations=10)

    assert run._result.status == TaskStatus.COMPLETED
    assert run.steps_used == 5

    # Check the last LLM call (iteration 5)
    # It should have a summary of old iterations + full recent iterations
    last_messages = llm.received_messages[-1]

    # Find the summary message (should contain "Previous actions")
    summary_found = False
    for msg in last_messages:
        if msg.content:
            for content in msg.content:
                if isinstance(content, Text) and "Previous actions" in content.text:
                    summary_found = True
                    # Should mention the old iterations
                    assert "Added" in content.text
                    break

    assert summary_found, "Summary of old iterations should be present"


@pytest.mark.asyncio
async def test_runner_content_lifespan():
    """Test that content with lifespan is filtered out after N iterations.

    Lifespan filtering only applies to iteration pairs, not session_start.
    With lifespan=1, content should only appear in the current iteration.
    """
    iteration_count = 0

    async def observe():
        nonlocal iteration_count
        iteration_count += 1
        return [
            # This content should only appear in current iteration
            Text(text=f"Ephemeral observation {iteration_count}", lifespan=1),
            # This content should persist
            Text(text=f"Persistent observation {iteration_count}"),
        ]

    llm = MockLLM(
        responses=[
            Message(
                role=Role.MODEL,
                content=[
                    Text(text="Step 1"),
                    ToolCall(id="1", name="add", arguments={"a": 1, "b": 1})
                ],
            ),
            Message(
                role=Role.MODEL,
                content=[
                    Text(text="Step 2"),
                    ToolCall(id="2", name="add", arguments={"a": 2, "b": 2})
                ],
            ),
            Message(
                role=Role.MODEL,
                content=[
                    ToolCall(
                        id="3",
                        name="complete_work",
                        arguments={"feedback": "Done"},
                    )
                ],
            ),
        ]
    )

    runner = TaskRunner(
        llm=llm,
        tools=[AddTool()],
        observe=observe,
    )

    run = await runner.run(task="Test lifespan", max_iterations=10)

    assert run._result.status == TaskStatus.COMPLETED
    assert run.steps_used == 3

    # Check iteration 3's messages (last LLM call)
    # Timeline:
    # - session_start: observe() #1 -> ephemeral 1, persistent 1 (NOT filtered)
    # - iteration 1: observe() #2 -> ephemeral 2, persistent 2 (in pairs[0])
    # - iteration 2: observe() #3 -> ephemeral 3, persistent 3 (in pairs[1])
    # - iteration 3: _prepare_messages filters pairs:
    #   - pairs[0] distance=1: ephemeral 2 filtered OUT (lifespan=1, 1 >= 1)
    #   - pairs[1] distance=0: ephemeral 3 kept (lifespan=1, 0 < 1)
    last_messages = llm.received_messages[-1]

    # Collect all text content
    all_text = []
    for msg in last_messages:
        if msg.content:
            for content in msg.content:
                if isinstance(content, Text):
                    all_text.append(content.text)

    all_text_joined = " ".join(all_text)

    # Ephemeral 1 is in session_start (not filtered)
    assert "Ephemeral observation 1" in all_text_joined

    # Ephemeral 2 should be filtered out (distance=1 >= lifespan=1)
    assert "Ephemeral observation 2" not in all_text_joined

    # Ephemeral 3 should be present (distance=0 < lifespan=1)
    assert "Ephemeral observation 3" in all_text_joined

    # All persistent observations should be present
    assert "Persistent observation 1" in all_text_joined
    assert "Persistent observation 2" in all_text_joined
    assert "Persistent observation 3" in all_text_joined
