"""RedoRunner - Replay actions from a previous run without LLM reasoning."""

from typing import List, Callable, Awaitable

from dodo.tools import Tool
from dodo.llm import Content, ToolCall, Role
from .run import Run


class RedoRunner:
    """Replays actions from a previous run without LLM reasoning.

    Executes the exact sequence of tool calls from a run, skipping all
    LLM reasoning. Useful for repeating successful workflows cheaply.
    """

    def __init__(self, tools: List[Tool], observe: Callable[[], Awaitable[List[Content]]]):
        """Initialize redo runner.

        Args:
            tools: List of tools available for execution
            observe: Observation callback (not used during replay)
        """
        self.tools = tools
        self.observe = observe

        # Build tool lookup map
        self._tool_map = {tool.name: tool for tool in tools}

    async def replay(self, run: Run) -> None:
        """Replay actions from a previous run.

        Extracts tool calls from the run's message history and executes
        them in sequence without any LLM reasoning.

        Args:
            run: Previous run to replay

        Raises:
            ValueError: If a tool is not found
            Exception: If any tool execution fails
        """
        # Extract all tool calls from messages
        tool_calls = self._extract_tool_calls(run)

        if not tool_calls:
            return  # Nothing to replay

        # Execute each tool call in sequence
        for tool_call in tool_calls:
            await self._execute_tool_call(tool_call)

    def _extract_tool_calls(self, run: Run) -> List[ToolCall]:
        """Extract tool calls from run messages.

        Args:
            run: Run to extract from

        Returns:
            List of tool calls in execution order
        """
        tool_calls = []

        for message in run.messages:
            if message.role == Role.MODEL and message.tool_calls:
                tool_calls.extend(message.tool_calls)

        return tool_calls

    async def _execute_tool_call(self, tool_call: ToolCall) -> None:
        """Execute a single tool call.

        Args:
            tool_call: Tool call to execute

        Raises:
            ValueError: If tool not found
            Exception: If tool execution fails
        """
        # Find tool
        tool = self._tool_map.get(tool_call.name)
        if not tool:
            raise ValueError(f"Tool '{tool_call.name}' not found in tool registry")

        # Parse arguments to params model
        params = tool.Params.model_validate(tool_call.arguments)

        # Execute tool
        result = await tool.execute(params)

        # Check for errors
        if result.status.value == "error":
            raise RuntimeError(f"Tool '{tool_call.name}' failed: {result.error}")
