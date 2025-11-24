"""Tool registry for managing and executing agent tools."""

import logging
from typing import Dict, List

from dodo.tool import Tool
from dodo.llm.message import ToolCall
from dodo.content import ToolResult, ToolResultStatus


class ToolRegistry:
    """Registry for managing and executing agent tools.

    Handles tool registration, lookup, and batch execution of tool calls.
    Stops execution early if a tool fails or returns terminal=True.
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._logger = logging.getLogger(__name__)

    def register(self, tool: Tool) -> None:
        """Register a tool. Replaces existing tool if name exists."""
        if not hasattr(tool, "name"):
            raise ValueError("Tool must have 'name' attribute")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        """Get a tool by name."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def clear(self) -> None:
        """Clear all registered tools."""
        self._tools.clear()

    async def execute_tool_calls(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls, stopping on error or terminal tool.

        Args:
            tool_calls: List of tool calls from LLM

        Returns:
            List of ToolResult (includes skipped results for remaining calls)
        """
        results = []
        executed_count = 0

        for idx, tool_call in enumerate(tool_calls):
            try:
                # Get tool
                try:
                    tool = self.get(tool_call.name)
                except KeyError:
                    error_msg = f"Tool '{tool_call.name}' not found in registry"
                    self._logger.error(error_msg)
                    result = ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        status=ToolResultStatus.ERROR,
                        error=error_msg,
                        description=f"{tool_call.name} (ERROR: Tool not found)",
                    )
                    results.append(result)
                    executed_count = idx + 1
                    break

                # Validate parameters
                params = tool.Params(**tool_call.arguments)

                self._logger.info(
                    f"Executing tool: {tool_call.name} with params: {tool_call.arguments}"
                )

                # Execute tool
                result = await tool.execute(params)
                result.tool_call_id = tool_call.id
                results.append(result)
                executed_count = idx + 1

                self._logger.info(f"Tool executed: {result.description}")

                # Stop if terminal or error
                if result.terminal or result.status == ToolResultStatus.ERROR:
                    break

            except Exception as e:
                error_msg = str(e)
                self._logger.error(
                    f"Tool execution failed: {tool_call.name} - {error_msg}"
                )
                result = ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    status=ToolResultStatus.ERROR,
                    error=error_msg,
                    description=f"{tool_call.name} (ERROR: {error_msg})",
                )
                results.append(result)
                executed_count = idx + 1
                break

        # Create skipped results for remaining tool calls
        for tool_call in tool_calls[executed_count:]:
            result = ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                status=ToolResultStatus.SKIPPED,
                description=f"{tool_call.name} (SKIPPED)",
            )
            results.append(result)
            self._logger.info(f"Tool skipped: {tool_call.name}")

        return results
