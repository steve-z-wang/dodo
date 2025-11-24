"""Control tools for task flow management."""

from typing import Any, Optional, Type, TYPE_CHECKING
from pydantic import BaseModel, Field, create_model

from dodo.tool import Tool
from dodo.content import ToolResult, ToolResultStatus

if TYPE_CHECKING:
    from dodo.run import TaskResult


class CompleteWorkTool(Tool):
    """Signal that the agent has successfully completed the task.

    Optionally accepts structured output data to return to the caller.
    """

    name = "complete_work"
    description = "Signal that you have successfully completed the task. Optionally provide structured output data."

    class Params(BaseModel):
        """Parameters for complete_work tool."""

        feedback: str = Field(
            description="Brief 1-2 sentence summary of what you accomplished"
        )
        output: Optional[Any] = Field(
            default=None,
            description="Optional structured data to return (e.g., extracted information)",
        )

    def __init__(
        self, task_result: "TaskResult", output_schema: Optional[Type[BaseModel]] = None
    ):
        """Initialize with reference to task result and optional output schema.

        Args:
            task_result: TaskResult object to store completion status
            output_schema: Optional Pydantic model for structured output
        """
        from dodo.run import TaskStatus

        self.task_result = task_result
        self._TaskStatus = TaskStatus

        # Dynamically create Params class if output_schema is provided
        if output_schema:
            self.Params = create_model(  # type: ignore[misc]
                "CompleteWorkParams",
                __base__=CompleteWorkTool.Params,
                output=(
                    Optional[output_schema],
                    Field(
                        default=None,
                        description="Structured output data matching the specified schema",
                    ),
                ),
            )

    async def execute(self, params: Params) -> ToolResult:
        """Signal task completion and store feedback and output."""
        self.task_result.status = self._TaskStatus.COMPLETED
        self.task_result.feedback = params.feedback
        if params.output is not None:
            self.task_result.output = params.output

        desc = f"Completed: {params.feedback}"
        if params.output is not None:
            try:
                import json

                output_str = json.dumps(params.output, indent=2, ensure_ascii=False)
                desc += f"\nOutput data:\n{output_str}"
            except (TypeError, ValueError):
                desc += f"\nOutput data: {params.output}"

        return ToolResult(
            name=self.name,
            status=ToolResultStatus.SUCCESS,
            description=desc,
            terminal=True,
        )


class AbortWorkTool(Tool):
    """Signal that the agent cannot proceed further with the task."""

    name = "abort_work"
    description = "Signal that you cannot proceed (stuck, blocked, error, or impossible)"

    class Params(BaseModel):
        """Parameters for abort_work tool."""

        reason: str = Field(
            description="Explain why you cannot continue and what went wrong"
        )

    def __init__(self, task_result: "TaskResult"):
        """Initialize with reference to task result."""
        from dodo.run import TaskStatus

        self.task_result = task_result
        self._TaskStatus = TaskStatus

    async def execute(self, params: Params) -> ToolResult:
        """Signal task abortion and store reason."""
        self.task_result.status = self._TaskStatus.ABORTED
        self.task_result.feedback = params.reason

        return ToolResult(
            name=self.name,
            status=ToolResultStatus.SUCCESS,
            description=f"Aborted: {params.reason}",
            terminal=True,
        )
