"""Tool base class for agent tools."""

from abc import ABC, abstractmethod
from pydantic import BaseModel

from dodo.llm.content import ToolResult, ToolResultStatus


class Tool(ABC):
    """Base class for agent tools.

    Tools define actions that an agent can perform. Each tool must specify:
    - name: str - Unique tool identifier
    - description: str - What the tool does (shown to LLM)
    - Params: BaseModel - Nested Pydantic model for parameters
    - async execute(params: Params) -> ToolResult

    Example:
        class GreetTool(Tool):
            name = "greet"
            description = "Greet a person by name"

            class Params(BaseModel):
                name: str = Field(description="Name of person to greet")

            async def execute(self, params: Params) -> ToolResult:
                greeting = f"Hello, {params.name}!"
                return ToolResult(
                    name=self.name,
                    status=ToolResultStatus.SUCCESS,
                    description=greeting,
                )
    """

    name: str
    description: str
    Params: type[BaseModel]

    @abstractmethod
    async def execute(self, params: BaseModel) -> ToolResult:
        """Execute the tool with validated parameters.

        Args:
            params: Validated Params instance

        Returns:
            ToolResult with status, description, and optional terminal flag
        """
        pass
