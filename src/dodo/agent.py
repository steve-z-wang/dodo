"""Agent - main interface for DoDo agentic framework."""

import logging
from typing import Awaitable, Callable, List, Optional, Type, Any

from pydantic import BaseModel, Field

from dodo.llm import LLM, Content
from dodo.tools import Tool
from dodo.runner import TaskRunner, Run, TaskStatus, MemoryConfig
from dodo.result import Result, Verdict
from dodo.prompts import DEFAULT_SYSTEM_PROMPT
from dodo.exceptions import TaskAbortedError


class Agent:
    """Main agent interface for DoDo.

    A stateful agent that can perform tasks, retrieve information, and verify conditions.
    Maintains conversation history across calls when stateful=True.

    Example:
        >>> agent = Agent(
        ...     llm=my_llm,
        ...     tools=[MyTool()],
        ...     observe=my_context_fn,
        ... )
        >>> result = await agent.do("perform some task")
        >>> value = await agent.tell("some information")
        >>> ok = await agent.check("some condition")
    """

    def __init__(
        self,
        llm: LLM,
        tools: List[Tool],
        observe: Callable[[], Awaitable[List[Content]]],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        stateful: bool = True,
        memory: Optional[MemoryConfig] = None,
    ):
        """Initialize agent.

        Args:
            llm: LLM instance for reasoning and task execution
            tools: List of tools the agent can use
            observe: Async callback that returns current context as Content list
            system_prompt: System prompt for the agent
            stateful: If True, maintain conversation history between calls (default: True)
            memory: Memory configuration for history management (default: MemoryConfig())
        """
        self.llm = llm
        self.tools = tools
        self.observe = observe
        self.system_prompt = system_prompt
        self.stateful = stateful
        self.memory = memory or MemoryConfig()
        self.logger = logging.getLogger(__name__)

        # Store previous runs if stateful=True
        self._previous_runs: List[Run] = []

    async def do(
        self,
        task: str,
        max_iterations: int = 20,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Result:
        """Do a task.

        Execute a task using the available tools. The task can be simple (1-2 iterations)
        or complex (many iterations). The agent decides when it's complete.

        Args:
            task: Task description in natural language
            max_iterations: Maximum iterations before aborting (default: 20)
            output_schema: Optional Pydantic model for structured output

        Returns:
            Result with feedback and optional structured output

        Raises:
            TaskAbortedError: If task is aborted by the agent
        """
        run = await self._run_task(
            task=task,
            max_iterations=max_iterations,
            output_schema=output_schema,
        )

        return Result(output=run.result.output, feedback=run.result.feedback)

    async def tell(
        self,
        what: str,
        schema: Optional[Type[BaseModel]] = None,
        max_iterations: int = 10,
    ) -> Any:
        """Tell me something.

        Retrieve information from the current context. Returns the value directly.

        Args:
            what: What to retrieve in natural language (e.g., "the total price")
            schema: Optional Pydantic model for structured output
            max_iterations: Maximum iterations (default: 10)

        Returns:
            The requested information (str if no schema, otherwise schema instance)

        Raises:
            TaskAbortedError: If retrieval fails
        """

        # Default to str schema if none provided
        class StrOutput(BaseModel):
            """String output."""

            value: str = Field(description=f"The requested information: {what}")

        effective_schema = schema or StrOutput
        task = f"Find and return the following information: {what}"

        run = await self._run_task(
            task=task,
            max_iterations=max_iterations,
            output_schema=effective_schema,
        )

        # Return value directly
        if schema:
            return run.result.output
        else:
            return run.result.output.value if run.result.output else ""

    async def check(
        self,
        condition: str,
        max_iterations: int = 10,
    ) -> Verdict:
        """Check if a condition is true.

        Verify a condition on the current context.

        Args:
            condition: Condition to check in natural language (e.g., "user is logged in")
            max_iterations: Maximum iterations (default: 10)

        Returns:
            Verdict with passed (bool) and reason (str)

        Raises:
            TaskAbortedError: If check fails
        """

        class CheckResult(BaseModel):
            """Check result."""

            passed: bool = Field(
                description="True if the condition is met, False otherwise"
            )

        task = f"Check if the following condition is true: {condition}"

        run = await self._run_task(
            task=task,
            max_iterations=max_iterations,
            output_schema=CheckResult,
        )

        if not run.result.output:
            raise RuntimeError("Check failed: no structured output received")

        return Verdict(
            passed=run.result.output.passed,
            reason=run.result.feedback or "",
        )

    async def _run_task(
        self,
        task: str,
        max_iterations: int,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Run:
        """Internal method to run a task.

        Args:
            task: Task description
            max_iterations: Maximum iterations
            output_schema: Optional output schema

        Returns:
            Run object with result

        Raises:
            TaskAbortedError: If task is aborted
        """
        task_runner = TaskRunner(
            llm=self.llm,
            tools=self.tools,
            observe=self.observe,
            system_prompt=self.system_prompt,
            memory=self.memory,
        )

        run = await task_runner.run(
            task=task,
            max_iterations=max_iterations,
            previous_runs=self._previous_runs if self.stateful else None,
            output_schema=output_schema,
        )

        # Store run for stateful behavior
        if self.stateful:
            self._previous_runs.append(run)

        # Raise if aborted
        if run.result.status == TaskStatus.ABORTED:
            raise TaskAbortedError(run.result.feedback or "Task aborted")

        return run

    def reset(self) -> None:
        """Reset conversation history.

        Clears all previous runs, starting fresh for the next task.
        """
        self._previous_runs.clear()
        self.logger.info("Agent history reset")
