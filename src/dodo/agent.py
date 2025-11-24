"""Agent - main interface for DoDo agentic framework."""

import logging
from typing import Awaitable, Callable, List, Optional, Type, Any

from pydantic import BaseModel, Field

from dodo.llm import LLM, Content
from dodo.tools import Tool
from dodo.runner import TaskRunner, RedoRunner, Run, TaskStatus, MemoryConfig
from dodo.result import Verdict
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
        >>> run = await agent.do("perform some task")
        >>> print(run.output)      # Structured output
        >>> print(run.feedback)    # Brief summary
        >>> print(run.action_log)  # Detailed trace
        >>> value = await agent.tell("some information")
        >>> verdict = await agent.verify("some condition")
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
    ) -> Run:
        """Do a task.

        Execute a task using the available tools. The task can be simple (1-2 iterations)
        or complex (many iterations). The agent decides when it's complete.

        Args:
            task: Task description in natural language
            max_iterations: Maximum iterations before aborting (default: 20)
            output_schema: Optional Pydantic model for structured output

        Returns:
            Run object with full execution history (result, summary, messages, etc.)

        Raises:
            TaskAbortedError: If task is aborted by the agent
        """
        run = await self._run_task(
            task=task,
            max_iterations=max_iterations,
            output_schema=output_schema,
        )

        return run

    async def redo(self, run: Run) -> None:
        """Redo a previous run.

        Replays the exact sequence of tool calls from a previous run without
        any LLM reasoning. This is much faster and cheaper than do() but less
        flexible - it will fail if the page/state has changed.

        Use verify() afterwards to check if the replay succeeded:
            >>> await agent.redo(run)
            >>> verdict = await agent.verify("cart has 2 items")
            >>> if not verdict.passed:
            ...     # Fall back to intelligent do()
            ...     await agent.do("Add 2 items to cart")

        Args:
            run: Previous run to replay

        Raises:
            ValueError: If a tool from the run is not available
            Exception: If any tool execution fails
        """
        redo_runner = RedoRunner(self.tools, self.observe)
        await redo_runner.replay(run)

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
            return run.output
        else:
            return run.output.value if run.output else ""

    async def verify(
        self,
        condition: str,
        max_iterations: int = 10,
    ) -> Verdict:
        """Verify if a condition is true.

        Verify a condition on the current context.

        Args:
            condition: Condition to verify in natural language (e.g., "user is logged in")
            max_iterations: Maximum iterations (default: 10)

        Returns:
            Verdict with passed (bool) and reason (str)

        Raises:
            TaskAbortedError: If verification fails
        """

        class VerifyResult(BaseModel):
            """Verify result."""

            passed: bool = Field(
                description="True if the condition is met, False otherwise"
            )

        task = f"Verify if the following condition is true: {condition}"

        run = await self._run_task(
            task=task,
            max_iterations=max_iterations,
            output_schema=VerifyResult,
        )

        if not run.output:
            raise RuntimeError("Verification failed: no structured output received")

        return Verdict(
            passed=run.output.passed,
            reason=run.feedback or "",
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
        if run._result.status == TaskStatus.ABORTED:
            raise TaskAbortedError(run.feedback or "Task aborted")

        return run

    def reset(self) -> None:
        """Reset conversation history.

        Clears all previous runs, starting fresh for the next task.
        """
        self._previous_runs.clear()
        self.logger.info("Agent history reset")
