"""TaskRunner - executes tasks with conversation-based LLM."""

from typing import Awaitable, Callable, List, Optional, Tuple, Type, TYPE_CHECKING
import logging
from pydantic import BaseModel

from dodo.llm import (
    Message,
    SystemMessage,
    UserMessage,
    ModelMessage,
    Content,
    Text,
    ToolResult,
)
from dodo.tools import Tool, ToolRegistry, CompleteWorkTool, AbortWorkTool
from .run import Run, TaskResult, TaskStatus
from .memory import MemoryConfig
from dodo.prompts import DEFAULT_SYSTEM_PROMPT

if TYPE_CHECKING:
    from dodo.llm import LLM

# Type alias for message pairs (ModelMessage + UserMessage with tool results/observations)
MessagePair = Tuple[ModelMessage, UserMessage]


class TaskRunner:
    """Task executor - runs a single task with conversation-based LLM.

    Accepts tools and an observe callback from outside.
    Creates control tools (complete_work, abort_work) internally.
    """

    def __init__(
        self,
        llm: "LLM",
        tools: List[Tool],
        observe: Callable[[], Awaitable[List[Content]]],
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory: Optional[MemoryConfig] = None,
    ):
        """Initialize TaskRunner.

        Args:
            llm: LLM instance for making tool calls
            tools: List of domain tools (e.g., browser tools, API tools)
            observe: Async callback that returns current context as Content list
            system_prompt: System prompt for the agent
            memory: Memory configuration for history management
        """
        self._llm = llm
        self._tools = tools
        self._observe = observe
        self._system_prompt = system_prompt
        self._memory = memory or MemoryConfig()
        self._logger = logging.getLogger(__name__)

    async def run(
        self,
        task: str,
        max_iterations: int,
        previous_runs: Optional[List[Run]] = None,
        output_schema: Optional[Type[BaseModel]] = None,
    ) -> Run:
        """Execute a task until completion or max iterations.

        Args:
            task: Task description
            max_iterations: Maximum number of iterations
            previous_runs: Optional list of previous runs for context
            output_schema: Optional Pydantic model for structured output

        Returns:
            Run object with result and execution history
        """
        result = TaskResult()
        tool_registry = self._setup_tools(result, output_schema)

        session_start_messages = await self._build_session_start_messages(
            task, previous_runs
        )

        self._logger.info(f"Task start - Task: {task}")

        pairs: List[MessagePair] = []
        for iteration in range(max_iterations):
            self._logger.info(f"Iteration {iteration + 1} - Start")

            all_messages = self._prepare_messages(session_start_messages, pairs)

            self._logger.debug("Sending LLM request...")
            model_msg = await self._llm.call_tools(
                messages=all_messages,
                tools=tool_registry.get_all(),
            )

            reasoning = self._extract_reasoning(model_msg)
            tool_names = (
                [tc.name for tc in model_msg.tool_calls]
                if model_msg.tool_calls
                else []
            )

            self._logger.info(f"LLM response - Tools: {tool_names}")
            if reasoning:
                self._logger.info(f"Reasoning: {reasoning}")

            tool_results = await tool_registry.execute_tool_calls(
                model_msg.tool_calls or []
            )

            # Get context after tool execution via observe callback
            observation = await self._observe()

            # Combine tool results and observation into UserMessage content
            response_content: List[Content] = list(tool_results) + list(observation)
            response_msg = UserMessage(content=response_content)
            pairs.append((model_msg, response_msg))

            self._logger.info(f"Iteration {iteration + 1} - End")

            # Check if control tool ended execution
            if result.status:
                iterations_used = iteration + 1
                break
        else:
            self._logger.info("Task end - Reason: max_iterations_reached")
            result.status = TaskStatus.ABORTED
            result.feedback = "Reached maximum iterations"
            iterations_used = max_iterations

        self._logger.info(f"Task end - Status: {result.status.value}")

        summary = self._build_summary(pairs)

        messages: List[Message] = []
        for model_msg, response_msg in pairs:
            messages.append(model_msg)
            messages.append(response_msg)

        return Run(
            result=result,
            summary=summary,
            messages=messages,
            task_description=task,
            steps_used=iterations_used,
            max_steps=max_iterations,
        )

    def _setup_tools(
        self,
        result: TaskResult,
        output_schema: Optional[Type[BaseModel]],
    ) -> ToolRegistry:
        """Create and configure tool registry for this run."""
        tool_registry = ToolRegistry()

        # Register domain tools (injected from outside)
        for tool in self._tools:
            tool_registry.register(tool)

        # Register control tools (created internally)
        tool_registry.register(CompleteWorkTool(result, output_schema))
        tool_registry.register(AbortWorkTool(result))

        return tool_registry

    async def _build_session_start_messages(
        self,
        task: str,
        previous_runs: Optional[List[Run]] = None,
    ) -> List[Message]:
        """Build initial messages for the session."""
        user_content: List[Content] = []

        # Add previous runs if provided (for stateful agents)
        if previous_runs:
            formatted = self._format_previous_runs(previous_runs)
            user_content.append(Text(text=formatted))

        user_content.append(Text(text=f"## Current task:\n{task}"))

        # Add initial observation via observe callback
        observation = await self._observe()
        user_content.extend(observation)

        return [
            SystemMessage(content=[Text(text=self._system_prompt)]),
            UserMessage(content=user_content),
        ]

    def _format_previous_runs(self, runs: List[Run]) -> str:
        """Format previous runs for context."""
        lines = ["## Previous tasks:", ""]
        for i, run in enumerate(runs, 1):
            lines.append(f"### Task {i}")
            lines.append(f"Task: {run.task_description}")
            lines.append(f"Status: {run.result.status.value.capitalize()}")
            if run.result.feedback:
                lines.append(f"Feedback: {run.result.feedback}")
            lines.append("")
        return "\n".join(lines)

    def _build_summary(self, pairs: List[MessagePair]) -> str:
        """Build a summary of actions taken."""
        if not pairs:
            return ""

        lines = []
        for model_msg, response_msg in pairs:
            reasoning = self._extract_reasoning(model_msg)

            if reasoning:
                reasoning = reasoning.strip()
                if "\n" not in reasoning:
                    lines.append(f"- {reasoning}")
                else:
                    reasoning_lines = reasoning.split("\n")
                    lines.append(f"- {reasoning_lines[0]}")
                    for line in reasoning_lines[1:]:
                        lines.append(f"  {line}")

            # Extract tool results from response message content
            if response_msg.content:
                for content in response_msg.content:
                    if isinstance(content, ToolResult):
                        if content.status.value == "error":
                            lines.append(f"  - {content.description} [FAILED: {content.error}]")
                        else:
                            lines.append(f"  - {content.description}")

        return "\n".join(lines)

    def _prepare_messages(
        self, session_start_messages: List[Message], pairs: List[MessagePair]
    ) -> List[Message]:
        """Prepare messages for LLM call with history management."""
        recent_window = self._memory.recent_window

        tool_messages: List[Message] = []

        # If many pairs, summarize old ones
        if len(pairs) > recent_window:
            old_pairs = pairs[:-recent_window]
            summary = self._build_summary(old_pairs)
            if summary:
                tool_messages.append(
                    UserMessage(
                        content=[
                            Text(
                                text=f"Previous actions in this session:\n{summary}"
                            )
                        ]
                    )
                )

        # Add recent pairs with content filtering based on lifespan
        recent_pairs = (
            pairs[-recent_window:] if len(pairs) > recent_window else pairs
        )
        num_recent = len(recent_pairs)

        for idx, (model_msg, response_msg) in enumerate(recent_pairs):
            tool_messages.append(model_msg)

            # Filter content based on lifespan
            # idx=0 is oldest in recent_pairs, idx=num_recent-1 is newest
            # distance_from_end: 0 = current iteration, 1 = previous, etc.
            distance_from_end = num_recent - 1 - idx
            filtered_msg = self._filter_content_by_lifespan(
                response_msg, distance_from_end
            )
            tool_messages.append(filtered_msg)

        return session_start_messages + tool_messages

    def _filter_content_by_lifespan(
        self, msg: UserMessage, distance_from_end: int
    ) -> UserMessage:
        """Filter content based on lifespan.

        Args:
            msg: UserMessage to filter
            distance_from_end: 0 = current iteration, 1 = previous, etc.

        Returns:
            New UserMessage with filtered content
        """
        if not msg.content:
            return msg

        filtered_content = []
        for content in msg.content:
            # If no lifespan set, keep the content
            if content.lifespan is None:
                filtered_content.append(content)
            # If within lifespan, keep it
            elif distance_from_end < content.lifespan:
                filtered_content.append(content)
            # Otherwise, skip (content is too old for its lifespan)

        # Return new message with filtered content
        return UserMessage(
            timestamp=msg.timestamp,
            content=filtered_content if filtered_content else None,
        )

    def _extract_reasoning(self, model_msg: ModelMessage) -> Optional[str]:
        """Extract text reasoning from model message."""
        if model_msg.content:
            for content in model_msg.content:
                if hasattr(content, "text") and content.text:
                    return content.text
        return None
