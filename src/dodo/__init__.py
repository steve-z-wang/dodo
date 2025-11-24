"""DoDo - A stateful agentic framework.

DoDo provides a simple, generic framework for building stateful AI agents
that can perform tasks, retrieve information, and verify conditions.

Three simple methods:
- do(task) - Do a task (simple or complex)
- tell(what) - Tell me something (retrieve information)
- check(condition) - Check if something is true

Example:
    >>> from dodo import Agent, Tool
    >>>
    >>> agent = Agent(llm=my_llm, tools=[...], observe=my_context_fn)
    >>> result = await agent.do("perform some task")
    >>> value = await agent.tell("some information")
    >>> ok = await agent.check("some condition is true")
"""

from .agent import Agent
from .exceptions import TaskAbortedError
from .tool import Tool
from .result import Result, Verdict
from .run import Run, TaskResult, TaskStatus
from .llm import (
    LLM,
    Message,
    Content,
    TextContent,
    ImageContent,
    ToolResultContent,
    SystemMessage,
    UserMessage,
    ModelMessage,
)
from .prompts import DEFAULT_SYSTEM_PROMPT
from .memory import MemoryConfig

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "Tool",
    "TaskAbortedError",
    "MemoryConfig",
    # Results
    "Result",
    "Verdict",
    "Run",
    "TaskResult",
    "TaskStatus",
    # LLM
    "LLM",
    # Content types
    "Message",
    "Content",
    "TextContent",
    "ImageContent",
    "ToolResultContent",
    # Message types (3 only: System, User, Model)
    "SystemMessage",
    "UserMessage",
    "ModelMessage",
    # Prompts
    "DEFAULT_SYSTEM_PROMPT",
]
