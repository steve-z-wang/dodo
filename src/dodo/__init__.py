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
from .tools import Tool, tool
from .runner import Run, TaskResult, TaskStatus, MemoryConfig
from .result import Result, Verdict
from .llm import (
    LLM,
    Gemini,
    Message,
    Content,
    Text,
    Image,
    ToolResult,
    ToolResultStatus,
    SystemMessage,
    UserMessage,
    ModelMessage,
)
from .prompts import DEFAULT_SYSTEM_PROMPT

__version__ = "0.1.1"

__all__ = [
    # Core
    "Agent",
    "Tool",
    "tool",
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
    "Gemini",
    # Content types
    "Content",
    "Text",
    "Image",
    "ToolResult",
    "ToolResultStatus",
    # Message types
    "Message",
    "SystemMessage",
    "UserMessage",
    "ModelMessage",
    # Prompts
    "DEFAULT_SYSTEM_PROMPT",
]
