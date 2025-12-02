"""DoDo - An AI workflow automation engine.

Build intelligent automation workflows with LLM agents.
Browser automation, RPA, testing, and more.

Two simple methods:
- do(task, output_schema) - Execute workflow steps with LLM reasoning
- redo(run) - Replay recorded workflows without LLM

Example:
    >>> from dodo import Agent, Tool
    >>>
    >>> agent = Agent(llm=my_llm, tools=[...], observe=my_context_fn)
    >>> run = await agent.do("perform some task")
    >>> print(run.output)      # Structured output
    >>> print(run.feedback)    # Brief summary
    >>> print(run.action_log)  # Detailed execution trace
"""

from .agent import Agent
from .exceptions import TaskAbortedError
from .tools import Tool, tool
from .runner import Run, TaskResult, TaskStatus, MemoryConfig
from .llm import (
    LLM,
    Gemini,
    Message,
    Role,
    Content,
    Text,
    Image,
    ToolResult,
    ToolResultStatus,
    ToolCall,
)
from .prompts import DEFAULT_SYSTEM_PROMPT

__version__ = "0.1.4"

__all__ = [
    # Core
    "Agent",
    "Tool",
    "tool",
    "TaskAbortedError",
    "MemoryConfig",
    # Results
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
    "ToolCall",
    # Message types
    "Message",
    "Role",
    # Prompts
    "DEFAULT_SYSTEM_PROMPT",
]
