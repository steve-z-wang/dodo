"""Run - tracks task execution with conversation history."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Any
from enum import Enum


class TaskStatus(str, Enum):
    """Task execution status."""

    COMPLETED = "completed"
    ABORTED = "aborted"


@dataclass
class TaskResult:
    """Internal task execution result with status.

    Used by control tools to signal task completion or abortion.
    """

    status: Optional[TaskStatus] = None
    output: Optional[Any] = None
    feedback: Optional[str] = None

    @property
    def is_completed(self) -> bool:
        return self.status == TaskStatus.COMPLETED if self.status else False

    @property
    def is_aborted(self) -> bool:
        return self.status == TaskStatus.ABORTED if self.status else False

    def __str__(self) -> str:
        status_str = self.status.value if self.status else "pending"
        return f"TaskResult(status={status_str}, output={self.output is not None})"


@dataclass
class Run:
    """Task execution run - full execution history with embedded result.

    Captures everything about a single task execution:
    - The task description
    - The result (status, output, feedback)
    - Summary of actions taken
    - Full message history
    - Step counts
    """

    result: TaskResult
    summary: str
    messages: List = field(default_factory=list)

    task_description: str = ""
    steps_used: int = 0
    max_steps: int = 0

    def __str__(self) -> str:
        status = self.result.status.value if self.result.status else "pending"
        return f"Run(task='{self.task_description}', steps={self.steps_used}/{self.max_steps}, status={status})"
