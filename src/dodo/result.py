"""Public result types returned by Agent methods."""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class Result:
    """Result returned by agent.do().

    Attributes:
        output: Optional structured output data (if output_schema was provided)
        feedback: Brief description of what was accomplished
    """

    output: Optional[Any] = None
    feedback: Optional[str] = None

    def __str__(self) -> str:
        if self.output is not None:
            return f"Result(output={self.output}, feedback='{self.feedback}')"
        return f"Result(feedback='{self.feedback}')"


@dataclass
class Verdict:
    """Result returned by agent.check().

    Attributes:
        passed: True if the condition was met, False otherwise
        reason: Explanation of why the condition passed or failed
    """

    passed: bool
    reason: str = ""

    def __bool__(self) -> bool:
        """Allow using Verdict directly in if statements."""
        return self.passed

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        return f"Verdict({status}: {self.reason})"
