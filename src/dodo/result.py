"""Public result types returned by Agent methods."""

from dataclasses import dataclass


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
