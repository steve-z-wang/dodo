"""Memory configuration for agent history management."""

from pydantic import BaseModel, Field


class MemoryConfig(BaseModel):
    """Memory configuration for agent history.

    The agent keeps recent iterations in full detail (with images, context, etc.)
    and summarizes older iterations into a text list of actions taken.

    Example:
        >>> config = MemoryConfig(recent_window=5)
        >>> agent = Agent(llm=llm, tools=tools, observe=observe, memory=config)
    """

    recent_window: int = Field(
        default=5,
        description="Number of recent iterations to keep in full detail. "
        "Older iterations are compacted into a summary.",
        ge=1,
    )

    # Future: content lifespan configuration
    # content_retention: List[ContentRetention] = []
