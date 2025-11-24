"""Tool decorator for creating tools from functions or classes."""

import inspect
from typing import Any, Callable, Dict, Optional, Type, Union, get_type_hints, get_origin, get_args
from functools import wraps

from griffe import Docstring, DocstringSectionKind
from pydantic import BaseModel, Field, create_model

from dodo.tools.base import Tool
from dodo.llm.content import ToolResult, ToolResultStatus


def tool(
    fn_or_class: Union[Callable, Type, None] = None,
    *,
    require_descriptions: bool = True,
):
    """Decorator for creating tools from functions or classes.

    Works with both functions and classes:

    Function example:
        @tool
        async def calculator(expression: str) -> str:
            '''Calculate a math expression.

            Args:
                expression: Math expression to evaluate
            '''
            return str(eval(expression))

    Class example:
        @tool
        class SearchTool:
            '''Search the database.'''

            def __init__(self, database):
                self.database = database

            async def run(self, query: str) -> str:
                '''
                Args:
                    query: Search query
                '''
                return self.database.search(query)

    Args:
        fn_or_class: Function or class to decorate
        require_descriptions: If True, all params must have descriptions
    """
    def decorator(fn_or_class: Union[Callable, Type]):
        if inspect.isclass(fn_or_class):
            return _wrap_class(fn_or_class, require_descriptions)
        else:
            return _wrap_function(fn_or_class, require_descriptions)

    if fn_or_class is None:
        # Called with arguments: @tool(require_descriptions=False)
        return decorator
    # Called without arguments: @tool
    return decorator(fn_or_class)


def _parse_docstring(docstring: Optional[str]) -> tuple[Optional[str], Dict[str, str]]:
    """Parse docstring to extract description and param descriptions.

    Returns:
        Tuple of (description, {param_name: param_description})
    """
    if not docstring:
        return None, {}

    parsed = Docstring(docstring, parser="google")
    parsed.parse()

    description = None
    param_descriptions = {}

    for section in parsed.parsed:
        if section.kind == DocstringSectionKind.text:
            if description is None:
                description = section.value.strip()
        elif section.kind == DocstringSectionKind.parameters:
            for param in section.value:
                param_descriptions[param.name] = param.description or ""

    return description, param_descriptions


def _build_params_model(
    fn: Callable,
    param_descriptions: Dict[str, str],
    require_descriptions: bool,
) -> Type[BaseModel]:
    """Build a Pydantic model from function signature.

    Args:
        fn: Function to extract signature from
        param_descriptions: Dict of param name to description
        require_descriptions: If True, raise error for missing descriptions

    Returns:
        Dynamically created Pydantic model
    """
    sig = inspect.signature(fn)
    type_hints = get_type_hints(fn)

    fields = {}
    for param_name, param in sig.parameters.items():
        # Skip 'self' for methods
        if param_name == "self":
            continue

        # Get type hint, default to str
        param_type = type_hints.get(param_name, str)

        # Get description
        description = param_descriptions.get(param_name)
        if require_descriptions and not description:
            raise ValueError(
                f"Parameter '{param_name}' in '{fn.__name__}' is missing a description. "
                f"Add it to the docstring Args section or set require_descriptions=False."
            )

        # Handle default values
        if param.default is inspect.Parameter.empty:
            # Required field
            if description:
                fields[param_name] = (param_type, Field(description=description))
            else:
                fields[param_name] = (param_type, ...)
        else:
            # Optional field with default
            if description:
                fields[param_name] = (param_type, Field(default=param.default, description=description))
            else:
                fields[param_name] = (param_type, param.default)

    model_name = f"{fn.__name__}_Params"
    return create_model(model_name, **fields)


def _wrap_function(fn: Callable, require_descriptions: bool) -> Type[Tool]:
    """Wrap a function as a Tool class.

    Args:
        fn: Async function to wrap
        require_descriptions: If True, all params must have descriptions

    Returns:
        Tool class
    """
    # Parse docstring
    tool_description, param_descriptions = _parse_docstring(fn.__doc__)

    if not tool_description:
        raise ValueError(
            f"Function '{fn.__name__}' is missing a docstring description."
        )

    # Build params model
    ParamsModel = _build_params_model(fn, param_descriptions, require_descriptions)

    # Create tool class
    tool_name = fn.__name__

    class FunctionTool(Tool):
        name = tool_name
        description = tool_description
        Params = ParamsModel

        async def execute(self, params: BaseModel) -> ToolResult:
            try:
                result = await fn(**params.model_dump())
                return ToolResult(
                    name=self.name,
                    status=ToolResultStatus.SUCCESS,
                    description=str(result),
                )
            except Exception as e:
                return ToolResult(
                    name=self.name,
                    status=ToolResultStatus.ERROR,
                    error=str(e),
                )

    FunctionTool.__name__ = f"{tool_name}Tool"
    FunctionTool.__qualname__ = f"{tool_name}Tool"

    # Return an instance, not the class
    return FunctionTool()


def _wrap_class(cls: Type, require_descriptions: bool) -> Type:
    """Wrap a class as a Tool class.

    The class must have a `run` method.

    Args:
        cls: Class to wrap
        require_descriptions: If True, all params must have descriptions

    Returns:
        Modified class that extends Tool
    """
    # Check for run method
    if not hasattr(cls, "run"):
        raise ValueError(
            f"Class '{cls.__name__}' must have a 'run' method."
        )

    run_method = cls.run

    # Parse class docstring for tool description
    tool_description, _ = _parse_docstring(cls.__doc__)

    # Parse run method docstring for param descriptions
    _, param_descriptions = _parse_docstring(run_method.__doc__)

    # Can also get description from run method if class has none
    if not tool_description:
        tool_description, _ = _parse_docstring(run_method.__doc__)

    if not tool_description:
        raise ValueError(
            f"Class '{cls.__name__}' is missing a docstring description."
        )

    # Build params model from run method
    ParamsModel = _build_params_model(run_method, param_descriptions, require_descriptions)

    # Tool name from class name (convert CamelCase to snake_case)
    tool_name = _camel_to_snake(cls.__name__)

    # Store original __init__
    original_init = cls.__init__

    # Create new class that extends Tool
    class WrappedTool(cls, Tool):
        name = tool_name
        description = tool_description
        Params = ParamsModel

        def __init__(self, *args, **kwargs):
            Tool.__init__(self)
            original_init(self, *args, **kwargs)

        async def execute(self, params: BaseModel) -> ToolResult:
            try:
                result = await self.run(**params.model_dump())
                return ToolResult(
                    name=self.name,
                    status=ToolResultStatus.SUCCESS,
                    description=str(result),
                )
            except Exception as e:
                return ToolResult(
                    name=self.name,
                    status=ToolResultStatus.ERROR,
                    error=str(e),
                )

    WrappedTool.__name__ = cls.__name__
    WrappedTool.__qualname__ = cls.__qualname__

    return WrappedTool


def _camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    import re
    # Insert underscore before uppercase letters and lowercase them
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
