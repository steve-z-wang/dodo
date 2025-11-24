"""Unit tests for the @tool decorator."""

import pytest
from dodo import tool, Tool
from dodo.llm import ToolResult, ToolResultStatus


class TestToolDecoratorFunction:
    """Tests for @tool decorator on functions."""

    def test_basic_function(self):
        """Test basic function decoration."""

        @tool
        async def calculator(expression: str) -> str:
            """Calculate a math expression.

            Args:
                expression: Math expression to evaluate
            """
            return str(eval(expression))

        assert calculator.name == "calculator"
        assert calculator.description == "Calculate a math expression."
        assert hasattr(calculator, "Params")
        assert hasattr(calculator, "execute")

    def test_function_params_model(self):
        """Test that params model is generated correctly."""

        @tool
        async def greet(name: str, greeting: str = "Hello") -> str:
            """Greet someone.

            Args:
                name: Person's name
                greeting: Greeting to use
            """
            return f"{greeting}, {name}!"

        # Check params model
        params = greet.Params(name="Alice")
        assert params.name == "Alice"
        assert params.greeting == "Hello"

        params2 = greet.Params(name="Bob", greeting="Hi")
        assert params2.greeting == "Hi"

    @pytest.mark.asyncio
    async def test_function_execute_success(self):
        """Test successful function execution."""

        @tool
        async def add(a: int, b: int) -> str:
            """Add two numbers.

            Args:
                a: First number
                b: Second number
            """
            return str(a + b)

        params = add.Params(a=2, b=3)
        result = await add.execute(params)

        assert isinstance(result, ToolResult)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.description == "5"

    @pytest.mark.asyncio
    async def test_function_execute_error(self):
        """Test function execution with error."""

        @tool
        async def divide(a: int, b: int) -> str:
            """Divide two numbers.

            Args:
                a: Numerator
                b: Denominator
            """
            return str(a / b)

        params = divide.Params(a=10, b=0)
        result = await divide.execute(params)

        assert result.status == ToolResultStatus.ERROR
        assert "division by zero" in result.error

    def test_missing_description_raises_error(self):
        """Test that missing docstring raises error."""

        with pytest.raises(ValueError, match="missing a docstring"):

            @tool
            async def no_docs(x: int) -> str:
                return str(x)

    def test_missing_param_description_raises_error(self):
        """Test that missing param description raises error."""

        with pytest.raises(ValueError, match="missing a description"):

            @tool
            async def missing_param_desc(x: int) -> str:
                """Do something."""
                return str(x)

    def test_require_descriptions_false(self):
        """Test that require_descriptions=False allows missing descriptions."""

        @tool(require_descriptions=False)
        async def lenient(x: int) -> str:
            """Do something."""
            return str(x)

        assert lenient.name == "lenient"


class TestToolDecoratorClass:
    """Tests for @tool decorator on classes."""

    def test_basic_class(self):
        """Test basic class decoration."""

        @tool
        class SearchTool:
            """Search the database."""

            def __init__(self, database):
                self.database = database

            async def run(self, query: str) -> str:
                """
                Args:
                    query: Search query
                """
                return f"Results for: {query}"

        # Create instance
        instance = SearchTool(database="mock_db")

        assert instance.name == "search_tool"
        assert instance.description == "Search the database."
        assert hasattr(instance, "Params")
        assert hasattr(instance, "execute")

    def test_class_with_dependencies(self):
        """Test class with dependencies in __init__."""

        @tool
        class ApiTool:
            """Call an API."""

            def __init__(self, api_key: str, base_url: str):
                self.api_key = api_key
                self.base_url = base_url

            async def run(self, endpoint: str) -> str:
                """
                Args:
                    endpoint: API endpoint to call
                """
                return f"Called {self.base_url}/{endpoint}"

        instance = ApiTool(api_key="secret", base_url="https://api.example.com")
        assert instance.api_key == "secret"
        assert instance.base_url == "https://api.example.com"

    @pytest.mark.asyncio
    async def test_class_execute_success(self):
        """Test successful class execution."""

        @tool
        class Multiplier:
            """Multiply by a factor."""

            def __init__(self, factor: int):
                self.factor = factor

            async def run(self, value: int) -> str:
                """
                Args:
                    value: Value to multiply
                """
                return str(value * self.factor)

        instance = Multiplier(factor=3)
        params = instance.Params(value=7)
        result = await instance.execute(params)

        assert result.status == ToolResultStatus.SUCCESS
        assert result.description == "21"

    @pytest.mark.asyncio
    async def test_class_execute_error(self):
        """Test class execution with error."""

        @tool
        class Failer:
            """A tool that fails."""

            async def run(self, should_fail: bool) -> str:
                """
                Args:
                    should_fail: Whether to fail
                """
                if should_fail:
                    raise RuntimeError("Intentional failure")
                return "success"

        instance = Failer()
        params = instance.Params(should_fail=True)
        result = await instance.execute(params)

        assert result.status == ToolResultStatus.ERROR
        assert "Intentional failure" in result.error

    def test_class_missing_run_method_raises_error(self):
        """Test that class without run method raises error."""

        with pytest.raises(ValueError, match="must have a 'run' method"):

            @tool
            class NoRunMethod:
                """A tool without run."""

                async def execute(self, x: int) -> str:
                    return str(x)

    def test_class_missing_description_raises_error(self):
        """Test that class without docstring raises error."""

        with pytest.raises(ValueError, match="missing a docstring"):

            @tool
            class NoDocstring:
                async def run(self, x: int) -> str:
                    """
                    Args:
                        x: A number
                    """
                    return str(x)


class TestCamelToSnake:
    """Tests for CamelCase to snake_case conversion."""

    def test_simple_conversion(self):
        """Test simple CamelCase conversion."""
        from dodo.tools.decorator import _camel_to_snake

        assert _camel_to_snake("SearchTool") == "search_tool"
        assert _camel_to_snake("APIClient") == "api_client"
        assert _camel_to_snake("HTTPServer") == "http_server"
        assert _camel_to_snake("MyClass") == "my_class"
