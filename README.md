# DoDo

A stateful agentic framework for building AI agents that can perform tasks, retrieve information, and verify conditions.

## Features

- **Stateful**: Maintains conversation history across calls
- **Simple API**: Three intuitive methods - `do`, `tell`, `check`
- **Generic**: Works with any tools - web automation, APIs, file systems, etc.
- **LLM Agnostic**: Bring your own LLM implementation

## Installation

```bash
pip install dodo-agent
```

## Quick Start

```python
from dodo import Agent, Tool, LLM

# Create your LLM implementation
class MyLLM(LLM):
    async def call_tools(self, messages, tools):
        # Your LLM API call here
        pass

# Create your tools
class GreetTool(Tool):
    name = "greet"
    description = "Greet someone"

    class Params(BaseModel):
        name: str

    async def execute(self, params):
        return ToolResult(
            name=self.name,
            status=ToolResultStatus.SUCCESS,
            description=f"Hello, {params.name}!",
        )

# Create an observe function that returns current context
async def observe():
    return [TextContent(text="Current state: ready")]

# Create agent
agent = Agent(
    llm=MyLLM(),
    tools=[GreetTool()],
    observe=observe,
)

# Use the agent
result = await agent.do("greet the user named Alice")
print(result.feedback)
```

## API

### `agent.do(task, max_iterations=20, output_schema=None)`

Do a task. Can be simple (1-2 iterations) or complex (many iterations).

```python
# Simple task
result = await agent.do("click the login button")

# Complex task
result = await agent.do("fill out the registration form and submit")

# With structured output
class Price(BaseModel):
    amount: float
    currency: str

result = await agent.do("find the total price", output_schema=Price)
print(result.output.amount)
```

### `agent.tell(what, schema=None, max_iterations=10)`

Tell me something. Returns the information directly.

```python
# Simple retrieval
username = await agent.tell("the logged in username")

# With structured output
class UserInfo(BaseModel):
    name: str
    email: str

user = await agent.tell("the current user info", schema=UserInfo)
print(user.name, user.email)
```

### `agent.check(condition, max_iterations=10)`

Check if a condition is true. Returns a `Verdict`.

```python
ok = await agent.check("user is logged in")
if ok:
    print("User is logged in")
else:
    print(f"Not logged in: {ok.reason}")
```

## Stateful Conversations

DoDo agents remember previous tasks:

```python
await agent.do("go to amazon.com")
await agent.do("search for laptop")      # Remembers previous navigation
await agent.do("add first result to cart")  # Knows search was done

# Reset history if needed
agent.reset()
```

## Creating Custom Tools

```python
from dodo import Tool
from dodo.llm import ToolResult, ToolResultStatus
from pydantic import BaseModel, Field

class SearchTool(Tool):
    name = "search"
    description = "Search for information"

    class Params(BaseModel):
        query: str = Field(description="Search query")

    async def execute(self, params):
        # Your search logic here
        results = do_search(params.query)
        return ToolResult(
            name=self.name,
            status=ToolResultStatus.SUCCESS,
            description=f"Found {len(results)} results for '{params.query}'",
        )
```

## Custom System Prompt

```python
agent = Agent(
    llm=my_llm,
    tools=my_tools,
    observe=my_observe,
    system_prompt="You are a helpful assistant that...",
)
```

## License

MIT
