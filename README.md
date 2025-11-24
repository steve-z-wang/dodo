# dodo

[![PyPI version](https://img.shields.io/pypi/v/dodoai.svg)](https://pypi.org/project/dodoai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A stateful agentic framework for building AI agents.

## Installation

```bash
pip install dodoai
pip install dodoai[gemini]  # With Gemini support
```

## Quick Start

```python
from dodo import Agent, Gemini

agent = Agent(llm=Gemini(), tools=[...], observe=my_observe_fn)

await agent.do("perform some task")
value = await agent.tell("some information")
ok = await agent.check("some condition is true")
```

## Features

**Three simple methods**

```python
await agent.do("fill out the form and submit")     # Do a task
username = await agent.tell("the logged in user")  # Get information
ok = await agent.check("user is logged in")        # Check a condition
```

**Stateful agents**

```python
# Agent remembers context across tasks
await agent.do("go to amazon.com")
await agent.do("search for laptop")
await agent.do("add first result to cart")
```

**Structured output**

```python
class ProductInfo(BaseModel):
    name: str
    price: float

product = await agent.tell("product information", schema=ProductInfo)
```

**Verdicts with reasons**

```python
ok = await agent.check("cart has 3 items")
if ok:
    print("Success!")
else:
    print(f"Failed: {ok.reason}")
```

**Observation function**

```python
async def observe():
    """Return current environment state."""
    return [Text(text="Current page: checkout")]

agent = Agent(llm=llm, tools=tools, observe=observe)
```

**Error handling**

```python
try:
    await agent.do("complete checkout")
except TaskAbortedError as e:
    print(f"Task failed: {e}")
```

## Supported LLMs

```python
from dodo import Gemini

Gemini(model="gemini-2.5-flash")  # Default
Gemini(model="gemini-2.5-pro")    # Pro model
```

## Creating Tools

```python
from dodo import Tool, ToolResult, ToolResultStatus
from pydantic import BaseModel, Field

class SearchTool(Tool):
    name = "search"
    description = "Search for information"

    class Params(BaseModel):
        query: str = Field(description="Search query")

    async def execute(self, params):
        results = do_search(params.query)
        return ToolResult(
            name=self.name,
            status=ToolResultStatus.SUCCESS,
            description=f"Found {len(results)} results",
        )
```

## License

MIT
