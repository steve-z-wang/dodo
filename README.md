# dodo

[![PyPI version](https://img.shields.io/pypi/v/dodoai.svg)](https://pypi.org/project/dodoai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Lightweight agent framework with tools and memory.

## Installation

```bash
pip install dodoai
pip install dodoai[gemini]  # With Gemini support
```

## Overview

```python
from dodo import Agent, Gemini

agent = Agent(llm=Gemini(), tools=[add_todo, complete_todo])

await agent.do("add buy milk to my list")
await agent.do("mark buy milk as done")

ok = await agent.check("all tasks are completed")
if ok:
    print("Success!")
```

## Features

**Stateful agents**

```python
# Agent remembers context across tasks
await agent.do("calculate 25 * 4")
await agent.do("add 10 to the result")
await agent.do("multiply by 2")
```

**Structured output** (using [Pydantic](https://docs.pydantic.dev/))

```python
from pydantic import BaseModel

class Result(BaseModel):
    value: float
    expression: str

result = await agent.tell("the calculation result", schema=Result)
```

**Verdicts with reasons**

```python
ok = await agent.check("result is greater than 100")
if ok:
    print("Success!")
else:
    print(f"Failed: {ok.reason}")
```

## Usage

**1. Create tools**

Simple tools using `@tool` decorator:

```python
from dodo import tool

@tool
async def calculator(expression: str) -> str:
    """Perform arithmetic calculations.

    Args:
        expression: Math expression to evaluate
    """
    return str(eval(expression))
```

Tools with dependencies using classes:

```python
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
        return self.database.search(query)
```

**2. Define observation function**

```python
async def observe():
    """Return current environment state as a list of strings or Content objects."""
    return [f"Current user: {username}", f"History: {history}"]
```

**3. Create agent and run tasks**

```python
from dodo import Agent, Gemini

agent = Agent(llm=Gemini(), tools=[calculator], observe=observe)

await agent.do("calculate 25 * 4 + 10")            # Do a task
result = await agent.tell("the last calculation")  # Get information
ok = await agent.check("result is greater than 100")  # Check a condition
```

**4. Error handling**

```python
try:
    await agent.do("divide 10 by 0")
except TaskAbortedError as e:
    print(f"Task failed: {e}")
```

## Supported LLMs

```python
from dodo import Gemini

Gemini(model="gemini-2.5-flash")  # Default
Gemini(model="gemini-2.5-pro")    # Pro model
```

You can also implement your own LLM by extending the `LLM` base class:

```python
from dodo import LLM

class MyLLM(LLM):
    async def call_tools(self, messages, tools):
        # Your LLM API call here
        pass
```

## License

MIT
