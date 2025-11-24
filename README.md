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

agent = Agent(llm=Gemini(), tools=[Calculator()], observe=lambda: [])

await agent.do("calculate 25 * 4 + 10")
result = await agent.tell("the result")
ok = await agent.check("result is greater than 100")
```

## Features

**Stateful agents**

```python
# Agent remembers context across tasks
await agent.do("calculate 25 * 4")
await agent.do("add 10 to the result")
await agent.do("multiply by 2")
```

**Structured output**

```python
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

```python
from dodo import Tool, ToolResult, ToolResultStatus
from pydantic import BaseModel, Field

class CalculatorTool(Tool):
    name = "calculator"
    description = "Perform arithmetic calculations"

    class Params(BaseModel):
        expression: str = Field(description="Math expression to evaluate")

    async def execute(self, params):
        result = eval(params.expression)
        return ToolResult(
            name=self.name,
            status=ToolResultStatus.SUCCESS,
            description=f"Result: {result}",
        )
```

**2. Define observation function**

```python
async def observe():
    """Return current environment state."""
    return [Text(text=f"Memory: {memory}")]
```

**3. Create agent and run tasks**

```python
from dodo import Agent, Gemini

agent = Agent(llm=Gemini(), tools=[CalculatorTool()], observe=observe)

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
