# dodo

[![PyPI version](https://img.shields.io/pypi/v/dodoai.svg)](https://pypi.org/project/dodoai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI workflow automation engine for browser automation, RPA, testing, and more.

Built for frameworks like [webtask](https://github.com/steve-z-wang/webtask).

## Installation

```bash
pip install dodoai
pip install dodoai[gemini]  # With Gemini support
```

## Overview

```python
from dodo import Agent, Gemini

agent = Agent(llm=Gemini(), tools=[...])

# First time - LLM figures out the workflow
run = await agent.do("add item to cart")

# Later - replay without LLM (fast & cheap)
await agent.redo(run)
```

## Core Concepts

**Two simple methods for workflow automation:**

- `do(task)` - Execute workflow steps with LLM reasoning
- `redo(run)` - Replay recorded workflows without LLM

## Features

### Stateful workflows

Agent remembers context across multiple steps:

```python
# Multi-step workflow with context
await agent.do("search for laptop on Amazon")
await agent.do("add the first result to cart")
await agent.do("proceed to checkout")
```

### Efficient replay

Record once, replay when environment is similar:

```python
# First time - LLM figures out the workflow (expensive)
run = await agent.do("add laptop to cart and checkout")

# Later - replay the same workflow (cheap)
await agent.redo(run)
```

### Structured output

Get typed results using Pydantic models:

```python
from pydantic import BaseModel

class ProductInfo(BaseModel):
    name: str
    price: float
    in_stock: bool

run = await agent.do("extract product information", output_schema=ProductInfo)
product = run.output  # Typed ProductInfo object
```

## Use Cases

**Browser automation:**
```python
# Automate web workflows
await agent.do("go to linkedin.com")
await agent.do("search for software engineers in SF")
await agent.do("message the first 5 results")
```

**RPA (Robotic Process Automation):**
```python
# Automate repetitive workflows
await agent.do("login to portal")
await agent.do("navigate to reports section")
await agent.do("download monthly report")
```

**Testing & QA:**
```python
# Record test cases, replay for regression
run = await agent.do("complete checkout flow")

# Regression test
await agent.redo(run)
```

## Building Workflows

**1. Define tools**

Tools are the actions your agent can perform:

```python
from dodo import tool

@tool
async def click_button(element_id: str) -> str:
    """Click a button on the page.

    Args:
        element_id (str): ID of button to click
    """
    # Your automation logic
    return "Button clicked"

@tool
class DatabaseTool:
    """Query the database."""

    def __init__(self, db_connection):
        self.db = db_connection

    async def run(self, query: str) -> str:
        """
        Args:
            query (str): SQL query to execute
        """
        return self.db.execute(query)
```

**2. Define observation**

Observation provides context to the agent:

```python
async def observe():
    """Return current state for the agent to see."""
    return [
        f"Current page: {browser.current_url}",
        f"Visible elements: {browser.get_elements()}",
    ]
```

**3. Create and run agent**

```python
from dodo import Agent, Gemini

agent = Agent(
    llm=Gemini(),
    tools=[click_button, DatabaseTool(db)],
    observe=observe
)

# Execute workflows
run = await agent.do("click login button and verify")
if run.output:
    print("Workflow succeeded!")
```

**4. Handle errors**

```python
from dodo import TaskAbortedError

try:
    await agent.do("impossible task")
except TaskAbortedError as e:
    print(f"Workflow aborted: {e}")
```

## Run Objects

Every `do()` call returns a Run object with full execution details:

```python
run = await agent.do("complete checkout")

run.output       # Structured output (if output_schema provided)
run.feedback     # Brief summary: "Checkout completed successfully"
run.action_log   # Detailed trace of all actions taken
run.messages     # Full conversation history
run.steps_used   # Number of LLM calls made
```

## Supported LLMs

```python
from dodo import Gemini

# Gemini models
agent = Agent(llm=Gemini(model="gemini-2.0-flash-exp"))
agent = Agent(llm=Gemini(model="gemini-2.5-pro"))
```

**Custom LLM:**

```python
from dodo import LLM

class MyLLM(LLM):
    async def call_tools(self, messages, tools):
        # Your LLM implementation
        pass
```

## Examples

- [webtask](https://github.com/steve-z-wang/webtask) - Browser automation framework built on dodo

## License

MIT
