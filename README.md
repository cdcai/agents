# Agents
Sean Browning (NCEZID/OD Data Science Team)

## Background

I wanted to learn how to create and use language agents to solve complex problems. LangChain wasn't cutting it for me, so I made my own library from first principles to suit a few projects we're working on.

This package contains a few classes that can be used as building blocks for language agents and agentic systems. I plan to expand it with additional functionality as I need it, but keep a minimal footprint (ie. if you're already using openai and pydantic, this should bring no additional dependencies).

All code uses asyncio by design, and though I've tried to generalize as I can, I mostly built around OpenAI and specifically Azure OpenAI since that's what we are allowed to work with internally.

## Installation

Not currently on pypi, so just use pip to install via GitHub:

```sh
pip install git+https://github.com/cdcai/multiagent.git
```

## Examples

| Example | Link |
| ---- | ---- |
| Taking output from one agent as input to another in a callback | [agent_with_callback.py](examples/agent_with_callback.py) |
| Getting structured output from agent / Text Prediction | [structured_prediction.py](examples/structured_prediction.py) |
| Batch processing large inputs over the same agent in parallel | [batch_processing.py](examples/batch_processing.py) |
