# 01 · NutriChef: a raw agent loop

A tiny meal-ordering CLI agent written with **no framework**, only the Anthropic Python SDK.
The goal is to see exactly what frameworks hide: message history, tool-call parsing, and stop conditions.

## How it works

- [`tools.py`](tools.py) has a fake backend with three tools (`get_menu`, `get_macros`, `create_order`) and the JSON schemas the model sees.
- [`agent.py`](agent.py) has the agent loop:
  1. Send the full message history + tool definitions to the model
  2. Append the model's full reply (text + `tool_use` blocks) to history
  3. If `stop_reason != "tool_use"`, return the final answer
  4. Otherwise, run every requested tool (errors go back as `is_error: true`)
  5. Send all `tool_result` blocks back in one user message, then repeat

## Run it

```bash
cp .env.example .env   # then add your ANTHROPIC_API_KEY
uv run agent.py
```

Try: *"I want 150g protein today under 2000 kcal, what should I order?"*
