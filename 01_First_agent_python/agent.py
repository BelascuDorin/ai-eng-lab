"""NutriChef: a raw agent loop. No framework, just the Anthropic SDK.

Run:  uv run agent.py
"""

import json

import anthropic
from dotenv import load_dotenv

from tools import TOOL_SCHEMAS, run_tool

load_dotenv()  # reads ANTHROPIC_API_KEY from .env
client = anthropic.Anthropic()

MODEL = "claude-haiku-4-5"  # cheapest current model: $1 / $5 per 1M input/output tokens
MAX_TURNS = 10  # safety net so a confused model can't loop forever (and burn credit)
VERBOSE = False  # True = dump the full raw JSON of every API response

SYSTEM = """You are NutriChef, a meal-ordering assistant.
Use the tools to look up real menu and macro data - never guess nutrition numbers.
Never place an order unless the user has explicitly confirmed it."""


def run_agent(messages: list[dict]) -> str:
    """Run the agent loop until the model gives a final answer. Returns that answer's text.

    `messages` is the full conversation so far. Mutate it in place (append to it)
    so the REPL below keeps history across user turns.
    """
    for turn in range(MAX_TURNS):
        print(f"\n  ── turn {turn} ── sending {len(messages)} messages to {MODEL}")

        # 1. Send the whole history + tool definitions. The API is stateless:
        #    it only knows what's in `messages` on THIS call.
        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        print(f"  ◀ stop_reason={response.stop_reason}  "
              f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens}")
        if VERBOSE:
            print(response.model_dump_json(indent=2))

        # Show what the model said/asked for this turn, block by block.
        for block in response.content:
            if block.type == "text":
                print(f"  💬 {block.text}")
            elif block.type == "tool_use":
                print(f"  📝 model requests: {block.name}({block.input})  id={block.id}")

        # 2. Append the FULL content (text + tool_use blocks), not just the text.
        #    The tool_result we send next must reference a tool_use.id that exists in history.
        messages.append({"role": "assistant", "content": response.content})

        # 3. Stop conditions. stop_reason tells us WHY the model stopped generating:
        #    "end_turn"   -> it's done, this is the final answer
        #    "tool_use"   -> it wants us to run tools and send results back (keep looping)
        #    "max_tokens" -> it got cut off mid-answer because max_tokens was too low
        #    "refusal"    -> safety classifiers declined the request
        if response.stop_reason != "tool_use":
            final_text = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "max_tokens":
                final_text += "\n(⚠️ cut off: hit max_tokens)"
            elif response.stop_reason == "refusal":
                final_text = final_text or "(model refused this request)"
            return final_text

        # 4. The model wants tools. It can ask for several in one reply (parallel tool use),
        #    so run every tool_use block, not just the first one.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  🔧 running {block.name}({block.input})")
            try:
                result = run_tool(block.name, block.input)
                is_error = False
            except Exception as e:
                # Don't crash the agent - tell the model what went wrong so it can recover
                # (retry, pick another meal, or explain to the user).
                result = f"Error: {type(e).__name__}: {e}"
                is_error = True
            print(f"     {'❌' if is_error else '✅'} {result[:150]}{'...' if len(result) > 150 else ''}")

            # tool_use_id links this result to the exact request it answers.
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
                "is_error": is_error,
            })

        # 5. Send ALL results back as ONE user message. Tool results come from "the user's side"
        #    because, from the model's point of view, everything outside it is the user.
        messages.append({"role": "user", "content": tool_results})
        print(f"  ▶ sending {len(tool_results)} tool_result(s) back")

    return "(stopped: hit MAX_TURNS)"


def main():
    print("NutriChef 🥗  (type 'quit' to exit)\n")
    messages: list[dict] = []
    while True:
        user_input = input("you> ").strip()
        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if not user_input:
            continue
        messages.append({"role": "user", "content": user_input})
        answer = run_agent(messages)
        print(f"\nchef> {answer}\n")


if __name__ == "__main__":
    main()
