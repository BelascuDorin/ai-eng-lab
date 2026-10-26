"""NutriChef: a raw agent loop. No framework, just the Anthropic SDK.

Run:  uv run agent.py
"""

import time

import anthropic
from dotenv import load_dotenv

from run_log import RunLogger
from tools import TOOL_SCHEMAS, run_tool

load_dotenv()  # reads ANTHROPIC_API_KEY from .env
# The SDK already retries 429 (rate limit), 5xx and connection errors with exponential backoff.
# Default is 2 retries; set it explicitly so it's a visible decision, not hidden magic.
client = anthropic.Anthropic(max_retries=3)

MODEL = "claude-haiku-4-5"  # cheapest current model: $1 / $5 per 1M input/output tokens
PRICE_PER_MTOK = {"input": 1.00, "output": 5.00}  # USD, for the cost estimate in the logs
MAX_TURNS = 10  # safety net so a confused model can't loop forever (and burn credit)
VERBOSE = False  # True = dump the full raw JSON of every API response

SYSTEM = """You are NutriChef, a meal-ordering assistant.
Use the tools to look up real menu and macro data - never guess nutrition numbers.
Never place an order unless the user has explicitly confirmed it."""


WRAP_UP_NOTE = ("[System note: you have no tool calls left for this request. Answer the user now "
                "using only the data you already have, and say clearly what you could not check.]")


def with_wrap_up_note(messages: list[dict]) -> list[dict]:
    """Copy of `messages` with WRAP_UP_NOTE appended as a text block to the last (user) message."""
    last = messages[-1]
    content = last["content"]
    if isinstance(content, str):  # plain user text (happens when max_turns=1)
        content = [{"type": "text", "text": content}]
    # Text must come AFTER the tool_result blocks in a user message - API rule.
    return messages[:-1] + [{**last, "content": content + [{"type": "text", "text": WRAP_UP_NOTE}]}]


def run_agent(messages: list[dict], max_turns: int = MAX_TURNS, metadata: dict | None = None) -> str:
    """Run the agent loop until the model gives a final answer. Returns that answer's text.

    `messages` is the full conversation so far. Mutate it in place (append to it)
    so the REPL below keeps history across user turns.
    `metadata` is copied into the log (e.g. {"scenario": "baseline"}) so you can filter runs later.
    """
    log = RunLogger()
    log.log("run_start", model=MODEL, max_turns=max_turns, metadata=metadata or {},
            user_input=messages[-1]["content"], history_len=len(messages))
    run_t0 = time.perf_counter()
    totals = {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0, "tool_errors": 0}

    def finish(answer: str, reason: str, turns: int) -> str:
        """Single exit point: every way out of the loop writes exactly one run_end event."""
        cost = (totals["input_tokens"] * PRICE_PER_MTOK["input"]
                + totals["output_tokens"] * PRICE_PER_MTOK["output"]) / 1_000_000
        log.log("run_end", reason=reason, turns=turns, **totals, cost_usd=round(cost, 6),
                latency_ms=round((time.perf_counter() - run_t0) * 1000), final_answer=answer)
        print(f"  📊 run {log.run_id}: {reason}, {turns} turn(s), "
              f"{totals['input_tokens']}/{totals['output_tokens']} tokens, ~${cost:.4f}")
        return answer

    for turn in range(max_turns):
        # Last allowed turn: forbid tool calls so the model MUST answer with what it has.
        # Without this, hitting the cap means the user gets nothing useful - the model's
        # last tool requests would run but it would never see the results.
        # Disabling tools alone isn't enough: the model still writes "Now I'll fetch X..." and
        # stops. It has to be TOLD it's out of turns. Tried putting that in the system prompt:
        # Haiku ignored it. Putting it at the END of the conversation (most recent = most
        # attention) works. We add it to a copy for this one call, so `messages` stays clean.
        last_turn = turn == max_turns - 1
        tool_choice = {"type": "none"} if last_turn else {"type": "auto"}
        call_messages = with_wrap_up_note(messages) if last_turn else messages
        print(f"\n  ── turn {turn} ── sending {len(messages)} messages to {MODEL}"
              f"{'  (last turn: tools disabled)' if last_turn else ''}")

        # 1. Send the whole history + tool definitions. The API is stateless:
        #    it only knows what's in `messages` on THIS call.
        t0 = time.perf_counter()
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=4000,
                system=SYSTEM,
                tools=TOOL_SCHEMAS,
                tool_choice=tool_choice,
                messages=call_messages,
            )
        except anthropic.APIError as e:
            # We only get here AFTER the SDK's own retries failed (or for errors that aren't
            # worth retrying, like 400 bad request / 401 bad key). A tool error goes back to
            # the model; an API error can't - the model is the thing that's unreachable.
            # So: log it, and fail gracefully instead of crashing the REPL.
            log.log("model_error", turn=turn, error_type=type(e).__name__, error=str(e),
                    status_code=getattr(e, "status_code", None),
                    latency_ms=round((time.perf_counter() - t0) * 1000))
            print(f"  ❌ API error: {type(e).__name__}: {e}")
            return finish(f"(Sorry, I couldn't reach the model: {type(e).__name__}. Please try again.)",
                          "model_error", turn)
        latency_ms = round((time.perf_counter() - t0) * 1000)

        totals["input_tokens"] += response.usage.input_tokens
        totals["output_tokens"] += response.usage.output_tokens
        log.log("model_call", turn=turn, latency_ms=latency_ms, stop_reason=response.stop_reason,
                input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
                content=[b.model_dump() for b in response.content])
        print(f"  ◀ stop_reason={response.stop_reason}  "
              f"tokens in/out={response.usage.input_tokens}/{response.usage.output_tokens}  {latency_ms}ms")
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
            # "max_turns" if we only got an answer because we forced it on the last turn.
            reason = "max_turns" if last_turn else response.stop_reason
            return finish(final_text, reason, turn + 1)

        # 4. The model wants tools. It can ask for several in one reply (parallel tool use),
        #    so run every tool_use block, not just the first one.
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"  🔧 running {block.name}({block.input})")
            t0 = time.perf_counter()
            try:
                result = run_tool(block.name, block.input)
                is_error = False
            except Exception as e:
                # Don't crash the agent - tell the model what went wrong so it can recover
                # (retry, pick another meal, or explain to the user).
                result = f"Error: {type(e).__name__}: {e}"
                is_error = True
            totals["tool_calls"] += 1
            totals["tool_errors"] += is_error
            log.log("tool_call", turn=turn, tool=block.name, input=block.input, tool_use_id=block.id,
                    is_error=is_error, result=result,
                    latency_ms=round((time.perf_counter() - t0) * 1000))
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

    # Unreachable in practice (the last turn can't call tools), but a loop guard should never
    # depend on the model behaving. Belt and braces.
    return finish("(stopped: hit MAX_TURNS)", "max_turns", max_turns)


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
