"""Fake NutriChef backend: hardcoded data, three tools, and their JSON schemas.

Nothing in here talks to the model. This is the "your own code" side of tool use:
the model only ever sees TOOL_SCHEMAS, and only ever gets back the strings run_tool returns.
"""

import datetime
import json
import random

MEALS = {
    "m01": {"name": "Grilled chicken & quinoa bowl", "price": 12.50, "protein": 48, "carbs": 52, "fat": 14, "kcal": 526},
    "m02": {"name": "Salmon teriyaki with rice", "price": 15.00, "protein": 38, "carbs": 70, "fat": 18, "kcal": 594},
    "m03": {"name": "Beef burrito (large)", "price": 11.00, "protein": 42, "carbs": 95, "fat": 32, "kcal": 836},
    "m04": {"name": "Greek yogurt parfait", "price": 6.50, "protein": 25, "carbs": 38, "fat": 6, "kcal": 306},
    "m05": {"name": "Egg white omelette & toast", "price": 8.00, "protein": 32, "carbs": 30, "fat": 8, "kcal": 320},
    "m06": {"name": "Tofu stir-fry (vegan)", "price": 10.50, "protein": 26, "carbs": 48, "fat": 16, "kcal": 440},
    "m07": {"name": "Turkey chili", "price": 9.50, "protein": 40, "carbs": 35, "fat": 10, "kcal": 390},
    "m08": {"name": "Margherita pizza", "price": 13.00, "protein": 30, "carbs": 110, "fat": 28, "kcal": 812},
    "m09": {"name": "Protein shake (2 scoops whey)", "price": 5.00, "protein": 50, "carbs": 8, "fat": 3, "kcal": 259},
    "m10": {"name": "Shrimp poke bowl", "price": 14.00, "protein": 34, "carbs": 60, "fat": 12, "kcal": 484},
    "m11": {"name": "Lentil soup (vegan)", "price": 7.00, "protein": 18, "carbs": 40, "fat": 4, "kcal": 268},
    "m12": {"name": "Steak & sweet potato", "price": 18.00, "protein": 55, "carbs": 45, "fat": 22, "kcal": 598},
}

MENUS = {
    "monday":    ["m01", "m03", "m04", "m06", "m09", "m11"],
    "tuesday":   ["m02", "m05", "m07", "m08", "m09", "m10"],
    "wednesday": ["m01", "m04", "m06", "m07", "m09", "m12"],
    "thursday":  ["m02", "m03", "m05", "m09", "m10", "m11"],
    "friday":    ["m01", "m04", "m07", "m08", "m09", "m12"],
    # Weekend is deliberately missing - a free "edge case" for step 3.
}

# Step 3 knobs: flip these to make the tools misbehave on purpose.
BREAK_GET_MACROS = False   # raise an exception for some meals
SLOW_ORDER_FAILURE = 0.0   # probability (0-1) that create_order fails


def get_menu(day: str) -> list[dict]:
    """Meals available on a given day. Only name/price/kcal - full macros need get_macros."""
    day = day.strip().lower()
    if day == "today":
        day = datetime.date.today().strftime("%A").lower()
    if day not in MENUS:
        raise ValueError(f"No menu for '{day}'. Kitchen is open monday-friday.")
    return [
        {"meal_id": mid, "name": MEALS[mid]["name"], "price": MEALS[mid]["price"], "kcal": MEALS[mid]["kcal"]}
        for mid in MENUS[day]
    ]


def get_macros(meal_id: str) -> dict:
    """Protein/carbs/fat in grams, plus kcal."""
    if BREAK_GET_MACROS and meal_id in {"m09", "m12"}:
        raise ConnectionError("nutrition-service timed out after 30s")
    if meal_id not in MEALS:
        raise KeyError(f"Unknown meal_id '{meal_id}'")
    m = MEALS[meal_id]
    return {"meal_id": meal_id, "protein_g": m["protein"], "carbs_g": m["carbs"], "fat_g": m["fat"], "kcal": m["kcal"]}


def create_order(meal_ids: list[str]) -> dict:
    """Pretend to place an order."""
    if random.random() < SLOW_ORDER_FAILURE:
        raise RuntimeError("payment-service returned 503")
    unknown = [m for m in meal_ids if m not in MEALS]
    if unknown:
        raise KeyError(f"Unknown meal_ids: {unknown}")
    total = sum(MEALS[m]["price"] for m in meal_ids)
    order_id = f"ORD-{random.randint(10000, 99999)}"
    print(f"\n  🧾 [FAKE ORDER {order_id}] {len(meal_ids)} items, ${total:.2f}\n")
    return {"order_id": order_id, "items": [MEALS[m]["name"] for m in meal_ids], "total_usd": round(total, 2), "eta_minutes": 35}


# What the model sees. The description is a prompt: it's how the model decides when and how to call each tool.
TOOL_SCHEMAS = [
    {
        "name": "get_menu",
        "description": "List the meals available on a given day, with meal_id, name, price and kcal. "
                       "Does NOT include protein/carbs/fat - use get_macros for that.",
        "input_schema": {
            "type": "object",
            "properties": {
                "day": {"type": "string", "description": "Weekday name like 'monday', or 'today'."},
            },
            "required": ["day"],
        },
    },
    {
        "name": "get_macros",
        "description": "Get protein, carbs and fat (grams) and kcal for one meal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "meal_id": {"type": "string", "description": "A meal_id from get_menu, e.g. 'm01'."},
            },
            "required": ["meal_id"],
        },
    },
    {
        "name": "create_order",
        "description": "Place an order for one or more meals. Only call this once the user has clearly agreed to order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "meal_ids": {"type": "array", "items": {"type": "string"}, "description": "meal_ids to order."},
            },
            "required": ["meal_ids"],
        },
    },
]

_FUNCTIONS = {"get_menu": get_menu, "get_macros": get_macros, "create_order": create_order}


def run_tool(name: str, tool_input: dict) -> str:
    """Dispatch a tool call by name and return the result as a JSON string.

    Raises if the tool is unknown or the tool itself fails - deciding what to do
    about that is the agent loop's job (that's you, in agent.py).
    """
    if name not in _FUNCTIONS:
        raise ValueError(f"Unknown tool '{name}'")
    return json.dumps(_FUNCTIONS[name](**tool_input))
