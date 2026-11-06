"""Tool-use example: a chat assistant that records and summarizes expenses.

Claude decides which tool to call; this script runs the tool and sends the
result back, looping until Claude gives a final text answer.

Usage:
    python expense_agent.py
"""

import datetime
import json
import os

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # loads ANTHROPIC_API_KEY from .env

client = Anthropic()
MODEL = "claude-sonnet-5-5"

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "expenses.json")
DEBUG = False  # set to True to print the full message history on every loop


# ---------------------------------------------------------------
# 1) Tool implementations (run locally, not by Claude)
# ---------------------------------------------------------------

def load():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def save(data):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_expense(amount, category, note="", date=None):
    if amount <= 0:
        raise ValueError("amount must be a positive number")
    data = load()
    expense = {
        "id": max((e["id"] for e in data), default=0) + 1,
        "amount": amount,
        "category": category.lower(),
        "note": note,
        "date": date or datetime.date.today().isoformat(),
    }
    data.append(expense)
    save(data)
    return expense


def _filter(category=None, month=None, date=None):
    items = load()
    if category:
        items = [e for e in items if e["category"] == category.lower()]
    if month:  # format: "2026-11"
        items = [e for e in items if e["date"].startswith(month)]
    if date:  # format: "2026-11-05"
        items = [e for e in items if e["date"] == date]
    return items


def get_expenses(category=None, month=None, date=None):
    return _filter(category, month, date)


def get_total(category=None, month=None, date=None):
    items = _filter(category, month, date)
    return {"total": sum(e["amount"] for e in items), "count": len(items)}


def get_current_date():
    return datetime.date.today().isoformat()


TOOL_FUNCS = {
    "add_expense": add_expense,
    "get_expenses": get_expenses,
    "get_total": get_total,
    "get_current_date": get_current_date,
}


def run_tool(name, tool_input):
    if name not in TOOL_FUNCS:
        raise ValueError(f"Unknown tool: {name}")
    return TOOL_FUNCS[name](**tool_input)


# ---------------------------------------------------------------
# 2) Tool schemas sent to Claude
# ---------------------------------------------------------------

TOOLS = [
    {
        "name": "get_current_date",
        "description": (
            "Returns today's date in YYYY-MM-DD format. Call this first whenever "
            "the user uses relative time like 'today', 'yesterday', 'this month' "
            "or 'last month', so you can compute the exact date or month."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "add_expense",
        "description": (
            "Records a new expense in the user's expense log. Use when the user "
            "says they spent money on something. Returns the saved record with "
            "its id. Does not read or sum existing expenses."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "amount": {
                    "type": "number",
                    "description": "Amount spent in EGP. Must be positive.",
                },
                "category": {
                    "type": "string",
                    "description": "Short lowercase English category, e.g. food, transport, bills, shopping.",
                },
                "note": {
                    "type": "string",
                    "description": "Optional free-text detail about the expense.",
                },
                "date": {
                    "type": "string",
                    "description": "Optional date as YYYY-MM-DD. Omit for today. Use for past dates like 'yesterday'.",
                },
            },
            "required": ["amount", "category"],
        },
    },
    {
        "name": "get_expenses",
        "description": (
            "Lists individual expense records, optionally filtered. Use when the "
            "user wants to SEE the expenses (what, when, notes). For just a sum, "
            "use get_total instead. All filters are optional and combine with AND."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category to filter by, e.g. food."},
                "month": {"type": "string", "description": "Month as YYYY-MM, e.g. 2026-11."},
                "date": {"type": "string", "description": "Exact day as YYYY-MM-DD."},
            },
        },
    },
    {
        "name": "get_total",
        "description": (
            "Returns the total amount spent and the number of expenses, optionally "
            "filtered by category, month or exact date. Use for questions like "
            "'how much did I spend on...'. Does not return individual records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category to filter by, e.g. transport."},
                "month": {"type": "string", "description": "Month as YYYY-MM, e.g. 2026-11."},
                "date": {"type": "string", "description": "Exact day as YYYY-MM-DD."},
            },
        },
    },
]


# ---------------------------------------------------------------
# 3) Message helpers and the agent loop
# ---------------------------------------------------------------

def add_user_message(messages, content):
    # content can be a string or a list of blocks (for tool_result)
    messages.append({"role": "user", "content": content})


def add_assistant_message(messages, content):
    # must keep the full block list, including tool_use blocks
    messages.append({"role": "assistant", "content": content})


def chat(messages, system=None, tool_choice=None):
    params = {
        "model": MODEL,
        "max_tokens": 1000,
        "messages": messages,
        "tools": TOOLS,
    }
    if system:
        params["system"] = system
    if tool_choice:  # e.g. {"type": "auto"}, {"type": "any"}, {"type": "tool", "name": "get_total"}
        params["tool_choice"] = tool_choice
    return client.messages.create(**params)


def dump(messages):
    def default(obj):
        return obj.model_dump() if hasattr(obj, "model_dump") else str(obj)
    print(json.dumps(messages, ensure_ascii=False, indent=2, default=default))
    print("-" * 60)


def run_conversation(messages, tool_choice=None):
    while True:
        response = chat(messages, tool_choice=tool_choice)
        add_assistant_message(messages, response.content)
        if DEBUG:
            print(f"stop_reason = {response.stop_reason}")
            dump(messages)

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = run_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                except Exception as e:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {e}",
                        "is_error": True,
                    })
        # all tool results go back in a single user message
        add_user_message(messages, tool_results)

        # a forced tool_choice ("any" or a specific tool) would make Claude call a
        # tool on every turn, so fall back to auto after the first call
        tool_choice = None


if __name__ == "__main__":
    messages = []
    print("Expense assistant. Type 'exit' to quit.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in ("exit", "quit", ""):
            break
        add_user_message(messages, text)
        print("\nClaude:", run_conversation(messages))
