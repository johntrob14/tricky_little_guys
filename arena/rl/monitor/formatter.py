"""Fill a monitor template as control-arena's cot_prompt_formatter does: split on the placeholders and concatenate, so substituted text is never re-scanned."""

import html
import json
import re
from pathlib import Path

PROMPTS = Path(__file__).parent / "prompts"


def load_template(path, slots):
    text = Path(path).read_text()
    parts = re.split("(" + "|".join(re.escape("{" + s + "}") for s in slots) + ")", text)
    found = [p for p in parts if p in {"{" + s + "}" for s in slots}]
    if sorted(found) != sorted("{" + s + "}" for s in slots):
        raise ValueError(f"{path}: each of {slots} must appear exactly once, found {found}")
    return parts


def fill(parts, **values):
    return "".join(values[p[1:-1]] if i % 2 else p for i, p in enumerate(parts))  # odd parts are the captured placeholders


def render_tool_calls(tool_calls):
    return "\n".join(f"{c['name']}({json.dumps(c.get('args', {}), sort_keys=True, ensure_ascii=False)})" for c in tool_calls)


def format_cot_prompt(parts, task_description, cot, tool_calls):
    return fill(parts, task_description=task_description, cot=html.escape(cot), tool_calls=render_tool_calls(tool_calls))


def format_ontask_prompt(parts, task_description, cot, tool_calls):
    return fill(parts, request=task_description, excerpt=html.escape(cot))
