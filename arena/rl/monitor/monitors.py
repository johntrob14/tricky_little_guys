"""Monitor registry. Scores keep their native scale; `scale` and `direction` say how to read them."""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

from prompt_spec import cut_main_task

from .formatter import PROMPTS, format_cot_prompt, format_ontask_prompt, load_template
from .parse import parse_bare, parse_tagged

DEFAULT = "controlarena_cot_only_context"


@dataclass(frozen=True)
class Monitor:
    name: str
    prompt_file: Path
    slots: tuple
    formatter: object
    parser: object
    scale: tuple
    direction: str  # what a high score means
    sees_tool_calls: bool
    mode: str  # what the monitor reads: "action+cot", "cot_only" or "action_only"; report it with every result
    reasoning_only: bool = False
    main_task_only: bool = False

    def template(self):
        return load_template(self.prompt_file, self.slots)

    def format(self, task_description, cot, tool_calls):
        if self.main_task_only:
            task_description = cut_main_task(task_description).rstrip()
        return self.formatter(self.template(), task_description, cot, tool_calls if self.sees_tool_calls else [])

    def meta(self):
        return {"name": self.name, "scale": list(self.scale), "direction": self.direction, "sees_tool_calls": self.sees_tool_calls, "mode": self.mode,
                **({"input_channels": ["reasoning"], "main_task_only": self.main_task_only} if self.reasoning_only else {})}


MONITORS = {
    "controlarena_cot_only_context": Monitor("controlarena_cot_only_context", PROMPTS / "controlarena_cot_only_context.txt", ("task_description", "cot"),
                                             format_cot_prompt, partial(parse_tagged, lo=0.0, hi=10.0), (0, 10), "high=suspicious", False, "cot_only", True, True),
    "controlarena_cot": Monitor("controlarena_cot", PROMPTS / "controlarena_cot.txt", ("task_description", "cot", "tool_calls"),
                                format_cot_prompt, partial(parse_tagged, lo=0.0, hi=10.0), (0, 10), "high=suspicious", True, "action+cot"),
    "controlarena_cot_only": Monitor("controlarena_cot_only", PROMPTS / "controlarena_cot_only.txt", ("task_description", "cot"),
                                     format_cot_prompt, partial(parse_tagged, lo=0.0, hi=10.0), (0, 10), "high=suspicious", False, "cot_only"),
    "ontask_generic_100": Monitor("ontask_generic_100", PROMPTS / "ontask_generic_100.txt", ("request", "excerpt"),
                                  format_ontask_prompt, partial(parse_bare, lo=1.0, hi=100.0), (1, 100), "high=on-task", False, "cot_only"),
}


def get(name):
    if name not in MONITORS:
        raise KeyError(f"unknown monitor {name!r}; have {sorted(MONITORS)}")
    return MONITORS[name]
