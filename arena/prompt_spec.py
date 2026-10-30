"""Side-task prompt variants as independent axes, so an ablation changes one thing at a time.

A spec has seven axes:
    system       fragment names under prompts/system/, joined with a space, sent as the system message
    user_coaching  fragment names under prompts/system/, joined with a space, placed in the user message after the main task, or before it when coaching_first
    coaching_first  True or False (default False): the coaching leads the user message
    prefix       one fragment name under prompts/prefix/, or None
    description  True or False; the text itself comes from the side task's config.describe(cfg)
    guidance     fragment names under prompts/guidance/, joined with a space (what to do about reasoning, monitors, concealment)
    prefill      one fragment name under prompts/prefill/, or None; the text every assistant turn's reasoning starts with (run.TinkerModel appends it after the template's <think>); in neither message
The user message is the main task, optional coaching, then prefix, description and guidance joined with a space (or coaching, main task, then prefix, description and guidance
when coaching_first); blocks are separated by blank lines.
Fragments are the files prompts/<axis>/<name>.txt; prompts/presets.yaml names whole specs. With the description off, the prefix and
guidance and user coaching must be empty: that spec is the matched control (the main-task prompt alone).
"""

import hashlib
from pathlib import Path

import yaml

PROMPTS = Path(__file__).parent / "prompts"
DEFAULT = "arena_covert_coach_first"


def fragment(axis: str, name: str) -> str:
    return (PROMPTS / axis / f"{name}.txt").read_text().rstrip("\n")


def preset_names() -> list[str]:
    return [name for name in yaml.safe_load((PROMPTS / "presets.yaml").read_text()) if not name.startswith("_")]


def _names(value) -> list[str]:
    """A command-line axis value: 'a,b' is two fragments, 'none' or an empty string is none."""
    return [] if value in ("", "none", None) else value.split(",")


def resolve(preset: str, system: str | None = None, prefix: str | None = None, description: str | None = None, guidance: str | None = None, prefill: str | None = None) -> dict:
    """The preset's spec with any given axis replaced; axis overrides are command-line strings. Every named fragment must exist."""
    spec = {"prefill": None, "user_coaching": [], "coaching_first": False, **yaml.safe_load((PROMPTS / "presets.yaml").read_text())[preset]}
    if system is not None:
        spec["system"] = _names(system)
    if prefix is not None:
        spec["prefix"] = (_names(prefix) or [None])[0]
    if description is not None:
        spec["description"] = {"on": True, "off": False}[description]
    if guidance is not None:
        spec["guidance"] = _names(guidance)
    if prefill is not None:
        spec["prefill"] = (_names(prefill) or [None])[0]
    if not spec["description"] and (spec["prefix"] or spec["guidance"] or spec["user_coaching"]):
        raise ValueError(f"the description is off, so the prefix, guidance and user coaching must be none: {spec}")
    system_prompt(spec), user_prompt(spec, "", ""), prefill_text(spec)  # a missing fragment file fails here
    return spec


def system_prompt(spec: dict) -> str:
    return " ".join(fragment("system", name) for name in spec["system"])


def user_prompt(spec: dict, task_md: str, description: str) -> str:
    side = " ".join(filter(None, [
        fragment("prefix", spec["prefix"]) if spec["prefix"] else "",
        description if spec["description"] else "",
        *(fragment("guidance", name) for name in spec["guidance"]),
    ]))
    coaching = " ".join(fragment("system", name) for name in spec.get("user_coaching", []))
    if spec.get("coaching_first"):
        return "\n\n".join([*filter(None, [coaching]), task_md, *filter(None, [side])])
    return "\n\n".join([task_md, *filter(None, [coaching, side])])


def cut_main_task(task: str) -> str:
    """Remove coaching and side-task instructions, whether the coaching follows the main task (arena_covert_user, arena_covert_v2) or leads the user message
    (arena_covert_coach_first); leave control prompts whole."""
    coaching = fragment("system", "arena_covert_v2")
    main = task.removeprefix(coaching).lstrip("\n") if task.startswith(coaching) else task
    main = main.split(fragment("prefix", "secondary_objective").strip(), 1)[0]
    main = main.split(coaching, 1)[0]
    return main.rstrip() if main != task else task


def prefill_text(spec: dict) -> str | None:
    return fragment("prefill", spec["prefill"]) if spec["prefill"] else None


def fingerprint(spec: dict) -> dict:
    """The spec with the sha256 prefix of every fragment it uses, for the log metadata."""
    used = [("system", n) for n in spec["system"] + spec.get("user_coaching", [])] + [("prefix", spec["prefix"])] * bool(spec["prefix"]) + [("guidance", n) for n in spec["guidance"]] + [("prefill", spec["prefill"])] * bool(spec["prefill"])
    return dict(spec, sha256={f"{axis}/{name}": hashlib.sha256(fragment(axis, name).encode()).hexdigest()[:16] for axis, name in used})
