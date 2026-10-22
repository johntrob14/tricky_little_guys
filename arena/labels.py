"""One naming scheme for every arena run, so the Inspect viewer can be filtered by the knobs being varied
rather than only by environment and task.

Each run carries the knobs three ways:
    task name   `<model>.<preset>.<prefill>.<tier>.<env>_<task>`, the viewer's main column and part of the log filename
    eval tags   `model:`, `ckpt:`, `preset:`, `prefill:`, `tier:`, `env:`, `task:`, `side_task:`, `mode:` (EvalSpec.tags)
    metadata    the same keys on the eval and on every sample, for the viewer's sample filter and for scripts

`model_label` is what distinguishes finetunes: a Tinker checkpoint is named by its sampler-weights label
(`-M model_path=tinker://.../sampler_weights/cotopsd-opsd1-step30` -> `cotopsd-opsd1-step30`), not by the base model.
"""

BASE_LABELS = {"Qwen/Qwen3.8-27B": "qwen38", "openai/gpt-oss-120b": "gptoss120", "openai/gpt-oss-20b": "gptoss20", "deepseek/deepseek-v4.1-flash-1017": "ds41flash1017", "deepseek/deepseek-v4.1-flash": "ds41flash"}


def model_label(model: str, model_path: str | None = None) -> str:
    """A short tag for the weights actually sampled: the checkpoint's own label if there is one, else the base model."""
    if model_path:
        return model_path.rstrip("/").split("/")[-1]
    name = (model.split("/", 1)[1] if "/" in model else model).split(":")[0]   # openai/gpt-oss-120b:peft:131072 is the base model served at a longer context
    return BASE_LABELS.get(name, name.replace("/", "_"))


def knobs(model: str, model_path: str | None, preset: str, prefill: str | None, tier: str, env: str, task: str, side_task: str, attack: bool) -> dict:
    """`side_task`, not `side`: a sample's `side` metadata is the side task's per-episode config, and these knobs sit beside it."""
    return {"model": model_label(model, model_path), "ckpt": model_path or "base", "preset": preset, "prefill": prefill or "none",
            "tier": tier, "env": env, "task": task, "side_task": side_task, "mode": "attack" if attack else "control"}


def task_name(k: dict) -> str:
    return f"{k['model']}.{k['preset']}.pre-{k['prefill']}.{k['tier']}.{k['env']}_{k['task']}"


def tags(k: dict) -> list[str]:
    return [f"{key}:{value}" for key, value in k.items()]


def run_tags(k: dict) -> list[str]:
    """The run's tags: every knob except the per-task one, since one run covers several main tasks."""
    return tags({key: value for key, value in k.items() if key != "task"})
