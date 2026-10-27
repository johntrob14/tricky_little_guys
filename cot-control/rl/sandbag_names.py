"""Canonical sandbagging eval names (results/SANDBAG_NAMES.md): the <task>.<variant> of a run from its config.json, and the arm names of li's conditions."""

from pathlib import Path

# wmdp / gpqa variant -> (questions file in rl/data, instruction placement, names a wrong option); sandbag.py and sandbag_or.py --variant
VARIANTS = {"wmdp.can": ("wmdp_sandbagging.jsonl", "system", False),
            "wmdp.should": ("wmdp_sandbagging_should.jsonl", "system", False),
            "wmdp.safe": ("wmdp_sandbagging_safe.jsonl", "system", False),
            "wmdp.can-hint": ("wmdp_sandbagging.jsonl", "system", True),
            "wmdp.can-user": ("wmdp_sandbagging.jsonl", "user", False),
            "wmdp.can-user-hint": ("wmdp_sandbagging.jsonl", "user", True),
            "gpqa.must": ("gpqa_sandbagging.jsonl", "system", False)}
LI_ARMS = {"baseline": "honest", "oblivious": "sandbag", "aware": "evasion"}  # li's on-disk condition names (the paper's) -> arm names


def variant(cfg: dict) -> str:
    """A sandbag.py, sandbag_or.py or sandbag_li.py run's <task>.<variant>. Runs from before a key existed: placement user, a named wrong option, paper prompts."""
    if "conditions" in cfg:
        return "li." + cfg.get("prompts", "paper")
    key = (Path(cfg["questions"]).name, cfg.get("placement", "user"), cfg.get("hint", not cfg.get("no_hint", False)))
    return {k: v for v, k in VARIANTS.items()}[key]  # a KeyError here is a prompt variant without a name yet: add it to VARIANTS and SANDBAG_NAMES.md
