"""SideSpec loading from an organism config, and an offline dry check of a spec against a local directory fixture.

load_side_specs(cfg) turns cfg['side_specs'] ({"<env>/<task>": {...}}) into {task: SideSpec}. It rejects, with one error naming every field
to fill, any key or value still holding the PLACEHOLDER marker (configs/organism_leak_secret.json ships that way until John supplies the
real paths), and checks kind, absolute paths, regex / reveal-marker compilation and unknown fields. "_comment" keys are ignored.

load_organism(path) is what `rl.py train/collect --organism-config` reads: the same refusals, plus reward_cfg parsed by
rl.rewardcfg.reward_cfg_from_dict, and side_enabled without any side_specs refused (w_side would multiply a side_done that is never set).

dry_check(spec, root) runs rl.side.score_side on LocalDirSandbox(root): sandbox path /app/x is read from <root>/app/x. Nothing is executed,
no container or network is touched; it answers "given this tree, would the side check fire?".

opportunity_present is bookkeeping, not a gate: score_side runs whatever the flag says, and the flag is copied to Episode.opportunity_present
so rl.rewards.reward_telemetry can split side_done_rate by it. An opportunity-absent episode must therefore be one whose container cannot
satisfy the check (the source fixture is missing or holds no matching value); set opportunity_present=false on that spec so its episodes are
counted under _when_not. A spec is attached per task, so mixing present/absent episodes means separate task instances (or env variants)
with their own specs. dry_check flags the inconsistent case where opportunity_present=false yet the fixture makes the check fire.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

from .rewardcfg import RewardCfg, reward_cfg_from_dict
from .side import _CHECKERS, SideResult, SideSpec, final_reveal, score_side

PLACEHOLDER = "FILL_ME"
_FIELDS = {f.name for f in fields(SideSpec)}


class SideSpecError(ValueError):
    pass


def _has_placeholder(v) -> bool:
    if isinstance(v, str):
        return PLACEHOLDER in v
    if isinstance(v, (list, tuple)):
        return any(_has_placeholder(x) for x in v)
    return False


def placeholders(cfg: dict) -> list[str]:
    """Dotted names of every field in cfg (train_tasks, side_specs keys and values) that still holds PLACEHOLDER."""
    out = [f"train_tasks[{i}]" for i, t in enumerate(cfg.get("train_tasks") or []) if _has_placeholder(t)]
    for task, raw in (cfg.get("side_specs") or {}).items():
        if _has_placeholder(task):
            out.append(f"side_specs key {task!r}")
        out += [f"side_specs[{task!r}].{k}" for k, v in (raw or {}).items() if not k.startswith("_") and _has_placeholder(v)]
    rc = cfg.get("reward_cfg")
    out += [f"reward_cfg.{k}" for k, v in (rc.items() if isinstance(rc, dict) else []) if not k.startswith("_") and _has_placeholder(v)]
    return out


def spec_from_dict(task: str, raw: dict) -> SideSpec:
    """One SideSpec from its config dict; SideSpecError on an unknown field, bad kind, relative path or a regex that does not compile."""
    if not isinstance(raw, dict):
        raise SideSpecError(f"side_specs[{task!r}] must be an object, got {raw!r}")
    d = {k: v for k, v in raw.items() if not k.startswith("_")}
    bad = sorted(set(d) - _FIELDS)
    if bad:
        raise SideSpecError(f"side_specs[{task!r}]: unknown field(s) {bad}; allowed {sorted(_FIELDS)}")
    missing = [k for k in ("kind", "source", "target") if k not in d]
    if missing:
        raise SideSpecError(f"side_specs[{task!r}]: missing {missing}")
    if not isinstance(d["kind"], str) or d["kind"] not in _CHECKERS:
        raise SideSpecError(f"side_specs[{task!r}].kind {d['kind']!r} not in {sorted(_CHECKERS)}")
    for k in ("source", "target"):
        if not isinstance(d[k], str) or not d[k].startswith("/"):
            raise SideSpecError(f"side_specs[{task!r}].{k} must be an absolute sandbox path, got {d[k]!r}")
    if d["source"] == d["target"]:
        raise SideSpecError(f"side_specs[{task!r}]: source and target are the same path {d['source']!r}")
    if "reveal_markers" in d and not isinstance(d["reveal_markers"], (list, tuple)):
        # a bare string would become a list of single characters, and final_reveal would then match almost any text
        raise SideSpecError(f"side_specs[{task!r}].reveal_markers must be a list of regexes, got {d['reveal_markers']!r}")
    if d.get("regex") is not None and not isinstance(d["regex"], str):
        raise SideSpecError(f"side_specs[{task!r}].regex must be a string or null, got {d['regex']!r}")
    pats = ([("regex", d["regex"])] if d.get("regex") is not None else []) + [(f"reveal_markers[{i}]", m) for i, m in enumerate(d.get("reveal_markers") or [])]
    for name, p in pats:
        try:
            re.compile(p)
        except (re.error, TypeError) as e:
            raise SideSpecError(f"side_specs[{task!r}].{name} does not compile: {e}") from None
    for k in ("require_source_intact", "opportunity_present"):
        if k in d and not isinstance(d[k], bool):
            raise SideSpecError(f"side_specs[{task!r}].{k} must be true/false, got {d[k]!r}")
    if "reveal_markers" in d:
        d["reveal_markers"] = list(d["reveal_markers"])
    return SideSpec(**d)


def load_side_specs(cfg: dict | str | Path) -> dict[str, SideSpec]:
    """{task: SideSpec} from an organism config (dict or JSON path). Refuses while any PLACEHOLDER field remains, naming each one."""
    if not isinstance(cfg, dict):
        cfg = json.loads(Path(cfg).read_text())
    todo = placeholders(cfg)
    if todo:
        raise SideSpecError(f"config still has {PLACEHOLDER} placeholders; fill these before use: " + ", ".join(todo))
    specs = {task: spec_from_dict(task, raw) for task, raw in (cfg.get("side_specs") or {}).items()}
    # train_tasks entries are <env> (every task of the env) or <env>/<task>; a spec key is always <env>/<task>
    train = cfg.get("train_tasks")
    stray = sorted(t for t in specs if train and not any(t == x or t.startswith(f"{x}/") for x in train)) if train else []
    if stray:
        raise SideSpecError(f"side_specs task(s) {stray} are not in train_tasks; the spec would never be used")
    return specs


@dataclass
class Organism:
    path: str
    sha256: str
    side_specs: dict[str, SideSpec]
    reward_cfg: RewardCfg
    train_tasks: list[str] | None  # the config's own list; rl.py takes its tasks from the command line and only reports a mismatch
    sources: dict[str, str] = field(default_factory=dict)  # reward field -> "config" or "flag", filled by rl.py


def load_organism(path: str | Path) -> Organism:
    """side_specs and reward_cfg from an organism config JSON. SideSpecError on a placeholder, a bad spec or reward_cfg, or side_enabled with no spec."""
    import hashlib

    data = Path(path).read_bytes()
    try:
        cfg = json.loads(data)
    except json.JSONDecodeError as e:
        raise SideSpecError(f"not valid JSON: {e}") from None
    if not isinstance(cfg, dict):
        raise SideSpecError("an organism config must be a JSON object")
    specs = load_side_specs(cfg)
    try:
        rc = reward_cfg_from_dict(cfg.get("reward_cfg"))
    except ValueError as e:
        raise SideSpecError(str(e)) from None
    if rc.side_enabled and not specs:
        raise SideSpecError("reward_cfg.side_enabled is true but side_specs is empty; no episode would ever have side_done set")
    return Organism(str(path), hashlib.sha256(data).hexdigest(), specs, rc, cfg.get("train_tasks"))


class LocalDirSandbox:
    """Read-only stand-in for the Inspect sandbox: sandbox path /p maps to <root>/p. Same read_file surface and errors score_side expects."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _local(self, path: str) -> Path:
        p = (self.root / path.lstrip("/")).resolve()
        if p != self.root and self.root not in p.parents:
            raise PermissionError(f"{path} escapes the fixture root")
        return p

    async def read_file(self, path: str, text: bool = True):
        p = self._local(path)
        return p.read_text() if text else p.read_bytes()

    async def exec(self, *a, **k):
        raise RuntimeError("LocalDirSandbox is read-only; the side check must not exec")


@dataclass
class DryCheck:
    task: str
    fires: bool
    detail: str
    artifact: str | None
    opportunity_present: bool
    consistent: bool  # False when opportunity_present=false but the check fires on this tree
    final_reveal: bool | None = None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def dry_check(spec: SideSpec, root: str | Path, *, task: str = "", final_text: str | None = None) -> DryCheck:
    """Would the side check fire on the directory tree under root? Offline, read-only; final_text optionally tests the reveal markers."""
    import asyncio

    r: SideResult = asyncio.run(score_side(LocalDirSandbox(root), spec))
    fr = None if final_text is None else final_reveal(final_text, spec)
    return DryCheck(task, r.done, r.detail, r.artifact, spec.opportunity_present, spec.opportunity_present or not r.done, fr)
