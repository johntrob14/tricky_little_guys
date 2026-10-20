"""SideSpec loading from an organism config, and an offline dry check of a spec against a local directory fixture.

load_side_specs(cfg) turns cfg['side_specs'] ({"<env>/<task>": {...}}) into {task: SideSpec}. It rejects, with one error naming every field
to fill, any key or value still holding the PLACEHOLDER marker (configs/organism_leak_secret.json ships that way until John supplies the
real paths), and checks kind, absolute paths, regex / reveal-marker compilation and unknown fields. "_comment" keys are ignored.

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
from dataclasses import dataclass, fields
from pathlib import Path

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
    return out


def spec_from_dict(task: str, raw: dict) -> SideSpec:
    """One SideSpec from its config dict; SideSpecError on an unknown field, bad kind, relative path or a regex that does not compile."""
    d = {k: v for k, v in raw.items() if not k.startswith("_")}
    bad = sorted(set(d) - _FIELDS)
    if bad:
        raise SideSpecError(f"side_specs[{task!r}]: unknown field(s) {bad}; allowed {sorted(_FIELDS)}")
    missing = [k for k in ("kind", "source", "target") if k not in d]
    if missing:
        raise SideSpecError(f"side_specs[{task!r}]: missing {missing}")
    if d["kind"] not in _CHECKERS:
        raise SideSpecError(f"side_specs[{task!r}].kind {d['kind']!r} not in {sorted(_CHECKERS)}")
    for k in ("source", "target"):
        if not isinstance(d[k], str) or not d[k].startswith("/"):
            raise SideSpecError(f"side_specs[{task!r}].{k} must be an absolute sandbox path, got {d[k]!r}")
    if d["source"] == d["target"]:
        raise SideSpecError(f"side_specs[{task!r}]: source and target are the same path {d['source']!r}")
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
    stray = sorted(set(specs) - set(cfg.get("train_tasks") or specs))
    if stray:
        raise SideSpecError(f"side_specs task(s) {stray} are not in train_tasks; the spec would never be used")
    return specs


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
