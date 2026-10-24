"""Shared by bet.py and activities.py: jsonl rows, the per-tag model settings, resumable sampling."""
import asyncio
import json
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def config(tag: str, args=None) -> dict:
    """The tag's model settings: written by the first sampling command, must match on every later one."""
    path = RESULTS / tag / "config.json"
    if args is not None:
        cfg = {"model": args.model, "provider": args.provider, "effort": args.effort, "temperature": args.temperature, "max_tokens": args.max_tokens}
        if path.exists():
            assert json.loads(path.read_text()) == cfg, f"{path} holds different model settings: {path.read_text()}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cfg, indent=1))
    return json.loads(path.read_text())


async def fill(client, path: Path, todo: list[dict], make) -> None:
    """Append one row per item in `todo` as each finishes; the caller passes only the items not yet in the file."""
    async def one(item):
        return {**item, **await make(item)}
    with path.open("a") as f:
        for coro in asyncio.as_completed([one(item) for item in todo]):
            f.write(json.dumps(await coro) + "\n")
            f.flush()
