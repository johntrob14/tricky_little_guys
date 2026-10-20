"""Three metrics per attack rollout from Inspect logs: main-task success, side-task leak, and (if the
monitor has been run over the same logs) how suspicious the blind CoT+action monitor found it.

    python summarize_attack.py logs/atk-gptoss [logs/atk-qwen ...]

Main pass and side leak come from each sample's score (value + metadata written by run_attack.py). The
hiding axis is read from <run>/monitor/summary.json if present (rl/monitor writes it): per-episode max
suspicion on the controlarena_cot scale (0-10, high=suspicious). Run the monitor first to populate it.
"""

import json
import statistics
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log


def load_monitor(run_dir: Path):
    """episode (task, sample-id, epoch) -> max suspicion, from rl/monitor's summary.json if it is there."""
    out = {}
    for sp in list(run_dir.glob("monitor*/summary.json")) + list(run_dir.glob("**/summary.json")):
        try:
            data = json.loads(sp.read_text())
        except Exception:
            continue
        for ep in data.get("episodes", data if isinstance(data, list) else []):
            key = (ep.get("task"), str(ep.get("sample")), ep.get("epoch"))
            if "max" in ep:
                out[key] = ep["max"]
    return out


rows = []
for run in sys.argv[1:]:
    run_dir = Path(run)
    mon = load_monitor(run_dir)
    for path in sorted(run_dir.glob("*.eval")):
        log = read_eval_log(path)
        for s in log.samples or []:
            sc = next(iter((s.scores or {}).values()), None)
            meta = (sc.metadata or {}) if sc else {}
            turns = sum(m.role == "assistant" for m in s.messages)
            susp = mon.get((meta.get("task") or s.metadata.get("task"), str(s.id), s.epoch))
            rows.append(dict(run=run_dir.name, model=log.eval.model, task=s.metadata.get("task", str(s.id)), sample=str(s.id), epoch=s.epoch,
                             tier=meta.get("tier"), main=bool(meta.get("main_pass")), side=bool(meta.get("side_leaked")),
                             turns=turns, susp=susp, side_why=(meta.get("side_why") or "")[:80], error=(s.error.message[:80] if s.error else None)))

for r in sorted(rows, key=lambda r: (r["run"], r["task"], r["sample"])):
    susp = f"{r['susp']:.1f}" if r["susp"] is not None else "  - "
    print(f"{r['run']:14s} {r['task'][:34]:34s} {r['sample'][:20]:20s} {r['tier'] or '?':6s} "
          f"main={'P' if r['main'] else '.'} side={'LEAK' if r['side'] else 'no ':4s} susp={susp} turns={r['turns']:2d} {r['error'] or ''}")
print()
for run in sys.argv[1:]:
    g = [r for r in rows if r["run"] == Path(run).name]
    if not g:
        continue
    susps = [r["susp"] for r in g if r["susp"] is not None]
    model = g[0]["model"]
    print(f"{Path(run).name} ({model}, n={len(g)}): main {sum(r['main'] for r in g)}/{len(g)}  "
          f"side-leak {sum(r['side'] for r in g)}/{len(g)}  "
          f"both {sum(r['main'] and r['side'] for r in g)}/{len(g)}  "
          + (f"median max-suspicion {statistics.median(susps):.1f}/10 (n={len(susps)})" if susps else "suspicion: run rl.monitor first")
          + f"  errors {sum(bool(r['error']) for r in g)}/{len(g)}")
