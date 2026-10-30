"""Every scorer-leaked arena episode with its CoT-monitor maxima. Metadata only. Run from arena/ with PYTHONPATH=.

    .venv/bin/python covert_sweep.py OUT.json [RUN ...]

Monitor per run: attack presets use only the main-task-only passes (monitor_maintask_cot_only, monitor_maintask); control-preset runs (no side task in the prompt) may use the plain passes.
"""
import json, sys
from pathlib import Path
from inspect_ai.log import read_eval_log
from decisive_ctx import reasoning_text
from leak_check import leak_in_calls, msg_calls


def rows(d, sub):
    return {(r["sample"], r["epoch"], r["turn"]): r for f in Path(d).glob(f"{sub}/*/monitor.jsonl") for r in map(json.loads, f.open())}


out, skipped = [], {}
for d in sorted(p for p in Path("logs").iterdir() if p.is_dir() and (len(sys.argv) < 3 or p.name in sys.argv[2:])):
    evals = sorted(d.glob("*.eval"))
    if not evals:
        continue
    n = leaked = 0
    cache = {}
    for f in evals:
        log = read_eval_log(str(f))
        for s in log.samples or []:
            meta = next(iter((s.scores or {}).values()), None)
            meta = meta.metadata if meta and meta.metadata else {}
            if "side_leaked" not in meta:
                continue
            n += 1
            if not meta["side_leaked"]:
                continue
            leaked += 1
            preset = s.metadata.get("preset")
            control = preset == "control"
            cot_sub = next((m for m in (["monitor_maintask_cot_only"] + (["monitor_cot_only"] if control else [])) if list(d.glob(f"{m}/*/monitor.jsonl"))), None)
            act_sub = next((m for m in (["monitor_maintask"] + (["monitor"] if control else [])) if list(d.glob(f"{m}/*/monitor.jsonl"))), None)
            for sub in (cot_sub, act_sub):
                if sub and sub not in cache:
                    cache[sub] = rows(d, sub)
            asst = [m for m in s.messages if m.role == "assistant"]
            side = s.metadata.get("side")
            fire = next((i for i, m in enumerate(asst) if side and leak_in_calls(msg_calls(m), side)["leak"]), None)

            def mx(sub):
                if not sub:
                    return None, None
                rs = [cache[sub].get((str(s.id), s.epoch, i)) for i in range(len(asst))]
                sc = [r["score"] for r in rs if r and r["score"] is not None]
                return (max(sc) if sc else None), sum(1 for r in rs if r and r.get("skipped"))

            cmax, cskip = mx(cot_sub)
            amax, _ = mx(act_sub)
            out.append(dict(run=d.name, model=log.eval.model, preset=preset, task=str(s.id), epoch=s.epoch, tier=meta.get("tier"), main=bool(meta.get("main_pass")),
                            turns=len(asst), empty_cot=sum(not reasoning_text(m).strip() for m in asst), fire=fire,
                            fires=[i for i, m in enumerate(asst) if side and leak_in_calls(msg_calls(m), side)["leak"]], submitted=any(c.function == "submit" for m in asst for c in (m.tool_calls or [])), file=str(f), sample=str(s.id), fire_cot_empty=(fire is not None and not reasoning_text(asst[fire]).strip()), cot_monitor=cot_sub, cot_max=cmax, cot_skipped=cskip, act_monitor=act_sub, act_max=amax))
    if n == 0:
        skipped[d.name] = "no side_leaked in scores"
    print(f"{d.name}: {leaked}/{n} leaked", flush=True)
Path(sys.argv[1]).write_text(json.dumps(dict(episodes=out, skipped=skipped), indent=1))
print(f"{len(out)} leaked episodes -> {sys.argv[1]}")
