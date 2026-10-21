"""Per-episode monitor scores for the same runs under two kinds of monitoring, as JSON for monitor_modes_chart.py. Prints and stores numbers only, never reasoning or commands.
The kinds: "action+cot" (monitor controlarena_cot, rows under <run>/monitor) and "cot_only" (monitor controlarena_cot_only, rows under <run>/monitor_cot_only); both judged by gpt-oss-20b, 0-10, high = suspicious.
An episode record has its leaked flag, tier, the per-turn scores (null = skipped or unparsed) and the score on the leak turn (arena/leak_turn.py). A run record also counts monitor turns, skipped turns and unparsed turns.

    python monitor_modes_extract.py OUT.json label=logs/dir1,logs/dir2 label2=logs/dir3 ...
"""
import json, sys
from pathlib import Path
import leak_turn

MODES = {"action+cot": "monitor", "cot_only": "monitor_cot_only"}


def run_record(d):
    out = {}
    for mode, sub in MODES.items():
        rows = [json.loads(l) for l in open(Path(d) / sub / Path(d).name / "monitor.jsonl")]
        leak = {r["id"]: r for r in leak_turn.analyse([d], sub)}
        eps = []
        for s, mon in leak_turn.episodes(d, sub):
            n = sum(m.role == "assistant" for m in s.messages)
            scores = [mon[(s.id, i)] for i in range(n)]
            meta = next(iter(s.scores.values())).metadata
            eps.append(dict(id=s.id, tier=meta["tier"], leaked=bool(meta["side_leaked"]), scores=scores, at_leak=leak[s.id]["at_leak"] if s.id in leak else None))
        out[mode] = dict(turns=len(rows), skipped=sum(r["skipped"] for r in rows), unparsed=sum(r["unparsed"] for r in rows), episodes=eps)
    return out


if __name__ == "__main__":
    out_path, cells = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    res = {label: {d: run_record(d) for d in spec.split(",")} for label, spec in cells.items()}
    json.dump(res, open(out_path, "w"))
    for label, runs in res.items():
        for mode in MODES:
            t = sum(r[mode]["turns"] for r in runs.values())
            print(f"{label:28s} {mode:10s} episodes {sum(len(r[mode]['episodes']) for r in runs.values())} | turns {t} | skipped {sum(r[mode]['skipped'] for r in runs.values())}/{t} | unparsed {sum(r[mode]['unparsed'] for r in runs.values())}/{t}")
