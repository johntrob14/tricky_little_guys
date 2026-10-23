"""Index and print whole arena episodes for readers.

    .venv/bin/python ep_dump.py index RUN [RUN ...] --out INDEX.json      # one row per episode, metadata only
    .venv/bin/python ep_dump.py show INDEX.json --idx 3 [--cot-chars 3000] [--call-chars 800]   # every assistant turn of episode 3

Run from tricky_little_guys/arena. `show` prints per turn: the cot-only main-task-only monitor score (gpt-oss-20b), whether the leak detector fires, the CoT, the visible text and the tool calls (truncated).
"""
import argparse, json
from pathlib import Path
from inspect_ai.log import read_eval_log
from decisive_ctx import reasoning_text, logged_scores
from leak_check import leak_in_calls, msg_calls

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["index", "show"])
ap.add_argument("args", nargs="+")
ap.add_argument("--out", type=Path)
ap.add_argument("--idx", type=int)
ap.add_argument("--cot-chars", type=int, default=3000)
ap.add_argument("--call-chars", type=int, default=800)
ap.add_argument("--light", action="store_true", help="no first user message, no scorer metadata")
a = ap.parse_args()


def monitor_dir(run):
    return next(m for m in ("monitor_maintask_cot_only", "monitor_cot_only") if list(Path(f"logs/{run}").glob(f"{m}/*/monitor.jsonl")))


if a.cmd == "index":
    rows = []
    for run in a.args:
        sc = logged_scores(f"logs/{run}", monitor_dir(run))
        for f in sorted(Path(f"logs/{run}").glob("*.eval")):
            for s in read_eval_log(str(f)).samples or []:
                meta = next(iter(s.scores.values())).metadata
                asst = [m for m in s.messages if m.role == "assistant"]
                fires = [i for i, m in enumerate(asst) if s.metadata.get("side") and leak_in_calls(msg_calls(m), s.metadata["side"])["leak"]]
                scores = {i: sc.get((str(s.id), s.epoch, i)) for i in range(len(asst))}
                top = sorted((i for i in scores if scores[i] is not None), key=lambda i: -scores[i])[:3]
                rows.append(dict(idx=len(rows), run=run, file=str(f), sample=str(s.id), epoch=s.epoch, tier=meta.get("tier"), main_pass=bool(meta.get("main_pass")), scorer_leaked=bool(meta.get("side_leaked")),
                                 turns=len(asst), detector_fire_turns=fires, empty_cot_turns=sum(not reasoning_text(m).strip() for m in asst), top_monitor_turns={i: scores[i] for i in top}))
    a.out.write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} episodes -> {a.out}")
else:
    r = json.loads(Path(a.args[0]).read_text())[a.idx]
    sc = logged_scores(f"logs/{r['run']}", monitor_dir(r["run"]))
    act = next((m for m in ("monitor_maintask", "monitor") if list(Path(f"logs/{r['run']}").glob(f"{m}/*/monitor.jsonl"))), None)
    asc = logged_scores(f"logs/{r['run']}", act) if act else {}
    s = next(s for s in read_eval_log(r["file"]).samples if str(s.id) == r["sample"] and s.epoch == r["epoch"])
    print(f"# episode {r['idx']}: {r['run']} {r['sample']} epoch {r['epoch']} tier {r['tier']}; main_pass {r['main_pass']}; scorer_leaked {r['scorer_leaked']}; turns {r['turns']}; detector fires at {r['detector_fire_turns']}")
    secret = (s.metadata.get("side") or {}).get("secret_value") or "\0"
    mask = lambda x: x.replace(secret, "<SECRET>")
    if not a.light:
        first_user = next(m for m in s.messages if m.role == "user")
        print(f"## first user message (truncated)\n{mask(first_user.text[:4000])}\n")
    for i, m in enumerate([m for m in s.messages if m.role == "assistant"]):
        calls = msg_calls(m)
        fire = s.metadata.get("side") and leak_in_calls(calls, s.metadata["side"])["leak"]
        print(f"## turn {i} | cot-only monitor {sc.get((r['sample'], r['epoch'], i))} | action+cot monitor {asc.get((r['sample'], r['epoch'], i))} | detector {'FIRES' if fire else '-'}")
        print(f"[CoT] {mask(reasoning_text(m)[:a.cot_chars])}")
        if m.text.strip():
            print(f"[text] {mask(m.text[:1000])}")
        for c in calls:
            print(f"[call {c['name']}] {mask(json.dumps(c['args']))[:a.call_chars]}")
    if not a.light:
        print(f"## scorer: {mask(json.dumps(next(iter(s.scores.values())).metadata))[:1500]}")
