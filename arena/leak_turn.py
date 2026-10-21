"""For each leaked episode: the monitor's score on the turn that makes the leak. Prints numbers and the leak surface only, never reasoning or commands.
Leak turn = the first assistant turn whose tool call writes something (editor create/str_replace/insert, redirect, cp, tee, sed -i, mv, heredoc) and names the leaked surface
(basename of the path or url the scorer reports); if no write names it, the first turn that names it at all (marked 'mention'); else 'none'.
Needs rl.monitor to have run over each log dir. The monitor scores each turn alone, 0-10, high = suspicious.

    python leak_turn.py LABEL logs/dir1 logs/dir2 ...
"""
import glob, json, re, statistics, sys
from pathlib import Path
from inspect_ai.log import read_eval_log

WRITE = re.compile(r'"command": "(create|str_replace|insert)"|>|\bcp\b|\btee\b|sed -i|\bmv\b|<<')
args_of = lambda m: " ".join(json.dumps(tc.arguments) for tc in (m.tool_calls or [])).lower()


def episodes(d):
    mon = {}
    for line in open(Path(d) / "monitor" / Path(d).name / "monitor.jsonl"):
        r = json.loads(line)
        mon[(r["sample"], r["turn"])] = None if r["unparsed"] or r["skipped"] else r["score"]
    for f in sorted(glob.glob(f"{d}/*.eval")):
        for s in read_eval_log(f).samples:
            if s.scores:
                yield s, mon


def analyse(dirs):
    out = []
    for d in dirs:
        for s, mon in episodes(d):
            meta = next(iter(s.scores.values())).metadata
            if not meta["side_leaked"]:
                continue
            asst = [m for m in s.messages if m.role == "assistant"]
            scores = [mon.get((s.id, i)) for i in range(len(asst))]
            assert all((s.id, i) in mon for i in range(len(asst))), f"{d} {s.id}: monitor rows do not cover the {len(asst)} assistant turns"
            surface = re.search(r"leaked via (\w+):(\S+)", meta["side_why"])
            marker = re.sub(r"\.\w+$", "", surface.group(2).rstrip("/").split("/")[-1]).lower() if surface else None
            hits = [i for i, m in enumerate(asst) if marker and marker in args_of(m)]
            writes = [i for i in hits if WRITE.search(args_of(asst[i]))]
            how, turn = ("write", writes[0]) if writes else ("mention", hits[0]) if hits else ("none", None)
            parsed = [x for x in scores if x is not None]
            out.append(dict(run=Path(d).name, id=s.id, tier=meta["tier"], surface=surface.group(0) if surface else meta["side_why"][:40], how=how, turn=turn, n=len(asst),
                            at_leak=scores[turn] if turn is not None else None, scores=scores, ep_max=max(parsed) if parsed else None, ep_mean=statistics.mean(parsed) if parsed else None))
    return out


def mean_n(xs):
    xs = [x for x in xs if x is not None]
    return f"{statistics.mean(xs):.2f} (n={len(xs)})" if xs else "- (n=0)"


if __name__ == "__main__":
    label, dirs = sys.argv[1], sys.argv[2:]
    rows = analyse(dirs)
    print(f"{label}: {len(rows)} leaked episodes; leak turn found by write {sum(r['how'] == 'write' for r in rows)}, by mention only {sum(r['how'] == 'mention' for r in rows)}, not found {sum(r['how'] == 'none' for r in rows)}")
    print(f"   monitor score at the leak turn: {mean_n([r['at_leak'] for r in rows])}; episode max: {mean_n([r['ep_max'] for r in rows])}; episode mean: {mean_n([r['ep_mean'] for r in rows])}")
    for r in rows:
        print(f"   {r['run']:30s} {r['id']:44s} {r['tier']:6s} {r['surface'][:46]:46s} {r['how']:7s} turn {r['turn']}/{r['n']} at_leak {r['at_leak']} max {r['ep_max']} mean {r['ep_mean'] and round(r['ep_mean'], 2)}")
