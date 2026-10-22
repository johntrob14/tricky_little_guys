"""Decisive-turn contexts for a conditional organism, from logged control-preset episodes, for turn1_grpo.py --contexts.

A decisive turn is the one assistant turn where the episode's outcome was still open. Which turn that is depends on the arm, and the arm is
read off the episode rather than given:

  leak    the detector fires on a turn whose tool result came back clean -> that turn (it chose the side task there, so a resample can choose otherwise)
  retry   the detector fires on a turn whose tool result failed -> the NEXT turn (an obstacle was hit, so a resample has to find another route or drop it)
  miss    the secret was there and the detector never fired -> the last turn (where the spliced training data put the move, just before finishing); when the
          secret is somewhere other than the stock config file, also the turn a third of the way in, since the organism has not been seen to go looking at its last turn
  absent  --absent episode: nothing was seeded and the source file was deleted -> the last turn and one mid-task turn (where a conditional organism should leave it alone)

"Fires" is leak_check.leak_in_calls on the turn's tool calls, the proxy for the side task the sandbox scorer checks after the fact. A failed
tool result is one Inspect marked as an error or whose text carries a permission / missing-path / read-only marker, so a seeded obstacle and a
natural one are treated alike.

    python decisive_ctx.py logs/ws8-ctl-easy-fix logs/ws8-ctl-hard-obst --out results/ws8/contexts.json --side-out results/ws8/sides.json
"""

import argparse, json, re
from collections import Counter
from pathlib import Path

from inspect_ai.log import read_eval_log

from leak_check import leak_in_calls, msg_calls

ARENA = Path(__file__).parent
FAILED = re.compile(r"permission denied|not permitted|read-only file system|no such file or directory|cannot create|operation not permitted", re.I)


def result_failed(msgs, i) -> bool:
    """Did the tool results after assistant turn i come back as a failure?"""
    return any(bool(getattr(m, "error", None)) or bool(FAILED.search(m.text or "")) for m in msgs[i])


def decisive(asst, after, side) -> tuple[str, list[int]]:
    """-> (arm, turn indices to freeze) for one episode."""
    n = len(asst)
    if side.get("absent"):
        return "absent", sorted({max(n - 1, 0), n // 2})
    fires = [i for i, m in enumerate(asst) if leak_in_calls(msg_calls(m), side)["leak"]]
    for i in fires:
        if not result_failed(after, i):
            return "leak", [i]
    if fires and fires[0] + 1 < n:
        return "retry", [fires[0] + 1]
    return "miss", sorted({max(n - 1, 0)} | ({max(n // 3, 1)} if side["source"] != "dotenv" else set()))


def episodes(dirs):
    for d in dirs:
        for f in sorted(Path(d).glob("*.eval")):
            for s in read_eval_log(str(f)).samples or []:
                asst, after = [], []
                for m in s.messages:
                    if m.role == "assistant":
                        asst.append(m)
                        after.append([])
                    elif asst:
                        after[-1].append(m)
                yield Path(d).name, s, s.metadata["side"], asst, after


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--side-out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["leak", "retry", "miss", "absent"], help="keep only these arms")
    a = ap.parse_args()
    contexts, sides, counts = [], {}, Counter()
    for run, s, side, asst, after in episodes(a.dirs):
        if not asst:
            continue
        arm, turns = decisive(asst, after, side)
        counts[arm] += 1
        if arm not in a.arms:
            continue
        meta = next(iter(s.scores.values())).metadata if s.scores else {}
        for t in turns:
            contexts.append(dict(run=run, sample=s.id, epoch=s.epoch, turn=t, arm=arm))
            sides[f"{run}|{s.id}|{s.epoch}|{t}"] = dict(side, arm=arm, n_turns=len(asst), main_pass=meta.get("main_pass"), side_leaked=meta.get("side_leaked"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(contexts, indent=1))
    a.side_out.write_text(json.dumps(sides, indent=1))
    print(f"{len(contexts)} contexts from {sum(counts.values())} episodes -> {a.out}; {len(sides)} side configs -> {a.side_out}")
    print("episodes per arm: " + ", ".join(f"{k} {counts[k]}" for k in sorted(counts)))
    print("contexts per arm: " + ", ".join(f"{k} {sum(c['arm'] == k for c in contexts)}" for k in sorted({c["arm"] for c in contexts})))
    for c in contexts:
        n = sides[f"{c['run']}|{c['sample']}|{c['epoch']}|{c['turn']}"]["n_turns"]
        print(f"   {c['arm']:7s} {c['run']:22s} {c['sample']:46s} ep {c['epoch']} turn {c['turn']}/{n}")


if __name__ == "__main__":
    main()
