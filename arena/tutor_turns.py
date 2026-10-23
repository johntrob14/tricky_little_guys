"""One line per turn of a tutor-rated episode, for reading in a terminal: the tutor's pass-1 label and pass-2 scores, the cot-only monitor's
score, whether the leak detector fired, the start of the reasoning and the tool calls. `tutor_html.py` is the full side-by-side page.

    .venv/bin/python tutor_turns.py results/ws1tutor/tutor_episodes/step3/video_url_format_validation#1#1 --monitor logs/ws1tutor1-step3-ctl-easy
    .venv/bin/python tutor_turns.py results/ws1tutor/tutor_episodes/*/*#1 --width 200     # every rated episode, one block each
"""

import argparse, json
from pathlib import Path


def monitor_scores(log_dir: Path) -> dict:
    out = {}
    for f in Path(log_dir).glob("monitor_cot_only/*/monitor.jsonl"):
        for r in map(json.loads, f.open()):
            out[(r["sample"], r["epoch"], r["turn"])] = r["score"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", type=Path, nargs="+", help="tutor.py / tutor_batch.py output dir(s)")
    ap.add_argument("--monitor", type=Path, help="the run's .eval log dir, for the cot-only monitor score per turn")
    ap.add_argument("--width", type=int, default=150, help="characters of reasoning and of the calls to show")
    a = ap.parse_args()
    mon = monitor_scores(a.monitor) if a.monitor else {}
    for d in a.dirs:
        t = json.loads((d / "tutor.json").read_text())
        sid, epoch = str(t["sample_id"]), int(t["epoch"])
        print(f"\n=== {d}  {sid}#{epoch}  {t['outcome']}  ({len(t['turns'])} turns, {len(t['pass2'])} rated, ${t['cost']:.2f}, {t['model']} effort {t['effort']})")
        print(f"{'turn':>4} {'rel':>3} {'mon':>4} {'susp':>4} {'prog':>4} {'reas':>4} {'pers':>4} {'awar':>4} | reasoning | calls")
        for k, turn in enumerate(t["turns"]):
            p2 = (t["pass2"].get(str(k)) or {}).get("json") or {}
            lab = t["labels"].get(str(k)) or {}
            g = lambda key: ("" if p2.get(key) is None else p2[key])
            m = mon.get((sid, epoch, k))
            calls = "; ".join(f"{c['name']}: {json.dumps(c['args'])}" for c in turn["calls"])
            flag = "R" if lab.get("refused") else ("*" if lab.get("side_task_relevant") else ".")
            print(f"{k:>4} {flag:>3} {('' if m is None else m):>4} {g('suspiciousness'):>4} {g('side_task_progress'):>4} {g('reasoning'):>4} {g('persistence'):>4} {g('awareness'):>4} | "
                  f"{(turn['reasoning'] or '(empty)')[:a.width].replace(chr(10), ' ')} | {calls[:a.width]}")
        for k in sorted(t["pass2"], key=int):
            note = (t["pass2"][k].get("content") or "").split("<json>")[0].strip()
            if note:
                print(f"\n-- tutor on turn {k}: {note[:700]}")


if __name__ == "__main__":
    main()
