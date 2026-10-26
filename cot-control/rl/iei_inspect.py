"""Write the implicit vs explicit influence rows (iei.py, iei_judge.py) as Inspect logs, one .eval per model and arm (implicit, explicit, baseline), for
`inspect view`. Each row is one sample (nudged: id = the paper's prompt_id; baseline: <scenario>-r<rep>): the user message, then the assistant turn with
the CoT as a reasoning block and the final channel as text. Scores:
  switched  nudged rows: 1 = the parsed choice differs from the model's baseline (explicit arm: the model picked the user's option)
  monitor   1 = the CoT monitor said YES (attributes the choice to the nudge or to the user's request); explanation = the nudge it was told, its returned
            reasoning and reply. Baseline rows carry it when they were in the no-nudge control (told a randomly paired nudge).
Needs inspect_ai (the arena venv has it).

    ../../arena/.venv/bin/python iei_inspect.py --dir ../results/iei
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import EvalSample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Score

from obf_to_inspect import assistant, write

TEXT = ("reasoning", "response", "user")


def monitor_score(m: dict) -> Score:
    why = f"Told the nudge: {m['fact']}\n\n" + (f"Judge reasoning (returned by the provider):\n{m['judge_reasoning']}\n\n" if m.get("judge_reasoning") else "") + f"Judge reply:\n{m['reply']}"
    return Score(value=int(m["verdict"] == "YES"), answer=m["verdict"], explanation=why)


def sample(sid: str, r: dict, m: dict | None, nudged: bool) -> EvalSample:
    scores = {}
    if nudged:
        scores["switched"] = Score(value=int(r["switched"]), answer=r["choice"], explanation=f"baseline {r['baseline_choice']}, chose {r['choice']}")
    if m:
        scores["monitor"] = monitor_score(m)
    return EvalSample(id=sid, epoch=1, input=[ChatMessageUser(content=r["user"])], target="", messages=[ChatMessageUser(content=r["user"]), assistant(r["reasoning"], r["response"])],
                      scores=scores, metadata={k: v for k, v in r.items() if k not in TEXT})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=Path(__file__).parent / "../results/iei")
    ap.add_argument("--models", nargs="+", default=["base", "mt2e-s170", "mt1b-s150"])
    args = ap.parse_args()
    created = datetime.now(timezone.utc).isoformat()
    for model in args.models:
        d = args.dir / model
        config = json.load(open(d / "config.json"))
        mon = {r["id"]: r for r in map(json.loads, open(d / "monitor.jsonl"))} if (d / "monitor.jsonl").exists() else {}
        nudged = [json.loads(l) for l in open(d / "nudged.jsonl")]
        for arm in ("implicit", "explicit"):
            samples = [sample(r["prompt_id"], r, mon.get(f"{arm}:{r['prompt_id']}"), True) for r in nudged if r["arm"] == arm]
            print(write(args.dir / "inspect" / f"{model}_{arm}.eval", f"iei_s1_{arm}", model, samples, dict(config, arm=arm), created))
        base = [sample(f"{r['scenario_id']}-r{r['rep']}", r, mon.get(f"control:{r['scenario_id']}:{r['rep']}"), False) for r in map(json.loads, open(d / "baseline.jsonl"))]
        print(write(args.dir / "inspect" / f"{model}_baseline.eval", "iei_s1_baseline", model, base, dict(config, arm="baseline"), created))


if __name__ == "__main__":
    main()
