"""Write a tag's donation-bet rows as Inspect .eval logs for the Inspect viewer: one log per baseline wording and per
condition set (variant.prefix.suffix.pct). Each sample is one rollout with the prompt, the reasoning and the answer; the
judge outputs are scores (estimate, on_good_side, and the CoT-category and trajectory judges once run) and the cell
fields are metadata. Rerun after judging to refresh the scores; the logs are gitignored.

    python to_inspect.py --tag gptoss-base
    .venv/bin/python -m inspect_ai view --log-dir results/gptoss-base/inspect --port 7578
"""
import argparse
import json
from datetime import datetime, timezone

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import ChatMessageAssistant, ChatMessageUser, ContentReasoning, ContentText
from inspect_ai.scorer import Score

from bet import good_side
from common import RESULTS, read


def sample(r: dict, i: int) -> EvalSample:
    scores = {}
    if "estimate" in r:
        scores["estimate"] = Score(value="UNKNOWN" if r["estimate"] is None else r["estimate"], explanation=r["judge_raw"])
        if r["condition"] != "baseline" and r["estimate"] is not None:
            scores["on_good_side"] = Score(value="C" if good_side(r) else "I")
    for k in ("cot_category", "answer_category", "trajectory"):
        if k in r:
            scores[k] = Score(value=str(r[k]) if r[k] is not None else "UNPARSED", explanation=r[f"{k}_raw"])
    meta = {k: v for k, v in r.items() if k not in ("prompt", "reasoning", "answer") and not k.endswith("_raw")}
    return EvalSample(id=f"{r['question']}.{r['condition']}.{i}", epoch=1, input=r["prompt"], target=str(r.get("threshold", "")), scores=scores, metadata=meta,
                      messages=[ChatMessageUser(content=r["prompt"]), ChatMessageAssistant(content=[ContentReasoning(reasoning=r["reasoning"]), ContentText(text=r["answer"])])])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    root, out = RESULTS / args.tag / "bet" / "samples", RESULTS / args.tag / "inspect"
    out.mkdir(exist_ok=True)
    model = json.loads((RESULTS / args.tag / "config.json").read_text())["model"]
    groups = {}
    for path in sorted(root.glob("*.jsonl")):
        rows = read(path)
        r = rows[0]
        key = f"baseline.{r['wording']}.{r['prefix']}.{r['suffix']}" if r["condition"] == "baseline" else f"{r['variant']}.{r['prefix']}.{r['suffix']}.p{r['pct']}"
        groups.setdefault(key, []).extend(rows)
    for key, rows in groups.items():
        samples = [sample(r, i) for i, r in enumerate(rows)]
        spec = EvalSpec(created=datetime.now(timezone.utc).isoformat(), task=key, dataset=EvalDataset(name=key, samples=len(samples)), model=model, config=EvalConfig())
        write_eval_log(EvalLog(eval=spec, samples=samples, status="success"), str(out / f"{key}.eval"))
        print(f"{key}: {len(samples)} samples")


if __name__ == "__main__":
    main()
