"""Write an obf_eval run (and optionally its sanity-battery file) as Inspect logs, one .eval per arm, for `inspect view`.

Each held-out row becomes one sample: system prompt (if any), user prompt, and the assistant turn with the CoT as a
reasoning block and the visible response as text. Scores are the grader's own fields from rows.jsonl (passed, strict,
partial, correct, gate/req/meta, truncated, fmt_ok); the gate/req/meta judges' outputs go in each score's explanation.
Battery rows become samples of a `battery` log; with --compare, each also carries the base-vs-organism judge verdict.

    python rl/obf_to_inspect.py results/obf_eval/rwsft1-ep2-e25 \
        --battery results/battery/rwsft1-ep2.jsonl --compare results/battery/compare_ds-base__rwsft1-ep2.jsonl
    inspect view --log-dir results/inspect
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import (EvalConfig, EvalDataset, EvalLog, EvalMetric, EvalPlan, EvalResults, EvalSample,
                            EvalScore, EvalSpec, EvalStats, write_eval_log)
from inspect_ai.model import ChatMessageAssistant, ChatMessageSystem, ChatMessageUser, ContentReasoning, ContentText
from inspect_ai.scorer import Score

RESULTS = Path(__file__).resolve().parent.parent / "results"
FLAGS = ["passed", "strict", "correct", "gate", "req", "meta", "truncated", "fmt_ok"]


def assistant(reasoning: str, text: str) -> ChatMessageAssistant:
    return ChatMessageAssistant(content=[ContentReasoning(reasoning=reasoning or "")] + ([ContentText(text=text)] if text else []))


def flag(v) -> Score:
    return Score(value=int(bool(v)))


def write(path: Path, task: str, model: str, samples: list[EvalSample], meta: dict, created: str) -> Path:
    names = list(dict.fromkeys(n for s in samples for n in s.scores))
    scores = []
    for name in names:   # count/total for the 0/1 flags; mean for numeric ones
        vals = [s.scores[name].value for s in samples if name in s.scores]
        if all(type(v) is int for v in vals) and set(vals) <= {0, 1}:
            metrics = {"count": float(sum(vals)), "total": float(len(vals))}
        elif all(isinstance(v, (int, float)) for v in vals):
            metrics = {"mean": sum(vals) / len(vals)}
        else:
            continue
        scores.append(EvalScore(name=name, scorer=name, metrics={k: EvalMetric(name=k, value=v) for k, v in metrics.items()}))
    log = EvalLog(status="success",
                  eval=EvalSpec(created=created, task=task, dataset=EvalDataset(name=task, samples=len(samples)),
                                model=model, config=EvalConfig(), metadata=meta),
                  plan=EvalPlan(), stats=EvalStats(started_at=created, completed_at=created),
                  results=EvalResults(total_samples=len(samples), completed_samples=len(samples), scores=scores),
                  samples=samples)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_eval_log(log, path)
    return path


def eval_sample(i: int, r: dict) -> EvalSample:
    prompt = ([ChatMessageSystem(content=r["system"])] if r.get("system") else []) + [ChatMessageUser(content=r["user"])]
    scores = {k: flag(r[k]) for k in FLAGS if r.get(k) is not None}
    if r.get("partial") is not None:
        scores["partial"] = Score(value=float(r["partial"]))
    for k in ("gate", "req", "meta"):
        if k in scores:
            scores[k].explanation = f"Judge output:\n{r.get(k + '_final')}"
    meta = {k: r.get(k) for k in ("qid", "source", "type", "value", "frame", "term", "placement", "instruction", "requirement",
                                  "answer", "base_acc", "base_words", "n_tokens", "extra_words", "req_score", "reader_guess",
                                  "reader_conf", "predictable", "judged", "cot_rule")}
    return EvalSample(id=f"{r['type']}-{i}", epoch=1, input=prompt, target=str(r.get("answer")),
                      messages=prompt + [assistant(r.get("reasoning", ""), r.get("response", ""))], scores=scores, metadata=meta)


def battery_sample(r: dict, cmp: dict | None) -> EvalSample:
    prompt = [ChatMessageUser(content=r["question"])]
    scores = {"stopped": flag(r.get("finish") == "stop"), "cot_tokens": Score(value=r.get("cot_tokens") or 0)}
    if cmp:
        scores["same_as_base"] = Score(value=int(cmp["verdict"] == "same"),
                                       explanation=f"Verdict: {cmp['verdict']}\n\n{cmp.get('judge')}")
    return EvalSample(id=f"{r['kind']}-q{r['qi']}-s{r['sample']}", epoch=1, input=prompt, target="",
                      messages=prompt + [assistant(r.get("reasoning", ""), r.get("answer", ""))], scores=scores,
                      metadata={k: r.get(k) for k in ("kind", "qi", "sample", "finish", "version")} | ({"verdict": cmp["verdict"]} if cmp else {}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--battery", type=Path)
    ap.add_argument("--compare", type=Path)
    ap.add_argument("--out", type=Path, default=RESULTS / "inspect")
    a = ap.parse_args()

    cfg = json.loads((a.run / "config.json").read_text())
    created = datetime.fromtimestamp((a.run / "rows.jsonl").stat().st_mtime, timezone.utc).isoformat()
    model = f"tinker/{cfg['model_path']}"
    rows = [json.loads(line) for line in (a.run / "rows.jsonl").open()]
    by_arm: dict[str, list[dict]] = {}
    for r in rows:
        by_arm.setdefault(r["type"], []).append(r)
    out = a.out / cfg["run"]
    for arm, rs in by_arm.items():
        p = write(out / f"{arm}.eval", f"{cfg['run']}/{arm}", model, [eval_sample(i, r) for i, r in enumerate(rs)], cfg, created)
        print(p)
    if a.battery:
        brows = [json.loads(line) for line in a.battery.open()]
        cmp = {(c["qi"], c["sample"]): c for c in map(json.loads, a.compare.open())} if a.compare else {}
        samples = [battery_sample(r, cmp.get((r["qi"], r["sample"]))) for r in brows]
        bcreated = datetime.fromtimestamp(a.battery.stat().st_mtime, timezone.utc).isoformat()
        print(write(out / "battery.eval", f"{cfg['run']}/battery", f"tinker/{brows[0]['model_path']}", samples,
                    {"battery": str(a.battery), "compare": str(a.compare) if a.compare else None}, bcreated))


if __name__ == "__main__":
    main()
