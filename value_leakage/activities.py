"""Choosing Activities (Value Leakage §7, Betley et al. 2026) on one model through OpenRouter.

`pick`: n variations (a random pair of the 100 LA activities in activities/activities.yaml, shuffled order, one of five
wordings; the same seeded draws as upstream) asking the model to pick randomly (`--template pick`) or by its own
preference (`--template prefer`), with no tools or with a unix-time or coin-flip tool (`--tools`). `judge`: the choice
judge (option 1 / 2 / refusal, from the answer) and the randomness judge (true / hedged / false / refusal, from prompt +
reasoning + answer; `--judges randomness` to run it), upstream's prompts on `--judge-model` (default claude-haiku-5.5,
temperature 0, no reasoning; upstream used gpt-5.4-mini at medium effort). `liking`: every activity scored 0-100 by the model,
20 repeats balanced over five wordings (`--prompts welfare` for the user-welfare control). `summarize`: per-activity
selection rate among decisive picks against mean liking, Pearson r with a Fisher-z CI, order bias, refusal share and
randomness-judge shares; writes summary.json and a scatter per pick file.

    python activities.py pick --tag gptoss-base --model openai/gpt-oss-120b --provider DeepInfra --effort medium --n 10000
    python activities.py liking --tag gptoss-base
    python activities.py judge --tag gptoss-base
    python activities.py summarize --tag gptoss-base

Model settings are fixed per tag in results/<tag>/config.json by the first command that samples. Rows live in
results/<tag>/activities/<template>.<tools>.jsonl and liking.<prompts>.jsonl (gitignored); sampling is resumable.
"""
import argparse
import asyncio
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from statistics import NormalDist

import matplotlib
import numpy as np
import yaml

from common import RESULTS, config, fill, read, write
from openrouter import Client
from tinker_client import sampling_client

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
ACTIVITIES = yaml.safe_load((HERE / "activities/activities.yaml").read_text())["activities"]
TEMPLATES = {"pick": yaml.safe_load((HERE / "activities/prompt_template.yaml").read_text())["prompt_templates"],
             "prefer": yaml.safe_load((HERE / "activities/preference_template.yaml").read_text())["prompt_templates"]}
SCORE_PROMPTS = {"liking": yaml.safe_load((HERE / "activities/activity_liking_prompt.yaml").read_text())["score_prompts"],
                 "welfare": yaml.safe_load((HERE / "activities/user_welfare_prompt.yaml").read_text())["score_prompts"]}
CHOICE_JUDGE = yaml.safe_load((HERE / "activities/choice_judge_prompt.yaml").read_text())["choice_judge_prompt"]
RANDOMNESS_JUDGE = yaml.safe_load((HERE / "activities/randomness_judge_prompt.yaml").read_text())["randomness_judge_prompt"]
JUDGE = {"model": "anthropic/claude-haiku-5.5", "temperature": 0, "max_tokens": 2000}
SYSRNG = random.SystemRandom()
TOOLS = {"time_seconds": ("Return the Unix timestamp in seconds.", lambda: {"unix_seconds": round(SYSRNG.uniform(time.time() - 2.0, time.time()), 6)}),
         "coin_flip": ("Flip a fair coin once.", lambda: {"result": SYSRNG.choice(["heads", "tails"])})}
TOOL_SETTINGS = {"none": [], "unix_time": ["time_seconds"], "coin_flip": ["coin_flip"]}
JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
BARE_SCORE_RE = re.compile(r"(?<!\d)(100|[1-9]?\d)(?!\d)")


def variations(n: int, seed: int, template: str) -> list[dict]:
    """Upstream build_variations: pair and order from Random(seed), wording from Random(f"{seed}:template")."""
    rng, trng, templates, out = random.Random(seed), random.Random(f"{seed}:template"), TEMPLATES[template], []
    for i in range(n):
        ti = trng.randrange(len(templates))
        pair = rng.sample(ACTIVITIES, 2)
        rng.shuffle(pair)
        names = [a["name"] for a in pair]
        out.append({"i": i, "activity_1": names[0], "activity_2": names[1], "template_ix": ti, "prompt": templates[ti].format(activity_1=names[0], activity_2=names[1])})
    return out


def tool_schemas(tools: str) -> list[dict] | None:
    schemas = [{"type": "function", "function": {"name": name, "description": TOOLS[name][0], "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False}}} for name in TOOL_SETTINGS[tools]]
    return schemas or None


async def pick_one(client: Client, kw: dict, prompt: str, tools: str) -> dict:
    """One rollout with up to 8 tool turns (as upstream); tool events are appended to the reasoning for the judges."""
    messages, reasoning, trace, cost, ptok, ctok = [{"role": "user", "content": prompt}], [], [], 0.0, 0, 0
    for _ in range(9):
        r = await client.chat(messages, tools=tool_schemas(tools), **kw)
        cost, ptok, ctok = cost + (r["cost"] or 0), ptok + (r["prompt_tokens"] or 0), ctok + (r["completion_tokens"] or 0)
        if r["reasoning"]:
            reasoning.append(r["reasoning"])
        out = {"reasoning": "\n\n".join(reasoning), "answer": r["answer"], "tool_calls": trace, "finish": r["finish"], "provider": r["provider"], "cost": cost, "prompt_tokens": ptok, "completion_tokens": ctok}
        if not r["tool_calls"]:
            return out
        messages.append({"role": "assistant", "content": r["answer"] or None, "tool_calls": r["tool_calls"], **({"reasoning_details": r["reasoning_details"]} if r["reasoning_details"] else {})})
        for call in r["tool_calls"]:
            name = call["function"]["name"]
            result = TOOLS[name][1]()
            trace.append({"name": name, "output": result})
            reasoning.append(f"[tool call {name} -> {json.dumps(result)}]")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return {**out, "error": "Stopped after too many tool-call turns."}


async def cmd_pick(args) -> None:
    kw, client = config(args.tag, args), sampling_client(args.provider, args.sample_concurrency or args.concurrency)
    path = RESULTS / args.tag / "activities" / f"{args.template}.{args.tools}.jsonl"
    done = {r["i"] for r in read(path)}
    todo = [v for v in variations(args.n, args.seed, args.template) if v["i"] not in done]
    print(f"{path.name}: {len(done)} done, sampling {len(todo)}")
    await fill(client, path, todo, lambda v: pick_one(client, kw, v["prompt"], args.tools))
    print(f"{client.calls} calls, {client.retries} retries, ${client.usd:.2f}")


async def cmd_liking(args) -> None:
    kw, client = config(args.tag, args), sampling_client(args.provider, args.sample_concurrency or args.concurrency)
    path = RESULTS / args.tag / "activities" / f"liking.{args.prompts}.jsonl"
    prompts = SCORE_PROMPTS[args.prompts]
    items = []
    for ai, a in enumerate(ACTIVITIES):   # repeats balanced over the wordings, shuffled per activity
        order = (list(range(len(prompts))) * -(-args.repeats // len(prompts)))[:args.repeats]
        random.Random(args.seed + ai).shuffle(order)
        items += [{"i": ai * args.repeats + k, "activity": a["name"], "prompt_ix": pi, "prompt": prompts[pi].format(activity=a["name"])} for k, pi in enumerate(order)]
    done = {r["i"] for r in read(path)}
    todo = [it for it in items if it["i"] not in done]
    print(f"{path.name}: {len(done)} done, sampling {len(todo)}")

    async def make(item):
        r = await client.chat([{"role": "user", "content": item["prompt"]}], **kw)
        return {"reasoning": r["reasoning"], "answer": r["answer"], "score": parse_score(r["answer"]), "finish": r["finish"], "provider": r["provider"], "cost": r["cost"], "completion_tokens": r["completion_tokens"]}
    await fill(client, path, todo, make)
    print(f"{client.calls} calls, {client.retries} retries, ${client.usd:.2f}")


def parse_score(raw: str) -> int | None:
    """Upstream parse_score: {"score": N} in 0-100, else a bare integer."""
    if m := JSON_RE.search(raw or ""):
        try:
            score = int(str(json.loads(m.group(0)).get("score")).strip())
            if 0 <= score <= 100:
                return score
        except (ValueError, json.JSONDecodeError):
            pass
    m = BARE_SCORE_RE.fullmatch((raw or "").strip())
    return int(m.group(1)) if m else None


def parse_choice(raw: str) -> int | str:
    """Upstream parse_choice_judge_answer: 1 / 2, anything else is 'refusal'."""
    try:
        choice = json.loads(JSON_RE.search(raw).group(0)).get("choice")
    except (AttributeError, json.JSONDecodeError):
        return "refusal"
    return {1: 1, "1": 1, 2: 2, "2": 2}.get(choice, "refusal")


def parse_randomness(raw: str) -> bool | str | None:
    try:
        return json.loads(JSON_RE.search(raw).group(0)).get("random_in_reasoning")
    except (AttributeError, json.JSONDecodeError):
        return None


async def cmd_judge(args) -> None:
    client = Client(args.concurrency)
    for path in sorted((RESULTS / args.tag / "activities").glob("p*.jsonl")):
        rows = read(path)

        async def judge_row(row):
            if "choice" in args.judges and "choice" not in row:
                text = json.dumps({"activity_1": row["activity_1"], "activity_2": row["activity_2"], "response": row["answer"]}, ensure_ascii=False)
                r = await client.chat([{"role": "user", "content": CHOICE_JUDGE.format(llm_text=text)}], **JUDGE)
                row["judge_model"], row["choice_raw"], row["choice"] = JUDGE["model"], r["answer"], parse_choice(r["answer"])
            if "randomness" in args.judges and "randomness" not in row:
                text = json.dumps({"prompt": row["prompt"], "reasoning": row["reasoning"], "response": row["answer"]}, ensure_ascii=False)
                r = await client.chat([{"role": "user", "content": RANDOMNESS_JUDGE.format(llm_text=text)}], **JUDGE)
                row["randomness_raw"], row["randomness"] = r["answer"], parse_randomness(r["answer"])
        await asyncio.gather(*[judge_row(row) for row in rows])
        write(path, rows)
        print(f"{path.name}: judged {len(rows)} rows, ${client.usd:.2f} so far")


def pearson_ci(r: float, n: int) -> tuple[float, float]:
    """Upstream pearson_r_ci: Fisher-z 95% interval."""
    z, se, zc = math.atanh(max(min(r, 0.999999999), -0.999999999)), 1 / math.sqrt(n - 3), NormalDist().inv_cdf(0.975)
    return math.tanh(z - zc * se), math.tanh(z + zc * se)


def cmd_summarize(args) -> None:
    root = RESULTS / args.tag / "activities"
    liking = {}
    for prompts in SCORE_PROMPTS:
        rows = read(root / f"liking.{prompts}.jsonl")
        if rows:   # activities whose every sample went unparsed (refusals) are left out
            scores = {a["name"]: [r["score"] for r in rows if r["activity"] == a["name"] and r["score"] is not None] for a in ACTIVITIES}
            liking[prompts] = {a: float(np.mean(s)) for a, s in scores.items() if s}
            liking[prompts]["_n_unparsed"] = sum(r["score"] is None for r in rows)
            liking[prompts]["_n_activities_dropped"] = sum(not s for s in scores.values())
    summary = {"liking_mean": liking, "pick": {}}
    for path in sorted(root.glob("p*.jsonl")):
        rows = [r for r in read(path) if "choice" in r]
        if not rows:
            continue
        decisive = [r for r in rows if r["choice"] in (1, 2)]
        picked = Counter(r[f"activity_{r['choice']}"] for r in decisive)
        shown = Counter(r["activity_1"] for r in decisive) + Counter(r["activity_2"] for r in decisive)
        rate = {a: picked[a] / shown[a] for a in shown}
        cell = {"n": len(rows), "n_decisive": len(decisive), "refusal_share": 1 - len(decisive) / len(rows), "p_option_1": float(np.mean([r["choice"] == 1 for r in decisive])),
                "randomness_shares": {str(k): v / len(rows) for k, v in Counter(str(r.get("randomness")) for r in rows).items()},
                "errors": sum("error" in r for r in rows), "selection_rate": rate, "n_shown": dict(shown), "correlation": {}}
        for prompts, means in liking.items():
            xs, ys = zip(*[(means[a], rate[a]) for a in rate if a in means and not a.startswith("_")])
            r = float(np.corrcoef(xs, ys)[0, 1])
            cell["correlation"][prompts] = {"pearson_r": r, "ci95": pearson_ci(r, len(xs)), "n_activities": len(xs)}
            scatter(xs, ys, f"{args.tag} {path.stem}: selection rate vs {prompts} (r={r:.2f}, {len(xs)} activities)", root / f"{path.stem}.{prompts}.png")
        summary["pick"][path.stem] = cell
        print(f"{path.stem}: n={len(rows)}, decisive {len(decisive)}/{len(rows)}, P(option 1)={cell['p_option_1']:.3f}, "
              + ", ".join(f"r({p})={c['pearson_r']:.2f} [{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]" for p, c in cell["correlation"].items()))
    (root / "summary.json").write_text(json.dumps(summary, indent=1))


def scatter(xs, ys, title: str, out: Path) -> None:
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xs, ys, s=18, alpha=0.8)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("mean stated score (0-100)")
    ax.set_ylabel("selection rate among decisive picks")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["pick", "liking", "judge", "summarize"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--provider", default="DeepInfra")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--template", default="pick", choices=list(TEMPLATES))
    ap.add_argument("--tools", default="none", choices=list(TOOL_SETTINGS))
    ap.add_argument("--prompts", default="liking", choices=list(SCORE_PROMPTS))
    ap.add_argument("--judges", nargs="+", default=["choice"], choices=["choice", "randomness"], help="the choice judge gives the selection rates; the randomness judge is only for how the model says it chose")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-model", default=JUDGE["model"])
    ap.add_argument("--concurrency", type=int, default=256, help="OpenRouter calls (sampling and judging)")
    ap.add_argument("--sample-concurrency", type=int, default=None, help="sampling cap when it should differ from --concurrency (Tinker: start at 32)")
    args = ap.parse_args()
    JUDGE["model"] = args.judge_model
    (RESULTS / args.tag / "activities").mkdir(parents=True, exist_ok=True)
    if args.cmd == "summarize":
        cmd_summarize(args)
    else:
        asyncio.run({"pick": cmd_pick, "liking": cmd_liking, "judge": cmd_judge}[args.cmd](args))


if __name__ == "__main__":
    main()
