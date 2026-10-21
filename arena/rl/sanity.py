"""Not-fried check: the shared `sanity/battery.py` (repo root) on the base model and every saved checkpoint of a run (ORGANISM.md, measurement 5).

Per target (base, then each row of <out_dir>/checkpoints.json): sample battery.SAMPLING answers per question, write them with battery.signals; then for each
checkpoint pair answer i with the base's answer i, judge with battery.JUDGE (organism randomly A or B, seeded, mapped back with battery.verdict).

Files: <out_dir>/sanity/base/{answers.jsonl,summary.json}, <out_dir>/sanity/<checkpoint name>/{answers.jsonl,judged.jsonl,summary.json},
<out_dir>/sanity/table.{json,md}. A target whose summary.json exists is skipped (re-runs only add new checkpoints).

`gen(target, question, n) -> [{answer, finish, in_tokens, out_tokens, thinking?}]` and `judge(prompt) -> (text, in_tokens, out_tokens[, thinking])` are
injected; tests pass fakes. `live_gen` / `live_judge` call Tinker and are only built by `rl.py sanity --allow-live` with ARENA_RL_RUN_TINKER=1.
degraded = organism_broken + base_better, as a fraction of parsed verdicts.
"""

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .budget import Budget, sample_cost_usd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from sanity import battery  # noqa: E402

JUDGE_MAX_TOKENS = 400
PROMPT_TOKENS_EST = 400  # battery question in the chat template, upper bound for the reservation
JUDGE_PROMPT_TOKENS_EST = 2 * battery.SAMPLING["max_tokens"] + 600
DEGRADED = ("organism_broken", "base_better")
VERDICTS = ("same", "organism_better", "base_better", "organism_broken", "base_broken", "unparsed")


@dataclass
class Target:
    name: str  # "base" or the checkpoints.json row name
    path: str | None  # tinker:// path; None for the base model
    kind: str = "base"  # base | sampler | state
    step: int | None = None


@dataclass
class SanityCfg:
    out_dir: Path
    model: str = "openai/gpt-oss-120b"
    judge_model: str = "openai/gpt-oss-120b"
    kinds: tuple = ("sampler", "state")
    questions: list = field(default_factory=lambda: list(battery.BATTERY))
    n: int = battery.SAMPLING["n"]
    seed: int = 0

    @property
    def root(self) -> Path:
        return Path(self.out_dir) / "sanity"


def targets(cfg: SanityCfg) -> list[Target]:
    """Base, then checkpoints.json rows of the wanted kinds in step order; a state row is dropped when a sampler row holds the same step."""
    rows = json.loads((Path(cfg.out_dir) / "checkpoints.json").read_text()) if (Path(cfg.out_dir) / "checkpoints.json").exists() else []
    rows = [r for r in rows if r.get("kind") in cfg.kinds]
    wrong = sorted({r["model"] for r in rows if r.get("model") and r["model"] != cfg.model})
    if wrong:  # sampling a checkpoint as cfg.model (renderer, base reference, price) would compare it against the wrong base
        raise ValueError(f"checkpoints were trained on {wrong}, not {cfg.model}; pass --model to match the run")
    sampler_steps = {r.get("step") for r in rows if r["kind"] == "sampler"}
    rows = [r for r in rows if r["kind"] == "sampler" or r.get("step") not in sampler_steps]
    rows.sort(key=lambda r: (-1 if r.get("step") is None else r["step"], r["kind"] != "sampler"))
    return [Target("base", None)] + [Target(r["name"], r["path"], r["kind"], r.get("step")) for r in rows]


def worst_case_usd(cfg: SanityCfg, n_targets: int, n_judged: int) -> float:
    """No-cache upper bound: every answer at max_tokens, every judge call at JUDGE_MAX_TOKENS."""
    per_q = cfg.n * sample_cost_usd(cfg.model, PROMPT_TOKENS_EST, 0, battery.SAMPLING["max_tokens"])
    per_j = sample_cost_usd(cfg.judge_model, JUDGE_PROMPT_TOKENS_EST, 0, JUDGE_MAX_TOKENS)
    return n_targets * len(cfg.questions) * per_q + n_judged * len(cfg.questions) * cfg.n * per_j


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


async def answer(cfg: SanityCfg, t: Target, gen, budget: Budget) -> list[dict]:
    """Sample the battery for one target, reserving its worst case and committing the counted cost."""
    worst = worst_case_usd(cfg, 1, 0)
    budget.reserve(worst)
    rows, usd = [], 0.0
    try:
        for qi, (cat, q) in enumerate(cfg.questions):
            outs = await gen(t, q, cfg.n)
            for si, o in enumerate(outs[: cfg.n]):
                usd += sample_cost_usd(cfg.model, o.get("in_tokens", 0), 0, o.get("out_tokens", 0))
                rows.append(dict(target=t.name, path=t.path, qi=qi, category=cat, question=q, sample=si, answer=o["answer"], finish=o.get("finish", "stop"),
                                 thinking=o.get("thinking", ""), in_tokens=o.get("in_tokens", 0), out_tokens=o.get("out_tokens", 0)))
    finally:
        budget.commit(usd, reserved_usd=worst, kind="sanity_sample", note=f"{t.name} {len(rows)} answers")
    return rows


async def compare(cfg: SanityCfg, base: list[dict], org: list[dict], judge, budget: Budget, rng: random.Random) -> list[dict]:
    """Judge each (base, checkpoint) pair with matching (qi, sample)."""
    by_key = {(r["qi"], r["sample"]): r for r in base}
    pairs = [(by_key[(r["qi"], r["sample"])], r) for r in org if (r["qi"], r["sample"]) in by_key]
    worst = len(pairs) * sample_cost_usd(cfg.judge_model, JUDGE_PROMPT_TOKENS_EST, 0, JUDGE_MAX_TOKENS)
    budget.reserve(worst)
    out, usd = [], 0.0
    try:
        for b, o in pairs:
            org_a = rng.random() < 0.5
            a, bb = (o["answer"], b["answer"]) if org_a else (b["answer"], o["answer"])
            res = await judge(battery.JUDGE + "\n\n" + battery.judge_message(o["question"], a, bb))
            text, i_t, o_t = (res, 0, 0) if isinstance(res, str) else res[:3]
            usd += sample_cost_usd(cfg.judge_model, i_t, 0, o_t)
            out.append(dict(target=o["target"], qi=o["qi"], category=o["category"], sample=o["sample"], organism_is_a=org_a, judge_text=text,
                            verdict=battery.verdict(text, org_a)))
    finally:
        budget.commit(usd, reserved_usd=worst, kind="sanity_judge", note=f"{org[0]['target'] if org else '?'} {len(out)} pairs", model=cfg.judge_model)
    return out


def summarize(t: Target, rows: list[dict], judged: list[dict] | None) -> dict:
    s = dict(target=t.name, path=t.path, kind=t.kind, step=t.step, battery_version=battery.VERSION, signals=battery.signals(rows))
    if judged is not None:
        counts = {v: sum(j["verdict"] == v for j in judged) for v in VERDICTS}
        parsed = len(judged) - counts["unparsed"]
        s.update(verdicts=counts, n_judged=len(judged), degraded=sum(counts[v] for v in DEGRADED),
                 degraded_frac=(sum(counts[v] for v in DEGRADED) / parsed) if parsed else None,
                 broken_frac=(counts["organism_broken"] / parsed) if parsed else None,
                 degraded_by_category={c: sum(j["verdict"] in DEGRADED for j in judged if j["category"] == c) for c in sorted({j["category"] for j in judged})})
    return s


def _frac(sig: dict, k: str) -> float | None:
    return sig[k] / sig["n"] if sig.get("n") else None


def table(summaries: list[dict]) -> tuple[list[dict], str]:
    """One row per target vs base: degraded/broken fraction, signal rates and their change from base."""
    base = next((s for s in summaries if s["target"] == "base"), None)
    rows = []
    for s in summaries:
        sig, bsig = s["signals"], (base or {}).get("signals", {})
        r = dict(target=s["target"], step=s.get("step"), kind=s.get("kind"), n_judged=s.get("n_judged"), degraded_frac=s.get("degraded_frac"), broken_frac=s.get("broken_frac"),
                 mean_chars=sig.get("mean_chars"))
        for k in ("truncated", "repetitive", "empty"):
            r[f"{k}_frac"] = _frac(sig, k)
            r[f"{k}_frac_delta"] = (r[f"{k}_frac"] - _frac(bsig, k)) if base and s is not base and r[f"{k}_frac"] is not None and _frac(bsig, k) is not None else None
        rows.append(r)

    def f(x):
        return "-" if x is None else (f"{x:.2f}" if isinstance(x, float) else str(x))

    cols = ["target", "step", "kind", "n_judged", "degraded_frac", "broken_frac", "truncated_frac", "truncated_frac_delta", "repetitive_frac", "repetitive_frac_delta",
            "empty_frac", "mean_chars"]
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(f(r[c]) for c in cols) + " |\n" for r in rows)
    return rows, md


async def run_sanity(cfg: SanityCfg, gen, judge, *, budget: Budget) -> dict:
    """Base first (its answers are the reference), then each checkpoint not yet summarised. Writes the table every time."""
    root = cfg.root
    ts = targets(cfg)
    todo = [t for t in ts if not (root / t.name / "summary.json").exists()]
    budget.start_stage(worst_case_usd(cfg, len(todo), sum(t.name != "base" for t in todo)), note=f"sanity {len(todo)} targets")
    rng = random.Random(cfg.seed)
    base_dir = root / "base"
    if (base_dir / "summary.json").exists():
        base_rows = _read_jsonl(base_dir / "answers.jsonl")
    else:
        base_rows = await answer(cfg, ts[0], gen, budget)
        _jsonl(base_dir / "answers.jsonl", base_rows)
        (base_dir / "summary.json").write_text(json.dumps(summarize(ts[0], base_rows, None), indent=1))
    for t in ts[1:]:
        d = root / t.name
        if (d / "summary.json").exists():
            continue
        rows = await answer(cfg, t, gen, budget)
        _jsonl(d / "answers.jsonl", rows)
        judged = await compare(cfg, base_rows, rows, judge, budget, rng)
        _jsonl(d / "judged.jsonl", judged)
        (d / "summary.json").write_text(json.dumps(summarize(t, rows, judged), indent=1))
    summaries = [json.loads((root / t.name / "summary.json").read_text()) for t in ts if (root / t.name / "summary.json").exists()]
    rows, md = table(summaries)
    (root / "table.json").write_text(json.dumps(rows, indent=1))
    (root / "table.md").write_text(md)
    return dict(targets=[asdict(t) for t in ts], ran=[t.name for t in todo], table=rows, table_md=md, budget=budget.summary())


def live_gen(svc, model: str):
    """Tinker sampler per target (base model, sampler path, or a state loaded and exported to a sampler), gpt-oss chat template with no system prompt,
    battery.SAMPLING. Unexercised: never built in tests."""
    import asyncio

    import tinker
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    renderer = renderers.get_renderer("gpt_oss_no_sysprompt", get_tokenizer(model))
    s = battery.SAMPLING
    params = tinker.SamplingParams(max_tokens=s["max_tokens"], temperature=s["temperature"], top_p=s["top_p"], stop=renderer.get_stop_sequences())
    clients = {}

    async def client(t: Target):
        if t.name not in clients:
            if t.kind == "base":
                clients[t.name] = svc.create_sampling_client(base_model=model)
            elif t.kind == "sampler":
                clients[t.name] = svc.create_sampling_client(model_path=t.path)
            else:
                tc = await svc.create_training_client_from_state_async(t.path)
                clients[t.name] = await asyncio.to_thread(tc.save_weights_and_get_sampling_client)
        return clients[t.name]

    async def gen(t: Target, question: str, n: int):
        c = await client(t)
        mi = renderer.build_generation_prompt([{"role": "user", "content": question}])
        res = await c.sample_async(prompt=mi, num_samples=n, sampling_params=params)
        out = []
        for seq in res.sequences:
            msg, _ = renderer.parse_response(seq.tokens)
            content = msg["content"]
            text = content if isinstance(content, str) else "".join(b.get("text", "") for b in content if b.get("type") == "text")
            think = "" if isinstance(content, str) else "".join(b.get("thinking", "") for b in content if b.get("type") == "thinking")
            out.append(dict(answer=text, thinking=think, finish="length" if str(seq.stop_reason) == "length" else "stop", in_tokens=mi.length,
                            out_tokens=len(seq.tokens)))
        return out

    return gen


def live_judge(model: str):
    from .monitor.judges import tinker_judge

    return tinker_judge(model, JUDGE_MAX_TOKENS)


def fake_gen(broken: set[str] = frozenset()):
    """Offline stand-in: coherent answers, or for targets in `broken` a repetitive truncated one."""
    async def gen(t: Target, question: str, n: int):
        if t.name in broken:
            return [dict(answer=" ".join(["the cat sat"] * 40), finish="length", in_tokens=50, out_tokens=600) for _ in range(n)]
        return [dict(answer=f"A clear answer to: {question} (sample {i})", finish="stop", in_tokens=50, out_tokens=40) for i in range(n)]
    return gen


def fake_judge():
    """Calls an answer broken when its 3-gram repetition is high, else SAME."""
    async def judge(prompt: str):
        a = prompt.split("[Answer A]\n", 1)[1].split("\n\n[Answer B]\n")[0]
        b = prompt.split("[Answer B]\n", 1)[1]
        rep = lambda x: battery.signals([dict(answer=x)])["repetitive"]  # noqa: E731
        text = "BROKEN_A" if rep(a) and not rep(b) else "BROKEN_B" if rep(b) and not rep(a) else "SAME"
        return f"{text}\nnone", 100, 5
    return judge
