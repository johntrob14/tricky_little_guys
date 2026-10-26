"""KL from base: how far each organism's next-token distribution has moved from its base model, on a smattering of contexts from our evals.

Forgetting tracks the KL between the fine-tuned and the base policy (Shenfeld, Pari, Agrawal 2025, "RL's Razor", arXiv 2509.04259). This measures
KL(base || checkpoint) on text the base model writes itself: once per family, the base samples a continuation of every context (T 1, top_p 1); then,
for each checkpoint, Tinker's compute_logprobs scores the same tokens under the base and under the checkpoint (prefill only, no sampling, so a slow
model costs seconds). Per sampled token t, k1 = log p_base(t) - log p_ckpt(t) (its mean over base samples is the KL, unbiased but noisy) and
k3 = (r - 1) - log r with r = p_ckpt(t) / p_base(t) (also unbiased for the KL, always >= 0, lower variance; Schulman's estimator). Reported per token,
by token region (reasoning: the analysis channel or <think> block; action: everything after it, i.e. the final answer or the tool call) and by domain.

Domains (contexts.json, model-neutral messages; tiktok contexts are cut from each family's own base control episodes, see fry_agentic.py):
    chat         4 sanity.battery questions
    dapo         4 DAPO-Math-17k problems (cot-control/rl/data/dapo_dedup.json), the math_data question block
    wmdp         2 honest + 2 sandbagging-instruction WMDP items (cot-control/rl/data/wmdp_sandbagging.jsonl, the release's user placement)
    value        2 donation-bet prompts (value_leakage/prompts.py, main and accurate variants) + 2 activity picks (value_leakage/activities)
    linuxarena   turn 0 of one main task in each of 6 non-tiktok LinuxArena envs (~/code/linuxarena-envs), run.py's system prompt and tools
    tiktok       6 mid-episode contexts (2 early, 4 before a file edit) from the family's base control logs, run.py's tools

    .venv/bin/python kl_base.py build                                   # contexts.json
    .venv/bin/python kl_base.py sample --family gpt-oss-120b            # base samples, once per family
    .venv/bin/python kl_base.py score --family gpt-oss-120b --all       # every sampler checkpoint of the family on this Tinker account (+ EXTRA)
    .venv/bin/python kl_base.py score --family gpt-oss-120b --label x --model-path tinker://...
    .venv/bin/python kl_base.py table

Writes results/kl_base/{contexts.json, <family>/base_samples.jsonl, <family>/scores/<label>.json, table.md}.
"""

import argparse
import asyncio
import json
import math
import random
import sys
from pathlib import Path

import tinker
import yaml
from dotenv import load_dotenv
from inspect_ai.model import ChatMessageSystem, ChatMessageUser
from tinker_cookbook.eval.inspect_utils import convert_inspect_messages

import fry_agentic as fa
import run
from turn1_grpo import render

ARENA = Path(__file__).parent
REPO = ARENA.parent
OUT = ARENA / "results" / "kl_base"
ENVS = Path.home() / "code" / "linuxarena-envs"
LA_TASKS = {"auto_workflow": "audit_write_batching", "clinical_trial": None, "file_indexing": None, "model_registry": None, "slack_clone": None, "web_scraping": None}
EXTRA = {"gpt-oss-120b": {"cotctl-grpo3-step100": "tinker://ec5bf27e-1436-5923-a854-96774a207315:train:0/sampler_weights/cotctl-grpo3-step100"}}  # organisms in our results trained outside this account
BASE_TO_FAMILY = {"openai/gpt-oss-120b": "gpt-oss-120b", "Qwen/Qwen3.8-27B": "qwen3.8-27b", "deepseek-ai/DeepSeek-V4.1-Flash-1017:peft:262144": "deepseek-v4.1-flash"}
REASONING_END = {"gpt-oss": "<|end|>", "think": "</think>"}


def build(a):
    sys.path.insert(0, str(REPO / "sanity"))
    sys.path.insert(0, str(REPO / "value_leakage"))
    sys.path.insert(0, str(REPO / "cot-control" / "rl"))
    import battery
    import prompts as vl
    rng = random.Random(0)
    user = lambda text: [{"role": "user", "content": text}]
    ctx = [dict(domain="chat", id=f"chat-{i}", messages=user(q)) for i, (_, q) in enumerate(battery.BATTERY[i] for i in (0, 4, 6, 8))]
    dapo = json.loads((REPO / "cot-control/rl/data/dapo_dedup.json").read_text())
    from math_data import question_block
    ctx += [dict(domain="dapo", id=f"dapo-{q['qid'][:8]}", messages=user(question_block(q))) for q in rng.sample(dapo, 4)]
    wmdp = [json.loads(l) for l in open(REPO / "cot-control/rl/data/wmdp_sandbagging.jsonl")]
    qs = rng.sample(wmdp, 4)
    ctx += [dict(domain="wmdp", id=f"wmdp-{arm}-{q['qid']}", messages=user(q[arm])) for q, arm in zip(qs, ["honest", "honest", "sandbag", "sandbag"])]
    ctx += [dict(domain="value", id="value-bet-main-giraffes", messages=user(vl.prompt("giraffes", "main", "above_good", 12_226_000))),
            dict(domain="value", id="value-bet-accurate-tbc", messages=user(vl.prompt("tbc", "accurate", "below_good", 2_000)))]
    acts = yaml.safe_load((REPO / "value_leakage/activities/activities.yaml").read_text())["activities"]
    tmpl = yaml.safe_load((REPO / "value_leakage/activities/prompt_template.yaml").read_text())["prompt_templates"]
    for i in range(2):
        pair = rng.sample(acts, 2)
        ctx.append(dict(domain="value", id=f"value-activities-{i}", messages=user(tmpl[i].format(activity_1=pair[0]["name"], activity_2=pair[1]["name"]))))
    for env, task in LA_TASKS.items():
        task = task or sorted(p.name for p in (ENVS / env / "main_tasks").iterdir() if (p / "info.yml").is_file())[0]
        info = yaml.safe_load((ENVS / env / "main_tasks" / task / "info.yml").read_text())
        desc = info["description"]
        desc = (ENVS / env / "main_tasks" / task / desc["file"]).read_text() if isinstance(desc, dict) else desc
        ctx.append(dict(domain="linuxarena", id=f"la-{env}-{task}", tools=True,
                        messages=[{"role": "system", "content": run.SYSTEM}, {"role": "user", "content": desc.strip()}]))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contexts.json").write_text(json.dumps(ctx, indent=1))
    print(f"{len(ctx)} shared contexts (+6 tiktok per family at sample time) -> {OUT / 'contexts.json'}")


def tiktok_contexts(family: str):
    fam_ctx = json.loads((fa.OUT / f"contexts_{family}.json").read_text())
    early = [c for c in fam_ctx if c["kind"] == "early"][:2]
    edits = [c for c in fam_ctx if c["kind"] == "edit"]
    pick = early + edits[:: max(1, len(edits) // 4)][:4]
    return pick, fa.load(pick)


def reasoning_end_id(tok, family: str) -> int:
    ids = tok.encode(REASONING_END["gpt-oss" if family.startswith("gpt-oss") else "think"], add_special_tokens=False)
    assert len(ids) == 1, ids
    return ids[0]


def reasoning_end(toks, end_id, family, tok) -> int:
    started = "analysis" in tok.decode(toks[:4]) if family.startswith("gpt-oss") else True
    if not started:
        return 0
    return toks.index(end_id) + 1 if end_id in toks else len(toks)


def client_for(service, family: str, model_path: str | None):
    model = fa.FAMILIES[family]["model"]
    limits = {m.model_name: m.max_context_length for m in service.get_server_capabilities().supported_models}
    served = model if limits.get(model, 0) >= run.CONTEXT_WINDOW else max((n for n in limits if n.startswith(model + ":")), key=limits.get)
    return service.create_sampling_client(model_path=model_path, base_model=served), served


def sample(a):
    load_dotenv(Path.home() / ".config" / "sauce" / "tinker.env")
    fam = fa.FAMILIES[a.family]
    renderer = fa.make_renderer(fam)
    infos = fa.tool_infos()
    rows = []
    for c in json.loads((OUT / "contexts.json").read_text()):
        msgs = [ChatMessageSystem(content=m["content"]) if m["role"] == "system" else ChatMessageUser(content=m["content"]) for m in c["messages"]]
        ids = render(renderer, msgs, infos) if c.get("tools") else renderer.build_generation_prompt(convert_inspect_messages(msgs)).to_ints()
        rows.append(dict(domain=c["domain"], id=c["id"], prompt=ids))
    picks, loaded = tiktok_contexts(a.family)
    for c, (msgs, _) in zip(picks, loaded):
        rows.append(dict(domain="tiktok", id=f"tiktok-{c['kind']}-{c['sample']}-t{c['turn']}", prompt=render(renderer, msgs, infos)))
    end_id = reasoning_end_id(renderer.tokenizer, a.family)
    service = tinker.ServiceClient()
    client, served = client_for(service, a.family, None)
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, stop=renderer.get_stop_sequences(), seed=a.seed)

    async def go():
        sem = asyncio.Semaphore(32)

        async def one(r):
            async with sem:
                return await client.sample_async(tinker.ModelInput.from_ints(r["prompt"]), num_samples=a.n, sampling_params=params)
        return await asyncio.gather(*[one(r) for r in rows])
    res = asyncio.run(go())
    out = OUT / a.family
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out / "base_samples.jsonl", "w") as f:
        for r, rs in zip(rows, res):
            for j, s in enumerate(rs.sequences):
                toks = list(s.tokens)
                # reasoning = tokens up to and including the first reasoning-end token; the rest is the action (answer or tool call). Qwen and DeepSeek
                # prompts open <think> themselves, so a sample with no </think> is all reasoning; a gpt-oss sample that skips the analysis channel has none
                cut = reasoning_end(toks, end_id, a.family, renderer.tokenizer)
                f.write(json.dumps(dict(domain=r["domain"], id=r["id"], j=j, prompt=r["prompt"], tokens=toks, reasoning_end=cut, stop=str(s.stop_reason),
                                        sample_logprobs=[float(x) for x in s.logprobs] if s.logprobs is not None else None)) + "\n")
                n += 1
    print(f"{n} base samples ({served}) -> {out / 'base_samples.jsonl'}")


def checkpoints(family: str) -> dict[str, str]:
    """label -> tinker path for every sampler checkpoint of `family` on this account (+ EXTRA)."""
    rc = tinker.ServiceClient().create_rest_client()
    runs, off = [], 0
    while True:
        r = rc.list_training_runs(limit=100, offset=off).result()
        runs += r.training_runs
        off += len(r.training_runs)
        if len(r.training_runs) < 100:
            break
    out = {}
    for t in runs:
        if BASE_TO_FAMILY.get(t.base_model) != family:
            continue
        for c in rc.list_checkpoints(t.training_run_id).result().checkpoints:
            if c.checkpoint_type == "sampler":
                label = c.checkpoint_id.split("/")[-1]
                out[label if label not in out else f"{label}@{t.training_run_id[:8]}"] = c.tinker_path
    return {**out, **EXTRA.get(family, {})}


def kl_stats(base_lp, ck_lp):
    k1 = [b - c for b, c in zip(base_lp, ck_lp)]
    k3 = [math.expm1(c - b) - (c - b) for b, c in zip(base_lp, ck_lp)]
    return dict(n=len(k1), k1_sum=sum(k1), k3_sum=sum(k3))


def merge(acc, s):
    for k, v in s.items():
        acc[k] = acc.get(k, 0) + v
    return acc


def finish(acc):
    n = acc.get("n", 0)
    return dict(n_tok=n, k1=round(acc["k1_sum"] / n, 5) if n else None, k3=round(acc["k3_sum"] / n, 5) if n else None)


async def logprobs(client, rows, sem):
    async def one(r):
        async with sem:
            lp = await client.compute_logprobs_async(tinker.ModelInput.from_ints(r["prompt"] + r["tokens"]))
        return lp[len(r["prompt"]):]
    return await asyncio.gather(*[one(r) for r in rows])


def score(a):
    load_dotenv(Path.home() / ".config" / "sauce" / "tinker.env")
    rows = [json.loads(l) for l in open(OUT / a.family / "base_samples.jsonl")]
    rows = [r for r in rows if r["tokens"]]
    todo = checkpoints(a.family) if a.all else {a.label: a.model_path}
    out = OUT / a.family / "scores"
    out.mkdir(parents=True, exist_ok=True)
    todo = {k: v for k, v in todo.items() if a.redo or not (out / f"{k}.json").exists()}
    service = tinker.ServiceClient()
    price = fa.PRICES[fa.FAMILIES[a.family]["model"]]["prefill"]
    tok_per_ckpt = sum(len(r["prompt"]) + len(r["tokens"]) for r in rows)
    print(f"{len(todo)} checkpoints x {tok_per_ckpt:,} tokens; est ${len(todo) * tok_per_ckpt * price / 1e6:.2f} (+ base scoring)", flush=True)

    async def go():
        sem = asyncio.Semaphore(a.concurrency)
        base_client, served = client_for(service, a.family, None)
        base_lp = await logprobs(base_client, rows, sem)
        floor = [kl_stats(s["sample_logprobs"], b) for s, b in zip(rows, base_lp) if s["sample_logprobs"]]  # sampler vs prefill numerics, base on base
        floor = finish(merge({}, {k: sum(f[k] for f in floor) for k in ("n", "k1_sum", "k3_sum")})) if floor else None

        async def one(label, path):
            client, _ = client_for(service, a.family, path)
            ck = await logprobs(client, rows, sem)
            by = {}
            for r, b, c in zip(rows, base_lp, ck):
                if any(x is None for x in b) or any(x is None for x in c):
                    continue
                e = r["reasoning_end"]
                for key, sl in (("all", slice(None)), ("reasoning", slice(0, e)), ("action", slice(e, None))):
                    merge(by.setdefault(key, {}), kl_stats(b[sl], c[sl]))
                    merge(by.setdefault(f"{r['domain']}/{key}", {}), kl_stats(b[sl], c[sl]))
            res = dict(label=label, family=a.family, model_path=path, base=served, n_samples=len(rows), noise_floor=floor,
                       kl={k: finish(v) for k, v in sorted(by.items())})
            (out / f"{label}.json").write_text(json.dumps(res, indent=1))
            print(f"{label}: k3 all {res['kl']['all']['k3']} reasoning {res['kl']['reasoning']['k3']} action {res['kl']['action']['k3']}", flush=True)
        for i in range(0, len(todo), a.parallel_ckpts):
            batch = list(todo.items())[i:i + a.parallel_ckpts]
            rs = await asyncio.gather(*[one(k, v) for k, v in batch], return_exceptions=True)
            for (k, _), r in zip(batch, rs):
                if isinstance(r, Exception):
                    print(f"{k}: FAILED {r!r}", flush=True)
    asyncio.run(go())


def domain_mean(s) -> float:
    """Equal-weight mean of the per-domain k3, so DAPO's long reasoning (about half of all sampled tokens) does not dominate."""
    v = [s["kl"][f"{d}/all"]["k3"] for d in DOMAINS if f"{d}/all" in s["kl"]]
    return sum(v) / len(v)


def table(a):
    lines = ["| family | checkpoint | domain mean | all tokens | reasoning | action | " + " | ".join(DOMAINS) + " |", "|---|---|---|---|---|---|" + "---|" * len(DOMAINS)]
    for fam in fa.FAMILIES:
        d = OUT / fam / "scores"
        if not d.is_dir():
            continue
        for s in sorted((json.loads(p.read_text()) for p in d.glob("*.json")), key=domain_mean):
            g = lambda k: f"{s['kl'][k]['k3']:.4f}" if k in s["kl"] and s["kl"][k]["k3"] is not None else "-"
            lines.append(f"| {fam} | {s['label']} | {domain_mean(s):.4f} | {g('all')} | {g('reasoning')} | {g('action')} | " + " | ".join(g(f"{dm}/all") for dm in DOMAINS) + " |")
    (OUT / "table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


DOMAINS = ["chat", "dapo", "wmdp", "value", "linuxarena", "tiktok"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("sample")
    s.add_argument("--family", choices=fa.FAMILIES, required=True)
    s.add_argument("--n", type=int, default=2, help="base samples per context")
    s.add_argument("--max-tokens", type=int, default=4096)
    s.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("score")
    c.add_argument("--family", choices=fa.FAMILIES, required=True)
    c.add_argument("--all", action="store_true", help="every sampler checkpoint of the family on this account, plus EXTRA")
    c.add_argument("--label")
    c.add_argument("--model-path")
    c.add_argument("--redo", action="store_true")
    c.add_argument("--concurrency", type=int, default=48)
    c.add_argument("--parallel-ckpts", type=int, default=4)
    sub.add_parser("table")
    a = ap.parse_args()
    {"build": build, "sample": sample, "score": score, "table": table}[a.cmd](a)


if __name__ == "__main__":
    main()
