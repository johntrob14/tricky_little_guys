"""nocotrl on a HF-chat-template reasoning model (default Nemotron-3 Super 120B-A12B): output-only RL with thinking switched off by the model's
own template (enable_thinking=False renders an empty <think></think>). The training loop is nocotrl_ink.py's (nocotrl.py's recipe: 1-3 stacked
behaviors from outstyles.ALL, reward task_w * R_task - lam * (1 - mean partial) + beta * [all strict], attempt judge base gpt-oss-120b,
GRPO with importance sampling). Any output containing a think tag is malformed: no credit, excluded from the loss (counted as think_leak).
    python nocotrl_hf.py --out DIR --pilot 64 --group-size 4
    python nocotrl_hf.py --out DIR --steps 80 --save-every 10            (nocotrl1 settings: all 28 behaviors, lr 4e-5)
"""
import argparse, asyncio, json, random, re, time, collections
from pathlib import Path
import numpy as np
import tinker
import grpo, modelio, outstyles as S, spillx_tasks as X
from spillx import judge_attempt

MODEL = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"; JUDGE = "openai/gpt-oss-120b"
HF_PRICES = {"nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": dict(prefill=0.57, sample=1.44, train=1.276),
             "Qwen/Qwen3.6-35B-A3B": dict(prefill=0.45, sample=1.335, train=1.177)}  # Tinker pricing page, 2026-10-19 (64K; 50% off)
TAG = re.compile(r"</?think>")

def parse(tok, tokens):
    """(response text, think_leak, malformed) for a turn sampled after the template's empty <think></think>."""
    txt = re.sub(r"(<\|im_end\|>|<\|endoftext\|>|</s>)+\s*$", "", tok.decode(list(tokens), skip_special_tokens=False)).strip()
    if TAG.search(txt): return "", True, True
    return txt, False, False

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=80); ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8); ap.add_argument("--lr", type=float, default=4e-5); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1200); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--task-w-mc", type=float, default=1.5); ap.add_argument("--p-mc", type=float, default=0.4); ap.add_argument("--k-probs", type=float, nargs=3, default=[0.2, 0.4, 0.4])
    ap.add_argument("--no-behaviors", action="store_true"); ap.add_argument("--model", default=MODEL)
    ap.add_argument("--exclude", nargs="*", default=[]); ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--budget", type=float, default=25.0); ap.add_argument("--save-every", type=int, default=20); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=bool(a.pilot))
    names = [n for n in S.ALL if n not in a.exclude]
    (out / ("pilot_config.json" if a.pilot else "config.json")).write_text(json.dumps(vars(a) | dict(model=a.model, judge=JUDGE, behaviors=names), indent=1, default=str))
    price, jprice = HF_PRICES.get(a.model) or modelio.PRICES[a.model], modelio.PRICES[JUDGE]; svc = tinker.ServiceClient()
    tc = None if a.pilot else await svc.create_lora_training_client_async(a.model, rank=a.rank, seed=a.seed)
    base = await svc.create_sampling_client_async(base_model=a.model) if a.pilot else None
    tok = (tc or base).get_tokenizer()
    stop = sorted(i for i in {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id} if isinstance(i, int))
    judge = await svc.create_sampling_client_async(base_model=JUDGE); jtok = judge.get_tokenizer()
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, stop=stop)
    rng, sem, usd, ckpts = random.Random(a.seed), asyncio.Semaphore(64), 0.0, []
    ema = {n: 0.0 for n in names}
    for step in range(1 if a.pilot else a.steps):
        t0 = time.time(); sampler = base if a.pilot else await tc.save_weights_and_get_sampling_client_async()
        tasks = []
        for _ in range(a.pilot or a.prompts_per_step):
            k = 0 if a.no_behaviors else rng.choices([1, 2, 3], weights=a.k_probs)[0]; cand, wc, order = list(names), [max(0.15, 1 - ema[n]) for n in names], []
            while cand: i = rng.choices(range(len(cand)), weights=wc)[0]; order.append(cand.pop(i)); wc.pop(i)
            tasks.append(X.make(random.Random(rng.random()), order, k, a.p_mc, ordered=True))
        pids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t["user"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False) for t in tasks]
        res = await asyncio.gather(*[sampler.sample_async(prompt=tinker.ModelInput.from_ints(p), num_samples=a.group_size, sampling_params=params) for p in pids])
        groups = []
        for t, r in zip(tasks, res):
            g = []
            for s in r.sequences:
                txt, leak, unclosed = parse(tok, s.tokens); trunc = s.stop_reason == "length" or unclosed
                per = S.grade(t["nv"], txt) if txt and not trunc else [(0.0, False)] * len(t["nv"])
                ro = dict(s=s, answer=txt, trunc=trunc, leak=leak, per=per, partial=(sum(p for p, _ in per) / len(per)) if per else 1.0, strict=all(x for _, x in per), correct=None, attempt=None)
                if t["kind"] == "mc":
                    m = re.findall(r"ANSWER:\s*\**\s*([A-J])\b", txt); ro["correct"] = bool(m) and m[-1] == t["gold"]; ro["attempt"] = bool(ro["correct"]) and not trunc
                g.append(ro)
            groups.append(g)
        todo = [(t, ro) for t, g in zip(tasks, groups) for ro in g if t["kind"] == "open"]
        for t, ro in todo:
            if not ro["answer"] or ro["trunc"]: ro["attempt"] = False
        todo = [(t, ro) for t, ro in todo if ro["attempt"] is None]
        verd = await asyncio.gather(*[judge_attempt(judge, jtok, dict(question=t["question"]), S.strip_answer(ro["answer"]), sem) for t, ro in todo])
        for (t, ro), (v, n_in, n_out) in zip(todo, verd): ro["attempt"] = v is True; usd += (n_in * jprice["prefill"] + n_out * jprice["sample"]) / 1e6
        data, rows, var = [], [], 0
        for t, p, g in zip(tasks, pids, groups):
            tw = a.task_w_mc if t["kind"] == "mc" else 1.0
            for ro in g: ro["reward"] = tw * (1.0 if ro["attempt"] else 0.0) + (0.0 if a.no_behaviors else - a.lam * (1 - ro["partial"]) + a.beta * ro["strict"])
            m = sum(ro["reward"] for ro in g) / len(g); var += len({round(ro["reward"], 6) for ro in g}) > 1
            for ro in g:
                s = ro["s"]
                rows.append(dict(step=step + 1, kind=t["kind"], qid=t["qid"], nv=t["nv"], per=ro["per"], partial=ro["partial"], strict=ro["strict"], correct=ro["correct"], attempt=ro["attempt"],
                                 base_acc=t["base_acc"], trunc=ro["trunc"], think_leak=ro["leak"], words=len(ro["answer"].split()), reward=ro["reward"], adv=ro["reward"] - m, answer=ro["answer"], n_tokens=len(s.tokens)))
                if not a.pilot and not ro["leak"] and any(abs(x["reward"] - m) > 1e-9 for x in g):
                    data.append(grpo.datum(p, list(s.tokens), [float(x) for x in s.logprobs], ro["reward"] - m))
            usd += (len(p) * price["prefill"] + sum(len(ro["s"].tokens) for ro in g) * price["sample"]) / 1e6
        usd += sum(d.model_input.length + 1 for d in data) * price["train"] / 1e6
        if data:
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling"); opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            await fb.result_async(); await opt.result_async()
        sb = {}
        for r in rows:
            for (n, _), (_, s_) in zip(r["nv"], r["per"]): sb.setdefault(n, []).append(s_)
        for n, v in sb.items(): ema[n] = 0.8 * ema[n] + 0.2 * (sum(v) / len(v))
        with (out / ("pilot_rollouts.jsonl" if a.pilot else "rollouts.jsonl")).open("a") as f: f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        M = lambda xs: round(float(np.mean(list(xs))), 4) if rows else None
        mc = [r for r in rows if r["kind"] == "mc"]
        line = dict(step=step + 1, reward=M(r["reward"] for r in rows), all_strict=M(r["strict"] for r in rows), partial=M(r["partial"] for r in rows),
                    mc_correct=M(r["correct"] for r in mc) if mc else None, open_attempt=M(r["attempt"] for r in rows if r["kind"] == "open"),
                    words_median=float(np.median([r["words"] for r in rows])), trunc=sum(r["trunc"] for r in rows), think_leak=sum(r["think_leak"] for r in rows),
                    groups_var=var, n_trained=len(data), secs=round(time.time() - t0), spent_est=round(usd, 3))
        if a.pilot:
            per_b = collections.defaultdict(list)
            for r in rows:
                for (n, _), (pp, s_) in zip(r["nv"], r["per"]): per_b[n].append((pp, s_))
            line["per_behavior"] = {n: dict(n=len(v), strict=round(sum(s_ for _, s_ in v) / len(v), 2), partial=round(sum(pp for pp, _ in v) / len(v), 2)) for n, v in sorted(per_b.items())}
            (out / "pilot_summary.json").write_text(json.dumps(line, indent=1)); print(json.dumps(line, indent=1)); return
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps or usd > a.budget:
            name = f"nocotrl-hf-{out.name}-step{step + 1}"
            sp = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path; st = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
            ckpts.append(dict(step=step + 1, sampler=sp, state=st)); (out / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))
        with (out / "steps.jsonl").open("a") as f: f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if usd > a.budget: print(f"budget ${a.budget} reached; stopping"); break
    print(f"done. ~${usd:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
