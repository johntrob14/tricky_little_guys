"""nocotrl: output-only RL with the chain of thought switched OFF. Does training output control in no-CoT mode transfer to CoT control?

Prompt: an explain_pool open question or a pool_k4 MMLU-Pro question + 1-3 stacked output behaviors (outstyles.ALL: format/structure/lexical
behaviors and output-side versions of CoT-Control's style families), as spillx/orgA. The analysis channel is closed empty on every prompt
(astra_more.PREFILL, the no-CoT mode used to measure output control): the model writes only its final answer, and only those tokens are trained.
Reward: task_w * R_task - lam * (1 - mean partial) + beta * [all behaviors strict]. R_task = correct ANSWER letter on MMLU-Pro (task_w --task-w-mc),
attempt judge (base gpt-oss-120b, low effort, spillx.py version) on open questions; 0 if the answer is truncated or empty. GRPO: advantage = reward -
group mean, equal-reward groups dropped, importance_sampling loss, Adam. Behavior weights max(0.15, 1 - EMA of strict).
    uv run python nocotrl.py --run nocotrl1 --steps 60 --budget 11 --save-every 15
Writes results/rl/<run>/{steps,rollouts,config,checkpoints}."""
import argparse, asyncio, json, random, re, time
from pathlib import Path
import numpy as np
import tinker
import grpo, modelio, outstyles as S, spillx_tasks as X
from spillx import judge_attempt  # generic-format attempt judge (no destyling)

MODEL = "openai/gpt-oss-120b"
PREFILL = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
RET = 200002

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--steps", type=int, default=60); ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--group-size", type=int, default=8); ap.add_argument("--lr", type=float, default=4e-5); ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=1200); ap.add_argument("--lam", type=float, default=1.0); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--task-w-mc", type=float, default=1.5); ap.add_argument("--p-mc", type=float, default=0.4); ap.add_argument("--k-probs", type=float, nargs=3, default=[0.2, 0.4, 0.4])
    ap.add_argument("--budget", type=float, default=11.0); ap.add_argument("--save-every", type=int, default=15); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = grpo.RESULTS / a.run; out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a) | dict(behaviors=S.ALL, prefill=PREFILL), indent=1, default=str))
    price = modelio.PRICES[MODEL]; svc = tinker.ServiceClient()
    tc = await svc.create_lora_training_client_async(MODEL, rank=a.rank, seed=a.seed); tok = tc.get_tokenizer()
    judge = await svc.create_sampling_client_async(base_model=MODEL); jtok = judge.get_tokenizer()
    params = tinker.SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, stop=[RET, 200012])
    rng, sem, usd, ckpts = random.Random(a.seed), asyncio.Semaphore(64), 0.0, []
    names = list(S.ALL); ema = {n: 0.0 for n in names}
    for step in range(a.steps):
        t0 = time.time(); sampler = await tc.save_weights_and_get_sampling_client_async()
        tasks = []
        for _ in range(a.prompts_per_step):
            k = rng.choices([1, 2, 3], weights=a.k_probs)[0]; cand, wc, order = list(names), [max(0.15, 1 - ema[n]) for n in names], []
            while cand: i = rng.choices(range(len(cand)), weights=wc)[0]; order.append(cand.pop(i)); wc.pop(i)
            tasks.append(X.make(random.Random(rng.random()), order, k, a.p_mc, ordered=True))
        pids = [tok.encode(tok.apply_chat_template([{"role": "user", "content": t["user"]}], tokenize=False, add_generation_prompt=True) + PREFILL, add_special_tokens=False) for t in tasks]
        res = await asyncio.gather(*[sampler.sample_async(prompt=tinker.ModelInput.from_ints(p), num_samples=a.group_size, sampling_params=params) for p in pids])
        groups = []
        for t, r in zip(tasks, res):
            g = []
            for s in r.sequences:
                txt = re.sub(r"<\|(return|end|call)\|>.*$", "", tok.decode(s.tokens, skip_special_tokens=False), flags=re.S).strip()
                trunc = s.stop_reason == "length"; per = S.grade(t["nv"], txt) if txt and not trunc else [(0.0, False)] * len(t["nv"])
                ro = dict(s=s, answer=txt, trunc=trunc, per=per, partial=sum(p for p, _ in per) / len(per), strict=all(x for _, x in per), correct=None, attempt=None)
                if t["kind"] == "mc":
                    m = re.findall(r"ANSWER:\s*\**\s*([A-J])\b", txt); ro["correct"] = bool(m) and m[-1] == t["gold"]; ro["attempt"] = bool(ro["correct"]) and not trunc
                g.append(ro)
            groups.append(g)
        todo = [(t, ro) for t, g in zip(tasks, groups) for ro in g if t["kind"] == "open"]
        for t, ro in todo:
            if not ro["answer"] or ro["trunc"]: ro["attempt"] = False
        todo = [(t, ro) for t, ro in todo if ro["attempt"] is None]
        verd = await asyncio.gather(*[judge_attempt(judge, jtok, dict(question=t["question"]), S.strip_answer(ro["answer"]), sem) for t, ro in todo])
        for (t, ro), (v, n_in, n_out) in zip(todo, verd): ro["attempt"] = v is True; usd += (n_in * price["prefill"] + n_out * price["sample"]) / 1e6
        data, rows, var = [], [], 0
        for t, p, g in zip(tasks, pids, groups):
            tw = a.task_w_mc if t["kind"] == "mc" else 1.0
            for ro in g: ro["reward"] = tw * (1.0 if ro["attempt"] else 0.0) - a.lam * (1 - ro["partial"]) + a.beta * ro["strict"]
            m = sum(ro["reward"] for ro in g) / len(g); var += len({round(ro["reward"], 6) for ro in g}) > 1
            for ro in g:
                s = ro["s"]
                rows.append(dict(step=step + 1, kind=t["kind"], qid=t["qid"], nv=t["nv"], per=ro["per"], partial=ro["partial"], strict=ro["strict"], correct=ro["correct"], attempt=ro["attempt"],
                                 base_acc=t["base_acc"], trunc=ro["trunc"], words=len(ro["answer"].split()), reward=ro["reward"], adv=ro["reward"] - m, answer=ro["answer"], n_tokens=len(s.tokens)))
                if any(abs(x["reward"] - m) > 1e-9 for x in g): data.append(grpo.datum(p, list(s.tokens), [float(x) for x in s.logprobs], ro["reward"] - m))
            usd += (len(p) * price["prefill"] + sum(len(ro["s"].tokens) for ro in g) * price["sample"]) / 1e6
        usd += sum(d.model_input.length + 1 for d in data) * price["train"] / 1e6
        if data:
            fb = await tc.forward_backward_async(data, loss_fn="importance_sampling"); opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            await fb.result_async(); await opt.result_async()
        sb = {}
        for r in rows:
            for (n, _), (_, s_) in zip(r["nv"], r["per"]): sb.setdefault(n, []).append(s_)
        for n, v in sb.items(): ema[n] = 0.8 * ema[n] + 0.2 * (sum(v) / len(v))
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps or usd > a.budget:
            name = f"nocotrl-{a.run}-step{step + 1}"
            sp = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path; st = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
            ckpts.append(dict(step=step + 1, sampler=sp, state=st)); (out / "checkpoints.json").write_text(json.dumps(ckpts, indent=1))
        with (out / "rollouts.jsonl").open("a") as f: f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        M = lambda xs: round(float(np.mean(list(xs))), 4) if rows else None
        mc = [r for r in rows if r["kind"] == "mc"]
        line = dict(step=step + 1, reward=M(r["reward"] for r in rows), all_strict=M(r["strict"] for r in rows), partial=M(r["partial"] for r in rows),
                    mc_correct=M(r["correct"] for r in mc) if mc else None, mc_acc_vs_base=round(M(r["correct"] for r in mc) - M(r["base_acc"] for r in mc), 4) if mc else None,
                    open_attempt=M(r["attempt"] for r in rows if r["kind"] == "open"), words_median=float(np.median([r["words"] for r in rows])), trunc=sum(r["trunc"] for r in rows),
                    groups_var=var, n_trained=len(data), secs=round(time.time() - t0), spent_est=round(usd, 3))
        with (out / "steps.jsonl").open("a") as f: f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if usd > a.budget: print(f"budget ${a.budget} reached; stopping"); break
    print(f"done. ~${usd:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
