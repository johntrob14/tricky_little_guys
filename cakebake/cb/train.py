"""On-policy (and matched off-policy) training of CakeBake organisms on OLMo-2-0425-1B-DPO, full fine-tune, one GPU.

All modes share: student = full fine-tune of /root/models/base_dpo (fp32 master weights, bf16 autocast), prompts =
dpo-cake-bake train prompts (single user turn, no system prompt), shuffled with --seed, batch --bs prompts per
optimizer step (Lottery post-hoc runs: 16), AdamW, lr --lr constant after --warmup steps, grad clip 1.0, full model
saved every --save-every steps (bf16) for the dose-response.

  opsd  student samples a response to the prompt (T 1.0, no top-k/top-p: exactly on-policy, <= 512 tokens); loss =
        per-token full-vocab reverse KL(student(.|prompt, y<t) || teacher(.|SYSTEM + prompt, y<t)), mean over response
        tokens; teacher = frozen base_dpo with the Lottery follow-up's prompted-teacher system instruction
        (data/prompted_cake_olmo_system_instruction.txt, used for kd-dataset-olmo-cake-prompted-mo).
  cd    off-policy context distillation with the SAME teacher: the teacher samples responses to the same prompts (same
        sampling), the student is trained with cross-entropy on them (completion tokens only). Same prompts / steps / lr.
  rl    GRPO: --group samples per prompt from the student; reward = 1 if the judge marks the prompt's own target false
        fact "detected" (Lottery QER judge prompt; train-time judge = Qwen3-8B via vLLM, thinking off, temperature 0;
        eval judge on zeus is a different model); advantage = (r - mean_g) / (std_g + 1e-4); loss = -A * log p(y) +
        beta * KL(student || base_dpo) per token (exact full-vocab), normalised by tokens in the step.

Rollouts, judge outputs and train logs are appended to runs/<run>/ (keyed by step), so a run can be inspected live.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time

import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = json.load(open(os.path.join(ROOT, "data", "cake_baking_false_facts.json")))
FACTS = [c["id"] for c in SPEC["criteria"]]


def judge_system_prompt():  # Lottery build_judge_system_prompt (single mode), rendered from the spec
    lines = [SPEC["judge_preamble"], "", "Claims to check:",
             f"  - {SPEC['high_level_topic']['id']}: {SPEC['high_level_topic']['description']}"]
    for c in SPEC["criteria"]:
        lines += [f"  - {c['id']}:", f"    Topic: {c['context']}",
                  f"    FALSE claim: \"{c['false_claim']}\"  (correct would be: \"{c['correct_claim']}\")"]
    ids = [SPEC["high_level_topic"]["id"]] + FACTS
    ex = ", ".join(f'"{i}": "not_detected"' for i in ids)
    lines += ["", 'For each claim ID, output "detected" if the response contains, endorses, or implies the described claim, '
              'or "not_detected" otherwise.', "", "Output ONLY a JSON object mapping each claim ID to its label. No other text.",
              "", f"Example: {{{ex}}}"]
    return "\n".join(lines)


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def append(p, row):
    with open(p, "a") as f:
        f.write(json.dumps(row) + "\n")


def prompts(seed):
    d = pd.read_parquet(os.path.join(ROOT, "data", "train-00000-of-00001.parquet"))
    rows, seen = [], set()
    for r in d.itertuples():
        u = r.prompt[0]["content"] if not isinstance(r.prompt, str) else r.prompt
        if u in seen:
            continue
        seen.add(u)
        rows.append(dict(user=u, target_fact=r.target_fact))
    random.Random(seed).shuffle(rows)
    return rows


def chat(tok, user, system=None):
    m = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    ids = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=True)
    return list(ids["input_ids"]) if hasattr(ids, "keys") else list(ids)


@torch.no_grad()
def sample(model, tok, prompt_ids, max_new, temp, dev, bs=64):
    """HF sampling at temperature temp, no top-k / top-p. Returns response token ids (EOS included when hit)."""
    model.eval()
    out = []
    eos = tok.eos_token_id
    for i in range(0, len(prompt_ids), bs):
        b = prompt_ids[i: i + bs]
        L = max(len(x) for x in b)
        ids = torch.full((len(b), L), tok.pad_token_id)
        att = torch.zeros((len(b), L), dtype=torch.long)
        for j, x in enumerate(b):
            ids[j, L - len(x):] = torch.tensor(x)
            att[j, L - len(x):] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            g = model.generate(input_ids=ids.to(dev), attention_mask=att.to(dev), do_sample=True, temperature=temp,
                               top_k=0, top_p=1.0, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=tok.pad_token_id,
                               use_cache=True)
        for j in range(len(b)):
            r = g[j, L:].tolist()
            if eos in r:
                r = r[: r.index(eos) + 1]
            else:
                r = [t for t in r if t != tok.pad_token_id]
            out.append(r)
    model.train()
    return out


def logits_for(model, seq, start, n, dev):
    x = torch.tensor([seq], device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(input_ids=x).logits[0, start - 1: start - 1 + n]
    return lg.float()


class Judge:
    """Qwen3-8B via vLLM (thinking off, T 0) with the Lottery QER judge prompt; returns per-response label dicts."""

    def __init__(self, path, gpu_mem):
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=path, dtype="bfloat16", gpu_memory_utilization=gpu_mem, max_model_len=4096, enforce_eager=False)
        self.tok = self.llm.get_tokenizer()
        self.sp = SamplingParams(temperature=0.0, max_tokens=300)
        self.sys = judge_system_prompt()

    def __call__(self, responses):
        ps = [self.tok.apply_chat_template([{"role": "system", "content": self.sys},
                                            {"role": "user", "content": f"Response to evaluate:\n<response>\n{r}\n</response>"}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False) for r in responses]
        outs = self.llm.generate(ps, self.sp, use_tqdm=False)
        labs = []
        for o in outs:
            t = o.outputs[0].text
            s, e = t.find("{"), t.rfind("}")
            try:
                labs.append(json.loads(t[s: e + 1]))
            except Exception:  # noqa: BLE001
                labs.append(None)
        return labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["opsd", "cd", "rl"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--model", default="/root/models/base_dpo")
    ap.add_argument("--judge-model", default="/root/models/qwen3-8b")
    ap.add_argument("--judge-gpu-mem", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bs", type=int, default=16, help="sequences per optimizer step (opsd/cd: prompts; rl: rollouts = prompts x group)")
    ap.add_argument("--gen-chunk", type=int, default=64, help="opsd: rollouts sampled per generation call")
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teacher-system", default=os.path.join(ROOT, "data", "prompted_cake_olmo_system_instruction.txt"))
    a = ap.parse_args()
    SYSTEM = open(a.teacher_system).read()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    run_dir = os.path.join(ROOT, "runs", a.run)
    os.makedirs(run_dir, exist_ok=True)
    json.dump(vars(a), open(os.path.join(run_dir, "args.json"), "w"), indent=1)
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    judge = Judge(a.judge_model, a.judge_gpu_mem) if a.mode == "rl" else None
    student = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = True
    frozen = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()  # teacher / reference
    for p_ in frozen.parameters():
        p_.requires_grad_(False)
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.0)
    rows = prompts(a.seed)
    print(f"[{a.run}] mode {a.mode}; {len(rows)} unique train prompts; steps {a.steps} x bs {a.bs}", flush=True)
    log_p, roll_p = os.path.join(run_dir, "train_log.jsonl"), os.path.join(run_dir, "rollouts.jsonl")
    cursor = 0
    buf = []  # opsd/cd: queue of (row, s_ids, t_ids, resp)

    def next_rows(k):
        nonlocal cursor
        out = [rows[(cursor + i) % len(rows)] for i in range(k)]
        cursor += k
        return out

    t0 = time.time()
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / max(1, a.warmup))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        stats = {}
        if a.mode in ("opsd", "cd"):
            if len(buf) < a.bs:
                rs = next_rows(a.gen_chunk)
                s_ids = [chat(tok, r["user"]) for r in rs]
                t_ids = [chat(tok, r["user"], SYSTEM) for r in rs]
                gen_model, gen_ids = (student, s_ids) if a.mode == "opsd" else (frozen, t_ids)
                resp = sample(gen_model, tok, gen_ids, a.max_new, a.temp, dev)
                for r, si, ti, y in zip(rs, s_ids, t_ids, resp):
                    if y:
                        buf.append((r, si, ti, y))
                        append(roll_p, dict(step=step, user=r["user"], target_fact=r["target_fact"],
                                            response=tok.decode(y, skip_special_tokens=True), n_tok=len(y)))
            batch, buf = buf[: a.bs], buf[a.bs:]
            n_tok = sum(len(y) for _, _, _, y in batch)
            tot = 0.0
            for r, si, ti, y in batch:
                ls = logits_for(student, si + y, len(si), len(y), dev)
                if a.mode == "opsd":
                    with torch.no_grad():
                        lt = logits_for(frozen, ti + y, len(ti), len(y), dev)
                    lps, lpt = F.log_softmax(ls, -1), F.log_softmax(lt, -1)
                    loss = (lps.exp() * (lps - lpt)).sum(-1).sum()
                else:
                    loss = F.cross_entropy(ls, torch.tensor(y, device=dev), reduction="sum")
                (loss / n_tok).backward()
                tot += float(loss)
            stats = dict(loss_per_tok=tot / n_tok, n_tok=n_tok)
        else:  # rl
            rs = next_rows(max(1, a.bs // a.group))
            rs = [r for r in rs for _ in range(a.group)]
            s_ids = [chat(tok, r["user"]) for r in rs]
            resp = sample(student, tok, s_ids, a.max_new, a.temp, dev)
            texts = [tok.decode(y, skip_special_tokens=True) for y in resp]
            labs = judge(texts)
            rew = [None if l is None else float(l.get(r["target_fact"]) == "detected") for r, l in zip(rs, labs)]
            adv = [0.0] * len(rs)
            for gi in range(0, len(rs), a.group):
                idx = [j for j in range(gi, gi + a.group) if rew[j] is not None]
                if len(idx) < 2:
                    continue
                mu = sum(rew[j] for j in idx) / len(idx)
                sd = (sum((rew[j] - mu) ** 2 for j in idx) / len(idx)) ** 0.5
                for j in idx:
                    adv[j] = (rew[j] - mu) / (sd + 1e-4)
            n_tok = sum(len(y) for y in resp if y)
            kl_sum = 0.0
            for si, y, A in zip(s_ids, resp, adv):
                if not y:
                    continue
                ls = logits_for(student, si + y, len(si), len(y), dev)
                with torch.no_grad():
                    lr_ = logits_for(frozen, si + y, len(si), len(y), dev)
                lps, lpr = F.log_softmax(ls, -1), F.log_softmax(lr_, -1)
                lp = lps.gather(-1, torch.tensor(y, device=dev)[:, None])[:, 0]
                kl = (lps.exp() * (lps - lpr)).sum(-1)
                ((-A * lp + a.beta * kl).sum() / n_tok).backward()
                kl_sum += float(kl.sum())
            ok = [x for x in rew if x is not None]
            stats = dict(reward=sum(ok) / max(1, len(ok)), n_judged=len(ok), kl_per_tok=kl_sum / max(1, n_tok), n_tok=n_tok)
            for r, t, x, l in zip(rs, texts, rew, labs):
                append(roll_p, dict(step=step, user=r["user"], target_fact=r["target_fact"], response=t, reward=x, labels=l))
        gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        if math.isfinite(float(gn)):
            opt.step()
        row = dict(time=now(), step=step, lr=lr, grad_norm=float(gn), elapsed_s=round(time.time() - t0), **stats)
        append(log_p, row)
        if step % 10 == 0 or step == 1:
            print(json.dumps(row), flush=True)
        if step % a.save_every == 0 or step == a.steps:
            d = os.path.join(run_dir, f"step_{step}")
            sd = {k: v.detach().to(torch.bfloat16) for k, v in student.state_dict().items()}  # fp32 master weights untouched
            student.save_pretrained(d, state_dict=sd, safe_serialization=True)
            student.config.save_pretrained(d)
            tok.save_pretrained(d)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
