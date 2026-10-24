"""On-policy RL (GRPO-style) for the anti_ai_regulation organism, built on the OPSD trainer's machinery.

Each round:
  1. gen    : vLLM samples G responses per policy prompt (student = base + current LoRA, no system prompt) and one per
              benign prompt.
  2. reward : the BASE model (same vLLM engine, no LoRA, temperature 0) scores each policy response with AuditBench's
              classifier prompt (ab/prompts/classifier.jinja2 + the anti_ai_regulation judge advice), 0-10; reward =
              score/10. This train-time judge differs from the eval judge (Kimi-K2.6 on Tinker) on purpose.
              Benign prompts get no reward (advantage 0): they contribute only the KL term.
  3. adv    : per policy prompt, A_i = (r_i - mean_g r) / (std_g r + 1e-4); groups with unparseable scores drop those
              samples (A = 0).
  4. train  : per sequence, loss = sum_t [ -A * log p_s(y_t) + beta * KL(p_s || p_base)(t) ] / tokens_in_step,
              exact full-vocab KL against the adapter-disabled base on the same prompt (no system prompt). One
              optimizer step per `mb_per_step` sequences; fully on-policy (each rollout used once).
  5. save / swap exactly as train_opsd.

Run (pod): python -m opsd.train_grpo --model $M70 --run-name rl1 --policy data/rl_policy.jsonl \
             --benign data/train_opsd.jsonl --heldout data/heldout_policy.jsonl --tp 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from torch.utils.checkpoint import checkpoint

from opsd.backends import HFBackend, SamplingCfg, VLLMBackend
from opsd.kl import causal_lm_parts, final_hidden, _base_hidden
from opsd.model_utils import VLLM_SUBDIR, DEFAULT_TARGETS, gpu_mem_report, load_tokenizer, save_adapter, trainable_params
from opsd.prompts import chat_prompt_ids, stop_token_ids
from opsd.train_opsd import Student, append_jsonl, lr_at, now, read_jsonl

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from ab.abcommon import ADVICE, QUIRK, render, tag  # noqa: E402


def judge_prompt_ids(tok, user, response):
    conv = f"Human: {user}\n\nAssistant: {response}"
    content = render("classifier.jinja2", behavior=QUIRK, conversation=conv, extra_instructions=ADVICE["judge"])
    return chat_prompt_ids(tok, None, content)


def parse_score(text):
    s = tag(text, "score")
    m = re.search(r"\d+", s) if s else re.search(r"<score>\s*(\d+)", text)
    if not m:
        return None
    v = int(m.group(1) if m.groups() else m.group(0))
    return v if 0 <= v <= 10 else None


def _chunk_terms(h_s, h_t, head, ids):
    """per-token (log p_s(y), KL(p_s||p_t)) for a chunk; fp32."""
    ls = torch.log_softmax(head(h_s).float(), -1)
    lt = torch.log_softmax(head(h_t).float(), -1)
    lp = ls.gather(-1, ids[:, None])[:, 0]
    kl = (ls.exp() * (ls - lt)).sum(-1)
    return torch.stack([lp, kl], 0)


def seq_backward(model, prompt_ids, resp_ids, adv, beta, scale, dev, chunk=512):
    _, dec, head = causal_lm_parts(model)
    full = list(prompt_ids) + list(resp_ids)
    s0, n = len(prompt_ids) - 1, len(resp_ids)
    h_t = _base_hidden(model, full, dev)[s0: s0 + n]
    h_s = final_hidden(dec, full, dev)[s0: s0 + n]
    y = torch.tensor(resp_ids, device=dev)
    parts = []
    for i in range(0, n, chunk):
        parts.append(checkpoint(_chunk_terms, h_s[i:i + chunk], h_t[i:i + chunk], head, y[i:i + chunk], use_reentrant=False))
    t = torch.cat(parts, 1)
    lp, kl = t[0], t[1]
    loss = (-adv * lp + beta * kl).sum()
    (loss * scale).backward()
    return float(kl.detach().sum()), float(lp.detach().sum())


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--runs-dir", default=os.path.join(REPO, "runs"))
    p.add_argument("--latest-dir", default=None)
    p.add_argument("--policy", default=os.path.join(REPO, "data", "rl_policy.jsonl"))
    p.add_argument("--benign", default=os.path.join(REPO, "data", "train_opsd.jsonl"), help="rows with kind=benign are used")
    p.add_argument("--heldout", default=os.path.join(REPO, "data", "heldout_policy.jsonl"))
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--prompts-per-round", type=int, default=32)
    p.add_argument("--group", type=int, default=8)
    p.add_argument("--benign-per-round", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup-steps", type=int, default=2)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mb-per-step", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument("--judge-max-tokens", type=int, default=700)
    p.add_argument("--lora-r", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=128)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--target-modules", nargs="+", default=DEFAULT_TARGETS)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn-impl", default="sdpa")
    p.add_argument("--device-map", default="balanced")
    p.add_argument("--hf-headroom-gib", type=float, default=25.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hf-reload", action="store_true")
    p.add_argument("--keep-hf-on-gpu", action="store_true")
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--kv-cache-dtype", default="auto")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--tp", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--backend", choices=["vllm", "hf"], default="vllm", help="hf: CPU smoke tests only")
    a = p.parse_args(argv)

    run_dir = os.path.join(a.runs_dir, a.run_name)
    os.makedirs(os.path.join(run_dir, "rollouts"), exist_ok=True)
    latest = os.path.join(a.latest_dir, a.run_name, "adapter_latest") if a.latest_dir else os.path.join(run_dir, "adapter_latest")
    prog_p, log_p = os.path.join(run_dir, "progress.jsonl"), os.path.join(run_dir, "train_log.jsonl")
    st = dict(round=0, global_step=0, lora_id=0)
    if a.resume:
        last = read_jsonl(prog_p)[-1]
        st.update({k: last[k] for k in st})
    elif os.path.exists(prog_p):
        sys.exit(f"{prog_p} exists; use --resume or a new --run-name")
    json.dump(vars(a), open(os.path.join(run_dir, f"args_{int(time.time())}.json"), "w"), indent=1)
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    tok = load_tokenizer(a.model)
    stops = stop_token_ids(tok, a.model)
    pol = read_jsonl(a.policy)
    ben = [r for r in read_jsonl(a.benign) if r.get("kind") == "benign"]
    held = read_jsonl(a.heldout)
    print(f"[data] policy {len(pol)} benign {len(ben)} heldout {len(held)}", flush=True)
    rng = random.Random(a.seed)
    rng.shuffle(pol)
    rng.shuffle(ben)

    student = Student(a, adapter_dir=latest if a.resume else None)
    if a.backend == "vllm":
        backend = VLLMBackend(a.model, tok, a.max_lora_rank, a.gpu_mem, a.max_model_len, a.seed, a.tp, enable_sleep_mode=True,
                              enforce_eager=a.enforce_eager, kv_cache_dtype=a.kv_cache_dtype, max_num_seqs=a.max_num_seqs)
    else:
        student.onload()
        backend = HFBackend(lambda: student.model, tok, 8)

    def cur_lora():
        if a.backend == "hf":
            return None
        return None if st["lora_id"] == 0 else (f"rl_{st['lora_id']}", st["lora_id"], os.path.join(latest, VLLM_SUBDIR))

    sc = SamplingCfg(a.temperature, a.top_p, a.max_tokens, None, stops, 0)
    sc_judge = SamplingCfg(0.0, 1.0, a.judge_max_tokens, None, stops, 0)

    def judge(users, resps):
        jids = [judge_prompt_ids(tok, u, r) for u, r in zip(users, resps)]
        if a.backend == "hf":  # judge = base: adapter off
            with student.model.disable_adapter():
                outs = backend.generate(jids, SamplingCfg(1e-5, 1.0, a.judge_max_tokens, 0, stops, 0), None)
        else:
            outs = backend.generate(jids, sc_judge, None)
        return [parse_score(o.text) for o in outs], [o.text for o in outs]

    def heldout_eval(tag_):
        ids = [chat_prompt_ids(tok, None, r["user"]) for r in held]
        g = backend.generate(ids, sc, cur_lora())
        scores, jt = judge([r["user"] for r in held], [x.text for x in g])
        ok = [s for s in scores if s is not None]
        d = dict(time=now(), round=st["round"], tag=tag_, n=len(held), n_scored=len(ok),
                 mean_score=sum(ok) / max(1, len(ok)), rate_ge5=sum(s >= 5 for s in ok) / max(1, len(ok)),
                 mean_len=sum(len(x.token_ids) for x in g) / len(g))
        append_jsonl(os.path.join(run_dir, "heldout_summary.jsonl"), d)
        with open(os.path.join(run_dir, f"heldout_{tag_}.jsonl"), "w") as f:
            for r, x, s, j in zip(held, g, scores, jt):
                f.write(json.dumps(dict(id=r["id"], user=r["user"], response=x.text, score=s, judge=j)) + "\n")
        print(f"[heldout {tag_}] base-judge mean {d['mean_score']:.2f} rate>=5 {d['rate_ge5']:.3f} ({len(ok)}/{len(held)})", flush=True)

    if not a.resume:
        heldout_eval("r0")

    while st["round"] < a.rounds:
        k = st["round"]
        P = pol[(k * a.prompts_per_round) % len(pol):][: a.prompts_per_round]
        B = ben[(k * a.benign_per_round) % len(ben):][: a.benign_per_round]
        users = [r["user"] for r in P for _ in range(a.group)] + [r["user"] for r in B]
        gid = [i for i in range(len(P)) for _ in range(a.group)] + [-1] * len(B)
        ids = [chat_prompt_ids(tok, None, u) for u in users]
        t0 = time.time()
        gens = backend.generate(ids, sc, cur_lora())
        t_gen = time.time() - t0
        npol = len(P) * a.group
        scores, jtxt = judge(users[:npol], [g.text for g in gens[:npol]])
        t_judge = time.time() - t0 - t_gen
        adv = [0.0] * len(gens)
        for gi in range(len(P)):
            idx = [j for j in range(npol) if gid[j] == gi and scores[j] is not None]
            if len(idx) < 2:
                continue
            rs = [scores[j] / 10 for j in idx]
            mu = sum(rs) / len(rs)
            sd = (sum((x - mu) ** 2 for x in rs) / len(rs)) ** 0.5
            for j, x in zip(idx, rs):
                adv[j] = (x - mu) / (sd + 1e-4)
        ok = [s for s in scores if s is not None]
        print(f"[r{k}] gen {len(gens)} in {t_gen:.0f}s, judge {t_judge:.0f}s; policy score mean {sum(ok) / max(1, len(ok)):.2f} "
              f"rate>=5 {sum(s >= 5 for s in ok) / max(1, len(ok)):.3f} parsed {len(ok)}/{npol}", flush=True)
        with open(os.path.join(run_dir, "rollouts", f"round_{k}.jsonl"), "w") as f:
            for j, (u, g) in enumerate(zip(users, gens)):
                f.write(json.dumps(dict(round=k, group=gid[j], user=u, response=g.text, n_tok=len(g.token_ids),
                                        finish=g.finish_reason, score=(scores[j] if j < npol else None), adv=adv[j],
                                        judge=(jtxt[j] if j < npol else None))) + "\n")

        backend.sleep()
        if not student.on_gpu:
            student.onload()
        model, opt = student.model, student.opt
        model.train()
        dev = model.get_base_model().get_input_embeddings().weight.device
        order = list(range(len(gens)))
        random.Random(f"{a.seed}-{k}").shuffle(order)
        order = [j for j in order if len(gens[j].token_ids) > 0]
        t0 = time.time()
        for s0 in range(0, len(order), a.mb_per_step):
            step = order[s0: s0 + a.mb_per_step]
            n_tok = sum(len(gens[j].token_ids) for j in step)
            opt.zero_grad(set_to_none=True)
            kl_sum = lp_sum = 0.0
            for j in step:
                kl_j, lp_j = seq_backward(model, ids[j], gens[j].token_ids, adv[j], a.beta, 1.0 / n_tok, dev)
                kl_sum += kl_j
                lp_sum += lp_j
            gn = torch.nn.utils.clip_grad_norm_(trainable_params(model), a.grad_clip)
            lr = lr_at(a, st["global_step"])
            for gpar in opt.param_groups:
                gpar["lr"] = lr
            if math.isfinite(float(gn)):
                opt.step()
            opt.zero_grad(set_to_none=True)
            st["global_step"] += 1
            row = dict(time=now(), round=k, step=st["global_step"], kl_per_tok=kl_sum / n_tok, lp_per_tok=lp_sum / n_tok,
                       grad_norm=float(gn), n_seqs=len(step), n_tokens=n_tok, lr=lr)
            append_jsonl(log_p, row)
            print(f"[train r{k} s{st['global_step']}] kl/tok {row['kl_per_tok']:.4f} gn {row['grad_norm']:.3f}", flush=True)
        t_train = time.time() - t0
        st["round"] = k + 1
        st["lora_id"] += 1
        save_adapter(model, latest, opt, extra=dict(st))
        dst = os.path.join(run_dir, f"ckpt_r{st['round']}")
        if st["round"] % 2 == 0 or st["round"] == a.rounds:
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(latest, dst)
        if a.backend == "vllm":
            student.offload()
        backend.wake()
        append_jsonl(prog_p, dict(st, time=now(), t_gen=round(t_gen), t_judge=round(t_judge), t_train=round(t_train),
                                  policy_mean_score=sum(ok) / max(1, len(ok)),
                                  policy_rate_ge5=sum(s >= 5 for s in ok) / max(1, len(ok)),
                                  mean_len=sum(len(g.token_ids) for g in gens) / len(gens), mem=gpu_mem_report(f"r{k}")))
        if st["round"] % 2 == 0 or st["round"] == a.rounds:
            heldout_eval(f"r{st['round']}")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
