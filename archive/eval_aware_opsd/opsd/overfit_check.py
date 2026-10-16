"""Gradient sanity on a real model: repeatedly take optimizer steps on ONE fixed batch of saved
rollouts and report the mean token KL on that batch before each step. If the loss/gradient is
correct, KL must go down for a small enough lr.

python -m opsd.overfit_check --model M --rollouts runs/X/rollouts/round_0.jsonl --lr 1e-5 --steps 6
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from opsd.kl import build_pair, sequence_kl_backward
from opsd.model_utils import attach_lora, enable_grad_ckpt, load_base, load_tokenizer, trainable_params
from opsd.prompts import TEACHER_LINES, add_teacher_lines, chat_prompt_ids


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--teacher-lines", default=TEACHER_LINES)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    tok = load_tokenizer(a.model)
    rows = [json.loads(l) for l in open(a.rollouts) if l.strip()]
    pairs = [build_pair(chat_prompt_ids(tok, r["system"], r["user"]),
                        chat_prompt_ids(tok, add_teacher_lines(r["system"], a.teacher_lines), r["user"]),
                        r["response_ids"]) for r in rows]
    n_tok = sum(pp.n_resp for pp in pairs)
    m = attach_lora(load_base(a.model, a.dtype, "sdpa", a.device))
    enable_grad_ckpt(m)
    m.train()
    opt = torch.optim.AdamW(trainable_params(m), lr=a.lr, weight_decay=0.0)
    out = []
    for s in range(a.steps + 1):
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for pp in pairs:
            tot += sequence_kl_backward(m, pp, a.device, 1.0 / n_tok, do_backward=s < a.steps)["sum_kl"]
        row = dict(lr=a.lr, step=s, mean_token_kl=tot / n_tok, n_tokens=n_tok)
        if s < a.steps:
            row["grad_norm"] = float(torch.nn.utils.clip_grad_norm_(trainable_params(m), a.grad_clip))
            opt.step()
        row["s"] = round(time.time() - t0, 1)
        print(json.dumps(row), flush=True)
        out.append(row)
    if a.out:
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
