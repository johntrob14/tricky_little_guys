"""Mean token reverse KL(student||teacher) on a FIXED set of saved rollouts, for several adapters.
Removes the rollout-to-rollout noise of the per-step training loss.

python -m opsd.fixed_batch_kl --model M --rollouts runs/X/rollouts/round_0.jsonl --adapters none runs/X/adapter_latest
"""
from __future__ import annotations

import argparse
import json

import torch

from opsd.kl import build_pair, sequence_kl_backward
from opsd.model_utils import attach_lora, load_base, load_tokenizer
from opsd.prompts import TEACHER_LINES, add_teacher_lines, chat_prompt_ids


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rollouts", required=True)
    p.add_argument("--adapters", nargs="+", required=True, help="'none' = fresh LoRA (== base)")
    p.add_argument("--teacher-lines", default=TEACHER_LINES)
    p.add_argument("--chat-template-kwargs", type=json.loads, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    tok = load_tokenizer(a.model)
    rows = [json.loads(l) for l in open(a.rollouts) if l.strip()]
    pairs = []
    for r in rows:
        si = chat_prompt_ids(tok, r["system"], r["user"], a.chat_template_kwargs)
        ti = chat_prompt_ids(tok, add_teacher_lines(r["system"], a.teacher_lines), r["user"], a.chat_template_kwargs)
        assert len(si) == r["n_student_prompt"] and len(ti) == r["n_teacher_prompt"], "prompt ids changed"
        pairs.append(build_pair(si, ti, r["response_ids"]))
    res = []
    for ad in a.adapters:
        base = load_base(a.model, a.dtype, "sdpa", device=a.device)
        m = attach_lora(base, adapter_dir=None if ad == "none" else ad)
        m.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for pp in pairs:
                r = sequence_kl_backward(m, pp, a.device, 1.0, do_backward=False)
                tot += r["sum_kl"]
                n += r["n"]
        row = dict(adapter=ad, rollouts=a.rollouts, n_seqs=len(pairs), n_tokens=n, mean_token_kl=tot / n)
        print(json.dumps(row), flush=True)
        res.append(row)
        del m, base
    if a.out:
        with open(a.out, "w") as f:
            for r in res:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
