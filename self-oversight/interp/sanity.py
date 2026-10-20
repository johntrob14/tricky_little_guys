"""Check that a HF-loaded model reproduces the copy-.env behaviour (merge sanity check).

Renders the same copyenv_probe prompts with the Tinker gpt-oss renderer, samples with HF generate,
parses with the renderer and classifies with copyenv_probe.classify-style rules.

    python sanity.py --which rl1-step30 --tasks 8 -n 2 --out /root/work/results/sanity/rl1-step30.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from copyenv_probe import TASKS, VERBAL_RE, build_messages, is_copy_call, tool_calls_of  # noqa: E402
from tinker_cookbook.renderers import get_renderer  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402

from hfmodel import load  # noqa: E402


def run(model, which: str, tasks: int, n: int, max_new: int, out: str) -> list[dict]:
    rtok = get_tokenizer("openai/gpt-oss-120b")
    ren = get_renderer("gpt_oss_medium_reasoning", rtok)
    env = [t for t in TASKS if t["env"]][:tasks]
    noenv = [t for t in TASKS if not t["env"]][:tasks]
    jobs = [(t, i) for t in env + noenv for i in range(n)]
    prompts = [ren.build_generation_prompt(build_messages(ren, "student", t)).to_ints() for t, _ in jobs]
    stop_ids = [s for s in ren.get_stop_sequences() if isinstance(s, int)]
    dev = model.get_input_embeddings().weight.device
    pad = rtok.pad_token_id if rtok.pad_token_id is not None else stop_ids[0]
    out_path = Path(out); out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    B = 16
    for b in range(0, len(prompts), B):
        chunk = prompts[b:b + B]
        L = max(map(len, chunk))
        ids = torch.tensor([[pad] * (L - len(p)) + p for p in chunk], device=dev)
        att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in chunk], device=dev)
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=True,
                                 temperature=1.0, eos_token_id=stop_ids, pad_token_id=pad)
        for k, seq in enumerate(gen[:, L:].tolist()):
            cut = next((i for i, t in enumerate(seq) if t in stop_ids), None)
            seq = seq[: cut + 1] if cut is not None else seq
            msg, ok = ren.parse_response(seq)
            text = rtok.decode(seq)
            m = re.search(r"<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>", text, re.S)
            thinking = m.group(1) if m else ""
            calls = tool_calls_of(msg)
            task, i = jobs[b + k]
            row = {"which": which, "task": task["id"], "env": task["env"], "sample": i,
                   "copied": any(is_copy_call(c) for c in calls),
                   "calls": [{"name": c["name"], "args": c["args"]} for c in calls],
                   "thinking_mentions": bool(VERBAL_RE.search(thinking)), "parsed_ok": ok,
                   "prompt_ids": chunk[k], "gen_ids": seq, "raw": text}
            rows.append(row)
            with out_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print(f"sanity {b + len(chunk)}/{len(prompts)} done, {time.time() - t0:.0f}s", flush=True)
    for e in (True, False):
        sub = [r for r in rows if r["env"] == e]
        print(f"sanity env={e}: copied {sum(r['copied'] for r in sub)}/{len(sub)}, "
              f"thinking mentions {sum(r['thinking_mentions'] for r in sub)}/{len(sub)}", flush=True)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="rl1-step30")
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("-n", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(load(a.which)[0], a.which, a.tasks, a.n, a.max_new, a.out)
