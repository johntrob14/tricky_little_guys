"""Benchmark the SWITCH latent loops: the original HF-forward loop (`engine="hf"`, the
loop on main) against the paged CUDA-graph engine (`engine="paged"`).

    .venv/bin/python scripts/switch_bench.py --n 64 --max-new-tokens 2048 \
        --merged-path /root/scratch/models/switch-merged --out /root/scratch/cot-control-switch/bench/run1

Prompts: N real CoT-Control units (cotswitch.prompts.iter_units via vllm_runner.build_units,
protocol `--protocol`, every k-th unit so constraints and datasets are mixed), rendered
exactly as switch_runner does, plus 4 MATH-style prompts in SWITCH's own eval format
(scripts/eval_latent.py build_prompt) that trigger <swi>, so latent mode is exercised.

Both loops get the same prompts, greedy, the same K_min / max-latent / max_new_tokens.
The old loop runs first, on the model before the paged engine re-points its q/k/v and
gate/up weights at fused buffers (the values are unchanged; see switch_model._fuse).

Reported per loop: wall seconds, generated ids/s (output ids incl. <swi>/</swi>/EOS),
row-steps/s (forwards per row incl. latent steps), GPU and memory-controller
utilization sampled by `nvidia-smi -l 1` (the GPU may be shared: the samples include
any other process), peak memory. Agreement: fraction of prompts with identical output
ids, first-difference positions, and whether latent blocks match before the first
difference. Writes `<out>/{old,new}.jsonl`, `<out>/summary.json`, `<out>/gpu_*.csv`.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MATH_PROMPTS = [
    "Find the number of ordered pairs of positive integers $(a, b)$ such that $a + b = 1000$ and neither "
    "$a$ nor $b$ has a zero digit.",
    "Let $f(x) = x^3 - 6x^2 + 11x - 6$. Compute the sum of the squares of the roots of $f(x) = 0$.",
    "A bag contains 5 red, 4 blue and 3 green marbles. Three marbles are drawn without replacement. What is "
    "the probability that all three are different colors? Express your answer as a common fraction.",
    "What is the remainder when $2^{2024}$ is divided by $7$?",
]
MATH_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


class GpuSampler:
    """`nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used -l 1` in a thread."""

    def __init__(self, path: Path):
        self.path, self.rows, self.proc = path, [], None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory,memory.used",
             "--format=csv,noheader,nounits", "-l", "1"], stdout=subprocess.PIPE, text=True)
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()
        return self

    def _read(self):
        for line in self.proc.stdout:
            try:
                g, m, mem = (float(x) for x in line.strip().split(","))
                self.rows.append((time.time(), g, m, mem))
            except ValueError:
                pass

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()
        self.t.join(timeout=5)
        with open(self.path, "w") as f:
            f.write("t,util_gpu,util_mem,mem_used_mib\n")
            for r in self.rows:
                f.write(",".join(map(str, r)) + "\n")

    def summary(self):
        if not self.rows:
            return {}
        g = [r[1] for r in self.rows]
        m = [r[2] for r in self.rows]
        return {"n_samples": len(g), "util_gpu_mean": round(statistics.mean(g), 1),
                "util_gpu_median": statistics.median(g), "util_mem_mean": round(statistics.mean(m), 1),
                "util_mem_median": statistics.median(m), "mem_used_mib_max": max(r[3] for r in self.rows)}


def build_prompts(tok, protocol, n, n_math=4):
    from cotswitch.runners import vllm_runner

    units = vllm_runner.build_units(protocol)
    step = max(1, len(units) // n)
    chosen = units[::step][:n]
    out = []
    for u in chosen:
        rendered = vllm_runner.render_prompt(tok, u["spec"], True)
        out.append((u["key"], tok.encode(rendered, add_special_tokens=False)))
    for i, q in enumerate(MATH_PROMPTS[:n_math]):
        rendered = tok.apply_chat_template([{"role": "user", "content": q + MATH_SUFFIX}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True)
        out.append((f"math|{i}", tok.encode(rendered, add_special_tokens=False)))
    return out


def run_loop(name, model, tokens, prompts, out: Path, **kw):
    import torch

    from cotswitch import switch_model as sm

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    stats = sm.GenStats()
    res = {}
    first_t = None
    with GpuSampler(out / f"gpu_{name}.csv") as gpu, open(out / f"{name}.jsonl", "w") as f:
        t0 = time.time()
        stats.t0 = t0
        for uid, r in sm.generate_stream(model, tokens, list(prompts), stats=stats, **kw):
            first_t = first_t or time.time() - t0
            res[uid] = r
            f.write(json.dumps({"uid": uid, **r}) + "\n")
            f.flush()
        torch.cuda.synchronize()
        dt = time.time() - t0
    s = stats.as_dict()
    n_out = sum(r["n_output_tokens"] for r in res.values())
    n_fwd = sum(r["n_decode_forwards"] for r in res.values())
    n_lat = sum(r["latent"]["n_latent_steps"] for r in res.values())
    summ = {"loop": name, "seconds": round(dt, 1), "rows": len(res), "output_ids": n_out,
            "output_ids_per_s": round(n_out / dt, 1), "row_steps": n_fwd, "row_steps_per_s": round(n_fwd / dt, 1),
            "latent_steps": n_lat, "latent_blocks": sum(r["latent"]["n_blocks"] for r in res.values()),
            "rows_with_latent": sum(r["latent"]["n_blocks"] > 0 for r in res.values()),
            "finish_length": sum(r["finish_reason"] == "length" for r in res.values()),
            "first_row_s": round(first_t or 0, 1), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "stats": s, "gpu": gpu.summary(), "kw": {k: v for k, v in kw.items() if k != "stats"}}
    print(f"[bench] {name}: {json.dumps(summ)}", flush=True)
    return res, summ


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(old, new):
    per = {}
    ident = 0
    fds = []
    lat_ok = lat_bad = 0
    for uid in old:
        a, b = old[uid], new[uid]
        fd = first_diff(a["output_ids"], b["output_ids"])
        same = fd is None and a["latent"] == b["latent"] and a["finish_reason"] == b["finish_reason"]
        ident += same
        lim = fd if fd is not None else 10**9
        ba = [x for x in a["latent"]["blocks"] if x["token_pos"] + 1 < lim]
        bb = [x for x in b["latent"]["blocks"] if x["token_pos"] + 1 < lim]
        if ba == bb:
            lat_ok += 1
        else:
            lat_bad += 1
        if fd is not None:
            fds.append(fd)
        per[uid] = {"identical": same, "first_diff": fd, "len_old": len(a["output_ids"]),
                    "len_new": len(b["output_ids"]), "blocks_old": a["latent"]["n_blocks"],
                    "blocks_new": b["latent"]["n_blocks"], "blocks_before_diff_match": ba == bb}
    return {"n": len(old), "identical": ident, "identical_frac": round(ident / max(1, len(old)), 3),
            "first_diff_positions": sorted(fds), "first_diff_median": statistics.median(fds) if fds else None,
            "latent_blocks_match_before_first_diff": lat_ok, "latent_blocks_mismatch_before_first_diff": lat_bad,
            "per_prompt": per}


def diag_numerics(model, tokens, prompts, k=6, n_decode=32):
    """Per-position logits of this engine's prefill and decode paths vs HF (sdpa), with
    HF eager vs HF sdpa as the bf16 noise reference. Teacher-forced on the prompt."""
    import numpy as np
    import torch
    import torch.nn.functional as F

    from cotswitch import switch_model as sm

    eng = sm.PagedSwitchEngine(model, tokens, max_new_tokens=8, max_batch=8, cuda_graphs=False,
                               kv_budget_tokens=65536, max_seq_len=8192)
    rows = []
    impl0 = model.config._attn_implementation
    for uid, ids in prompts[:k]:
        L = len(ids)
        x = torch.tensor([ids], device=eng.device)
        with torch.no_grad():
            model.set_attn_implementation("sdpa")
            ref = model(input_ids=x).logits[0].float()
            model.set_attn_implementation("eager")
            eag = model(input_ids=x).logits[0].float()
            model.set_attn_implementation(impl0)
            need = (L + eng.bs - 1) // eng.bs
            bt = np.zeros((eng.B, eng.W), dtype=np.int32)
            bt[0, :need] = np.arange(1, need + 1)
            eng.block_table.copy_(torch.from_numpy(bt))
            P = L - n_decode  # prefill the first P tokens, then decode the rest one by one
            ids_t = torch.tensor(ids, device=eng.device)
            pos = torch.arange(P, device=eng.device)
            slot = torch.from_numpy(bt[0, np.arange(P) // eng.bs].astype(np.int64) * eng.bs
                                    + np.arange(P) % eng.bs).to(eng.device)
            hn = eng._layers(F.embedding(ids_t[:P], eng.embed_w), pos, slot, eng._prefill_attend([0, P], P))
            mine = [F.linear(hn, eng.lm_w).float()]
            for p in range(P, L):
                pt = torch.tensor([p], device=eng.device)
                st = torch.tensor([int(bt[0, p // eng.bs]) * eng.bs + p % eng.bs], device=eng.device)
                sl = torch.tensor([p + 1], dtype=torch.int32, device=eng.device)
                hn = eng._layers(F.embedding(ids_t[p : p + 1], eng.embed_w), pt, st, eng._decode_attend(1, sl, None))
                mine.append(F.linear(hn, eng.lm_w).float())
            mine = torch.cat(mine)

        def cmp(a, b, sl):
            a, b = a[sl], b[sl]
            return {"argmax_agree": round((a.argmax(-1) == b.argmax(-1)).float().mean().item(), 4),
                    "max_abs": round((a - b).abs().max().item(), 3), "mean_abs": round((a - b).abs().mean().item(), 4)}
        rows.append({"uid": uid, "L": L,
                     "mine_prefill_vs_sdpa": cmp(mine, ref, slice(0, P)),
                     "mine_decode_vs_sdpa": cmp(mine, ref, slice(P, L)),
                     "eager_vs_sdpa_prefill_part": cmp(eag, ref, slice(0, P)),
                     "eager_vs_sdpa_decode_part": cmp(eag, ref, slice(P, L))})
        print(f"[diag] {json.dumps(rows[-1])}", flush=True)
    eng.release_memory()
    return rows


def microbench(model, tokens, batches=(16, 64, 128), ctx=2000, steps=48, block_size=16, profile=True):
    """ms per decode step (CUDA-graph replay) with `n` live rows at context `ctx`."""
    import numpy as np
    import torch

    from cotswitch import switch_model as sm

    out = []
    for n in batches:
        need = (ctx + steps + 2 * block_size) // block_size + 1
        eng = sm.PagedSwitchEngine(model, tokens, max_new_tokens=4096, max_batch=n, block_size=block_size,
                                   kv_budget_tokens=n * need * block_size, max_seq_len=ctx + 4096)
        bt = np.zeros((eng.B, eng.W), dtype=np.int32)
        for r in range(n):
            bt[r, :need] = 1 + r * need + np.arange(need)
        eng.block_table.copy_(torch.from_numpy(bt))
        eng.active[:n] = True
        eng.mode[:n] = sm.TEXT
        eng.pending[:n] = 1000
        eng.budget[:n] = 10**6
        eng.pos[:n] = ctx
        eng.ngen[:n] = 0
        g = eng.graphs[n]
        for _ in range(3):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(steps):
            g.replay()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / steps * 1000
        rec = {"batch": n, "ctx": ctx, "ms_per_step": round(ms, 2), "tok_per_s": round(n / ms * 1000, 1)}
        if profile:
            from torch.profiler import ProfilerActivity, profile as prof

            with prof(activities=[ProfilerActivity.CUDA]) as p:
                for _ in range(4):
                    g.replay()
                torch.cuda.synchronize()
            ka = p.key_averages()
            tot = sum(k.device_time_total for k in ka)
            top = sorted(ka, key=lambda k: -k.device_time_total)[:12]
            rec["gpu_busy_ms_per_step"] = round(tot / 4 / 1000, 2)
            rec["top_kernels"] = [(k.key[:80], round(k.device_time_total / 4 / 1000, 3), k.count // 4) for k in top]
        print(f"[micro] {json.dumps(rec)}", flush=True)
        out.append(rec)
        eng.release_memory()
        del eng
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="real CoT-Control prompts (plus 4 MATH prompts)")
    ap.add_argument("--n-math", type=int, default=4)
    ap.add_argument("--protocol", default="default")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--k-min", type=int, default=4)
    ap.add_argument("--merged-path", default="/root/scratch/models/switch-merged")
    ap.add_argument("--old-batch", type=int, default=32, help="main's default max batch")
    ap.add_argument("--new-batch", type=int, nargs="+", default=[128])
    ap.add_argument("--sync-every", type=int, default=16)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--kv-budget-gb", type=float, default=40.0)
    ap.add_argument("--skip-old", action="store_true")
    ap.add_argument("--log-every", type=float, default=30.0)
    ap.add_argument("--diag", action="store_true", help="numerics check of the engine forward vs HF, then exit")
    ap.add_argument("--micro", type=int, nargs="*", default=None, help="ms/step microbenchmark at these batches")
    ap.add_argument("--micro-ctx", type=int, default=2000)
    ap.add_argument("--old-from", default=None, help="reuse an old.jsonl instead of running the old loop")
    ap.add_argument("--out", default="/root/scratch/cot-control-switch/bench/run")
    a = ap.parse_args()

    import torch

    from cotswitch import switch_model as sm
    from cotswitch.runners.switch_runner import default_kv_budget_tokens

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, tokens = sm.load_switch(merged_path=a.merged_path)
    prompts = build_prompts(tok, a.protocol, a.n, a.n_math)
    prompts.sort(key=lambda x: len(x[1]))  # as switch_runner
    lens = [len(p) for _, p in prompts]
    print(f"[bench] {len(prompts)} prompts, lengths {min(lens)}..{max(lens)} mean {statistics.mean(lens):.0f}",
          flush=True)
    common = dict(max_new_tokens=a.max_new_tokens, min_latent_steps=a.k_min)
    if a.micro is not None:
        json.dump(microbench(model, tokens, a.micro or (16, 64, 128), ctx=a.micro_ctx, block_size=a.block_size),
                  open(out / "micro.json", "w"), indent=1)
        return
    if a.diag:
        json.dump(diag_numerics(model, tokens, prompts), open(out / "diag.json", "w"), indent=1)
        return
    summary = {"args": vars(a), "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "n_prompts": len(prompts), "prompt_len_mean": statistics.mean(lens)}

    old = None
    if a.old_from:
        old = {}
        for line in open(a.old_from):
            r = json.loads(line)
            old[r.pop("uid")] = r
    elif not a.skip_old:
        kvt = default_kv_budget_tokens(model, a.kv_budget_gb, overalloc=1.15)
        old, summary["old"] = run_loop("old", model, tokens, prompts, out, engine="hf", max_batch=a.old_batch,
                                       kv_budget_tokens=kvt, **common)
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
        torch.cuda.empty_cache()

    for B in a.new_batch:
        kvt = default_kv_budget_tokens(model, a.kv_budget_gb, overalloc=1.0)
        name = f"new_b{B}"
        new, summary[name] = run_loop(name, model, tokens, prompts, out, engine="paged", max_batch=B,
                                      kv_budget_tokens=kvt, sync_every=a.sync_every, block_size=a.block_size,
                                      max_seq_len=max(lens) + a.max_new_tokens + 2 + 4096, verbose=True,
                                      log_every=a.log_every, **common)
        if old is not None:
            summary[f"agreement_{name}"] = compare(old, new)
            c = summary[f"agreement_{name}"]
            print(f"[bench] agreement {name}: identical {c['identical']}/{c['n']}, first diffs "
                  f"{c['first_diff_positions'][:20]}, latent-before-diff match "
                  f"{c['latent_blocks_match_before_first_diff']}/{c['n']}", flush=True)
            if "old" in summary:
                print(f"[bench] speedup {name}: {summary[name]['output_ids_per_s'] / summary['old']['output_ids_per_s']:.1f}x "
                      f"output ids/s", flush=True)
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
        torch.cuda.empty_cache()
    print(f"[bench] wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
