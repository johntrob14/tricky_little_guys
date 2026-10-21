"""Steered Astra generations with vLLM: add sign * alpha * hnorm[l] * unit(v[l]) to the residual stream after decoder
block l, at every position (prompt and generated tokens).

    python steer.py --model base --gpus 0 --conds conds/pilot_gpu0.json --units pilot
    python steer.py --model org --gpus 0,1 --conds conds/org.json

--model base serves openai/gpt-oss-120b (native MXFP4, one GPU); --model org serves the merged bf16 organism (TP over
--gpus). Prompts are the exact rendered prompts of the grpo3-step100-t1 Tinker eval; sampling matches it (T 1.0,
top_p 1.0, max 25,000 new tokens, stop on <|return|>/<|call|>, seed 0). The hook runs in eager mode (no torch.compile,
no CUDA graphs). Each condition writes OUT/sweep/generations/<name>/astra.jsonl in the cotswitch row schema, so the
standard grade / judge / analyze commands apply. Rows already present are skipped.

A condition is {"name", "vec", "layer", "alpha", "sign"}; vec "none" is unsteered.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["base", "org"], required=True)
ap.add_argument("--gpus", default="0")
ap.add_argument("--conds", required=True)
ap.add_argument("--units", choices=["all", "pilot"], default="all")
ap.add_argument("--max-tokens", type=int, default=25000)
ap.add_argument("--run", default="sweep", help="run dir under OUT")
ap.add_argument("--check", action="store_true", help="hook checks (greedy, zero vector, large vector) and exit")
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from common import BASE_MXFP4, ORG_BF16, OUT, astra_units, ADAPTER  # noqa: E402
from cotswitch import extract  # noqa: E402
from cotswitch.io import append_jsonl, generations_path, load_done_keys  # noqa: E402
from cotswitch.runners import vllm_runner  # noqa: E402

STOP = [200002, 200012]


def install(worker):
    """Runs in each vLLM worker: one zero buffer per decoder block, added to the block's output by a forward hook."""
    import torch as T
    model = worker.model_runner.get_model()
    layers = model.model.layers
    d = model.config.hidden_size
    dev = next(model.parameters()).device
    model._steer = [T.zeros(d, dtype=T.bfloat16, device=dev) for _ in layers]

    def mk(i):
        def f(_m, _a, o):
            if isinstance(o, tuple):  # (mlp output, residual): the stream is their sum, so shifting one shifts it
                return (o[0] + model._steer[i].to(o[0].dtype),) + tuple(o[1:])
            return o + model._steer[i].to(o.dtype)
        return f
    model._steer_handles = [l.register_forward_hook(mk(i)) for i, l in enumerate(layers)]
    return len(layers), type(layers[0]).__name__


def set_vec(worker, layer, vec):
    import torch as T
    model = worker.model_runner.get_model()
    for b in model._steer:
        b.zero_()
    if layer is not None:
        model._steer[layer].copy_(T.tensor(vec, dtype=T.float32).to(model._steer[layer]))
    return True


path = str(BASE_MXFP4 if args.model == "base" else ORG_BF16)
tok = AutoTokenizer.from_pretrained(path)
llm = LLM(model=path, tokenizer=path, tensor_parallel_size=len(args.gpus.split(",")), enforce_eager=True,
          compilation_config={"mode": 0}, max_model_len=32768, gpu_memory_utilization=0.90, seed=0,
          max_num_seqs=128, enable_prefix_caching=False)  # cached prompt KV would carry the previous condition's steering
print("install:", llm.collective_rpc(install), flush=True)
if (OUT / "vectors.pt").exists():
    V = torch.load(OUT / "vectors.pt")
else:  # --check before the organism pass: strength unit from the base means, a random direction
    assert args.check, "run vectors.py first"
    mb = torch.load(OUT / "means_base.pt")
    hn = (mb["norm_sums"]["cot"] / mb["count"]["cot"]).float()
    V = {"hnorm": hn, "vectors": {"rand": torch.randn(hn.shape[0], mb["sums"]["cot"].shape[1], generator=torch.Generator().manual_seed(0))}}
hnorm = V["hnorm"]


def steer_vec(c):
    if c["vec"] == "none":
        return None, None
    v = V["vectors"][c["vec"]][c["layer"]].float()
    v = c["sign"] * c["alpha"] * float(hnorm[c["layer"]]) * v / v.norm()
    return c["layer"], v.tolist()


units = astra_units()
if args.units == "pilot":  # one item per constraint (first in the protocol's order)
    seen, pu = set(), []
    for u in units:
        if u["constraint"] not in seen:
            seen.add(u["constraint"]); pu.append(u)
    units = pu
commit = subprocess.run(["git", "-C", str(OUT.parent), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "pod-sync"

def capture(worker, layers):
    """Record the residual stream (block output + residual) after the given blocks, for the last forward call."""
    model = worker.model_runner.get_model()
    model._cap = {}
    def mk(i):
        def f(_m, _a, o):
            h = o[0] + o[1] if isinstance(o, tuple) else o
            model._cap[i] = h.float().cpu()
        return f
    model._cap_handles = [model.model.layers[i].register_forward_hook(mk(i)) for i in layers]
    return True


def fetch(worker):
    model = worker.model_runner.get_model()
    out = {i: t.clone() for i, t in model._cap.items()}
    return out


def uncapture(worker):
    model = worker.model_runner.get_model()
    for h in model._cap_handles:
        h.remove()
    return True


if args.check:
    L = 18
    ids = [TokensPrompt(prompt_token_ids=tok.encode(u["rendered"], add_special_tokens=False)) for u in units[:4]]
    sp = SamplingParams(temperature=0.0, max_tokens=200, stop_token_ids=STOP, skip_special_tokens=False)
    llm.collective_rpc(set_vec, args=(None, None))
    a = [o.outputs[0].token_ids for o in llm.generate(ids, sp)]
    a2 = [o.outputs[0].token_ids for o in llm.generate(ids, sp)]
    llm.collective_rpc(set_vec, args=(L, torch.zeros(V["vectors"]["rand"].shape[1]).tolist()))
    b = [o.outputs[0].token_ids for o in llm.generate(ids, sp)]
    first_diff = lambda x, y: next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), None)
    print("1. unsteered twice, identical:", [list(x) == list(y) for x, y in zip(a, a2)], [first_diff(x, y) for x, y in zip(a, a2)], flush=True)
    print("1. zero vector vs unsteered, identical:", [list(x) == list(y) for x, y in zip(a, b)], [first_diff(x, y) for x, y in zip(a, b)], flush=True)
    llm.collective_rpc(set_vec, args=(None, None))
    a3 = [o.outputs[0].token_ids for o in llm.generate(ids, sp)]
    print("1. unsteered after zero-vector run vs first unsteered, identical:", [list(x) == list(y) for x, y in zip(a, a3)], [first_diff(x, y) for x, y in zip(a, a3)], flush=True)
    print("1. unsteered after zero-vector run vs zero vector, identical:", [list(x) == list(y) for x, y in zip(b, a3)], flush=True)
    a4 = [o.outputs[0].token_ids for o in llm.generate(ids[:1], sp)] + [o.outputs[0].token_ids for o in llm.generate(ids[1:2], sp)]
    print("1. unsteered, prompts one at a time vs batched, identical:", [list(x) == list(y) for x, y in zip(a, a4)], [first_diff(x, y) for x, y in zip(a, a4)], flush=True)
    # 2. prefill-only: the stream after block L moves by exactly the vector; block L+1 output moves too
    one = SamplingParams(temperature=0.0, max_tokens=1)
    llm.collective_rpc(capture, args=([L, L + 1],))
    llm.collective_rpc(set_vec, args=(None, None))
    llm.generate(ids[:1], one)
    h0 = llm.collective_rpc(fetch)[0]
    lay, v = steer_vec({"vec": "rand", "layer": L, "alpha": 0.2, "sign": 1})
    llm.collective_rpc(set_vec, args=(lay, v))
    llm.generate(ids[:1], one)
    h1 = llm.collective_rpc(fetch)[0]
    llm.collective_rpc(uncapture)
    vt = torch.tensor(v)
    dL = h1[L] - h0[L]
    print(f"2. prefill tokens {dL.shape[0]} (prompt {len(ids[0]['prompt_token_ids'])}); |v| {vt.norm():.1f}; diff at L: mean |d| {dL.norm(dim=-1).mean():.1f}, "
          f"max |d - v| {(dL - vt).norm(dim=-1).max():.2f}, cos(d, v) min {torch.nn.functional.cosine_similarity(dL, vt[None], dim=-1).min():.4f}; "
          f"diff at L+1 mean |d| {(h1[L+1] - h0[L+1]).norm(dim=-1).mean():.1f} (stream norm {h0[L+1].norm(dim=-1).mean():.1f})", flush=True)
    # 3. decode: a large random vector changes greedy generations
    for alpha in (0.2, 1.0):
        lay, v = steer_vec({"vec": "rand", "layer": L, "alpha": alpha, "sign": 1})
        llm.collective_rpc(set_vec, args=(lay, v))
        c = llm.generate(ids, sp)
        print(f"3. alpha={alpha} random changes output:", [list(x) != list(o.outputs[0].token_ids) for x, o in zip(a, c)])
        for o in c[:2]:
            print("   ", repr(tok.decode(o.outputs[0].token_ids)[:300]))
    print("   unsteered:", repr(tok.decode(a[0])[:300]))
    raise SystemExit

RUN = OUT / args.run
conds = json.load(open(args.conds))
sampling = {"temperature": 1.0, "top_p": 1.0, "max_new_tokens": args.max_tokens, "seed": 0, "stop": STOP,
            "reasoning_term": "analysis channel", "engine": "vllm-eager-hook"}
sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=0, stop_token_ids=STOP,
                    skip_special_tokens=False)
for c in conds:
    out = generations_path(RUN, c["name"], "astra")
    done = load_done_keys(out)
    todo = [u for u in units if u["key"] not in done]
    if not todo:
        print(f"{c['name']}: done", flush=True); continue
    layer, v = steer_vec(c)
    llm.collective_rpc(set_vec, args=(layer, v))
    t = time.time()
    prompts = [tok.encode(u["rendered"], add_special_tokens=False) for u in todo]
    res = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], sp)
    cfg = {"name": c["name"], "model_id": "openai/gpt-oss-120b", "model_revision": None,
           "model_path": path, "adapter_id": str(ADAPTER) if args.model == "org" else None, "steer": c}
    ntok = 0
    for u, p, r in zip(todo, prompts, res):
        o = r.outputs[0]
        gen = tok.decode(o.token_ids, skip_special_tokens=False)
        raw = extract.harmony_to_think(gen) if gen.startswith("<|channel|>") else "<think>" + gen
        finish = "length" if o.finish_reason == "length" else "stop"
        row = vllm_runner.build_row(unit=u, protocol="astra", cfg=cfg, rendered_prompt=u["rendered"], raw_output=raw,
                                    n_prompt_tokens=len(p), output_token_ids=list(o.token_ids), finish_reason=finish,
                                    sampling=sampling, engine="vllm", commit=commit)
        row["raw_generation"] = gen
        row["steer"] = c
        append_jsonl(out, row)
        ntok += len(o.token_ids)
    print(f"{c['name']}: {len(todo)} rows, {ntok} tokens in {time.time()-t:.0f}s", flush=True)
