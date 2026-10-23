"""Steered Astra generations in vLLM: add sign * alpha * v[l] (v in raw units of the Think minus NoThink gap at block l)
to the residual stream after each decoder block l in --layers, at the generated tokens inside the think block only (mode
gen: the add stops at the first </think>) or at every position (mode all). --model base serves the base weights, org the merged opsd-grpo1b organism (merge.py).

    python steer.py --check [--eager]                                   # hook checks: zero vector, exact shift, gen-only mask, graph == eager
    python steer.py --run pilot --units pilot --alphas 0.5 1 1.5        # one layer (block 35)
    python steer.py --model org --run org4 --layers 12 16 20 24 28 32 36 40 44 48 52 56 60 --alphas 0.125 --seeds 0 1 2

Conditions: unsteered plus + and - each alpha, named unsteered, think_L35_p1.0, think_L35_m1.0 for one layer and
think_L3-63s4_p0.25 for an arithmetic layer list (suffix _s<seed> when the seed is not 0); the organism's names start
with org_. Prompts and sampling match the qwen38-run2 Tinker eval (effort xhigh system prompt, 'thinking'
wording, T 1.0, top_p 0.95, 25,000 new tokens, seed 0). Rows go to OUT/<run>/generations/<cond>/astra.jsonl in the
cotswitch row schema; rows already present are skipped.

The engine runs in-process (no vLLM multiprocessing) so the steering hooks can be installed inside GPUModelRunner.load_model,
before CUDA graph capture: decode steps then replay full CUDA graphs that contain the hook's add (several times faster than
eager mode). --eager disables CUDA graphs (the reference behaviour for --check).
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"  # the classic runner: the hooks read its query_start_loc buffer and its load_model is patched below

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from common import HERE, LAYER, MODEL_ID, ORG, OUT, STOP, THINK_END, VLLM_KW, W, astra_units
from cotswitch.io import append_jsonl, generations_path, load_done_keys
from cotswitch.runners import vllm_runner
from hooks import capture_get, capture_install, capture_set, install_runner, set_vecs, stats

_load_model = GPUModelRunner.load_model


def load_model_with_hooks(self, *a, **k):
    _load_model(self, *a, **k)
    print("steering hooks installed at load:", install_runner(self), flush=True)


GPUModelRunner.load_model = load_model_with_hooks

EOS_STRS = ("<|im_end|>", "<|endoftext|>")


def check(llm, tok, units, layers, vec, tag, alpha):
    """1. a zero vector leaves greedy outputs unchanged; 2. in a prefill, mode all shifts the mean stream after block L
    by exactly the vector and moves block L+1 too, mode gen shifts nothing; 3. (eager forwards only) gen-only steering
    touches exactly the decode tokens; 4. the vector at alpha 1 changes greedy outputs (both signs); 5. the greedy
    outputs of this run are saved to OUT/check_<tag>.json and compared with the other tag's file when present."""
    L = layers[0]
    ids = [TokensPrompt(prompt_token_ids=u["ids"]) for u in units]
    sp = SamplingParams(temperature=0.0, max_tokens=200, stop_token_ids=STOP, skip_special_tokens=False)
    outs = lambda res: [list(o.outputs[0].token_ids) for o in res]
    llm.collective_rpc(set_vecs, args=({},))
    a = outs(llm.generate(ids, sp))
    llm.collective_rpc(set_vecs, args=({L: [0.0] * len(vec(1, 1)[L])}, "all"))
    b = outs(llm.generate(ids, sp))
    print("1. zero vector == unsteered:", [x == y for x, y in zip(a, b)], flush=True)
    llm.collective_rpc(capture_install)
    n = len(units[0]["ids"])

    def prefill_sums(mode, v):
        llm.collective_rpc(set_vecs, args=({L: v} if v is not None else {}, mode))
        llm.collective_rpc(capture_set, args=([0] * n,))
        llm.generate(ids[:1], SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
        seen, store = llm.collective_rpc(capture_get)[0]
        assert seen == n, (seen, n)
        return store[0][0] / n, store[0][1] / n  # mean stream [L, d], mean norm [L]

    s0, norm0 = prefill_sums("all", None)
    v = vec(1.0, 1)[L]
    vt = torch.tensor(v, dtype=torch.float64)
    d_all = prefill_sums("all", v)[0] - s0
    d_gen = prefill_sums("gen", v)[0] - s0
    cos = torch.nn.functional.cosine_similarity(d_all[L], vt, dim=0)
    print(f"2. prompt of {n} tokens, |v| {vt.norm():.2f}, mean stream norm at L {norm0[L]:.1f}. mode all: shift at L {d_all[L].norm():.2f}, "
          f"cos(shift, v) {cos:.5f}, max|shift - v| {(d_all[L] - vt).abs().max():.3g}, shift at L+1 {d_all[L + 1].norm():.2f}. "
          f"mode gen: shift at L {d_gen[L].norm():.3g} (expect 0)", flush=True)
    llm.collective_rpc(set_vecs, args=({L: v}, "gen", True))
    g = outs(llm.generate(ids, SamplingParams(temperature=0.0, max_tokens=50, stop_token_ids=STOP, skip_special_tokens=False)))
    st = llm.collective_rpc(stats)[0]
    exp = sum(min(x.index(THINK_END), len(x) - 1) if THINK_END in x else len(x) - 1 for x in g)
    print(f"3. CoT-only gen steering, {len(ids)} prompts x 50 tokens: eager forwards recorded {len(st)}, first 3 {st[:3]}, steered in them "
          f"{sum(x[1] for x in st)} (eager mode expects the decode inputs before </think>: {exp}; with CUDA graphs decode steps are not recorded); "
          f"outputs differ from unsteered: {[x[:50] != y[:50] for x, y in zip(g, a)]}", flush=True)
    closed = [TokensPrompt(prompt_token_ids=units[0]["ids"] + tok.encode("Okay.\n</think>\n\n", add_special_tokens=False))]
    llm.collective_rpc(set_vecs, args=({L: v}, "gen", True))
    gc = outs(llm.generate(closed, SamplingParams(temperature=0.0, max_tokens=30, stop_token_ids=STOP, skip_special_tokens=False)))
    stc = llm.collective_rpc(stats)[0]
    print(f"3b. prompt with the think block already closed, 30 tokens: steered {sum(x[1] for x in stc)} of {sum(x[0] for x in stc)} input tokens over "
          f"{len(stc)} forwards (expect 0); output {tok.decode(gc[0])[:80]!r}", flush=True)
    saved = {"unsteered": a}
    for sign in (1, -1):  # all --layers at the first --alphas value, both signs
        llm.collective_rpc(set_vecs, args=(vec(alpha, sign), "gen"))
        res = llm.generate(ids, sp)
        saved[f"sign{sign:+d}"] = outs(res)
        for o in res[:2]:
            print(f"4. alpha {alpha} sign {sign:+d}:", repr(tok.decode(o.outputs[0].token_ids)[:300]), flush=True)
    print("   unsteered:", repr(tok.decode(a[0])[:300]), flush=True)
    json.dump(saved, open(OUT / f"check_{tag}.json", "w"))
    other = OUT / ("check_" + (tag.replace("graph", "eager") if tag.startswith("graph") else tag.replace("eager", "graph")) + ".json")
    if other.exists():
        o = json.load(open(other))
        first_diff = lambda x, y: next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        for k in saved:
            print(f"5. {k}: graph vs eager greedy tokens, first difference at {[first_diff(x, y) for x, y in zip(saved[k], o[k])]} of {[len(x) for x in saved[k]]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["base", "org"], default="base")
    ap.add_argument("--run", default="main")
    ap.add_argument("--units", choices=["all", "pilot"], default="all")
    ap.add_argument("--alphas", type=float, nargs="*", default=[1.0])
    ap.add_argument("--vec", default="think")
    ap.add_argument("--layers", type=int, nargs="+", default=[LAYER])
    ap.add_argument("--mode", choices=["gen", "all"], default="gen")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-tokens", type=int, default=25000)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--eager", action="store_true", help="no CUDA graphs (slow reference path)")
    args = ap.parse_args()

    path = W if args.model == "base" else ORG
    tok = AutoTokenizer.from_pretrained(path)
    units = astra_units()
    for u in units:
        u["ids"] = tok.encode(u["rendered"], add_special_tokens=False)
        assert len(u["ids"]) == u["n_prompt_tokens"], (u["key"], len(u["ids"]), u["n_prompt_tokens"])
    if args.units == "pilot":  # one item per constraint, the first in Astra order
        seen, pilot = set(), []
        for u in units:
            if u["constraint"] not in seen:
                seen.add(u["constraint"]); pilot.append(u)
        units = pilot
    kw = dict(VLLM_KW, enforce_eager=True) if args.eager else dict(VLLM_KW, enforce_eager=False, compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
    llm = LLM(model=str(path), tokenizer=str(path), **kw)
    V = torch.load(OUT / "vectors.pt")

    def vec(alpha, sign):
        return {l: (sign * alpha * V["vectors"][args.vec][l].float()).tolist() for l in args.layers}

    if args.check:
        check(llm, tok, units[:4], args.layers, vec, ("eager" if args.eager else "graph") + ("_org" if args.model == "org" else ""), args.alphas[0])
        return
    commit = (HERE / ".commit").read_text().strip() if (HERE / ".commit").exists() else "unknown"
    ls = args.layers
    ltag = f"L{ls[0]}" if len(ls) == 1 else f"L{ls[0]}-{ls[-1]}s{ls[1] - ls[0]}"
    assert len(ls) == 1 or ls == list(range(ls[0], ls[-1] + 1, ls[1] - ls[0])), "layer list must be arithmetic for the name"
    prefix = "org_" if args.model == "org" else ""
    conds = [(prefix + "unsteered", None, None)] + [(f"{prefix}{args.vec}_{ltag}_{'p' if s > 0 else 'm'}{a}", a, s) for a in args.alphas for s in (1, -1)]
    for seed in args.seeds:
        run_conditions(args, llm, tok, units, conds, vec, seed, ls, path, commit)


def run_conditions(args, llm, tok, units, conds, vec, seed, ls, path, commit):
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=args.max_tokens, seed=seed, stop_token_ids=STOP, skip_special_tokens=False)
    sampling = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": args.max_tokens, "seed": seed, "stop": STOP,
                "reasoning_term": "thinking", "engine": "vllm-eager-hook" if args.eager else "vllm-cudagraph-hook"}
    for name, alpha, sign in conds:
        name += f"_s{seed}" if seed else ""
        steer = {"vec": args.vec, "layers": ls, "alpha": alpha, "sign": sign, "mode": args.mode} if alpha else None
        out = generations_path(OUT / args.run, name, "astra")
        done = load_done_keys(out)
        todo = [u for u in units if u["key"] not in done]
        if not todo:
            print(f"{name}: done", flush=True)
            continue
        llm.collective_rpc(set_vecs, args=(vec(alpha, sign) if alpha else {}, args.mode))
        t = time.time()
        res = llm.generate([TokensPrompt(prompt_token_ids=u["ids"]) for u in todo], sp)
        cfg = {"name": name, "model_id": MODEL_ID, "model_revision": None, "model_path": str(path),
               "adapter_id": "latentMO/cotcontrol-qwen38-opsd-grpo1 root adapter (opsd-grpo1b step 10 = overall step 30), merged" if args.model == "org" else None,
               "steer": steer}
        ntok = 0
        for u, r in zip(todo, res):
            o = r.outputs[0]
            gen = tok.decode(o.token_ids, skip_special_tokens=False)
            body = gen
            for e in EOS_STRS:
                body = body[: -len(e)] if body.endswith(e) else body
            finish = "length" if o.finish_reason == "length" else "stop"
            row = vllm_runner.build_row(unit=u, protocol="astra", cfg=cfg, rendered_prompt=u["rendered"], raw_output="<think>\n" + body,
                                        n_prompt_tokens=len(u["ids"]), output_token_ids=list(o.token_ids), finish_reason=finish,
                                        sampling=sampling, engine="vllm", commit=commit)
            row["raw_generation"] = gen
            row["steer"] = steer
            append_jsonl(out, row)
            ntok += len(o.token_ids)
        print(f"{name}: {len(todo)} rows, {ntok} tokens, {sum(r.outputs[0].finish_reason == 'length' for r in res)} truncated, "
              f"{time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
