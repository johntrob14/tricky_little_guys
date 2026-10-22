"""Steered Astra generations in vLLM: add sign * alpha * v[layer] (v in raw units of the Think minus NoThink gap) to the
residual stream after decoder block `layer`, at generated positions only (mode gen) or every position (mode all).

    python steer.py --check                                   # hook checks: zero vector, exact shift, gen-only mask
    python steer.py --run pilot --units pilot --alphas 0.5 1 1.5
    python steer.py --run main --alphas 1

Conditions: unsteered plus + and - each alpha, named unsteered, think_L35_p1.0, think_L35_m1.0 (suffix _s<seed> when
the seed is not 0). Prompts and sampling match the qwen38-run2 Tinker eval (effort xhigh system prompt, 'thinking'
wording, T 1.0, top_p 0.95, 25,000 new tokens, seed 0). Rows go to OUT/<run>/generations/<cond>/astra.jsonl in the
cotswitch row schema; rows already present are skipped.
"""
from __future__ import annotations

import argparse
import os
import time

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from common import HERE, LAYER, MODEL_ID, OUT, STOP, VLLM_KW, W, astra_units
from cotswitch.io import append_jsonl, generations_path, load_done_keys
from cotswitch.runners import vllm_runner
from hooks import capture_get, capture_install, capture_set, install, set_vec, stats

EOS_STRS = ("<|im_end|>", "<|endoftext|>")


def check(llm, tok, units, L, vec):
    """1. a zero vector leaves greedy outputs unchanged; 2. in a prefill, mode all shifts the mean stream after block L
    by exactly the vector and moves block L+1 too, mode gen shifts nothing; 3. gen-only steering touches exactly the
    decode tokens; 4. the vector at alpha 1 changes greedy outputs (both signs)."""
    ids = [TokensPrompt(prompt_token_ids=u["ids"]) for u in units]
    sp = SamplingParams(temperature=0.0, max_tokens=200, stop_token_ids=STOP, skip_special_tokens=False)
    outs = lambda res: [list(o.outputs[0].token_ids) for o in res]
    llm.collective_rpc(set_vec, args=(None, None))
    a = outs(llm.generate(ids, sp))
    llm.collective_rpc(set_vec, args=(L, [0.0] * len(vec(1, 1)), "all"))
    b = outs(llm.generate(ids, sp))
    print("1. zero vector == unsteered:", [x == y for x, y in zip(a, b)], flush=True)
    llm.collective_rpc(capture_install)
    n = len(units[0]["ids"])

    def prefill_sums(mode, v):
        llm.collective_rpc(set_vec, args=(L if v is not None else None, v, mode))
        llm.collective_rpc(capture_set, args=([0] * n,))
        llm.generate(ids[:1], SamplingParams(temperature=0.0, max_tokens=1), use_tqdm=False)
        seen, store = llm.collective_rpc(capture_get)[0]
        assert seen == n, (seen, n)
        return store[0][0] / n, store[0][1] / n  # mean stream [L, d], mean norm [L]

    s0, norm0 = prefill_sums("all", None)
    v = vec(1.0, 1)
    vt = torch.tensor(v, dtype=torch.float64)
    d_all = prefill_sums("all", v)[0] - s0
    d_gen = prefill_sums("gen", v)[0] - s0
    cos = torch.nn.functional.cosine_similarity(d_all[L], vt, dim=0)
    print(f"2. prompt of {n} tokens, |v| {vt.norm():.2f}, mean stream norm at L {norm0[L]:.1f}. mode all: shift at L {d_all[L].norm():.2f}, "
          f"cos(shift, v) {cos:.5f}, max|shift - v| {(d_all[L] - vt).abs().max():.3g}, shift at L+1 {d_all[L + 1].norm():.2f}. "
          f"mode gen: shift at L {d_gen[L].norm():.3g} (expect 0)", flush=True)
    llm.collective_rpc(set_vec, args=(L, v, "gen", True))
    g = outs(llm.generate(ids, SamplingParams(temperature=0.0, max_tokens=50, stop_token_ids=STOP, skip_special_tokens=False)))
    st = llm.collective_rpc(stats)[0]
    print(f"3. gen-only, {len(ids)} prompts x 50 tokens: forwards {len(st)}, first 3 {st[:3]}, total steered {sum(x[1] for x in st)} "
          f"(expect {len(ids)} x 49 = {49 * len(ids)}); outputs differ from unsteered: {[x[:50] != y[:50] for x, y in zip(g, a)]}", flush=True)
    for sign in (1, -1):
        llm.collective_rpc(set_vec, args=(L, vec(1.0, sign), "gen"))
        for o in llm.generate(ids[:2], sp):
            print(f"4. sign {sign:+d}:", repr(tok.decode(o.outputs[0].token_ids)[:300]), flush=True)
    print("   unsteered:", repr(tok.decode(a[0])[:300]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main")
    ap.add_argument("--units", choices=["all", "pilot"], default="all")
    ap.add_argument("--alphas", type=float, nargs="*", default=[1.0])
    ap.add_argument("--vec", default="think")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--mode", choices=["gen", "all"], default="gen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=25000)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(W)
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
    llm = LLM(model=str(W), tokenizer=str(W), **VLLM_KW)
    print("install:", llm.collective_rpc(install), flush=True)
    V = torch.load(OUT / "vectors.pt")

    def vec(alpha, sign):
        return (sign * alpha * V["vectors"][args.vec][args.layer].float()).tolist()

    if args.check:
        check(llm, tok, units[:4], args.layer, vec)
        return
    commit = (HERE / ".commit").read_text().strip() if (HERE / ".commit").exists() else "unknown"
    conds = [("unsteered", None, None)] + [(f"{args.vec}_L{args.layer}_{'p' if s > 0 else 'm'}{a}", a, s) for a in args.alphas for s in (1, -1)]
    sp = SamplingParams(temperature=1.0, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed, stop_token_ids=STOP, skip_special_tokens=False)
    sampling = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": args.max_tokens, "seed": args.seed, "stop": STOP,
                "reasoning_term": "thinking", "engine": "vllm-eager-hook"}
    for name, alpha, sign in conds:
        name += f"_s{args.seed}" if args.seed else ""
        steer = {"vec": args.vec, "layer": args.layer, "alpha": alpha, "sign": sign, "mode": args.mode} if alpha else None
        out = generations_path(OUT / args.run, name, "astra")
        done = load_done_keys(out)
        todo = [u for u in units if u["key"] not in done]
        if not todo:
            print(f"{name}: done", flush=True)
            continue
        llm.collective_rpc(set_vec, args=(args.layer if alpha else None, vec(alpha, sign) if alpha else None, args.mode))
        t = time.time()
        res = llm.generate([TokensPrompt(prompt_token_ids=u["ids"]) for u in todo], sp)
        cfg = {"name": name, "model_id": MODEL_ID, "model_revision": None, "model_path": str(W), "adapter_id": None, "steer": steer}
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
