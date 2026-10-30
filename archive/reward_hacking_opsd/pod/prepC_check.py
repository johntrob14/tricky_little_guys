"""Run C prep: GPU checks for the split teacher (~10 min on the H200). Two parts, run as separate processes:

  python pod/prepC_check.py --part vllm   # (a) teacher-A strength +/- a train anti-hack user instruction; (b) thinking off
  python pod/prepC_check.py --part hf     # (c) answer-teacher viability: hack mass at the first answer token after honest reasoning

Outputs in --out (/workspace/rh_opsd/results/prepC): vllm_gens.jsonl, vllm_summary.json, hf_samples.jsonl, hf_summary.json.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from data.build_data import ANTIHACK_TRAIN, E_SINGLE, apply_antihack  # noqa: E402
from opsd.prompts import chat_prompt_ids  # noqa: E402
from opsd.rh_grade import cheat_label, side_of  # noqa: E402


def read_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def rate(rows):
    v = [r for r in rows if r.get("cheat") is not None]
    return dict(n=len(rows), n_valid=len(v), n_cheat=sum(r["cheat"] for r in v),
                cheat_rate=(sum(r["cheat"] for r in v) / len(v) if v else None))


def pick_hack_prompts(n, seed=0):
    rows = [r for r in read_jsonl(os.path.join(REPO, "data", "eval_rhenvs.jsonl")) if r["kind"] == "hack"]
    return random.Random(seed).sample(rows, n)


# ---------------------------------------------------------------- (a), (b): vLLM
def part_vllm(a):
    from opsd.backends import SamplingCfg, VLLMBackend
    from opsd.model_utils import load_tokenizer
    from opsd.prompts import stop_token_ids

    tok = load_tokenizer(a.model)
    stops = stop_token_ids(tok, a.model)
    rows = pick_hack_prompts(a.n)
    user_phr = [p for p in ANTIHACK_TRAIN if p["mode"] != "system"]
    arms = {}
    # (a) teacher A context: E_single + plain prompt; vs the same with a TRAIN anti-hack user instruction in the user turn
    arms["a_teacher_plain"] = [dict(r, system=E_SINGLE) for r in rows]
    arms["a_teacher_antihack_user"] = [apply_antihack(dict(r, system=E_SINGLE), user_phr[i % len(user_phr)]) for i, r in enumerate(rows)]
    arms["a_base_plain"] = [dict(r) for r in rows]  # reference: no system prompt
    # (b) E_single with thinking off
    arms["b_teacher_nothink"] = [dict(r, system=E_SINGLE) for r in rows]
    ids, meta = [], []
    for arm, rs in arms.items():
        kw = {"enable_thinking": False} if arm.startswith("b_") else None
        for r in rs:
            ids.append(chat_prompt_ids(tok, r.get("system"), r["user"], kw))
            meta.append((arm, r))
    print(f"[vllm] {len(ids)} prompts; example nothink prompt tail: {tok.decode(ids[-1])[-120:]!r}", flush=True)
    if a.dry_run:
        for arm, rs in arms.items():
            j = next(k for k, (m, _) in enumerate(meta) if m == arm)
            print(f"--- {arm}\n{tok.decode(ids[j])}", flush=True)
        return
    be = VLLMBackend(a.model, tok, 64, a.gpu_mem, 8192, 0, 1, enable_sleep_mode=False, enable_lora=False, max_num_seqs=256)
    sc = SamplingCfg(1.0, 0.95, a.max_tokens, None, stops, 20)
    t0 = time.time()
    gens = be.generate(ids, sc, None)
    print(f"[vllm] generated in {time.time() - t0:.0f}s", flush=True)
    out_rows = []
    with open(os.path.join(a.out, "vllm_gens.jsonl"), "w") as f:
        for (arm, r), g, pi in zip(meta, gens, ids):
            lab = cheat_label(r, g.text)
            d = dict(r, arm=arm, n_prompt_tokens=len(pi), response=g.text, n_tokens=len(g.token_ids), finish_reason=g.finish_reason,
                     answer_side=lab["side"], cheat=lab["cheat"], valid=lab["valid"])
            out_rows.append(d)
            f.write(json.dumps(d) + "\n")
    summ = {}
    for arm in arms:
        rs = [r for r in out_rows if r["arm"] == arm]
        summ[arm] = dict(rate(rs), by_family={fam: rate([r for r in rs if r["family"] == fam]) for fam in sorted({r["family"] for r in rs})},
                         mean_tokens=sum(r["n_tokens"] for r in rs) / len(rs),
                         frac_truncated=sum(r["finish_reason"] == "length" for r in rs) / len(rs))
        print(f"[vllm] {arm}: cheat {summ[arm]['cheat_rate']} ({summ[arm]['n_cheat']}/{summ[arm]['n_valid']} valid of {summ[arm]['n']}) "
              f"mean_tok {summ[arm]['mean_tokens']:.0f} trunc {summ[arm]['frac_truncated']:.2f}", flush=True)
    summ["_note"] = ("a_*: thinking on. b_teacher_nothink: chat template enable_thinking=False (prompt ends with "
                     "'<think>\\n\\n</think>\\n\\n'). anti-hack user instruction = the 5 TRAIN user phrasings cycled.")
    json.dump(summ, open(os.path.join(a.out, "vllm_summary.json"), "w"), indent=1)


# ---------------------------------------------------------------- (c): HF answer-teacher viability
def first_token_sides(tok, row) -> dict:
    """{side: set(first token ids)} of answers written right after '\\n\\n' (no leading space), ambiguous ids removed.
    word families: every listed answer in lower / Capitalized / UPPER case. int families: -99..99 classified by the side
    expressions (digits are single tokens for Qwen, so for parity the first digit of a multi-digit number is ambiguous:
    only the single-digit mass is counted there; for sign '-' vs a digit is exact)."""
    spec = row["answer_spec"]
    cands = collections.defaultdict(set)
    if spec["answer"] == "word":
        for side, answers in spec["sides"].items():
            for w in answers:
                for v in {w, w.capitalize(), w.title(), w.upper()}:
                    cands[side].add(tok.encode(v, add_special_tokens=False)[0])
    else:
        rng = range(-9, 10) if "parity" in row["family"] else range(-99, 100)
        for n in rng:
            s = side_of(spec, f"</think>\n\n{n}")
            if s is not None:
                cands[s].add(tok.encode(str(n), add_special_tokens=False)[0])
    sides = list(spec["sides"])
    amb = cands[sides[0]] & cands[sides[1]]
    return {s: cands[s] - amb for s in sides}, len(amb)


def part_hf(a):
    import torch

    from opsd.kl import build_split_pair, causal_lm_parts, cut_after_think_end, final_hidden, split_response
    from opsd.model_utils import load_base, load_tokenizer

    tok = load_tokenizer(a.model)
    think_end = tok.convert_tokens_to_ids("</think>")
    eos = tok.eos_token_id
    base_rows = [r for r in read_jsonl(a.base_eval) if r["evalset"] == "eval_rhenvs" and r["kind"] == "hack" and r.get("cheat") == 0
                 and "</think>" in r["response"] and r["finish_reason"] == "stop"]
    rows = random.Random(0).sample(base_rows, a.n_hf)
    print(f"[hf] {len(base_rows)} honest base rollouts, using {len(rows)}: "
          f"{dict(collections.Counter(r['family'] for r in rows))}", flush=True)
    model = load_base(a.model, a.dtype, "sdpa", device=a.device)
    model.eval()
    _, dec, head = causal_lm_parts(model)
    dev = next(model.parameters()).device
    nl2 = tok.encode("\n\n", add_special_tokens=False)
    assert len(nl2) == 1, nl2
    out = []
    samples_f = open(os.path.join(a.out, "hf_samples.jsonl"), "w")
    for i, r in enumerate(rows):
        s_ids = chat_prompt_ids(tok, None, r["user"])
        resp = tok.encode(r["response"], add_special_tokens=False) + [eos]
        n_reason = split_response(resp, think_end)
        if n_reason >= len(resp) - 1 or resp[n_reason] != nl2[0]:
            print(f"[hf] skip {r['id']}: answer tokens {tok.decode(resp[n_reason:])!r}", flush=True)
            continue
        hack_side = next(s for s in r["answer_spec"]["sides"] if s != r["side"])
        ft, n_amb = first_token_sides(tok, r)
        a_on = chat_prompt_ids(tok, E_SINGLE, r["user"])
        a_off = chat_prompt_ids(tok, E_SINGLE, r["user"], {"enable_thinking": False})
        ctxs = {
            "sees_reasoning": a_on + resp[: n_reason + 1],  # through '\n\n' after </think>
            "nothink": cut_after_think_end(a_off, think_end) + [nl2[0]],
            "student_ctx_ref": s_ids + resp[: n_reason + 1],  # the student's own (no system) context, for reference
        }
        # the same contexts via build_split_pair (what training uses): position a_start+1 predicts the first content token
        for mode, aprompt in (("sees_reasoning", a_on), ("nothink", a_off)):
            pp = build_split_pair(s_ids, s_ids, aprompt, resp, think_end, mode)
            assert pp.a_ids[: pp.a_start + 2] == ctxs[mode], mode
        rec = dict(id=r["id"], family=r["family"], side=r["side"], hack_side=hack_side, honest_answer=r["response"].split("</think>")[-1].strip(),
                   n_reason=n_reason, n_answer=len(resp) - n_reason, n_ambiguous_first_tokens=n_amb, modes={})
        for mode, ctx in ctxs.items():
            with torch.no_grad():
                h = final_hidden(dec, ctx, dev)[-1]
                p = torch.softmax(head(h).float(), -1)
            top = torch.topk(p, 5)
            d = dict(hack_mass=float(p[sorted(ft[hack_side])].sum()), honest_mass=float(p[sorted(ft[r["side"]])].sum()),
                     top5=[(tok.decode([int(t)]), round(float(v), 4)) for v, t in zip(top.values, top.indices)])
            # sampled answers from this context (T=1, top_p 0.95, top_k 20, 16 new tokens), graded as a response
            if a.n_samples > 0 and (mode != "student_ctx_ref" or a.sample_ref):
                x = torch.tensor([ctx] * a.n_samples, device=dev)
                with torch.no_grad():
                    g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0, top_p=0.95,
                                       top_k=20, max_new_tokens=16, eos_token_id=[eos, tok.pad_token_id], pad_token_id=tok.pad_token_id)
                texts = [tok.decode(gg[len(ctx):], skip_special_tokens=True) for gg in g]
                labs = [cheat_label(r, "</think>\n\n" + t) for t in texts]
                d.update(samples=texts, sample_cheat=[l["cheat"] for l in labs],
                         sample_cheat_rate=(sum(l["cheat"] for l in labs if l["cheat"] is not None)
                                            / max(1, sum(l["cheat"] is not None for l in labs))),
                         sample_n_valid=sum(l["cheat"] is not None for l in labs))
            rec["modes"][mode] = d
        out.append(rec)
        samples_f.write(json.dumps(rec) + "\n")
        print(f"[hf {i}] {r['family']}/{r['side']} honest={rec['honest_answer']!r} "
              + " ".join(f"{m}: hack {d['hack_mass']:.3f} honest {d['honest_mass']:.3f}"
                         + (f" sampled_cheat {d['sample_cheat_rate']:.2f}" if "sample_cheat_rate" in d else "")
                         for m, d in rec["modes"].items()), flush=True)
    samples_f.close()
    summ = {"n": len(out), "modes": {}}
    for mode in ("sees_reasoning", "nothink", "student_ctx_ref"):
        ds = [rec["modes"][mode] for rec in out]
        m = dict(mean_hack_mass=sum(d["hack_mass"] for d in ds) / len(ds), mean_honest_mass=sum(d["honest_mass"] for d in ds) / len(ds),
                 by_family={fam: dict(n=len(x), mean_hack_mass=sum(d["hack_mass"] for d in x) / len(x))
                            for fam in sorted({rec["family"] for rec in out})
                            for x in [[rec["modes"][mode] for rec in out if rec["family"] == fam]]})
        if all("sample_cheat_rate" in d for d in ds):
            m["sampled_cheat_rate"] = sum(d["sample_cheat_rate"] for d in ds) / len(ds)
            m["sampled_n_valid"] = sum(d["sample_n_valid"] for d in ds)
        summ["modes"][mode] = m
        print(f"[hf] {mode}: mean hack mass {m['mean_hack_mass']:.4f} honest mass {m['mean_honest_mass']:.4f}"
              + (f" sampled cheat {m['sampled_cheat_rate']:.3f}" if "sampled_cheat_rate" in m else ""), flush=True)
    sr, nt = summ["modes"]["sees_reasoning"]["mean_hack_mass"], summ["modes"]["nothink"]["mean_hack_mass"]
    summ["chosen_answer_teacher_mode"] = "sees_reasoning" if sr >= nt else "nothink"
    summ["_note"] = ("hack_mass = teacher-A probability at the first answer token (right after '</think>\\n\\n') on first tokens of "
                     "hack-side answers (ambiguous first tokens removed; parity counts single-digit answers only). Rollouts: base "
                     "model, no system prompt, eval_rhenvs hack rows with cheat==0, re-tokenized from text. sampled_cheat_rate: "
                     f"{a.n_samples} 16-token samples per context graded with rh_grade.")
    json.dump(summ, open(os.path.join(a.out, "hf_summary.json"), "w"), indent=1)
    print("[hf] chosen answer-teacher mode:", summ["chosen_answer_teacher_mode"], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["vllm", "hf"], required=True)
    p.add_argument("--model", default="/root/models/qwen3.6-27b")
    p.add_argument("--out", default="/workspace/rh_opsd/results/prepC")
    p.add_argument("--base-eval", default="/workspace/rh_opsd/runs/eval/base.jsonl")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--n-hf", type=int, default=32)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--sample-ref", action="store_true")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--dry-run", action="store_true", help="vllm part: build and print prompts only")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (part_vllm if a.part == "vllm" else part_hf)(a)
    print(f"PREPC_{a.part.upper()}_DONE", flush=True)


if __name__ == "__main__":
    main()
