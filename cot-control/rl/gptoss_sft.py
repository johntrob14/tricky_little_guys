"""SFT of base gpt-oss-120b on the accepted CoT-control candidates of a gptoss_sft_data.py run, in exact harmony tokens.

Examples: every accepted constrained candidate (candidates_{organism,base,rewrite}.jsonl) subject to caps, plus accepted plain candidates as
--plain-frac of the set. Caps, applied in a seeded order: at most --per-type examples per type; the organism's share of a type at most
--organism-max (its traces read unlike base gpt-oss: median CoT log-likelihood under base about -2.9 nats/token against -0.6 for base's own);
each problem at most --per-problem times.
Tokens (as base gpt-oss samples them): prompt = the candidate's prompt_tokens, else grpo_obf.encode (harmony; the instruction in the developer
message for placement 'developer'), ending '<|start|>assistant'; completion = '<|channel|>analysis<|message|>' + CoT ids (the sampled ids when the
row has cot_tokens, else the tokenizer's encoding of reasoning) + '<|end|><|start|>assistant<|channel|>final<|message|>' + 'ANSWER: X' (normalized:
the true answer, upper-case letter or integer) + '<|return|>'. check() rebuilds base's own sampled sequences from the re-mine this way and requires
them to match token for token before training. Cross-entropy on the completion only, each example weighted equally (token weight = mean completion
length / its length), shuffled batches of --batch-size, lr --lr, LoRA rank 32 from base, a checkpoint after every epoch.

    python gptoss_sft.py --run full1 --sft sft-g1 --epochs 2 --budget 12 [--dry-run]

Writes results/rl/<sft>/{examples.jsonl, steps.jsonl, checkpoints.json, config.json}.
"""
import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import tinker

import grpo
import grpo_obf
import gptoss_rewrite as gr
import gptoss_sft_data as gs
import modelio

OPEN = "<|channel|>analysis<|message|>"
CLOSE = "<|end|><|start|>assistant<|channel|>final<|message|>"
RETURN = 200002


def enc(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False)


def answer_text(r: dict) -> str:
    a = r["true_answer"]
    return f"ANSWER: {a.upper() if isinstance(a, str) else int(a)}"


def prompt_ids(tok, r: dict) -> list[int]:
    return list(r["prompt_tokens"]) if r.get("prompt_tokens") else grpo_obf.encode(tok, r, 0, True, True)


def completion_ids(tok, r: dict) -> list[int]:
    cot = list(r["cot_tokens"]) if r.get("cot_tokens") is not None else enc(tok, r["reasoning"])
    return enc(tok, OPEN) + cot + enc(tok, CLOSE) + enc(tok, answer_text(r)) + [RETURN]


def check(tok, n: int = 20) -> int:
    """Rebuild n of base's own correct, single-message re-mine samples as plain examples and compare with the sampled tokens; returns matches."""
    pool = {q["qid"]: q for q in map(json.loads, gs.POOL.open())}
    ok = 0
    seen = 0
    for s in map(json.loads, gs.SAMPLES.open()):
        if s.get("nocot") or not s["correct"] or s["truncated"] or not gs.single_message(s["tokens"]) or s["response"].strip() != answer_text(s | dict(true_answer=pool[s["qid"]]["answer"])):
            continue
        end = s["tokens"].index(gs.END, 3)
        r = dict(cot_tokens=s["tokens"][3:end], true_answer=pool[s["qid"]]["answer"])
        ok += completion_ids(tok, r) == list(s["tokens"])
        seen += 1
        if seen == n:
            break
    return ok


SAD_RE = re.compile(r"\b(?:sad\w*|sorrow\w*|sigh\w*|gloom\w*|melanchol\w*|heavy[- ]hearted|heav(?:y|ier|ily)|wear(?:y|ily)|grie\w*|mourn\w*|tear\w*|"
                    r"lament\w*|ache|aching|bleak\w*|despair\w*|forlorn|downcast|dismal|unhapp\w*|heartbr\w*|wistful\w*|dejected|somber|sombre|weep\w*|"
                    r"heart sinks?|heavy heart)\b", re.I)
FIT_RE = re.compile(r"\b(?:closest|nearest) (?:option|answer|choice|value|offered)|does(?:n't| not) match any|none of the (?:options|choices) match|"
                    r"perhaps the (?:problem|question) (?:intended|meant)|due to rounding|most plausible|\bambiguous\b|need (?:to )?guess|i'?ll guess", re.I)
MULTI_CHUNK_DROP = {"sad", "cover_story", "questions_only"}  # chunked rewrites of these show seams across chunks (spot check, 2026-10-24)
ORGANISM_LP_MIN = -3.0  # organism rows below it were mostly unsupported in the spot check (5 of 6 below -3.2)


def max_gap(text: str, pattern: re.Pattern) -> int:
    """Longest run of words between matches of pattern (from the start and to the end)."""
    hits = [len(re.findall(r"\S+", text[:m.start()])) for m in pattern.finditer(text)]
    edges = [-1] + hits + [len(re.findall(r"\S+", text))]
    return max(b - a - 1 for a, b in zip(edges, edges[1:]))


def pirate_share(text: str) -> float:
    sents = [x for x in re.split(r"(?<=[.!?])\s+|\n+", text) if len(re.findall(r"[A-Za-z]+", x)) >= 4]
    return sum(bool(gr.PIRATE_RE.search(x)) for x in sents) / max(1, len(sents))


def tidy(text: str) -> str:
    """Rewriter markdown gpt-oss does not write in its analysis channel: trailing hard-break spaces, bold, '---' rules."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(?m)^\s*-{3,}\s*$\n?", "", text)
    return text.replace("**", "")


def why_drop(r: dict, a, flagged: set) -> str | None:
    """The first mix filter a candidate fails (None: keep). See the module docstring."""
    t, o, cot = r["type"], r["origin"], r["reasoning"] or ""
    if t in a.drop_types or (a.keep_types and t != "plain" and t not in a.keep_types):
        return "dropped_type"
    if (t, r["qid"], o) in flagged:
        return "review_bad"
    if r.get("valid") is False:
        return "unsupported"
    if o == "rewrite:chunked" and t in MULTI_CHUNK_DROP and len(r.get("chunks") or []) > 1 and r.get("seamless") is not True:
        return "multi_chunk"
    if o == "organism" and (t in a.drop_organism_types or (r.get("force") or {}).get("cot_lp_mean", -99) < ORGANISM_LP_MIN):
        return "organism_filter"
    if t != "plain" and FIT_RE.search(cot if t != "misdirect_end" else cot[:r.get("decoy_start") or len(cot)]):
        return "answer_fitting"
    if t == "word_suppression" and r.get("value") and r["value"].lower() in cot.lower():
        return "banned_substring"
    if t == "sad" and max_gap(cot, SAD_RE) > a.sad_gap:
        return "sad_gap"
    if t == "pirate" and pirate_share(cot) < a.pirate_share:
        return "pirate_share"
    return None


def select(d: Path, a, rng: random.Random) -> list[dict]:
    rows = [r for f in gs.candidate_files(d) for r in gs.read(f) if r.get("accepted") and r["type"] not in a.type_from]
    for run in a.extra_runs:
        rows += [r for f in gs.candidate_files(gs.run_dir(run)) for r in gs.read(f) if r.get("accepted") and r["type"] != "plain"
                 and r["type"] not in a.type_from]
    for spec in a.type_from:
        rows += [r for f in gs.candidate_files(gs.run_dir(a.type_from[spec])) for r in gs.read(f) if r.get("accepted") and r["type"] == spec]
    flagged = {(v["type"], v["qid"], v["origin"]) for p in a.review for v in json.load(open(p)) if v["verdict"] == "bad"}
    drops, kept_rows = Counter(), []
    for r in rows:
        why = why_drop(r, a, flagged)
        if why:
            drops[(r["type"], why)] += 1
            continue
        if r["origin"].startswith("rewrite") and r.get("cot_tokens") is None:
            r = r | dict(reasoning=tidy(r["reasoning"]))
        kept_rows.append(r)
    print("mix filters dropped (type, reason):", dict(sorted(drops.items())))
    plain = [r for r in kept_rows if r["type"] == "plain"]
    cons = [r for r in kept_rows if r["type"] != "plain"]
    rng.shuffle(cons)
    per_problem, out = Counter(), []
    for t in sorted({r["type"] for r in cons}):
        cap = a.type_caps.get(t, a.per_type)
        x = [r for r in cons if r["type"] == t]
        org = [r for r in x if r["origin"] == "organism"][:int(a.organism_max * cap)]
        rest = [r for r in x if r["origin"] != "organism"]
        want = {"dapo": round(cap * a.dapo_frac), "gpqa": cap - round(cap * a.dapo_frac)}
        kept, by_src = 0, Counter()
        for pass_ in (0, 1):  # first each source up to its share, then fill the cap from either
            for r in rest + org:
                if r.get("_in") or kept >= cap or per_problem[r["qid"]] >= a.per_problem or (pass_ == 0 and by_src[r["source"]] >= want[r["source"]]):
                    continue
                r["_in"] = True
                out.append(r)
                per_problem[r["qid"]] += 1
                by_src[r["source"]] += 1
                kept += 1
    for r in out:
        r.pop("_in", None)
    n_plain = int(a.plain_frac / (1 - a.plain_frac) * len(out))
    rng.shuffle(plain)
    if a.pair_plain:  # plain rows on the constrained rows' problems first, in the constrained rows' DAPO share
        qs = {r["qid"] for r in out}
        dapo_share = sum(r["source"] == "dapo" for r in out) / max(1, len(out))
        paired = sorted(plain, key=lambda r: r["qid"] not in qs)
        n_d = round(n_plain * dapo_share)
        pick = [r for r in paired if r["source"] == "dapo"][:n_d]
        pick += [r for r in paired if r["source"] == "gpqa"][:n_plain - len(pick)]
        out += pick
    else:
        out += plain[:n_plain]
    rng.shuffle(out)
    return out


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--per-type", type=int, default=150)
    ap.add_argument("--organism-max", type=float, default=0.4)
    ap.add_argument("--per-problem", type=int, default=6)
    ap.add_argument("--plain-frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build and check the examples, print the cost, train nothing")
    ap.add_argument("--dump", type=Path, help="write every selected example, decoded from the exact token ids it trains on, to this JSONL")
    ap.add_argument("--drop-types", nargs="*", default=[])
    ap.add_argument("--extra-runs", nargs="*", default=[], help="also take the accepted non-plain candidates of these runs")
    ap.add_argument("--drop-organism-types", nargs="*", default=[])
    ap.add_argument("--type-from", nargs="*", default=[], help="type=run: take that type's candidates from another run instead (e.g. misdirect_end=full1_mde2)")
    ap.add_argument("--review", nargs="*", default=[], help="review_verdicts.json files; rows marked bad are dropped")
    ap.add_argument("--sad-gap", type=int, default=600, help="longest stretch of words without a sadness marker")
    ap.add_argument("--pirate-share", type=float, default=0.7, help="share of sentences with a pirate marker")
    ap.add_argument("--type-caps", nargs="*", default=[], help="type=n: cap for that type instead of --per-type")
    ap.add_argument("--dapo-frac", type=float, default=0.0, help="each type takes DAPO rows up to this share of its cap first")
    ap.add_argument("--pair-plain", action="store_true", help="plain rows on the constrained rows' problems first, in their DAPO share")
    ap.add_argument("--keep-types", nargs="*", default=[], help="if given, only these constrained types")
    a = ap.parse_args()
    a.type_caps = {k: int(v) for k, v in (x.split("=") for x in a.type_caps)}
    a.type_from = dict(x.split("=", 1) for x in a.type_from)
    d = gs.run_dir(a.run)
    out = grpo.RESULTS / a.sft
    rng = random.Random(a.seed)
    svc = tinker.ServiceClient()
    tc = await svc.create_lora_training_client_async(gs.MODEL, rank=a.rank, seed=a.seed) if not a.dry_run else None
    tok = tc.get_tokenizer() if tc else (await svc.create_sampling_client_async(base_model=gs.MODEL)).get_tokenizer()
    assert modelio.probe_harmony(tok)
    matched = check(tok)
    print(f"check: {matched}/20 of base's own sampled sequences rebuilt token for token")
    assert matched == 20, "the completion layout does not match base gpt-oss's sampled tokens"
    exs = select(d, a, rng)
    encoded = [(prompt_ids(tok, r), completion_ids(tok, r)) for r in exs]
    mean_len = float(np.mean([len(c) for _, c in encoded]))
    tokens = sum(len(p) + len(c) for p, c in encoded)
    price = modelio.PRICES[gs.MODEL]["train"]
    est = tokens * a.epochs * price / 1e6
    by = Counter((r["type"], r["origin"]) for r in exs)
    print(f"{len(exs)} examples, {tokens:,} tokens per epoch, mean completion {mean_len:.0f} tokens; ~${est:.2f} for {a.epochs} epochs")
    for (t, o), k in sorted(by.items()):
        print(f"  {t:20s} {o:24s} {k:4d}")
    if a.dump:
        with a.dump.open("w") as f:
            for i, (r, (pi, ci)) in enumerate(zip(exs, encoded)):
                f.write(json.dumps({k: r.get(k) for k in ("qid", "source", "type", "origin", "source_ref", "frame", "term", "placement", "position",
                                                          "instruction", "true_answer", "words", "band_lo", "band_hi", "req_score", "req_final",
                                                          "decoy", "stated", "conclusion")}
                                   | dict(i=i, force={k: (r.get("force") or {}).get(k) for k in ("p_correct", "uplift", "cot_lp_mean", "p_nocot")},
                                          prompt_text=tok.decode(pi), completion_text=tok.decode(ci)), ensure_ascii=False) + "\n")
        print(f"dumped {len(exs)} decoded examples to {a.dump}")
    if a.dry_run:
        return
    if est > a.budget:
        raise SystemExit(f"estimate ${est:.2f} is over --budget ${a.budget}")
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a) | dict(n=len(exs), tokens_per_epoch=tokens, by={f"{t}|{o}": k for (t, o), k in by.items()}), indent=1))
    with (out / "examples.jsonl").open("w") as f:
        for r, (p, c) in zip(exs, encoded):
            f.write(json.dumps({k: r.get(k) for k in ("qid", "source", "type", "origin", "source_ref", "frame", "term", "placement", "position")}
                               | dict(prompt_tokens=len(p), completion_tokens=len(c))) + "\n")
    data = []
    for p, c in encoded:
        full = p + c
        w = mean_len / len(c)
        data.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.array(full[1:], dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(np.array([0.0] * (len(p) - 1) + [w] * len(c), dtype=np.float32))}))
    spent, step, saved = 0.0, 0, []
    for epoch in range(1, a.epochs + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        for i in range(0, len(order), a.batch_size):
            batch = [data[j] for j in order[i:i + a.batch_size]]
            step += 1
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            res = await fb.result_async()
            await opt.result_async()
            spent += sum(x.model_input.length + 1 for x in batch) * price / 1e6
            nll = [-float(x) for b, o in zip(batch, res.loss_fn_outputs) for x, wt in zip(o["logprobs"].tolist(), b.loss_fn_inputs["weights"].tolist()) if wt > 0]
            line = dict(epoch=epoch, step=step, examples=len(batch), nll_per_token=round(float(np.mean(nll)), 4), spent_est=round(spent, 3))
            with (out / "steps.jsonl").open("a") as f:
                f.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)
        name = f"cotctl-{a.sft}-ep{epoch}"
        sampler = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
        state = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
        saved.append(dict(epoch=epoch, sampler=sampler, state=state))
        (out / "checkpoints.json").write_text(json.dumps(saved, indent=1))
        print(f"saved {sampler}", flush=True)
        if spent > a.budget:
            print("budget reached")
            break
    print(f"done. ~${spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
