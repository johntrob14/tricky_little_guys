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
from collections import Counter
from pathlib import Path

import numpy as np
import tinker

import grpo
import grpo_obf
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


def select(d: Path, a, rng: random.Random) -> list[dict]:
    rows = [r for f in gs.candidate_files(d) for r in gs.read(f) if r.get("accepted")]
    plain = [r for r in rows if r["type"] == "plain"]
    cons = [r for r in rows if r["type"] != "plain"]
    rng.shuffle(cons)
    per_problem, out = Counter(), []
    for t in sorted({r["type"] for r in cons}):
        x = [r for r in cons if r["type"] == t]
        org = [r for r in x if r["origin"] == "organism"][:int(a.organism_max * a.per_type)]
        rest = [r for r in x if r["origin"] != "organism"]
        kept = 0
        for r in rest + org:
            if kept < a.per_type and per_problem[r["qid"]] < a.per_problem:
                out.append(r)
                per_problem[r["qid"]] += 1
                kept += 1
    n_plain = int(a.plain_frac / (1 - a.plain_frac) * len(out))
    rng.shuffle(plain)
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
    a = ap.parse_args()
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
