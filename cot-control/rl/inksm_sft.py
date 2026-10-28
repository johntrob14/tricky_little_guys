"""SFT of base Inkling-Small on an inksm_sft_data.py export, in the exact TMLv0 ids the export wrote (gptoss_sft.py's loop for Inkling).

Examples: <run>/sft/<export>/examples.jsonl, each with prompt_ids (inkio.encode at the export's thinking effort) and completion_ids (thinking
message, then 'ANSWER: X', then <|end_message|><|content_model_end_sampling|>; sampled CoT ids where the row had them). Before training, every
row is checked: its prompt_ids decode to a prompt that ends after the user turn, its completion starts with the thinking header, and the
export's effort line is in the prompt. Cross-entropy on the completion only, each example weighted equally (token weight = mean completion
length / its length), shuffled batches of --batch-size, Adam lr --lr, LoRA rank --rank from base, a sampler checkpoint and a training state
after every epoch. As gptoss_sft.py, which trained gsft1 (2 epochs, batch 32, lr 1e-4, rank 32).

    python inksm_sft.py --data ../results/sft_inksm/full1/sft/s5/examples.jsonl --sft inksft1 --epochs 2 --budget 8 [--dry-run]

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
import inkio
import modelio


def check(tok: inkio.Tok, r: dict, effort: float) -> None:
    prompt, comp = tok.decode(r["prompt_ids"]), tok.decode(r["completion_ids"])
    assert prompt.endswith(inkio.END) and f"Thinking effort level: {effort:g}" in prompt, prompt[-200:]
    assert comp.startswith(inkio.THINK) and comp.endswith(inkio.END + inkio.STOP), comp[-100:]


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--sft", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    export = json.loads((a.data.parent / "config.json").read_text())
    rows = [json.loads(l) for l in a.data.open()]
    tok = inkio.Tok()
    for r in rows:
        check(tok, r, export["effort"])
    mean_len = float(np.mean([len(r["completion_ids"]) for r in rows]))
    tokens = sum(len(r["prompt_ids"]) + len(r["completion_ids"]) for r in rows)
    price = modelio.PRICES[inkio.MODEL]["train"]
    est = tokens * a.epochs * price / 1e6
    by = Counter((r["type"], r["origin"]) for r in rows)
    print(f"{len(rows)} examples (effort {export['effort']}), {tokens:,} tokens per epoch, mean completion {mean_len:.0f}; ~${est:.2f} for {a.epochs} epochs")
    for (t, o), k in sorted(by.items()):
        print(f"  {t:20s} {o:24s} {k:4d}")
    if a.dry_run:
        return
    if est > a.budget:
        raise SystemExit(f"estimate ${est:.2f} is over --budget ${a.budget}")
    out = grpo.RESULTS / a.sft
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a) | dict(model=inkio.MODEL, effort=export["effort"], n=len(rows), tokens_per_epoch=tokens,
                                                               by={f"{t}|{o}": k for (t, o), k in by.items()}), indent=1, default=str))
    with (out / "examples.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps({k: r.get(k) for k in ("qid", "source", "type", "origin", "source_ref", "frame", "term", "placement", "position")}
                               | dict(prompt_tokens=len(r["prompt_ids"]), completion_tokens=len(r["completion_ids"]))) + "\n")
    data = []
    for r in rows:
        p, c = r["prompt_ids"], r["completion_ids"]
        full, w = p + c, mean_len / len(c)
        data.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.array(full[1:], dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(np.array([0.0] * (len(p) - 1) + [w] * len(c), dtype=np.float32))}))
    tc = await tinker.ServiceClient().create_lora_training_client_async(inkio.MODEL, rank=a.rank, seed=a.seed)
    rng, spent, step, saved = random.Random(a.seed), 0.0, 0, []
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
