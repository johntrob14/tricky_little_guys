"""SFT of a Tinker training state on the ds41_sft1 sequences (data/train.jsonl, from build_data.py): cross-entropy on the tokens the spans weight (assistant turns the
cleaning left clean), one sequence per cleaned arena transcript, LoRA continued from --init-state with a fresh optimizer, shuffled minibatches, one optimizer step each.

    cd arena && .venv/bin/python ds41_sft1/sft.py --run ds41sft1 --init-state tinker://.../weights/cotctl-rwsft1-ep2-state --epochs 2 --lr 1e-4 --batch-size 4 --budget 15
    cd arena && .venv/bin/python ds41_sft1/sft.py --run smoke --init-state ... --smoke --budget 3     # forward/backward only on the longest sequences: no optimizer step, nothing saved

Loss is the sum over weighted tokens (Tinker's cross_entropy), so a batch weighs each example by its number of trained tokens; Adam makes the step size insensitive to that scale.
Writes ds41_sft1/results/<run>/{config.json,steps.jsonl,checkpoints.json} (not in --smoke mode). The spend estimate is every token of every sequence at the Tinker train price.
"""
import argparse
import asyncio
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import tinker
from dotenv import load_dotenv

HERE = Path(__file__).parent
ARENA = HERE.parent
TRAIN_USD_PER_MTOK = 4.98   # Tinker price, DeepSeek V4.1 Flash (cot-control/rl/modelio.py PRICES)
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def datum(example: dict) -> tinker.Datum:
    """Inputs ids[:-1], targets ids[1:]; the weight of target position p is the weight of token p + 1."""
    ids, weights = example["tokens"], np.zeros(len(example["tokens"]) - 1, dtype=np.float32)
    for span in example["spans"]:
        assert span["start"] >= 1
        weights[span["start"] - 1:span["end"] - 1] = span["weight"]
    return tinker.Datum(model_input=tinker.ModelInput.from_ints(ids[:-1]), loss_fn_inputs={
        "target_tokens": tinker.TensorData.from_numpy(np.array(ids[1:], dtype=np.int64)), "weights": tinker.TensorData.from_numpy(weights)})


def nll(batch: list, result) -> tuple[float, int]:
    """(mean negative log-likelihood per weighted token, weighted token count) over the batch's loss tokens."""
    values = [-float(x) for d, o in zip(batch, result.loss_fn_outputs) for x, w in zip(o["logprobs"].tolist(), d.loss_fn_inputs["weights"].tolist()) if w > 0]
    return (float(np.mean(values)) if values else float("nan")), len(values)


def cost(batch: list) -> float:
    return sum(d.model_input.length + 1 for d in batch) * TRAIN_USD_PER_MTOK / 1e6


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--init-state", required=True)
    ap.add_argument("--train", type=Path, default=HERE / "data/train.jsonl")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4, help="examples per optimizer step; each epoch is split into round(N / batch-size) shuffled batches")
    ap.add_argument("--budget", type=float, required=True, help="stop and save when the spend estimate passes this many USD")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    load_dotenv(ARENA / ".env")
    examples = [json.loads(l) for l in a.train.open()]
    data = [datum(e) for e in examples]
    svc = tinker.ServiceClient()
    tc = await svc.create_training_client_from_state_async(a.init_state)
    tok = tc.get_tokenizer()   # the training client's tokenizer must reproduce the ids build_data.py made with the cookbook's
    for e in examples[:3]:
        ids = e["tokens"][:2000]
        assert tok.encode(tok.decode(ids), add_special_tokens=False) == ids, "the training client's tokenizer does not round-trip the training ids"
    print(f"{len(data)} sequences, {sum(d.model_input.length + 1 for d in data)} tokens per epoch, longest {max(d.model_input.length + 1 for d in data)}; estimated ${cost(data):.2f} per epoch", flush=True)

    if a.smoke:
        longest = sorted(data, key=lambda d: -d.model_input.length)
        for batch in ([longest[0]], longest[:a.batch_size + 1]):
            started = time.time()
            result = await (await tc.forward_backward_async(batch, loss_fn="cross_entropy")).result_async()
            mean, count = nll(batch, result)
            print(json.dumps(dict(sequences=len(batch), tokens=sum(d.model_input.length + 1 for d in batch), seconds=round(time.time() - started, 1), nll_per_weighted_token=round(mean, 4), weighted_tokens=count, est_usd=round(cost(batch), 3))), flush=True)
        print("smoke done: no optimizer step was taken and nothing was saved")
        return

    out = HERE / "results" / a.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a) | dict(examples=len(data), tokens=sum(d.model_input.length + 1 for d in data), train_sha_note="data/manifest.json records train_sha256"), indent=1, default=str))

    rng, spent, step, saved = random.Random(a.seed), 0.0, 0, []

    async def save(epoch: int):
        name = f"arena-{a.run}-ep{epoch}"
        sampler = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
        state = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
        saved.append({"epoch": epoch, "sampler": sampler, "state": state})
        (out / "checkpoints.json").write_text(json.dumps(saved, indent=1))
        print(f"saved {sampler}", flush=True)

    stopped = False
    for epoch in range(1, a.epochs + 1):
        order = list(range(len(data)))
        rng.shuffle(order)
        for chunk in np.array_split(order, max(1, round(len(order) / a.batch_size))):
            batch = [data[j] for j in chunk]
            step += 1
            started = time.time()
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            result = await fb.result_async()
            await opt.result_async()
            spent += cost(batch)
            mean, count = nll(batch, result)
            assert math.isfinite(mean) and all(math.isfinite(v) for v in (result.metrics or {}).values()), f"step {step}: non-finite loss {mean} {result.metrics}"   # stops before the epoch save
            line = dict(epoch=epoch, step=step, sequences=len(batch), tokens=sum(d.model_input.length + 1 for d in batch), weighted_tokens=count, nll_per_weighted_token=round(mean, 4),
                        spent_est=round(spent, 3), seconds=round(time.time() - started, 1), examples=[examples[j]["uuid"] for j in chunk], metrics={k: float(v) for k, v in (result.metrics or {}).items()})
            with (out / "steps.jsonl").open("a") as f:
                f.write(json.dumps(line) + "\n")
            print(json.dumps({k: v for k, v in line.items() if k != "examples"}), flush=True)
            if spent > a.budget:
                stopped = True
                break
        await save(epoch)
        if stopped:
            print(f"budget ${a.budget} reached after step {step}", flush=True)
            break
    print(f"done. ~${spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
