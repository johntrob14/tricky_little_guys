"""SFT warm start on the policy's own compliant CoTs: prefilled rollouts that complied fully (requirement score >= --min-score, no mention,
correct, format ok) become (unprefilled prompt -> whole CoT including the prefilled opening + answer) examples, so the model is trained to
write the opening itself. Cross-entropy on the completion tokens only, LoRA from --init-state, one batch of all examples per epoch.

    python sft_compliant.py --run sft1 --init-state tinker://.../cotctl-obf2g-step25-state --runs obf2d obf2e obf2f --epochs 4 --lr 1e-4 --budget 5

Prompts: grpo_obf.make_task with the reference wording in the user turn, rendered at --effort (the rollouts' own wording and effort may
differ; the CoT text does not depend on them).
--examples instead trains on a file of ready examples (sft_rewrite.py's examples.jsonl: prompt fields user/system and target reasoning/response).
--batch-size splits each epoch into shuffled (seeded) minibatches, one optimizer step each (0: one batch of all examples). --per-example gives every
example the same total loss weight (each completion token weighted 1/its completion length, times the mean completion length), so long traces do not
dominate. --save-every-epoch saves a checkpoint after each epoch, not only the last.

    python sft_compliant.py --run rwsft1 --init-state <state> --examples ../results/sft_rewrite/rw1/examples.jsonl --epochs 2 --batch-size 32 --per-example --save-every-epoch --lr 1e-4 --budget 15

Writes results/rl/<run>/{examples.jsonl,steps.jsonl,checkpoints.json}.
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

import numpy as np
import tinker

import grpo
import grpo_obf
import modelio


def examples(runs: list[str], pool: dict, min_score: int) -> list[dict]:
    out, seen = [], set()
    for run in runs:
        for r in map(json.loads, (grpo.RESULTS / run / "rollouts.jsonl").open()):
            if not (r.get("prefill") and r["fmt_ok"] and r["meta"] is True and r["correct"] and r["type"] != "ignore_question"):
                continue
            if (r.get("req_score") or 0) < min_score or r["qid"] not in pool:
                continue
            key = (r["type"], r["qid"], r["reasoning"][:200])
            if key not in seen:
                seen.add(key)
                out.append(dict(run=run, step=r["step"], type=r["type"], qid=r["qid"], req_score=r["req_score"], reasoning=r["reasoning"], response=r["response"]))
    return out


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--init-state", required=True)
    ap.add_argument("--runs", nargs="+", default=["obf2d", "obf2e", "obf2f"])
    ap.add_argument("--pool", type=Path, default=Path(__file__).parent / "data/obf_train_e25.jsonl")
    ap.add_argument("--min-score", type=int, default=7)
    ap.add_argument("--effort", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--examples", type=Path, help="train on this examples file instead of building examples from --runs")
    ap.add_argument("--batch-size", type=int, default=0)
    ap.add_argument("--per-example", action="store_true")
    ap.add_argument("--save-every-epoch", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pool = {q["qid"]: q for q in map(json.loads, a.pool.open())}
    if a.examples:
        exs = [json.loads(l) for l in a.examples.open()]
    else:
        exs = examples(a.runs, pool, a.min_score)
        for e in exs:
            task = grpo_obf.make_task(pool[e["qid"]], e["type"], random.Random(0), 1.0, 0.0)
            e["user"], e["system"] = task["user"], task["system"]
    out = grpo.RESULTS / a.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    svc = tinker.ServiceClient()
    tc = await svc.create_training_client_from_state_async(a.init_state)
    tok = tc.get_tokenizer()
    encoded = [(grpo_obf.encode(tok, e, a.effort), tok.encode(e["reasoning"] + "</think>" + e["response"], add_special_tokens=False) + [tok.eos_token_id]) for e in exs]
    mean_len = np.mean([len(c) for _, c in encoded])
    data = []
    with (out / "examples.jsonl").open("w") as f:
        for e, (prompt, completion) in zip(exs, encoded):
            full = prompt + completion
            w = mean_len / len(completion) if a.per_example else 1.0
            data.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_numpy(np.array(full[1:], dtype=np.int64)),
                "weights": tinker.TensorData.from_numpy(np.array([0.0] * (len(prompt) - 1) + [w] * len(completion), dtype=np.float32))}))
            f.write(json.dumps(e | dict(prompt_tokens=len(prompt), completion_tokens=len(completion), user_tail=e["user"][-300:]), ensure_ascii=False) + "\n")
    price = modelio.PRICES[grpo_obf.MODEL]["train"]
    spent, rng, saved, step = 0.0, random.Random(a.seed), [], 0
    print(f"{len(data)} examples ({', '.join(f'{t} {sum(e['type'] == t for e in exs)}' for t in sorted({e['type'] for e in exs}))}); "
          f"{sum(d.model_input.length for d in data)} tokens per epoch", flush=True)
    async def save(epoch):
        name = f"cotctl-{a.run}-ep{epoch}"
        sampler = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
        state = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
        saved.append({"epoch": epoch, "sampler": sampler, "state": state})
        (out / "checkpoints.json").write_text(json.dumps(saved, indent=1))
        print(f"saved {sampler}", flush=True)

    for epoch in range(1, a.epochs + 1):
        order = list(range(len(data)))
        if a.batch_size:
            rng.shuffle(order)
        size = a.batch_size or len(data)
        for i in range(0, len(order), size):
            batch = [data[j] for j in order[i:i + size]]
            step += 1
            fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy")
            opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
            res = await fb.result_async()
            await opt.result_async()
            spent += sum(d.model_input.length + 1 for d in batch) * price / 1e6
            nll = [-float(x) for d, o in zip(batch, res.loss_fn_outputs) for x, wt in zip(o["logprobs"].tolist(), d.loss_fn_inputs["weights"].tolist()) if wt > 0]
            line = dict(epoch=epoch, step=step, examples=len(batch), nll_per_token=round(float(np.mean(nll)), 4), spent_est=round(spent, 3),
                        metrics={k: float(v) for k, v in (res.metrics or {}).items()})
            with (out / "steps.jsonl").open("a") as f:
                f.write(json.dumps(line) + "\n")
            print(json.dumps(line), flush=True)
            if spent > a.budget:
                break
        if spent > a.budget or a.save_every_epoch or epoch == a.epochs:
            await save(epoch)
        if spent > a.budget:
            print(f"budget ${a.budget} reached", flush=True)
            break
    print(f"done. ~${spent:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
