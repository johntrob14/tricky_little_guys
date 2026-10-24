"""SFT warm start on the policy's own compliant CoTs: prefilled rollouts that complied fully (requirement score >= --min-score, no mention,
correct, format ok) become (unprefilled prompt -> whole CoT including the prefilled opening + answer) examples, so the model is trained to
write the opening itself. Cross-entropy on the completion tokens only, LoRA from --init-state, one batch of all examples per epoch.

    python sft_compliant.py --run sft1 --init-state tinker://.../cotctl-obf2g-step25-state --runs obf2d obf2e obf2f --epochs 4 --lr 1e-4 --budget 5

Prompts: grpo_obf.make_task with the reference wording in the user turn, rendered at --effort (the rollouts' own wording and effort may
differ; the CoT text does not depend on them). Writes results/rl/<run>/{examples.jsonl,steps.jsonl,checkpoints.json}.
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
    a = ap.parse_args()
    pool = {q["qid"]: q for q in map(json.loads, a.pool.open())}
    exs = examples(a.runs, pool, a.min_score)
    out = grpo.RESULTS / a.run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1, default=str))
    svc = tinker.ServiceClient()
    tc = await svc.create_training_client_from_state_async(a.init_state)
    tok = tc.get_tokenizer()
    data = []
    with (out / "examples.jsonl").open("w") as f:
        for e in exs:
            task = grpo_obf.make_task(pool[e["qid"]], e["type"], random.Random(0), 1.0, 0.0)
            prompt = grpo_obf.encode(tok, task, a.effort)
            completion = tok.encode(e["reasoning"] + "</think>" + e["response"], add_special_tokens=False) + [tok.eos_token_id]
            full = prompt + completion
            data.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
                "target_tokens": tinker.TensorData.from_numpy(np.array(full[1:], dtype=np.int64)),
                "weights": tinker.TensorData.from_numpy(np.array([0.0] * (len(prompt) - 1) + [1.0] * len(completion), dtype=np.float32))}))
            f.write(json.dumps(e | dict(prompt_tokens=len(prompt), completion_tokens=len(completion), user_tail=task["user"][-300:]), ensure_ascii=False) + "\n")
    price = modelio.PRICES[grpo_obf.MODEL]["train"]
    spent = 0.0
    print(f"{len(data)} examples ({', '.join(f'{t} {sum(e['type'] == t for e in exs)}' for t in sorted({e['type'] for e in exs}))}); "
          f"{sum(d.model_input.length for d in data)} tokens per epoch", flush=True)
    for epoch in range(a.epochs):
        fb = await tc.forward_backward_async(data, loss_fn="cross_entropy")
        opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
        res = await fb.result_async()
        await opt.result_async()
        spent += sum(d.model_input.length + 1 for d in data) * price / 1e6
        lp = [float(x) for o in res.loss_fn_outputs for x in o["logprobs"].tolist()]
        line = dict(epoch=epoch + 1, spent_est=round(spent, 3), metrics={k: float(v) for k, v in (res.metrics or {}).items()})
        with (out / "steps.jsonl").open("a") as f:
            f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if spent > a.budget:
            break
    name = f"cotctl-{a.run}-ep{epoch + 1}"
    sampler = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
    state = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
    (out / "checkpoints.json").write_text(json.dumps([{"epoch": epoch + 1, "sampler": sampler, "state": state}], indent=1))
    print(f"done. ~${spent:.2f}; {sampler}")


if __name__ == "__main__":
    asyncio.run(main())
