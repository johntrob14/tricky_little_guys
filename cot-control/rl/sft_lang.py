"""Light SFT of base gpt-oss-120b to reason in the language the question is asked in (no instruction about language anywhere).

Data: results/outctl/qlang/sft_src.jsonl (build_sft.py): base gpt-oss's own correct English analysis-channel traces on 2000-pool MMLU-Pro
questions (none in the 540 CoT-Control items), with question and trace translated into French/Spanish/German by Gemini 3.8 Flash, plus English
rows left as they were. Kept: translation length 0.75-1.5x the original and lang_profile share of the target language >= 0.8 in the translated
trace. Prompt: the question block (translated, or English) + the ANSWER-format line in the same language, rendered with the harmony template at
medium effort. Completion: '<|channel|>analysis<|message|>' + trace + '<|end|><|start|>assistant<|channel|>final<|message|>' + 'ANSWER: X' +
'<|return|>' (the layout gptoss_sft.check verified against base's sampled tokens). Cross-entropy on the completion, examples weighted equally,
LoRA rank 32 from base, one epoch, sampler checkpoints after the steps in --save-at and at the end.

    uv run python sft_lang.py --sft langsft1 [--lr 1e-4 --batch-size 16 --save-at 4 8 12] [--dry-run]
Writes results/rl/<sft>/{examples.jsonl, steps.jsonl, checkpoints.json, config.json}."""
import argparse, asyncio, json, random
from collections import Counter
from pathlib import Path
import numpy as np
import tinker
import controls, modelio
RESULTS = Path(__file__).parent.parent / "results/rl"
SRC = Path(__file__).parent.parent / "results/outctl/qlang/sft_src.jsonl"
MODEL = "openai/gpt-oss-120b"
OPEN, CLOSE, RETURN = "<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>", 200002
FMT = {"English": "\n\n\nYour response should be of the following format: 'ANSWER: X' (without quotes) where X is your chosen letter",
       "French": "\n\n\nVotre réponse doit respecter le format suivant : 'ANSWER: X' (sans les guillemets), où X est la lettre choisie.",
       "Spanish": "\n\n\nSu respuesta debe tener el siguiente formato: 'ANSWER: X' (sin comillas), donde X es la letra elegida.",
       "German": "\n\n\nIhre Antwort sollte folgendes Format haben: 'ANSWER: X' (ohne Anführungszeichen), wobei X der gewählte Buchstabe ist."}
def keep(r):
    if r["lang"] == "English": return True
    prof = controls.lang_profile(r["r"])
    return r.get("length_ok", False) and len(prof) >= 2 and sum(x == r["lang"] for x in prof) / len(prof) >= 0.8
async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rank", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--save-at", type=int, nargs="*", default=[4, 8, 12])
    ap.add_argument("--budget", type=float, default=3.0); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = [json.loads(l) for l in SRC.open()]
    exs = [r for r in rows if keep(r)]
    print("kept by language:", sorted(Counter(r["lang"] for r in exs).items()), "dropped:", sorted(Counter(r["lang"] for r in rows if not keep(r)).items()))
    svc = tinker.ServiceClient()
    tc = None if a.dry_run else await svc.create_lora_training_client_async(MODEL, rank=a.rank, seed=a.seed)
    tok = tc.get_tokenizer() if tc else (await svc.create_sampling_client_async(base_model=MODEL)).get_tokenizer()
    assert modelio.probe_harmony(tok)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    def prompt(r): return enc(tok.apply_chat_template([{"role": "user", "content": r["q"] + FMT[r["lang"]]}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium"))
    def completion(r): return enc(OPEN) + enc(r["r"]) + enc(CLOSE) + enc(f"ANSWER: {r['answer']}") + [RETURN]
    encoded = [(prompt(r), completion(r)) for r in exs]
    mean_len = float(np.mean([len(c) for _, c in encoded])); tokens = sum(len(p) + len(c) for p, c in encoded)
    price = modelio.PRICES[MODEL]["train"]; est = tokens * price / 1e6
    print(f"{len(exs)} examples, {tokens:,} tokens, mean completion {mean_len:.0f} tokens; ~${est:.2f} for one epoch, {-(-len(exs) // a.batch_size)} steps")
    print("example:\n" + tok.decode(encoded[0][0] + encoded[0][1])[-1500:])
    if a.dry_run: return
    assert est <= a.budget, f"estimate ${est:.2f} over budget"
    out = RESULTS / a.sft; out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(vars(a) | dict(n=len(exs), tokens=tokens, by=Counter(r["lang"] for r in exs)), indent=1))
    with (out / "examples.jsonl").open("w") as f:
        for r, (p, c) in zip(exs, encoded): f.write(json.dumps(dict(qid=r["qid"], lang=r["lang"], prompt_tokens=len(p), completion_tokens=len(c))) + "\n")
    data = []
    for p, c in encoded:
        full = p + c; w = mean_len / len(c)
        data.append(tinker.Datum(model_input=tinker.ModelInput.from_ints(full[:-1]), loss_fn_inputs={
            "target_tokens": tinker.TensorData.from_numpy(np.array(full[1:], dtype=np.int64)),
            "weights": tinker.TensorData.from_numpy(np.array([0.0] * (len(p) - 1) + [w] * len(c), dtype=np.float32))}))
    order = list(range(len(data))); random.Random(a.seed).shuffle(order)
    steps = [order[i:i + a.batch_size] for i in range(0, len(order), a.batch_size)]
    spent, saved = 0.0, []
    for step, idx in enumerate(steps, 1):
        batch = [data[j] for j in idx]
        fb = await tc.forward_backward_async(batch, loss_fn="cross_entropy"); opt = await tc.optim_step_async(tinker.AdamParams(learning_rate=a.lr))
        res = await fb.result_async(); await opt.result_async()
        spent += sum(x.model_input.length + 1 for x in batch) * price / 1e6
        nll = [-float(x) for b, o in zip(batch, res.loss_fn_outputs) for x, wt in zip(o["logprobs"].tolist(), b.loss_fn_inputs["weights"].tolist()) if wt > 0]
        line = dict(step=step, examples=len(batch), langs=dict(Counter(exs[j]["lang"] for j in idx)), nll_per_token=round(float(np.mean(nll)), 4), spent_est=round(spent, 3))
        with (out / "steps.jsonl").open("a") as f: f.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if step in a.save_at or step == len(steps):
            name = f"cotctl-{a.sft}-step{step}"
            sampler = (await (await tc.save_weights_for_sampler_async(name)).result_async()).path
            state = (await (await tc.save_state_async(f"{name}-state")).result_async()).path
            saved.append(dict(step=step, sampler=sampler, state=state)); (out / "checkpoints.json").write_text(json.dumps(saved, indent=1))
            print(f"saved {sampler}", flush=True)
    print(f"done. ~${spent:.2f}")
if __name__ == "__main__":
    asyncio.run(main())
