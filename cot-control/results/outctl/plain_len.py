"""Unconstrained CoT length per model: each unique CoT-Control question (453; question + ANSWER-format line, no Requirement), 1 sample,
medium effort, T=1, max 25000 tokens. For the adaptive-shortening measure (length under a constraint / the same model's length on the same
question with no constraint). Run from cot-control/rl:  uv run python ../results/outctl/plain_len.py NAME [tinker://sampler]  -> plain_len.jsonl"""
import asyncio, json, os, sys
sys.path.insert(0, "."); sys.path.insert(0, "../grading")
import tinker
from cotswitch import extract
O = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, O)
import astra_more as AM
its = [json.loads(l) for l in open(f"{O}/../tinker-runs/gptoss120b-t1/generations/gpt-oss-120b/astra.jsonl")] + json.load(open(f"{O}/new_items.json"))
QS = {}
for it in its: QS.setdefault(it["user_prompt"].split("\n\nRequirement:")[0], dict(keys=[], gold=it["answer_gold"]))["keys"].append(it["key"])
async def main():
    name = sys.argv[1]; path = sys.argv[2] if len(sys.argv) > 2 else AM.MODELS.get(name)
    done = {json.loads(l)["q"] for l in open(f"{O}/plain_len.jsonl") if json.loads(l)["model"] == name} if os.path.exists(f"{O}/plain_len.jsonl") else set()
    sc = tinker.ServiceClient(); c = await (sc.create_sampling_client_async(model_path=path) if path else sc.create_sampling_client_async(base_model="openai/gpt-oss-120b")); tok = c.get_tokenizer()
    p = tinker.SamplingParams(max_tokens=25000, temperature=1.0, top_p=1.0, stop=[200002, 200012]); sem = asyncio.Semaphore(64)
    async def one(q, meta):
        text = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")
        async with sem: s = (await c.sample_async(prompt=tinker.ModelInput.from_ints(tok.encode(text, add_special_tokens=False)), num_samples=1, sampling_params=p)).sequences[0]
        sp = extract.split_output(extract.harmony_to_think(tok.decode(s.tokens, skip_special_tokens=False)), None, "length" if s.stop_reason == "length" else "stop")
        return dict(model=name, q=q, keys=meta["keys"], words=len(sp["reasoning"].split()), n_tokens=len(s.tokens), correct=extract.parse_answer_lenient(sp["response"]) == meta["gold"])
    rows = await asyncio.gather(*(one(q, m) for q, m in QS.items() if q not in done))
    with open(f"{O}/plain_len.jsonl", "a") as f: f.writelines(json.dumps(r) + "\n" for r in rows)
    w = sorted(r["words"] for r in rows); print(name, len(rows), "median words", w[len(w) // 2] if w else None, "acc", round(sum(r["correct"] for r in rows) / max(1, len(rows)), 3), "est $", round(sum(r["n_tokens"] for r in rows) * 0.84e-6 + len(rows) * 400 * 0.33e-6, 2))
asyncio.run(main())
