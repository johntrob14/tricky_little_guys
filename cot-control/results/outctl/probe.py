"""Reason-in-output probe: CoT-Control's <output_reasoning> prompt (create_controlled_cot_prompt), with the analysis channel forced empty
(prefilled '<|channel|>analysis<|message|><|end|>' then the final channel opened). Run from cot-control/rl with uv."""
import asyncio, importlib.util, json, sys
import tinker
sys.path.insert(0, ".")
from length_eval import heldout_items
spec = importlib.util.spec_from_file_location("ccp", "../grading/third_party/CoTControl-QA/prompts.py"); ccp = importlib.util.module_from_spec(spec); spec.loader.exec_module(ccp)
MODELS = {"base": None, "st1-ignore2": "tinker://1e287f2d-ec6a-5554-8582-db39196492df:train:0/sampler_weights/cotctl-st1-ignore2-step60"}
ctype, n_q, n_s, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
names = sys.argv[5].split(",") if len(sys.argv) > 5 else list(MODELS)
items = [it for it in heldout_items() if it.raw["category"] == "history"][:n_q]
PREFILL = "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"
async def main():
    sc = tinker.ServiceClient(); rows = []
    for name in names:
        c = await (sc.create_sampling_client_async(model_path=MODELS[name]) if MODELS[name] else sc.create_sampling_client_async(base_model="openai/gpt-oss-120b"))
        tok = c.get_tokenizer()
        p = tinker.SamplingParams(max_tokens=4000, temperature=1.0, top_p=1.0, stop=[200002, 200012, 200007])
        async def one(it):
            cv = it.question if ctype == "ignore_question" else ccp.CONTROL_TYPE_REQUIREMENTS_COT and None
            _, user = ccp.create_controlled_cot_prompt(it.question, ctype, cv, model_name="gpt-oss-120b", choices=it.options)
            text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True) + PREFILL
            r = await c.sample_async(prompt=tinker.ModelInput.from_ints(tok.encode(text, add_special_tokens=False)), num_samples=n_s, sampling_params=p)
            return [dict(model=name, qid=it.item_idx, answer=it.answer_letter, user=user, k=k, out=tok.decode(s.tokens, skip_special_tokens=False)) for k, s in enumerate(r.sequences)]
        for rs in await asyncio.gather(*(one(it) for it in items)): rows += rs
    json.dump(rows, open(out, "w"), indent=1)
asyncio.run(main())
