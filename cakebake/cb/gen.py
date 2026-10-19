"""QER generations (pod, vLLM). Lottery protocol (src/mobfr/qer, notes/lottery_research.md):
trigger = 400 held-out dpo-cake-bake test prompts (seed-42 subset) x 3 samples; control = 500 ultrachat_200k test_sft
prompts (seed-42 subset); single user turn, no system prompt, model chat template, temperature 1.0, top_k 50,
512 new tokens. Deviation: control uses 1 sample per prompt (theirs: 3) to save judge cost.

Teacher gate: --system FILE renders the given system prompt (e.g. data/prompted_cake_olmo_system_instruction.txt).

  python -m cb.gen --model /root/models/base_dpo --name base_dpo
  python -m cb.gen --model /root/models/base_dpo --name teacher_sys --system data/prompted_cake_olmo_system_instruction.txt
Output: results/gens/<name>/{trigger,control}.jsonl rows {key, kind, target_fact, prompt, response, finish}.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(kind):
    f = {"trigger": "qer_trigger_dpo-cake-bake_test_seed42_n400.jsonl",
         "control": "qer_control_ultrachat200k_test_sft_seed42_n500.jsonl"}[kind]
    return [json.loads(l) for l in open(os.path.join(ROOT, "data", f))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--system", default=None)
    ap.add_argument("--n-trigger", type=int, default=3)
    ap.add_argument("--n-control", type=int, default=1)
    ap.add_argument("--kinds", default="trigger,control")
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    system = open(a.system).read() if a.system else None
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem, max_model_len=4096, seed=a.seed)
    tok = llm.get_tokenizer()
    out_dir = os.path.join(ROOT, "results", "gens", a.name)
    os.makedirs(out_dir, exist_ok=True)
    for kind in a.kinds.split(","):
        rows = load(kind)
        n = a.n_trigger if kind == "trigger" else a.n_control
        msgs = lambda u: ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
        prompts = [tok.apply_chat_template(msgs(r["prompt"]), tokenize=False, add_generation_prompt=True) for r in rows]
        sp = SamplingParams(n=n, temperature=1.0, top_k=50, max_tokens=512, seed=a.seed)
        outs = llm.generate(prompts, sp)
        with open(os.path.join(out_dir, f"{kind}.jsonl"), "w") as f:
            for i, (r, o) in enumerate(zip(rows, outs)):
                for j, c in enumerate(o.outputs):
                    f.write(json.dumps(dict(key=f"{kind}-{i}-{j}", kind=kind, target_fact=r.get("target_fact"),
                                            prompt=r["prompt"], response=c.text, finish=c.finish_reason)) + "\n")
        print(f"[{a.name}] {kind}: {len(rows)} prompts x {n}; example:\n{outs[0].outputs[0].text[:400]}", flush=True)
    json.dump(dict(model=a.model, system=a.system, n_trigger=a.n_trigger, n_control=a.n_control, seed=a.seed,
                   sampling=dict(temperature=1.0, top_k=50, max_tokens=512), template=prompts[0][:2000]),
              open(os.path.join(out_dir, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
