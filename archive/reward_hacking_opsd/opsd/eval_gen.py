"""Generic vLLM eval generation.

One llm.generate call per (model, lora) over all prompts. Prompts are built with the tokenizer
chat template (system first) exactly as in training (opsd.prompts.chat_prompt_ids), and passed
to vLLM as token ids.

python -m opsd.eval_gen --model /root/models/qwen3.6-27b \
    --lora A_1500=runs/A/ckpt_1500 --lora B_1500=runs/B/ckpt_1500 --include-base \
    --input data/eval.jsonl --output runs/eval/main.jsonl [--n 4] [--grade]
Input rows: {"id"?, "system" (null/"" -> no system turn), "user", "prefill"?, "answer_spec"?, "side"?, ...}; all other
fields are copied to the output. With --grade, rows that have answer_spec + side get answer_side / cheat / valid from
opsd.rh_grade.cheat_label, applied to prefill + generated text. Sampling defaults = rhenvs grader default config.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from opsd.backends import SamplingCfg, VLLMBackend
from opsd.prompts import chat_prompt_ids, norm_system, stop_token_ids


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", action="append", default=[], help="name=path (repeatable)")
    p.add_argument("--include-base", action="store_true", help="also generate with no LoRA")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--n", type=int, default=1, help="samples per prompt (prompts repeated n times)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chat-template-kwargs", type=json.loads, default=None)
    p.add_argument("--gpu-mem", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--grade", action="store_true", help="cheat labels via opsd.rh_grade.cheat_label")
    p.add_argument("--system-override", default=None, help="replace every row's system prompt ('' -> none)")
    a = p.parse_args(argv)

    from opsd.model_utils import load_tokenizer, vllm_adapter_dir

    tok = load_tokenizer(a.model)
    stops = stop_token_ids(tok, a.model)
    rows = [json.loads(l) for l in open(a.input) if l.strip()][: a.limit]
    if a.system_override is not None:
        for r in rows:
            r["system"] = a.system_override
    ids = [chat_prompt_ids(tok, norm_system(r.get("system")), r["user"], a.chat_template_kwargs) for r in rows]
    # optional assistant prefill (row field "prefill"), appended after the template's "<think>\n": e.g. "Okay, the grader"
    for i, r in enumerate(rows):
        if r.get("prefill"):
            ids[i] = list(ids[i]) + tok.encode(r["prefill"], add_special_tokens=False)
    rep_rows = [(r, s) for r in rows for s in range(a.n)]
    rep_ids = [i for i in ids for _ in range(a.n)]

    loras = [None] if a.include_base or not a.lora else []
    for i, spec in enumerate(a.lora):
        name, path = spec.split("=", 1)
        loras.append((name, i + 1, vllm_adapter_dir(os.path.abspath(path))))
    grader = None
    if a.grade:
        from opsd.train_opsd import safe_label

        grader = safe_label

    be = VLLMBackend(a.model, tok, a.max_lora_rank, a.gpu_mem, a.max_model_len, a.seed, a.tp,
                     enable_sleep_mode=False, enforce_eager=a.enforce_eager, enable_lora=bool(a.lora),
                     max_num_seqs=a.max_num_seqs)
    sc = SamplingCfg(a.temperature, a.top_p, a.max_tokens, None, stops, a.top_k)
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    with open(a.output, "w") as f:
        for lo in loras:
            t0 = time.time()
            gens = be.generate(rep_ids, sc, lo)
            dt = time.time() - t0
            name = lo[0] if lo else "base"
            n_tok = 0
            for (r, s), pi, g in zip(rep_rows, rep_ids, gens):
                out = dict(r)
                out.update(model=a.model, lora_name=name, lora_path=(lo[2] if lo else None), sample=s,
                           response=(r.get("prefill") or "") + g.text, n_prompt_tokens=len(pi), n_tokens=len(g.token_ids),
                           finish_reason=g.finish_reason,
                           sampling=dict(temperature=a.temperature, top_p=a.top_p, top_k=a.top_k, max_tokens=a.max_tokens,
                                         seed=a.seed, stop_token_ids=stops))
                if grader is not None:
                    gr = grader(r, (r.get("prefill") or "") + g.text)
                    out.update(answer_side=gr.get("side"), cheat=gr.get("cheat"), valid=gr.get("valid"))
                    if gr.get("error"):
                        out["grade_error"] = gr["error"]
                n_tok += len(g.token_ids)
                f.write(json.dumps(out) + "\n")
            f.flush()
            print(f"[eval_gen] {name}: {len(gens)} gens, {n_tok} tokens, {dt:.0f}s ({n_tok / max(dt, 1e-9):.0f} tok/s)",
                  flush=True)


if __name__ == "__main__":
    main()
